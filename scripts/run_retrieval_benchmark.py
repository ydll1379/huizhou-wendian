"""多条件检索基准：固定意图的规范问法、自然改写与领域别名问法。

指标使用 eval/questions.json 中的期望关键词作可复现的相关性代理，不衡量生成答案事实准确率。
每个原始问题派生的变体共享同一意图，报告会同时给出按意图分组和按 query 统计结果。
"""
from __future__ import annotations
import hashlib, json, sys, time, math
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import ROOT, load_config
from core.embeddings import Embedder
from core.vector_store import VectorStore
from core.retriever import Retriever, _tokenize, expand_query
from core.reranker import Reranker

TOP_K = 12
OOD_QUESTIONS = [
    "明天合肥天气怎么样？", "今天合肥地铁末班车几点？", "怎样申请护照？",
    "今年安徽高考录取分数线是多少？", "附近哪家餐厅现在还营业？",
    "新能源汽车电池怎么保养？", "2026年春节火车票什么时候开售？",
    "如何办理营业执照？", "合肥市现在的房价是多少？", "明天渡江战役纪念馆几点开门？",
    "帮我写一首关于秋天的诗。", "最近有哪些电影上映？",
]


def variants(q):
    out=[("original",q)]
    out.append(("context_prefix", "请根据可查资料回答："+q))
    out.append(("conversational", "我想了解一下，"+q))
    paraphrase=q
    swaps=[("是什么？","指的是什么？"),("是什么？","具体是怎么回事？"),
           ("在哪里？","具体位于什么地方？"),("在哪个镇？","具体在哪个镇？"),
           ("是什么时候发生的？","发生时间是什么？"),("是什么时候？","具体时间是何时？"),
           ("有什么特色？","有哪些特点？"),("为什么有名？","出名的原因是什么？"),
           ("为什么重要？","重要意义在哪里？"),("是怎么形成的？","形成原因是什么？"),
           ("有什么关系？","两者之间是什么关系？"),("是什么样的？","具体是什么样子？"),
           ("有哪些？","具体包括哪些？"),("在哪里？","位于何处？")]
    for a,b in swaps:
        if a in q:
            paraphrase=q.replace(a,b,1); break
    if paraphrase==q:
        paraphrase="能否介绍一下："+q
    out.append(("paraphrase",paraphrase))
    return out


def alias_variant(q, terms):
    # Limit synthetic alias changes to unique named entities; broad city names are excluded.
    excluded={"合肥","阜阳","淮南","芜湖","新四军","金寨","小岗村"}
    for canonical, aliases in terms.items():
        if canonical in excluded: continue
        if canonical in q and aliases:
            return q.replace(canonical, aliases[0], 1)
        for alias in aliases:
            if len(alias)>=3 and alias in q:
                return q.replace(alias, canonical, 1)
    return None


def build_queries(questions, terms):
    allq=[]
    for n,item in enumerate(questions):
        intent=f"q{n+1:03d}"
        for kind,text in variants(item["question"]):
            allq.append({"intent":intent,"variant":kind,"question":text,"expect":item["expect"],"topic":item.get("topic","")})
        alt=alias_variant(item["question"],terms)
        if alt and alt!=item["question"]:
            allq.append({"intent":intent,"variant":"domain_alias","question":alt,"expect":item["expect"],"topic":item.get("topic","")})
    return allq


def metric(hits, expect):
    for rank,h in enumerate(hits[:TOP_K],1):
        blob=h.get("text","")+h.get("doc","")+h.get("title","")
        if any(e in blob for e in expect): return 1/rank,rank
    return 0.0,None


def ranked_relevance(hits, expect, chunks):
    """BEIR-style ranking metrics using explicit expected-term proxy qrels."""
    relevant={i for i,c in enumerate(chunks)
              if any(term in c.get("text","")+c.get("doc","")+c.get("title","") for term in expect)}
    ranked=[h.get("idx") for h in hits[:10]]
    if not relevant: return 0.0,0.0,0
    rel=[1 if idx in relevant else 0 for idx in ranked]
    dcg=sum(r/(math.log2(pos+2)) for pos,r in enumerate(rel))
    ideal=sum(1/math.log2(pos+2) for pos in range(min(len(relevant),10)))
    return dcg/ideal if ideal else 0.0, sum(rel)/len(relevant), len(relevant)


def ranker(kind,cfg,store,embedder,reranker=None):
    ret=Retriever(cfg,store,embedder)
    bm=ret._build_bm25()
    def run(query, qemb):
        dense=store.search(qemb,TOP_K)
        dmap={i:s for i,s in dense}
        if kind=="dense":
            idxs=[i for i,_ in dense]
        else:
            query_for_bm=query
            if kind in {"hybrid_alias","hybrid_alias_rerank"}:
                query_for_bm=expand_query(query,cfg.get("domain_terms",{}))
            scores=bm.get_scores(_tokenize(query_for_bm))
            sparse=sorted(range(len(scores)),key=lambda i:-scores[i])[:TOP_K]
            if kind=="bm25": idxs=sparse
            else:
                fused={}
                for rank,(i,_) in enumerate(dense): fused[i]=fused.get(i,0)+cfg["retrieval"]["dense_weight"]/(60+rank)
                for rank,i in enumerate(sparse): fused[i]=fused.get(i,0)+cfg["retrieval"]["bm25_weight"]/(60+rank)
                idxs=[i for i,_ in sorted(fused.items(),key=lambda x:-x[1])[:TOP_K]]
        hits=[{**store.get(i),"idx":i,"dense_score":dmap.get(i,0)} for i in idxs]
        if kind=="hybrid_alias_rerank": hits=reranker.rerank(query,hits,TOP_K)
        return hits
    return run


def summarize(rows,systems):
    result={}
    for system in systems:
        rr=[x for x in rows if x["system"]==system]
        d={}
        for group in sorted({x["variant"] for x in rr}):
            g=[x for x in rr if x["variant"]==group]
            d[group]={"n":len(g),"hit@5":sum(x["hit@5"] for x in g)/len(g),
                      "hit@10":sum(x["hit@10"] for x in g)/len(g),
                      "mrr@12":sum(x["mrr@12"] for x in g)/len(g),
                      "ndcg@10":sum(x["ndcg@10"] for x in g)/len(g),
                      "recall@10":sum(x["recall@10"] for x in g)/len(g)}
        d["overall"]={"n":len(rr),"hit@5":sum(x["hit@5"] for x in rr)/len(rr),
                      "hit@10":sum(x["hit@10"] for x in rr)/len(rr),
                      "mrr@12":sum(x["mrr@12"] for x in rr)/len(rr),
                      "ndcg@10":sum(x["ndcg@10"] for x in rr)/len(rr),
                      "recall@10":sum(x["recall@10"] for x in rr)/len(rr)}
        result[system]=d
    return result


def main():
    cfg=load_config(); base_questions=json.loads((ROOT/"eval/questions.json").read_text("utf-8"))
    alias_questions=json.loads((ROOT/"eval/alias_challenge.json").read_text("utf-8"))
    questions=base_questions+alias_questions
    qs=build_queries(questions,cfg.get("domain_terms",{}))
    kb=VectorStore(ROOT/cfg["data"]["kb_dir"])
    if not kb.load(): raise SystemExit("data/kb 无法加载；请先运行 scripts/build_kb.py")
    embedder=Embedder(cfg["embedding"])
    systems=["dense","bm25","hybrid","hybrid_alias"]
    cfg["retrieval"]={**cfg["retrieval"],"top_k":TOP_K}
    runners={s:ranker(s,cfg,kb,embedder) for s in systems}
    rows=[]; start=time.time()
    # Encode each unique query once, in batches, then share the same vector across all methods.
    query_texts=list(dict.fromkeys(q["question"] for q in qs))
    qembs={}
    for b in range(0,len(query_texts),32):
        vectors=embedder.embed(query_texts[b:b+32])
        qembs.update(zip(query_texts[b:b+32],vectors))
    for qi,q in enumerate(qs,1):
        for system in systems:
            hits=runners[system](q["question"],qembs[q["question"]])
            m,rank=metric(hits,q["expect"])
            ndcg,recall,relevant_count=ranked_relevance(hits,q["expect"],kb.chunks)
            rows.append({"intent":q["intent"],"variant":q["variant"],"question":q["question"],
                         "topic":q["topic"],"system":system,"mrr@12":m,"ndcg@10":ndcg,"recall@10":recall,"qrels_count":relevant_count,"first_hit_rank":rank,
                         "hit@5":rank is not None and rank<=5,"hit@10":rank is not None and rank<=10,
                         "top_doc":hits[0].get("doc","") if hits else ""})
        if qi%50==0: print(f"retrieval queries {qi}/{len(qs)}",flush=True)
    metrics=summarize(rows,systems)
    # Evaluate the cross-encoder on a separate stratified sample for practical CPU runtime.
    rerank_qs=[q for q in qs if q["variant"]=="original" and int(q["intent"][1:])%2==0]
    rerank_qs += [q for q in qs if q["variant"]=="domain_alias"]
    reranker=Reranker(cfg["reranker"])
    alias_runner=ranker("hybrid_alias",cfg,kb,embedder)
    before=[]; after=[]
    for qi,q in enumerate(rerank_qs,1):
        hits=alias_runner(q["question"],qembs[q["question"]])
        reranked=reranker.rerank(q["question"],hits,TOP_K)
        bm,br=metric(hits,q["expect"]); rm,rrank=metric(reranked,q["expect"])
        before.append({"intent":q["intent"],"variant":q["variant"],"mrr":bm,"rank":br})
        after.append({"intent":q["intent"],"variant":q["variant"],"mrr":rm,"rank":rrank})
        if qi%10==0: print(f"reranker sample {qi}/{len(rerank_qs)}",flush=True)
    rerank_metrics={
        "n":len(rerank_qs),
        "hybrid_alias_mrr@12":sum(x["mrr"] for x in before)/len(before),
        "hybrid_alias_rerank_mrr@12":sum(x["mrr"] for x in after)/len(after),
        "hybrid_alias_hit@5":sum(x["rank"] is not None and x["rank"]<=5 for x in before)/len(before),
        "hybrid_alias_rerank_hit@5":sum(x["rank"] is not None and x["rank"]<=5 for x in after)/len(after),
        "paired_mrr_wins":sum(y["mrr"]>x["mrr"] for x,y in zip(before,after)),
        "paired_mrr_ties":sum(y["mrr"]==x["mrr"] for x,y in zip(before,after)),
        "paired_mrr_losses":sum(y["mrr"]<x["mrr"] for x,y in zip(before,after)),
    }
    # Separate out-of-domain refusal screening at the current configured cosine threshold.
    threshold=cfg["retrieval"].get("refuse_threshold",0.35); ood=[]
    ood_vecs=embedder.embed(OOD_QUESTIONS)
    for question,emb in zip(OOD_QUESTIONS,ood_vecs):
        hits=kb.search(emb,TOP_K)
        score=hits[0][1] if hits else 0
        ood.append({"question":question,"top_dense_score":float(score),"would_pass_current_threshold":bool(score>=threshold)})
    timestamp=time.strftime("%Y%m%d-%H%M%S")
    out=ROOT/"eval"/"results"/f"benchmark-{timestamp}"; out.mkdir(parents=True)
    payload={"created_at":timestamp,"source_questions":len(questions),"base_questions":len(base_questions),"alias_challenge_questions":len(alias_questions),"query_variants":len(qs),
      "unique_intents":len(questions),"top_k":TOP_K,"duration_seconds":round(time.time()-start,2),
      "knowledge_chunks":len(kb.chunks),"systems":metrics,"reranker_subset":rerank_metrics,
      "ood_screening":{"n":len(ood),"threshold":threshold,"items":ood},"rows":rows}
    (out/"benchmark.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),"utf-8")
    def pct(x): return f"{x:.1%}"
    lines=[f"# 徽州问典检索多条件对比（{len(qs)} 条查询 / {len(questions)} 个基础意图）","",
      f"- 时间：{timestamp}；知识库片段：{len(kb.chunks)}；Top-K：{TOP_K}；耗时：{payload['duration_seconds']} 秒。",
      f"- 查询由 {len(base_questions)} 个既有问题与 {len(alias_questions)} 个手工别名挑战题派生；同一意图内改写不是独立样本。",
      "- 相关性判定：用期望关键词在知识片段上构造代理 qrels；Hit/MRR 看首个关键词命中，nDCG/Recall@10 按匹配片段计分。属于代理标注，不表示答案事实正确。",
      "- 混合排序为 BM25 与向量召回的 RRF；交叉编码器重排在单独分层子集上比较，避免把不同样本数混为一谈。","",
      "## 整体指标","","|方案|查询数|Hit@5|Hit@10|MRR@12|nDCG@10|Recall@10|","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in systems:
        m=metrics[s]["overall"]; lines.append(f"|{s}|{m['n']}|{pct(m['hit@5'])}|{pct(m['hit@10'])}|{m['mrr@12']:.4f}|{m['ndcg@10']:.4f}|{m['recall@10']:.4f}|")
    lines += ["","## 按查询类型拆分","","|方案|类型|n|Hit@5|MRR@12|nDCG@10|Recall@10|","|---|---|---:|---:|---:|---:|---:|"]
    for s in systems:
        for v,m in metrics[s].items():
            if v=="overall": continue
            lines.append(f"|{s}|{v}|{m['n']}|{pct(m['hit@5'])}|{m['mrr@12']:.4f}|{m['ndcg@10']:.4f}|{m['recall@10']:.4f}|")
    lines += ["","## 交叉编码器重排子集","",
      f"在 {rerank_metrics['n']} 条分层抽样查询上，混合检索+别名扩展 MRR@12={rerank_metrics['hybrid_alias_mrr@12']:.4f}、Hit@5={pct(rerank_metrics['hybrid_alias_hit@5'])}；加入 BGE cross-encoder 后 MRR@12={rerank_metrics['hybrid_alias_rerank_mrr@12']:.4f}、Hit@5={pct(rerank_metrics['hybrid_alias_rerank_hit@5'])}。逐题胜/平/负为 {rerank_metrics['paired_mrr_wins']}/{rerank_metrics['paired_mrr_ties']}/{rerank_metrics['paired_mrr_losses']}。该子集用于技术诊断，不与全量变体总体指标直接横比。","",
      "|意图|变体|重排前名次|重排后名次|","|---|---|---:|---:|"]
    for x,y in zip(before,after): lines.append(f"|{x['intent']}|{x['variant']}|{x['rank']}|{y['rank']}|")
    false=sum(x["would_pass_current_threshold"] for x in ood)
    lines += ["","## 知识库外问题阈值筛查","",f"当前阈值 {threshold:.2f} 下，{false}/{len(ood)} 条知识库外问题的最高向量分数仍会通过门控。该小样本仅用于发现拒答阈值风险，不用于宣称拒答准确率。","",
      "|问题|最高向量分数|当前阈值会放行|","|---|---:|---|"]
    for x in ood: lines.append(f"|{x['question']}|{x['top_dense_score']:.3f}|{'是' if x['would_pass_current_threshold'] else '否'}|")
    (out/"report.md").write_text("\n".join(lines)+"\n","utf-8")
    print("RESULT_DIR",out)
    for s in systems: print(s,metrics[s]["overall"])
    print("RERANK_SUBSET",rerank_metrics)
    print("OOD",false,"/",len(ood),"threshold",threshold)
    print("SECONDS",payload["duration_seconds"])

if __name__=="__main__": main()
