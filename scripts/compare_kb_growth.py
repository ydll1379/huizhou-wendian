"""Compare the current corpus snapshot with a rebuilt corpus on identical queries."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import ROOT, load_config
from core.embeddings import Embedder
from core.vector_store import VectorStore
from scripts.run_retrieval_benchmark import (
    OOD_QUESTIONS,
    TOP_K,
    build_queries,
    metric,
    ranked_relevance,
    ranker,
)


def load_store(path: Path) -> VectorStore:
    store = VectorStore(path)
    if not store.load():
        raise SystemExit(f"无法从 {path} 加载知识库快照")
    return store


def kb_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("chunks.jsonl", "embeddings.npy"):
        with (path / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def aggregate(rows: list[dict], system: str, corpus: str) -> dict:
    subset = [row for row in rows if row["system"] == system and row["corpus"] == corpus]
    if not subset:
        return {"n": 0, "hit@5": 0, "mrr@12": 0, "ndcg@10": 0, "recall@10": 0}
    return {
        "n": len(subset),
        "hit@5": sum(row["hit@5"] for row in subset) / len(subset),
        "mrr@12": sum(row["mrr@12"] for row in subset) / len(subset),
        "ndcg@10": sum(row["ndcg@10"] for row in subset) / len(subset),
        "recall@10": sum(row["recall@10"] for row in subset) / len(subset),
    }


def main() -> None:
    started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True, help="建库前知识库目录快照")
    parser.add_argument("--augmented", type=Path, default=None, help="建库后知识库目录，默认 data/kb")
    args = parser.parse_args()

    cfg = load_config()
    augmented_path = args.augmented or ROOT / cfg["data"]["kb_dir"]
    baseline = load_store(args.baseline)
    augmented = load_store(augmented_path)
    legacy = json.loads((ROOT / "eval/questions.json").read_text("utf-8"))
    legacy += json.loads((ROOT / "eval/alias_challenge.json").read_text("utf-8"))
    new_materials = json.loads((ROOT / "eval/jianghuai_materials.json").read_text("utf-8"))
    terms = cfg.get("domain_terms", {})
    queries = build_queries(legacy + new_materials, terms)
    query_group = {}
    for n, item in enumerate(legacy + new_materials, 1):
        group = "new_materials" if n > len(legacy) else "existing_eval"
        query_group[f"q{n:03d}"] = group

    embedder = Embedder(cfg["embedding"])
    texts = list(dict.fromkeys(item["question"] for item in queries))
    vectors = {}
    for start in range(0, len(texts), 32):
        batch = texts[start:start + 32]
        vectors.update(zip(batch, embedder.embed(batch)))

    systems = ["dense", "bm25", "hybrid", "hybrid_alias"]
    cfg["retrieval"] = {**cfg["retrieval"], "top_k": TOP_K}
    corpora = {"before": baseline, "after": augmented}
    runners = {
        (corpus, system): ranker(system, cfg, store, embedder)
        for corpus, store in corpora.items()
        for system in systems
    }
    rows = []
    for index, query in enumerate(queries, 1):
        for corpus, store in corpora.items():
            for system in systems:
                hits = runners[(corpus, system)](query["question"], vectors[query["question"]])
                reciprocal_rank, first_rank = metric(hits, query["expect"])
                ndcg, recall, relevant_count = ranked_relevance(hits, query["expect"], store.chunks)
                rows.append({
                    "corpus": corpus,
                    "group": query_group[query["intent"]],
                    "intent": query["intent"],
                    "topic": query["topic"],
                    "variant": query["variant"],
                    "question": query["question"],
                    "system": system,
                    "first_hit_rank": first_rank,
                    "hit@5": first_rank is not None and first_rank <= 5,
                    "mrr@12": reciprocal_rank,
                    "ndcg@10": ndcg,
                    "recall@10": recall,
                    "relevant_chunks": relevant_count,
                    "top_doc": hits[0].get("doc", "") if hits else "",
                })
        if index % 50 == 0:
            print(f"queries {index}/{len(queries)}", flush=True)

    threshold = cfg["retrieval"].get("refuse_threshold", 0.35)
    ood_vectors = embedder.embed(OOD_QUESTIONS)
    ood_rows = []
    for question, vector in zip(OOD_QUESTIONS, ood_vectors):
        item = {"question": question}
        for corpus, store in corpora.items():
            hits = store.search(vector, TOP_K)
            score = hits[0][1] if hits else 0.0
            item[f"{corpus}_top_dense_score"] = score
            item[f"{corpus}_passes_threshold"] = score >= threshold
        ood_rows.append(item)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = ROOT / "eval" / "results" / f"corpus-growth-{timestamp}"
    output_dir.mkdir(parents=True)
    result = {
        "created_at": timestamp,
        "baseline_kb_sha256": kb_hash(args.baseline),
        "augmented_kb_sha256": kb_hash(augmented_path),
        "baseline_chunks": baseline.size,
        "augmented_chunks": augmented.size,
        "legacy_intents": len(legacy),
        "new_material_intents": len(new_materials),
        "query_variants": len(queries),
        "top_k": TOP_K,
        "duration_seconds": round(time.time() - started, 2),
        "systems": systems,
        "rows": rows,
        "ood_threshold": threshold,
        "ood_rows": ood_rows,
    }
    (output_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")

    def summary_table(group: str) -> list[str]:
        lines = [f"### {group}：相同问题的语料前后对照", "", "|检索方案|建库前 Hit@5|导入后 Hit@5|建库前 MRR@12|导入后 MRR@12|", "|---|---:|---:|---:|---:|"]
        for system in systems:
            before = aggregate([r for r in rows if r["group"] == group], system, "before")
            after = aggregate([r for r in rows if r["group"] == group], system, "after")
            lines.append(f"|{system}|{before['hit@5']:.1%}|{after['hit@5']:.1%}|{before['mrr@12']:.4f}|{after['mrr@12']:.4f}|")
        return lines

    def algorithm_table(group: str) -> list[str]:
        lines = [f"### 导入后语料：检索算法对照（{group}）", "", "|检索方案|查询数|Hit@5|MRR@12|nDCG@10|Recall@10|", "|---|---:|---:|---:|---:|---:|"]
        for system in systems:
            stats = aggregate([r for r in rows if r["group"] == group], system, "after")
            lines.append(f"|{system}|{stats['n']}|{stats['hit@5']:.1%}|{stats['mrr@12']:.4f}|{stats['ndcg@10']:.4f}|{stats['recall@10']:.4f}|")
        return lines

    report = [
        "# 徽州地区资料导入与检索对照", "",
        f"- 建库前分块：{baseline.size}；导入后分块：{augmented.size}。",
        f"- 旧评测意图：{len(legacy)}；新增资料专项问题：{len(new_materials)}；问题改写后共 {len(queries)} 条查询。",
        f"- Top-K={TOP_K}；运行时间 {result['duration_seconds']} 秒。",
        "- 相关性按评测题期望词在分块中的出现构造代理 qrels；Hit@5/MRR 使用期望词命中，nDCG/Recall 使用匹配分块数。它衡量检索证据覆盖与排序，不等于生成答案正确率。",
        "- 语料前后表只列 Hit@5/MRR，避免因导入后相关分块候选数量增加而改变 Recall/nDCG 分母；算法表只在同一导入后语料内比较，口径一致。",
        "- `dense`、`bm25`、`hybrid`、`hybrid_alias` 在两个语料上使用相同问题、Embedding 模型及 Top-K。没有可用的在线 WeKnora 服务，因此本次不把离线结果冒充跨产品对测。", "",
    ]
    report += summary_table("existing_eval") + [""]
    report += summary_table("new_materials") + [""]
    report += algorithm_table("existing_eval") + [""]
    report += algorithm_table("new_materials") + [""]
    report += [
        "### 知识库外问题门控检查", "",
        f"当前向量阈值为 {threshold:.2f}；表中比较扩库前后这组 {len(ood_rows)} 条已有知识库外问题的最高 dense 分数是否通过阈值。它只是风险筛查小样本，不是拒答准确率。", "",
        "|问题|导入前最高分|导入后最高分|前会通过|后会通过|", "|---|---:|---:|---|---|",
    ]
    for item in ood_rows:
        report.append(f"|{item['question']}|{item['before_top_dense_score']:.3f}|{item['after_top_dense_score']:.3f}|{'是' if item['before_passes_threshold'] else '否'}|{'是' if item['after_passes_threshold'] else '否'}|")
    report += ["", "### 数据完整性", "", f"- 建库前快照 SHA-256：`{result['baseline_kb_sha256']}`", f"- 导入后快照 SHA-256：`{result['augmented_kb_sha256']}`", "- 原 PDF 文本层为空的页面以 OCR 转录 Markdown 进入索引，原 PDF 保留在 `data/raw`；答案应优先打开原 PDF 核验数字和专名。", "- 安庆 HTML 是在线方志平台的目录页，仅有目录及 PDF 链接，不含 PDF 正文。本次只索引目录，不统计为完整安庆通览正文。", ""]
    (output_dir / "report.md").write_text("\n".join(report), "utf-8")
    print(f"RESULT_DIR {output_dir}")
    print(f"CHUNKS {baseline.size} -> {augmented.size}; INTENTS {len(legacy)} + {len(new_materials)}; QUERIES {len(queries)}")
    for group in ("existing_eval", "new_materials"):
        print(group)
        for system in systems:
            print(system, aggregate([r for r in rows if r["group"] == group], system, "before"), "->", aggregate([r for r in rows if r["group"] == group], system, "after"))


if __name__ == "__main__":
    main()
