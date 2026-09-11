"""评测脚本：基线（纯向量） vs 优化（词表扩展+混合检索+Rerank）对比

用法：
  python scripts/run_eval.py                # 跑全量评测题，输出对比表
  python scripts/run_eval.py --quick        # 只跑前 5 题，快速验证

产出：
  eval/results/<时间戳>/baseline.json / optimized.json
  eval/results/<时间戳>/report.md          # 对比实验表（答辩硬通货）
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import ROOT, load_config
from core.embeddings import Embedder
from core.loader import load_directory
from core.chunker import chunk_document
from core.vector_store import VectorStore
from core.retriever import Retriever

cfg = load_config()


def build_store() -> VectorStore:
    """评测用独立临时库（不污染正式知识库）"""
    kb = Path("/tmp/luzhou_eval_kb")
    store = VectorStore(kb)
    if store.load():
        return store
    docs = load_directory(ROOT / cfg["data"]["raw_dir"])
    chunk_cfg = cfg["chunking"]
    chunks = []
    for title, text, _ in docs:
        chunks.extend(chunk_document(title, text, chunk_cfg["max_chars"], chunk_cfg["overlap"]))
    embedder = Embedder(cfg["embedding"])
    store.add(chunks, embedder.embed([c["text"] for c in chunks]))
    store.save()
    return store


def hit_at_k(question: str, expect: list[str], hits: list[dict], k: int) -> bool:
    """Top-k 内是否命中期望主题（用期望关键词做近似判断）"""
    top_texts = [h["text"] + h["doc"] for h in hits[:k]]
    joined = "\n".join(top_texts)
    return any(e in joined for e in expect)


def mrr(question: str, expect: list[str], hits: list[dict], k: int) -> float:
    """首个命中结果排名的倒数（0 = 未命中），对排序质量更敏感"""
    for rank, h in enumerate(hits[:k], start=1):
        joined = h["text"] + h["doc"]
        if any(e in joined for e in expect):
            return 1.0 / rank
    return 0.0


def run(mode: str, questions: list[dict], store: VectorStore, embedder: Embedder):
    r_cfg = dict(cfg)
    r_cfg["retrieval"] = {**cfg["retrieval"], "baseline": (mode == "baseline")}
    retriever = Retriever(r_cfg, store, embedder)
    rows = []
    for q in questions:
        hits = retriever.retrieve(q["question"], cfg["retrieval"]["top_k"])
        rows.append({
            "question": q["question"],
            "hit@5": hit_at_k(q["question"], q["expect"], hits, 5),
            "hit@10": hit_at_k(q["question"], q["expect"], hits, 10),
            "mrr": mrr(q["question"], q["expect"], hits, cfg["retrieval"]["top_k"]),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="只跑前 5 题")
    args = parser.parse_args()

    questions = json.loads((Path(__file__).resolve().parent.parent / "eval" / "questions.json").read_text("utf-8"))
    if args.quick:
        questions = questions[:5]

    print("构建评测知识库…")
    store = build_store()
    embedder = Embedder(cfg["embedding"])

    print(f"跑 {len(questions)} 题：基线 vs 优化（首次会下载模型，较慢）…")
    t0 = time.time()
    base = run("baseline", questions, store, embedder)
    opt = run("optimized", questions, store, embedder)

    out_dir = ROOT / "eval" / "results" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline.json").write_text(json.dumps(base, ensure_ascii=False, indent=2), "utf-8")
    (out_dir / "optimized.json").write_text(json.dumps(opt, ensure_ascii=False, indent=2), "utf-8")

    b5 = sum(r["hit@5"] for r in base) / len(base)
    o5 = sum(r["hit@5"] for r in opt) / len(opt)
    b10 = sum(r["hit@10"] for r in base) / len(base)
    o10 = sum(r["hit@10"] for r in opt) / len(opt)
    bmrr = sum(r["mrr"] for r in base) / len(base)
    omrr = sum(r["mrr"] for r in opt) / len(opt)

    report = f"""# 徽州问典 · 红色文化检索优化对比实验（{len(questions)} 题，耗时 {time.time()-t0:.0f}s）

| 指标 | 基线（纯向量） | 优化（词表扩展+混合检索+Rerank） | 提升 |
|---|---|---|---|
| Hit@5 | {b5:.1%} | {o5:.1%} | {o5-b5:+.1%} |
| Hit@10 | {b10:.1%} | {o10:.1%} | {o10-b10:+.1%} |
| MRR | {bmrr:.3f} | {omrr:.3f} | {omrr-bmrr:+.3f} |

> MRR（平均倒数排名）衡量首个正确结果的排名位置，对 Reranker 的排序增益更敏感。

## 逐题明细
| 问题 | 基线 Hit@5 | 优化 Hit@5 |
|---|---|---|
"""
    for b, o in zip(base, opt):
        report += f"| {b['question']} | {'✅' if b['hit@5'] else '❌'} | {'✅' if o['hit@5'] else '❌'} |\n"
    (out_dir / "report.md").write_text(report, "utf-8")
    print(f"\n完成。结果与对比表已保存到 {out_dir}")
    print(f"Hit@5: 基线 {b5:.1%} -> 优化 {o5:.1%}（{o5-b5:+.1%}）")
    print(f"Hit@10: 基线 {b10:.1%} -> 优化 {o10:.1%}（{o10-b10:+.1%}）")


if __name__ == "__main__":
    main()
