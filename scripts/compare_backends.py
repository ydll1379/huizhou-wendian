"""本地检索 vs WeKnora 检索：同一批评测题的召回对比。

只比较召回结果，不调用生成模型，避免回答质量干扰。
用法：
    .venv/bin/python scripts/compare_backends.py --limit 5
    .venv/bin/python scripts/compare_backends.py
    .venv/bin/python scripts/compare_backends.py --top-k 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import ROOT, load_config
from core.embeddings import Embedder
from core.retriever import Retriever
from core.vector_store import VectorStore
from core.weknora import WeKnoraRetriever


def hit_at_k(expect: list[str], hits: list[dict], k: int) -> bool:
    joined = "\n".join((h.get("text", "") + str(h.get("doc", ""))) for h in hits[:k])
    return any(e in joined for e in expect)


def mrr(expect: list[str], hits: list[dict], k: int) -> float:
    for rank, h in enumerate(hits[:k], start=1):
        joined = h.get("text", "") + str(h.get("doc", ""))
        if any(e in joined for e in expect):
            return 1.0 / rank
    return 0.0


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "hit@5": 0.0, "hit@10": 0.0, "mrr": 0.0}
    n = len(rows)
    return {
        "count": n,
        "hit@5": sum(1 for r in rows if r["hit@5"]) / n,
        "hit@10": sum(1 for r in rows if r["hit@10"]) / n,
        "mrr": sum(r["mrr"] for r in rows) / n,
    }


def run_backend(name: str, retriever, questions: list[dict], top_k: int) -> list[dict]:
    rows = []
    for q in questions:
        hits = retriever.retrieve(q["question"], top_k)
        # 保留简版命中信息，便于人工复核
        compact_hits = [
            {
                "rank": i + 1,
                "score": h.get("score"),
                "dense_score": h.get("dense_score"),
                "doc": h.get("doc", ""),
                "title": h.get("title", ""),
                "text": (h.get("text", "") or "")[:160],
            }
            for i, h in enumerate(hits)
        ]
        rows.append({
            "question": q["question"],
            "expect": q["expect"],
            "hit@5": hit_at_k(q["expect"], hits, 5),
            "hit@10": hit_at_k(q["expect"], hits, 10),
            "mrr": mrr(q["expect"], hits, top_k),
            "hits": compact_hits,
        })
    return rows


def make_local_retriever(cfg: dict):
    store = VectorStore(ROOT / cfg["data"]["kb_dir"])
    if not store.load():
        raise RuntimeError("本地知识库不存在，请先运行 scripts/build_kb.py")
    embedder = Embedder(cfg["embedding"])
    return Retriever(cfg, store, embedder)


def make_weknora_retriever(cfg: dict):
    r = WeKnoraRetriever(cfg)
    if not r.ready:
        raise RuntimeError("WeKnora 未配置：请检查 .env 的 WEKNORA_* 和 config.yaml 的 weknora.knowledge_base_id")
    return r


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题")
    parser.add_argument("--top-k", type=int, default=0, help="覆盖 config.retrieval.top_k")
    args = parser.parse_args()

    cfg = load_config()
    top_k = args.top_k or int(cfg["retrieval"]["top_k"])
    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    print(f"题目数：{len(questions)}，top_k={top_k}")
    output = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "top_k": top_k,
        "count": len(questions),
        "backends": {},
    }

    print("加载本地检索器（bge-m3）…")
    local = make_local_retriever(cfg)
    print("跑本地召回…")
    output["backends"]["local"] = run_backend("local", local, questions, top_k)

    print("连接 WeKnora…")
    wk = make_weknora_retriever(cfg)
    print("跑 WeKnora 召回…")
    output["backends"]["weknora"] = run_backend("weknora", wk, questions, top_k)

    summary = {name: summarize(rows) for name, rows in output["backends"].items()}
    output["summary"] = summary

    out_dir = ROOT / "eval" / "results" / f"backend-compare-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 本地检索 vs WeKnora 召回对比",
        "",
        f"- 题目数：{len(questions)}",
        f"- top_k：{top_k}",
        f"- 生成时间：{output['generated_at']}",
        "",
        "| 指标 | 本地 | WeKnora | 差异 |",
        "|---|---:|---:|---:|",
    ]
    l, w = summary["local"], summary["weknora"]
    for key, label in [("hit@5", "Hit@5"), ("hit@10", "Hit@10"), ("mrr", "MRR")]:
        lines.append(f"| {label} | {l[key]:.3f} | {w[key]:.3f} | {w[key]-l[key]:+.3f} |")
    lines += [
        "",
        "## 逐题对比",
        "",
        "| 问题 | 本地 Hit@5 | WeKnora Hit@5 | 本地 MRR | WeKnora MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for rl, rw in zip(output["backends"]["local"], output["backends"]["weknora"]):
        lines.append(
            f"| {rl['question']} | {'✅' if rl['hit@5'] else '❌'} | "
            f"{'✅' if rw['hit@5'] else '❌'} | {rl['mrr']:.3f} | {rw['mrr']:.3f} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n完成：{out_dir}")
    print(f"本地    Hit@5={l['hit@5']:.1%} Hit@10={l['hit@10']:.1%} MRR={l['mrr']:.3f}")
    print(f"WeKnora Hit@5={w['hit@5']:.1%} Hit@10={w['hit@10']:.1%} MRR={w['mrr']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
