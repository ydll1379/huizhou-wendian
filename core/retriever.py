"""混合检索：BM25 + 向量（RRF 融合）+ 领域词表查询扩展

baseline 模式（retrieval.baseline=true）只走纯向量检索，用于评测对比。
优化模式为：词表扩展查询 -> BM25 与向量分别召回 -> RRF 融合 -> Rerank。
"""
import jieba
from rank_bm25 import BM25Okapi

from .vector_store import VectorStore


def expand_query(query: str, domain_terms: dict) -> str:
    """词表扩展：把查询中的别名/规范名补全为规范名 + 别名集合。

    例：'庐州府志里包青天的故事' -> '庐州 合肥 包拯 包公 包青天 的故事'
    让 BM25 能命中更多写法不同的原文。
    """
    found = []
    for canonical, aliases in domain_terms.items():
        candidates = [canonical] + aliases
        if any(c in query for c in candidates):
            found.extend(candidates)
    return query + " " + " ".join(found) if found else query


def _tokenize(text: str) -> list[str]:
    return [w for w in jieba.lcut(text) if w.strip() and w not in "，。！？、；：""''（）"]


class Retriever:
    def __init__(self, cfg: dict, store: VectorStore, embedder):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder
        self._bm25 = None

    def _build_bm25(self):
        if self._bm25 is None:
            corpus = [_tokenize(c["text"]) for c in self.store.chunks]
            self._bm25 = BM25Okapi(corpus)
        return self._bm25

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """返回 [{'idx', 'text', 'title', 'doc', 'score', 'dense_score'}]"""
        q_emb = self.embedder.embed([query])[0]
        dense_hits = self.store.search(q_emb, top_k)
        dense_scores = {i: s for i, s in dense_hits}

        r = self.cfg["retrieval"]
        if r.get("baseline", False):
            # 基线：纯向量
            return [
                {**self.store.get(i), "idx": i, "score": s, "dense_score": s}
                for i, s in dense_hits
            ]

        # 优化：词表扩展 + BM25 混合
        expanded = expand_query(query, self.cfg.get("domain_terms", {}))
        bm25 = self._build_bm25()
        bm25_scores = bm25.get_scores(_tokenize(expanded))
        bm25_hits = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:top_k]

        # RRF 融合
        fused: dict[int, float] = {}
        for rank, (i, _) in enumerate(dense_hits):
            fused[i] = fused.get(i, 0.0) + r["dense_weight"] * (1.0 / (60 + rank))
        for rank, i in enumerate(bm25_hits):
            fused[i] = fused.get(i, 0.0) + r["bm25_weight"] * (1.0 / (60 + rank))
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]

        return [
            {**self.store.get(i), "idx": i, "score": s,
             "dense_score": dense_scores.get(i, 0.0)}
            for i, s in ranked
        ]
