"""重排序：BGE-Reranker（CrossEncoder）"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = CrossEncoder(
            cfg["model"],
            device=cfg.get("device", "cpu"),
            trust_remote_code=cfg.get("trust_remote_code", False),
        )

    def rerank(self, query: str, docs: list[dict], top_k: int) -> list[dict]:
        """docs: [{'idx','text','score',...}]，返回按重排分数降序的前 top_k 条"""
        if not docs:
            return []
        pairs = [[query, d["text"]] for d in docs]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
        result = []
        for d, s in ranked[:top_k]:
            d = dict(d)
            d["rerank_score"] = float(s)
            result.append(d)
        return result
