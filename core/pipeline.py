"""RAG 总管线：检索 -> 重排 -> 生成（含拒答）"""
from pathlib import Path

from .embeddings import Embedder
from .generator import Generator
from .reranker import Reranker
from .retriever import Retriever
from .vector_store import VectorStore


class RAGPipeline:
    def __init__(self, cfg: dict, kb_dir: Path):
        self.cfg = cfg
        self.store = VectorStore(kb_dir)
        if not self.store.load():
            raise RuntimeError(
                f"知识库为空：{kb_dir}。请先运行 python scripts/build_kb.py"
            )
        self.embedder = Embedder(cfg["embedding"])
        self.retriever = Retriever(cfg, self.store, self.embedder)
        self.reranker = Reranker(cfg["reranker"])
        self.generator = Generator(cfg)

    def answer(self, question: str, style: str = "guide") -> dict:
        r = self.cfg["retrieval"]
        hits = self.retriever.retrieve(question, r["top_k"])

        # 拒答：召回结果与问题相关性普遍过低（用余弦分判断），直接拒答防幻觉
        if not hits:
            return {
                "answer": "知识库还是空的，请先运行 python scripts/build_kb.py 建库。",
                "citations": [],
                "sources": [],
                "refused": True,
            }
        top_cosine = hits[0].get("dense_score", hits[0].get("score", 1.0))
        if top_cosine < r.get("refuse_threshold", 0.35):
            return {
                "answer": "抱歉，知识库里暂时没有与这个问题直接相关的资料。"
                          "可以换个问法，例如问问三河古镇、逍遥津、包公故事、寿县古城墙、芜湖古城、太和文庙等主题。",
                "citations": [],
                "sources": [],
                "refused": True,
            }

        # 重排（基线模式跳过重排，用于对比实验）
        reranked = hits
        if not r.get("baseline", False):
            reranked = self.reranker.rerank(
                question, hits, self.cfg["reranker"]["top_k"]
            )

        result = self.generator.generate(question, reranked, style)
        # 给前端提供可点击的原文片段
        for i, s in enumerate(result["sources"]):
            s["cite_id"] = i + 1
        return result
