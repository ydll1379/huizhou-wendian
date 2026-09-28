"""RAG 总管线：检索 -> 重排 -> 生成（含拒答）

检索层可切换（config.retrieval.backend）：
- local（默认）：自研 BM25 + 向量混合检索 + 本地 Reranker
- weknora：交给 WeKnora 的 hybrid-search 召回（它已完成重排），
  生成仍然由本地 DeepSeek Prompt 负责，风格/引用/拒答逻辑不变
"""
from __future__ import annotations

from pathlib import Path

from .embeddings import Embedder
from .generator import Generator
from .reranker import Reranker
from .retriever import Retriever
from .vector_store import VectorStore
from .weknora import WeKnoraRetriever


class RAGPipeline:
    def __init__(self, cfg: dict, kb_dir: Path | None = None):
        self.cfg = cfg
        self.backend = (cfg.get("retrieval") or {}).get("backend", "local")
        self.store = None
        self.embedder = None
        self.reranker = None

        if self.backend == "weknora":
            # 检索交给 WeKnora，本地不加载 bge-m3 / 本地向量库
            self.retriever = WeKnoraRetriever(cfg)
        else:
            self.store = VectorStore(kb_dir)
            if not self.store.load():
                raise RuntimeError(
                    f"知识库为空：{kb_dir}。请先运行 python scripts/build_kb.py"
                )
            self.embedder = Embedder(cfg["embedding"])
            self.retriever = Retriever(cfg, self.store, self.embedder)

        # 本地 Reranker：weknora 后端默认跳过（它已重排，且省一份模型内存）
        w_cfg = cfg.get("weknora") or {}
        if self.backend != "weknora" or w_cfg.get("local_rerank", False):
            self.reranker = Reranker(cfg["reranker"])

        self.generator = Generator(cfg)

    def answer(self, question: str, style: str = "teacher", output_context: str = "") -> dict:
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
        threshold = r.get("refuse_threshold", 0.35)
        if self.backend == "weknora":
            # WeKnora 返回的是 rerank 后的归一化分，与本地余弦分口径不同
            threshold = (self.cfg.get("weknora") or {}).get("refuse_threshold", threshold)
        top_cosine = hits[0].get("dense_score", hits[0].get("score", 1.0))
        if top_cosine < threshold:
            return {
                "answer": "抱歉，知识库里暂时没有与这个问题直接相关的资料。"
                          "可以换个问法，例如问问三河古镇、逍遥津、包公故事、寿县古城墙、芜湖古城、太和文庙等主题。",
                "citations": [],
                "sources": [],
                "refused": True,
            }

        # 重排（基线模式跳过重排，用于对比实验；weknora 后端默认已重排）
        reranked = hits
        if self.reranker is not None and not r.get("baseline", False):
            reranked = self.reranker.rerank(
                question, hits, self.cfg["reranker"]["top_k"]
            )
        elif self.backend == "weknora":
            # WeKnora 已重排，按与本地一致的条数截断后再交给 LLM
            reranked = hits[: self.cfg["reranker"]["top_k"]]

        result = self.generator.generate(question, reranked, style, output_context)
        # 给前端提供可点击的原文片段
        for i, s in enumerate(result["sources"]):
            s["cite_id"] = i + 1
        return result
