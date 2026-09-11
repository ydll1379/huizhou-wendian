"""轻量向量库：numpy 余弦相似度 + jsonl/npy 持久化

知识库规模为 80~150 份文档（数千个分块），暴力余弦检索毫秒级完成，
无需引入重型向量数据库，且全链路可控（便于做对比实验）。
"""
import json
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, kb_dir: Path):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[dict], embeddings: np.ndarray):
        self.chunks.extend(chunks)
        self.embeddings = (
            embeddings
            if self.embeddings is None
            else np.vstack([self.embeddings, embeddings])
        )

    def search(self, query_emb: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """返回 [(chunk_idx, cosine_score)]，按相似度降序"""
        if self.size == 0:
            return []
        sims = self.embeddings @ query_emb  # 已归一化 -> 余弦
        idx = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in idx]

    def get(self, idx: int) -> dict:
        return self.chunks[idx]

    def save(self):
        with open(self.kb_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for ch in self.chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")
        if self.embeddings is not None:
            np.save(self.kb_dir / "embeddings.npy", self.embeddings)

    def load(self) -> bool:
        chunks_file = self.kb_dir / "chunks.jsonl"
        emb_file = self.kb_dir / "embeddings.npy"
        if not (chunks_file.exists() and emb_file.exists()):
            return False
        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.embeddings = np.load(emb_file)
        return True
