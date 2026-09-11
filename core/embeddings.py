"""本地 Embedding（sentence-transformers / BGE 系列）"""
import os

# 国内网络默认走 hf-mirror 下载模型（外部已显式设置时不覆盖）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = SentenceTransformer(
            cfg["model"],
            device=cfg.get("device", "cpu"),
            trust_remote_code=cfg.get("trust_remote_code", False),
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        """返回已归一化的向量矩阵 (n, dim)"""
        return self.model.encode(
            texts,
            batch_size=self.cfg.get("batch_size", 16),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
