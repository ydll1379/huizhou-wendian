"""建库脚本：扫描 data/raw 下所有文档 -> 分块 -> Embedding -> 存入 data/kb"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import ROOT, load_config
from core.chunker import chunk_document
from core.embeddings import Embedder
from core.loader import load_directory
from core.vector_store import VectorStore

cfg = load_config()
raw_dir = ROOT / cfg["data"]["raw_dir"]
kb_dir = ROOT / cfg["data"]["kb_dir"]


def main():
    docs = load_directory(raw_dir)
    if not docs:
        print(f"未在 {raw_dir} 下找到任何文档（支持 .txt/.md/.html/.pdf）")
        return

    print(f"共 {len(docs)} 份文档，开始分块…")
    chunk_cfg = cfg["chunking"]
    chunks = []
    for title, text, path in docs:
        chunks.extend(chunk_document(title, text, chunk_cfg["max_chars"], chunk_cfg["overlap"]))

    print(f"共 {len(chunks)} 个分块，加载 Embedding 模型 {cfg['embedding']['model']} …")
    embedder = Embedder(cfg["embedding"])
    vectors = embedder.embed([c["text"] for c in chunks])

    store = VectorStore(kb_dir)
    store.add(chunks, vectors)
    store.save()
    print(f"完成：{store.size} 个分块已写入 {kb_dir}")


if __name__ == "__main__":
    main()
