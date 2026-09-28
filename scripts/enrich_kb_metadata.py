"""为已有 chunks.jsonl 补充原始资料出处元数据，不重新计算向量。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import ROOT, load_config
from core.loader import load_directory


def metadata_for(text: str, path: Path) -> dict:
    metadata = {"source_file": str(path.relative_to(ROOT))}
    front = re.match(r"\A---\s*\n(.*?)\n---\s*\n", text, re.S)
    if front:
        for key in ("source_url", "publisher", "published_at", "site", "page"):
            match = re.search(rf"(?m)^{key}:\s*['\"]?(.+?)['\"]?\s*$", front.group(1))
            if match:
                metadata[key] = match.group(1).strip().strip("'\"")
    url = re.search(r"https?://[^\s)]+", text)
    if url:
        metadata.setdefault("source_url", url.group(0).rstrip(".,，。"))
    return metadata


def main() -> None:
    cfg = load_config()
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    chunks_path = ROOT / cfg["data"]["kb_dir"] / "chunks.jsonl"
    if not chunks_path.exists():
        raise SystemExit(f"未找到索引：{chunks_path}")

    docs = {title: (text, path) for title, text, path in load_directory(raw_dir)}
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    enriched = 0
    for chunk in chunks:
        source = docs.get(chunk.get("doc", ""))
        if not source:
            continue
        text, path = source
        for key, value in metadata_for(text, path).items():
            if value and not chunk.get(key):
                chunk[key] = value
        enriched += 1

    tmp = chunks_path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks), encoding="utf-8")
    tmp.replace(chunks_path)
    print(f"已为 {enriched}/{len(chunks)} 个分块补充来源字段；向量文件未改动。")


if __name__ == "__main__":
    main()
