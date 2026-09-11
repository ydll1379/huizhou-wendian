"""分块策略：标题感知 + 滑动窗口

- 对含 Markdown 标题（# / ## / 【】）的文档先按标题切段，保留上下文；
- 对长段落用滑动窗口在句子边界处切分，带 overlap，避免切断语义。
"""
import re

HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$|^【(.+?)】\s*$")


def _split_sentences(text: str) -> list[str]:
    # 按中文句末标点切句，保留标点
    parts = re.split(r"(?<=[。！？!?；;])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_document(title: str, text: str, max_chars: int, overlap: int) -> list[dict]:
    """返回 [{'title': 文档标题, 'text': 块文本, 'doc': 文档名}]"""
    lines = text.splitlines()
    chunks: list[dict] = []
    current: list[str] = []
    current_heading: str = title

    def flush():
        nonlocal current
        body = "".join(current).strip()
        if body:
            chunks.append({"title": current_heading, "text": body, "doc": title})
        current = []

    for line in lines:
        m = HEADING_RE.match(line.strip())
        if m:
            flush()
            current_heading = (m.group(2) or m.group(3) or "").strip()
            continue
        current.append(line)
    flush()

    # 长块按滑动窗口二次切分
    result: list[dict] = []
    for ch in chunks:
        if len(ch["text"]) <= max_chars:
            result.append(ch)
            continue
        sentences = _split_sentences(ch["text"])
        buf = ""
        for sent in sentences:
            if buf and len(buf) + len(sent) > max_chars:
                result.append({"title": ch["title"], "text": buf, "doc": ch["doc"]})
                # overlap：保留上一块末尾若干字
                buf = buf[-overlap:] if overlap else ""
            buf += sent
        if buf:
            result.append({"title": ch["title"], "text": buf, "doc": ch["doc"]})
    return result
