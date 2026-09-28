"""文档加载：支持 .txt / .md / .html / .pdf / .docx，统一输出 (标题, 正文)"""
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree
from urllib.parse import urljoin
import re

from bs4 import BeautifulSoup


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_html(path: Path) -> str:
    raw_html = _read_text(path)
    soup = BeautifulSoup(raw_html, "html.parser")
    page_url_match = re.search(r"https?://[^\s\"'<>]+", raw_html)
    page_url = page_url_match.group(0).rstrip(")") if page_url_match else ""
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    # Saved government resource pages sometimes embed the real chapter PDFs only
    # as data-url attributes in a table of contents. Preserve those source links.
    source_items = []
    for item in soup.select("[data-url]"):
        label = item.get_text(" ", strip=True)
        target = item.get("data-url", "").strip()
        if label and target:
            source_items.append(f"{label}：{urljoin(page_url, target)}")
    if source_items:
        text += "\n\n目录所链接的官方原始资料（本 HTML 本身不包含这些 PDF 正文）：\n" + "\n".join(dict.fromkeys(source_items))
    return text


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    """Extract paragraphs and table cells from a DOCX without an extra runtime dependency."""
    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    }
    with ZipFile(path) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    body = document.find("w:body", ns)
    if body is None:
        return ""

    def paragraph_text(paragraph) -> str:
        # Preserve tabs and explicit line breaks while ignoring formatting markup.
        pieces = []
        for node in paragraph.iter():
            if node.tag == f"{{{ns['w']}}}t":
                pieces.append(node.text or "")
            elif node.tag == f"{{{ns['w']}}}tab":
                pieces.append("\t")
            elif node.tag in (f"{{{ns['w']}}}br", f"{{{ns['w']}}}cr"):
                pieces.append("\n")
        return "".join(pieces).strip()

    def heading_prefix(paragraph) -> str:
        style = paragraph.find("w:pPr/w:pStyle", ns)
        style_name = style.get(f"{{{ns['w']}}}val", "") if style is not None else ""
        if style_name.startswith("Heading"):
            digits = "".join(ch for ch in style_name if ch.isdigit())
            level = min(max(int(digits or "2"), 1), 4)
            return "#" * level + " "
        return ""

    blocks = []
    for child in body:
        if child.tag == f"{{{ns['w']}}}p":
            value = paragraph_text(child)
            if value:
                blocks.append(heading_prefix(child) + value)
        elif child.tag == f"{{{ns['w']}}}tbl":
            for row in child.findall("w:tr", ns):
                cells = []
                for cell in row.findall("w:tc", ns):
                    value = " ".join(filter(None, (paragraph_text(p) for p in cell.findall("w:p", ns))))
                    cells.append(value)
                if any(cells):
                    blocks.append(" | ".join(cells))
    return "\n".join(blocks)


def load_file(path: Path) -> tuple[str, str]:
    """返回 (标题, 正文)。标题默认取文件名（不含扩展名）。"""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md", ".markdown"):
        text = _read_text(path)
    elif suffix in (".html", ".htm"):
        text = _read_html(path)
    elif suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix == ".docx":
        text = _read_docx(path)
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")
    title = path.stem
    return title, text.strip()


def load_directory(raw_dir: Path) -> list[tuple[str, str, Path]]:
    """递归扫描目录，返回 [(标题, 正文, 文件路径)]"""
    docs = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".docx"):
            try:
                title, text = load_file(path)
            except Exception as e:  # 单文件失败不影响整体
                print(f"[跳过] {path}: {e}")
                continue
            if len(text) < 50:  # 过滤空文档
                print(f"[跳过-内容过少] {path}")
                continue
            docs.append((title, text, path))
    return docs
