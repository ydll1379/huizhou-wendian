"""文档加载：支持 .txt / .md / .html / .pdf，统一输出 (标题, 正文)"""
from pathlib import Path

from bs4 import BeautifulSoup


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_html(path: Path) -> str:
    soup = BeautifulSoup(_read_text(path), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_file(path: Path) -> tuple[str, str]:
    """返回 (标题, 正文)。标题默认取文件名（不含扩展名）。"""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md", ".markdown"):
        text = _read_text(path)
    elif suffix in (".html", ".htm"):
        text = _read_html(path)
    elif suffix == ".pdf":
        text = _read_pdf(path)
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")
    title = path.stem
    return title, text.strip()


def load_directory(raw_dir: Path) -> list[tuple[str, str, Path]]:
    """递归扫描目录，返回 [(标题, 正文, 文件路径)]"""
    docs = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".txt", ".md", ".markdown", ".html", ".htm", ".pdf"):
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
