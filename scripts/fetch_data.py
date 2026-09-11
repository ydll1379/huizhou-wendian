"""数据收集脚本（Phase 1）

从公开网站抓取合肥地方文化资料，保存为 md 文件到 data/raw/。

用法：
  python scripts/fetch_data.py ihchina          # 抓中国非物质文化遗产网
  python scripts/fetch_data.py hefei-museum     # 抓安徽博物院/包公园等（示例）

注意：请遵守各网站 robots.txt 与版权规定，仅用于学习演示；
部分站点有反爬，建议控制频率（脚本内置 1.5s 延迟）。
"""
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}


def _save(title: str, text: str):
    name = re.sub(r'[\\/:*?"<>|]', "_", title)
    path = OUT / f"{name}.md"
    path.write_text(f"# {title}\n\n{text}\n", encoding="utf-8")
    print(f"  [保存] {path.name} ({len(text)} 字)")


def fetch_ihchina():
    """中国非物质文化遗产网：国家级非遗名录检索页（示例入口，可按需扩展）"""
    # 说明：ihchina.cn 名录页为动态渲染，此处给出静态入口示例。
    # 实际可改用其 API 或手工下载名录 PDF。
    url = "https://www.ihchina.cn/project.html"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.encoding = r.apparent_encoding
    print(f"状态码 {r.status_code}，页面 {len(r.text)} 字节。"
          "该站为动态页面，请在此基础上扩展（或人工收集名录页另存为 html）。")


def fetch_hefei_museum():
    """示例：安徽博物院镇馆之宝页（静态页面，可直接存 md）"""
    urls = {
        "安徽博物院-楚大鼎": "https://www.ahm.cn/",
    }
    for title, url in urls.items():
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.encoding = r.apparent_encoding
            _save(title, f"来源页面：{url}\n\n（此页面为示例入口，请按实际栏目页替换为具体展品介绍）")
        except Exception as e:
            print(f"[失败] {title}: {e}")
        time.sleep(1.5)


JOBS = {
    "ihchina": fetch_ihchina,
    "hefei-museum": fetch_hefei_museum,
}

if __name__ == "__main__":
    job = sys.argv[1] if len(sys.argv) > 1 else "ihchina"
    OUT.mkdir(parents=True, exist_ok=True)
    JOBS[job]()
