# 庐州问典 —— 合肥地方文化智能讲解与传承平台
# 轻量自建 RAG：文档解析 -> 分块 -> Embedding -> 混合检索 -> Rerank -> DeepSeek 生成（溯源+多风格）

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_api_key(cfg: dict) -> str:
    key = os.environ.get(cfg["llm"]["api_key_env"], "")
    return key
