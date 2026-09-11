"""LLM 调用助手：DeepSeek（OpenAI 兼容），供路线规划/报告生成等模块复用"""
from __future__ import annotations

import json
import os

from openai import OpenAI


def make_client(cfg: dict) -> tuple[OpenAI, dict]:
    llm = cfg["llm"]
    key = os.environ.get(llm["api_key_env"], "")
    if not key:
        raise RuntimeError(
            f"未找到 {llm['api_key_env']}，请复制 .env.example 为 .env 并填入 key"
        )
    return OpenAI(api_key=key, base_url=llm["base_url"]), llm


def chat_json(client: OpenAI, llm: dict, system: str, user: str) -> dict:
    """要求模型输出 JSON 并解析"""
    resp = client.chat.completions.create(
        model=llm["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=llm.get("temperature", 0.3),
        max_tokens=llm.get("max_tokens", 1024),
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def chat_text(client: OpenAI, llm: dict, system: str, user: str, max_tokens: int = 2048) -> str:
    """普通文本生成"""
    resp = client.chat.completions.create(
        model=llm["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=llm.get("temperature", 0.3),
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
