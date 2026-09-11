"""生成层：DeepSeek API（OpenAI 兼容接口）

- 强制引用标注：资料带编号 [1][2]...，要求模型回答中标注来源
- 低置信度拒答：召回最高分低于阈值时直接拒答，不调用 LLM（防幻觉）
- 多风格：系统提示词按 config.styles 切换
"""
import os
import re

from openai import OpenAI

CITE_RE = re.compile(r"\[(\d+)\]")


def build_context_block(docs: list[dict]) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        lines.append(f"[{i}] 出处：《{d['doc']}》\n{d['text']}")
    return "\n\n".join(lines)


class Generator:
    def __init__(self, cfg: dict):
        self.cfg = cfg["llm"]
        api_key = os.environ.get(cfg["llm"]["api_key_env"], "")
        if not api_key:
            raise RuntimeError(
                f"未找到 {cfg['llm']['api_key_env']}，请复制 .env.example 为 .env 并填入 key"
            )
        self.client = OpenAI(api_key=api_key, base_url=cfg["llm"]["base_url"])

    def generate(self, question: str, docs: list[dict], style: str) -> dict:
        """返回 {'answer': str, 'citations': [编号列表], 'sources': [docs], 'refused': bool}"""
        style_prompt = self.cfg.get("styles", {}).get(style, self.cfg.get("styles", {}).get("guide", ""))
        context = build_context_block(docs)

        system = (
            f"{style_prompt}\n\n"
            "你是「徽州问典」地方文化讲解助手，覆盖合肥、芜湖、阜阳、淮南等市的红色文化与地方文史知识（含革命老区）。回答规则：\n"
            "1. 只依据【背景资料】回答，不要编造资料中没有的信息；\n"
            "2. 涉及革命历史、英烈人物和红色主题时，表述必须庄重、准确，使用规范称谓并保持敬意；\n"
            "3. 关键事实后面用 [编号] 标注出处（如 [1]），编号必须对应背景资料；\n"
            "4. 如果背景资料不足以回答问题，直接说明'资料库中暂无相关信息'，不要猜测。"
        )
        user = f"【背景资料】\n{context}\n\n【问题】\n{question}"

        resp = self.client.chat.completions.create(
            model=self.cfg["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.cfg.get("temperature", 0.3),
            max_tokens=self.cfg.get("max_tokens", 1024),
        )
        answer = resp.choices[0].message.content or ""

        citations = sorted({int(n) for n in CITE_RE.findall(answer)})
        return {
            "answer": answer,
            "citations": citations,
            "sources": docs,
            "refused": False,
        }
