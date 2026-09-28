"""WeKnora 检索后端适配器（可选）

把 WeKnora（腾讯开源知识平台）当作**检索层**使用：只取它召回的原文分块，
生成仍然由本地 DeepSeek Prompt 负责——三档讲解风格、[n] 引用编号、拒答
阈值这些应用层逻辑全部保留，换掉的只有检索这一环。

适配后的 hits 形状与 core/retriever.Retriever.retrieve() 完全一致：
    [{'idx', 'text', 'title', 'doc', 'score', 'dense_score', ...}]
因此 core/pipeline.py（拒答 + 重排）、core/generator.py、core/recommender.py、
core/route_planner.py 和前端都不需要改动。

启用方式（config.yaml）：
    retrieval:
      backend: weknora
    weknora:
      knowledge_base_id: "kb-xxxxxxxx"
并在 .env 配置：
    WEKNORA_BASE_URL=http://127.0.0.1:8080
    WEKNORA_API_KEY=sk-xxxx

接口依据（已对照 WeKnora 源码核对，见 internal/handler/knowledgebase.go
的 HybridSearch 与 internal/types/search.go 的 SearchParams / SearchResult）：
    POST /api/v1/knowledge-bases/{id}/hybrid-search
    请求 {"query_text": "...", "match_count": N}
    响应 {"success": true, "data": [{"id","content","knowledge_title","score",...}]}

注意：官方 docs/api/knowledge-search.md 里描述的 POST /api/v1/knowledge-search
在当前 main 分支上是「文档列表关键词搜索」（handler.SearchKnowledge 读的是
URL 上的 keyword/query 参数，返回 has_more/total），并非向量检索，故不使用。
"""
from __future__ import annotations

import os
import time

import requests


class WeKnoraUnavailable(RuntimeError):
    """WeKnora 未配置 / 不可用（上层会转成友好提示返回给前端）"""


class WeKnoraRetriever:
    """对接 WeKnora `POST /api/v1/knowledge-bases/{id}/hybrid-search` 的检索器。"""

    def __init__(self, cfg: dict):
        w = cfg.get("weknora") or {}
        self.cfg = w
        self.base_url = self._resolve_base_url(w)
        self.api_key = os.environ.get(w.get("api_key_env", "WEKNORA_API_KEY"), "")
        self.knowledge_base_id = (w.get("knowledge_base_id") or "").strip()
        self.timeout = float(w.get("timeout", 20))
        self.resource_urls = (w.get("resource_urls") or "").strip()
        self.vector_threshold = float(w.get("vector_threshold", 0.0))
        self.keyword_threshold = float(w.get("keyword_threshold", 0.0))
        self.session = requests.Session()
        self._last_call = 0.0
        self._min_interval = float(w.get("min_interval", 0.0))

    @staticmethod
    def _resolve_base_url(w: dict) -> str:
        env_name = w.get("base_url_env", "WEKNORA_BASE_URL")
        return (os.environ.get(env_name) or w.get("base_url") or "http://127.0.0.1:8080").rstrip("/")

    @property
    def ready(self) -> bool:
        return bool(self.api_key and self.knowledge_base_id)

    def describe(self) -> dict:
        """给 /api/health 用的运行时信息（不含密钥）"""
        return {
            "base_url": self.base_url,
            "knowledge_base_id": self.knowledge_base_id,
            "api_key_configured": bool(self.api_key),
        }

    # ---------- 内部 ----------

    def _endpoint(self) -> str:
        return (f"{self.base_url}/api/v1/knowledge-bases/"
                f"{self.knowledge_base_id}/hybrid-search")

    def _check_config(self):
        if not self.api_key:
            raise WeKnoraUnavailable(
                "未配置 WeKnora API Key。请在 WeKnora 页面「账户信息」获取后，"
                "填入 .env 的 WEKNORA_API_KEY 并重启服务。"
            )
        if not self.knowledge_base_id:
            raise WeKnoraUnavailable(
                "未配置 WeKnora 知识库 ID。请在 config.yaml 的 weknora.knowledge_base_id "
                "填入目标知识库 ID（形如 kb-xxxxxxxx）。"
            )

    def _throttle(self):
        if self._min_interval <= 0:
            return
        wait = self._min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def _parse(self, resp: requests.Response) -> dict:
        if resp.status_code in (401, 403):
            raise WeKnoraUnavailable(
                f"WeKnora 拒绝访问（HTTP {resp.status_code}）：请检查 WEKNORA_API_KEY 是否有效，"
                "以及该 Key 是否有此知识库的读取权限。"
            )
        try:
            body = resp.json()
        except ValueError:
            raise WeKnoraUnavailable(
                f"WeKnora 返回的不是 JSON（HTTP {resp.status_code}）：{resp.text[:200]}"
            )
        if body.get("error") or body.get("success") is False:
            err = body.get("error") or {}
            msg = err.get("message") or err.get("code") or str(body)[:200]
            raise WeKnoraUnavailable(f"WeKnora 检索失败：{msg}")
        return body

    def _post(self, payload: dict) -> dict:
        self._check_config()
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        params = {"resource_urls": self.resource_urls} if self.resource_urls else None

        last_err = ""
        for attempt in range(2):
            self._throttle()
            try:
                resp = self.session.post(
                    self._endpoint(), json=payload, params=params,
                    headers=headers, timeout=self.timeout,
                )
            except requests.RequestException as e:
                last_err = (f"无法连接 WeKnora（{self.base_url}）：{e}。"
                            "请确认 WeKnora 已启动（docker compose up -d）。")
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                raise WeKnoraUnavailable(last_err)

            if resp.status_code >= 500 and attempt == 0:
                last_err = f"WeKnora 服务端错误（HTTP {resp.status_code}），稍后重试一次。"
                time.sleep(1.0)
                continue
            return self._parse(resp)
        raise WeKnoraUnavailable(last_err or "WeKnora 检索请求失败。")

    # ---------- 对外：与 core/retriever.Retriever 同签名 ----------

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """检索并转换成项目内部的 hits 形状（下游无感）。"""
        payload: dict = {"query_text": query, "match_count": int(top_k)}
        if self.vector_threshold > 0:
            payload["vector_threshold"] = self.vector_threshold
        if self.keyword_threshold > 0:
            payload["keyword_threshold"] = self.keyword_threshold

        data = self._post(payload)
        results = data.get("data") or []

        hits: list[dict] = []
        for i, r in enumerate(results or []):
            text = r.get("content") or ""
            if not text.strip():
                continue
            title = r.get("knowledge_title") or r.get("knowledge_filename") or ""
            score = float(r.get("score") or 0.0)
            hits.append({
                "idx": r.get("id") or f"weknora-{i}",
                "text": text,
                "title": title,
                "doc": title,          # generator 输出「出处：《doc》」
                "score": score,        # WeKnora rerank 后的归一化最终分
                "dense_score": score,  # pipeline 拒答判定 + 前端「相关度」展示都读这个字段
                # 以下为 WeKnora 独有信息，便于前端/调试扩展
                "chunk_id": r.get("id", ""),
                "knowledge_id": r.get("knowledge_id", ""),
                "chunk_index": r.get("chunk_index"),
                "chunk_type": r.get("chunk_type", ""),
                "image_info": r.get("image_info", ""),
                "knowledge_filename": r.get("knowledge_filename", ""),
                "source": "weknora",
            })
        return hits

    def probe(self, query: str = "测试") -> list[dict]:
        """连通性自检：跑一次单条检索，供 scripts/check_weknora.py 使用。"""
        return self.retrieve(query, 1)
