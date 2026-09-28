"""端到端验证：用一个模拟 WeKnora 服务跑通 WeKnoraRetriever + RAGPipeline。

模拟服务的请求/响应严格按 WeKnora 源码契约：
  POST /api/v1/knowledge-bases/{id}/hybrid-search
  {"query_text": ..., "match_count": ...}  ->  {"success": true, "data": [SearchResult]}
"""
import copy
import json
import os
import sys
import threading
import unittest.mock as mock
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

RECEIVED = []
SCORE = {"value": 0.87}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        RECEIVED.append({
            "path": self.path,
            "api_key": self.headers.get("X-API-Key"),
            "payload": body,
        })
        if self.headers.get("X-API-Key") != "sk-test-key":
            self._send(401, {"success": False, "error": {"code": "UNAUTHORIZED", "message": "invalid api key"}})
            return
        data = [
            {"id": "chunk-1", "content": "中共小甸集特支于1923年冬在寿县小甸集成立，是安徽省最早的农村党组织。",
             "knowledge_id": "k-1", "chunk_index": 3, "knowledge_title": "安徽第一面党旗纪念园",
             "start_at": 0, "end_at": 40, "seq": 1, "score": SCORE["value"],
             "chunk_type": "text", "image_info": "", "knowledge_filename": "党旗纪念园.md",
             "knowledge_source": "file"},
            {"id": "chunk-2", "content": "小甸集特支点燃了安徽革命斗争的火种。",
             "knowledge_id": "k-2", "chunk_index": 0, "knowledge_title": "寿县小甸村",
             "start_at": 0, "end_at": 20, "seq": 2, "score": SCORE["value"] - 0.1,
             "chunk_type": "text", "image_info": "", "knowledge_filename": "小甸村.md",
             "knowledge_source": "file"},
        ] + [
            {"id": f"chunk-{i}", "content": f"补充分块 {i}", "knowledge_id": f"k-{i}",
             "chunk_index": i, "knowledge_title": f"文档{i}", "start_at": 0, "end_at": 10,
             "seq": i, "score": SCORE["value"] - 0.01 * i, "chunk_type": "text",
             "image_info": "", "knowledge_filename": f"doc{i}.md", "knowledge_source": "file"}
            for i in range(3, 9)
        ]
        self._send(200, {"success": True, "data": data})

    def _send(self, code, payload):
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    os.environ["WEKNORA_BASE_URL"] = f"http://127.0.0.1:{port}"
    os.environ["WEKNORA_API_KEY"] = "sk-test-key"

    from core import load_config
    from core.pipeline import RAGPipeline
    from core.weknora import WeKnoraRetriever

    cfg = copy.deepcopy(load_config())
    cfg["retrieval"]["backend"] = "weknora"
    cfg["weknora"]["knowledge_base_id"] = "kb-test-0001"

    results = []

    def check(name, cond, extra=""):
        results.append(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {name}{('  -> ' + str(extra)) if extra and not cond else ''}")

    # --- 1. 检索器本身：请求形状 + 字段映射 ---
    r = WeKnoraRetriever(cfg)
    hits = r.retrieve("小甸集特支", 12)
    check("retriever.ready", r.ready is True)
    check("请求路径含知识库 ID 与 hybrid-search",
          "/api/v1/knowledge-bases/kb-test-0001/hybrid-search" in RECEIVED[-1]["path"], RECEIVED[-1]["path"])
    check("默认带 resource_urls=handle（保持内部引用）",
          RECEIVED[-1]["path"].endswith("resource_urls=handle"), RECEIVED[-1]["path"])
    check("请求携带 X-API-Key", RECEIVED[-1]["api_key"] == "sk-test-key")
    check("请求体用 query_text/match_count",
          RECEIVED[-1]["payload"].get("query_text") == "小甸集特支"
          and RECEIVED[-1]["payload"].get("match_count") == 12, RECEIVED[-1]["payload"])
    check("hits 数量", len(hits) == 8, len(hits))
    check("字段映射 text/doc/score/dense_score",
          hits[0]["text"].startswith("中共小甸集特支")
          and hits[0]["doc"] == "安徽第一面党旗纪念园"
          and hits[0]["title"] == hits[0]["doc"]
          and hits[0]["score"] == 0.87 and hits[0]["dense_score"] == 0.87, hits[0])
    check("保留 WeKnora 原始分块信息", hits[0]["chunk_id"] == "chunk-1" and hits[0]["source"] == "weknora")

    # --- 2. 管线：不加载本地向量库/embedding/reranker，仅注入检索器 ---
    with mock.patch("core.generator.Generator.generate") as gen:
        # 真实 generate() 会把传入的 docs 原样放进 sources，这里照此模拟
        gen.side_effect = lambda question, docs, style, output_context="": {
            "answer": "小甸集特支是安徽最早的农村党组织 [1]。",
            "citations": [1], "sources": docs, "refused": False,
        }
        pipe = RAGPipeline(cfg, ROOT + "/data/kb")
        check("backend=weknora 时不加载本地 store/embedder/reranker",
              pipe.store is None and pipe.embedder is None and pipe.reranker is None)
        check("pipeline.retriever 是 WeKnoraRetriever", isinstance(pipe.retriever, WeKnoraRetriever))

        out = pipe.answer("小甸集特支为什么重要？", "guide")
        check("答案正常返回且未拒答", out["refused"] is False and out["answer"].startswith("小甸集特支"))
        check("sources 带 cite_id 供前端点击",
              [s["cite_id"] for s in out["sources"]] == [1, 2, 3, 4, 5],
              [s.get("cite_id") for s in out["sources"]])
        check("交给 LLM 的条数按 reranker.top_k 截断",
              len(gen.call_args[0][1]) == cfg["reranker"]["top_k"],
              len(gen.call_args[0][1]))

        # --- 3. 低分拒答：不调用 LLM ---
        gen.reset_mock()
        SCORE["value"] = 0.10
        out2 = pipe.answer("与知识库无关的问题", "guide")
        check("低分时拒答", out2["refused"] is True and not out2["sources"])
        check("拒答时不调用 LLM", gen.call_count == 0)
        SCORE["value"] = 0.87

    # --- 4. 鉴权失败 -> WeKnoraUnavailable（RuntimeError 子类，main.py 会转友好提示）---
    os.environ["WEKNORA_API_KEY"] = "sk-wrong"
    try:
        r2 = WeKnoraRetriever(cfg)
        r2.retrieve("x", 1)
        check("错误的 API Key 抛错", False)
    except Exception as e:
        check("错误的 API Key 抛 WeKnoraUnavailable/RuntimeError",
              isinstance(e, RuntimeError), f"{type(e).__name__}: {e}")

    # --- 5. 未配置知识库 ID -> 明确报错 ---
    os.environ["WEKNORA_API_KEY"] = "sk-test-key"
    cfg2 = copy.deepcopy(cfg)
    cfg2["weknora"]["knowledge_base_id"] = ""
    try:
        WeKnoraRetriever(cfg2).retrieve("x", 1)
        check("缺知识库 ID 报错", False)
    except RuntimeError as e:
        check("缺知识库 ID 给出明确提示", "知识库 ID" in str(e), e)

    # --- 6. 服务不可达 -> 友好报错 ---
    os.environ["WEKNORA_BASE_URL"] = "http://127.0.0.1:1"
    try:
        WeKnoraRetriever(cfg).retrieve("x", 1)
        check("服务不可达报错", False)
    except RuntimeError as e:
        check("服务不可达给出可操作提示", "无法连接 WeKnora" in str(e), e)
    os.environ["WEKNORA_BASE_URL"] = f"http://127.0.0.1:{port}"

    srv.shutdown()
    ok = all(results)
    print(f"\n{sum(results)}/{len(results)} 通过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
