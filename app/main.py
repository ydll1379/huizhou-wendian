"""庐州问典 Web 服务"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import ROOT, load_config
from core.amap import AmapUnavailable
from core.pipeline import RAGPipeline
from core.recommender import Recommender, identify_spots

cfg = load_config()
app = FastAPI(title="庐州问典")

# 知识库在后台线程懒加载，避免启动时卡住
_pipeline: RAGPipeline | None = None
_recommender: Recommender | None = None
_pipeline_lock = threading.RLock()  # 可重入：get_recommender 内部会调用 get_pipeline


def get_pipeline() -> RAGPipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = RAGPipeline(cfg, ROOT / cfg["data"]["kb_dir"])
    return _pipeline


def get_recommender() -> Recommender:
    global _recommender
    with _pipeline_lock:
        if _recommender is None:
            _recommender = Recommender(cfg, get_pipeline())
    return _recommender


class ChatRequest(BaseModel):
    question: str
    style: str = "guide"  # guide / scholar / child


class ChatResponse(BaseModel):
    answer: str
    style: str
    citations: list[int] = []
    sources: list[dict] = []
    refused: bool = False
    spots: list[str] = []  # 问题中识别到的景点（驱动前端周边推荐面板）


class RecommendRequest(BaseModel):
    spot: str
    kind: str = "food"    # food / hotel
    limit: Optional[int] = None


@app.get("/api/health")
def health():
    kb = (ROOT / cfg["data"]["kb_dir"] / "chunks.jsonl").exists()
    return {"status": "ok", "kb_ready": kb, "styles": list(cfg["styles"].keys())}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        return ChatResponse(answer="请输入问题。", style=req.style)
    style = req.style if req.style in cfg["styles"] else "guide"
    spots = identify_spots(req.question, cfg.get("spots", {}))
    try:
        result = get_pipeline().answer(req.question.strip(), style)
    except RuntimeError as e:
        # 未配置 API key / 知识库为空等可预期错误，返回友好提示
        return ChatResponse(answer=str(e), style=style, refused=True, spots=spots)
    result["style"] = style
    result["spots"] = spots
    return ChatResponse(**result)


@app.get("/api/spots")
def spots():
    """已收录景点（前端快捷入口用）"""
    return [{"name": n, "city": m.get("city", "")} for n, m in cfg.get("spots", {}).items()]


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    try:
        rec = get_recommender().recommend(req.spot, req.kind, req.limit)
        return {"ok": True, **rec}
    except (AmapUnavailable, ValueError) as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"推荐服务出错：{e}"}


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
