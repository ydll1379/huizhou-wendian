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
from core.route_planner import RouteError, RoutePlanner

cfg = load_config()
app = FastAPI(title="庐州问典")

# 知识库在后台线程懒加载，避免启动时卡住
_pipeline: RAGPipeline | None = None
_recommender: Recommender | None = None
_planner: RoutePlanner | None = None
_pipeline_lock = threading.RLock()


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


def get_planner() -> RoutePlanner:
    global _planner
    with _pipeline_lock:
        if _planner is None:
            _planner = RoutePlanner(cfg, get_pipeline(), ROOT)
    return _planner


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


class RouteGenRequest(BaseModel):
    city: str
    days: int = 1


class CheckinRequest(BaseModel):
    route_id: str
    spot: str
    done: bool
    note: str = ""


class ReportRequest(BaseModel):
    route_id: str


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
    """已收录景点（前端快捷入口用；tag=red 为红色主题景点）"""
    return [{"name": n, "city": m.get("city", ""), "tag": m.get("tag", "")}
            for n, m in cfg.get("spots", {}).items()]


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    try:
        rec = get_recommender().recommend(req.spot, req.kind, req.limit)
        return {"ok": True, **rec}
    except (AmapUnavailable, ValueError) as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"推荐服务出错：{e}"}


# ---------- 青年红色筑梦之旅：研学路线 / 打卡 / 实践报告 ----------

@app.get("/api/cities")
def cities():
    """有可编排路线景点的城市（红色景点数>=2 的优先）"""
    counter: dict[str, int] = {}
    for m in cfg.get("spots", {}).values():
        c = m.get("city", "")
        counter[c] = counter.get(c, 0) + (1 if m.get("tag") == "red" else 0)
    return [{"city": c, "red_count": n} for c, n in sorted(counter.items(), key=lambda x: -x[1]) if n > 0]


@app.post("/api/routes/generate")
def routes_generate(req: RouteGenRequest):
    try:
        route = get_planner().generate(req.city, max(1, min(req.days, 3)))
        return {"ok": True, "route": route}
    except (AmapUnavailable, RouteError, RuntimeError) as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"路线生成出错：{e}"}


@app.get("/api/routes")
def routes_list():
    return {"ok": True, "routes": get_planner().list_routes()}


@app.get("/api/routes/{route_id}")
def routes_get(route_id: str):
    try:
        route = get_planner().get_route(route_id)
        return {"ok": True, "route": route, "checkins": get_planner().get_checkins(route_id)}
    except RouteError as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/checkin")
def checkin(req: CheckinRequest):
    try:
        state = get_planner().checkin(req.route_id, req.spot, req.done, req.note)
        return {"ok": True, "checkin": state}
    except RouteError as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/report")
def report(req: ReportRequest):
    try:
        text = get_planner().gen_report(req.route_id)
        return {"ok": True, "report": text}
    except (RouteError, RuntimeError) as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"报告生成出错：{e}"}


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
