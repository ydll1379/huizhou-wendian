"""徽州问典 Web 服务"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import ROOT, load_config
from core.amap import AmapUnavailable
from core.pipeline import RAGPipeline
from core.recommender import Recommender, identify_spots
from core.route_planner import RouteError, RoutePlanner
from core.weknora import WeKnoraRetriever

cfg = load_config()
app = FastAPI(title="徽州问典")

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
    style: str = "teacher"  # teacher / tourist / promo
    audience: str = "通用"
    duration_minutes: int = 20
    learning_goal: str = ""
    content_format: str = "通用宣传文案"


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
    start_spot: str = ""
    end_spot: str = ""
    via_spots: list[str] = []


class CheckinRequest(BaseModel):
    route_id: str
    spot: str
    done: bool
    note: str = ""


class ReportRequest(BaseModel):
    route_id: str


class FeedbackRequest(BaseModel):
    category: str
    message: str


_feedback_lock = threading.Lock()


@app.get("/api/health")
def health():
    kb = (ROOT / cfg["data"]["kb_dir"] / "chunks.jsonl").exists()
    backend = (cfg.get("retrieval") or {}).get("backend", "local")
    info = {
        "status": "ok",
        "retrieval_backend": backend,
        "kb_ready": kb,
        "styles": ["teacher", "tourist", "promo"],
    }
    if backend == "weknora":
        info["weknora"] = WeKnoraRetriever(cfg).describe()
    return info


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        return ChatResponse(answer="请输入问题。", style=req.style)
    style = req.style if req.style in cfg["styles"] else "teacher"
    spots = identify_spots(req.question, cfg.get("spots", {}))
    try:
        context = ""
        if style == "teacher":
            context = (f"适用对象：{req.audience[:40]}；讲解时长：{max(5, min(req.duration_minutes, 120))}分钟；"
                       f"教学目标：{req.learning_goal[:300] or '帮助学生理解主题并联系地方历史'}。"
                       "请按‘核心知识—课堂讲解—提问互动—现场研学任务—注意事项’组织内容，注明适合讲解的时间分配。")
        elif style == "promo":
            context = (f"宣传素材类型：{req.content_format[:60]}。请给出可直接编辑使用的成稿；"
                       "活动时间、地点、报名方式等资料库未提供的信息请写成【待补充】，不要虚构。")
        result = get_pipeline().answer(req.question.strip(), style, context)
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
        route = get_planner().generate(req.city, max(1, min(req.days, 3)), req.start_spot, req.end_spot, req.via_spots)
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


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    """Save short prototype feedback locally for later manual review."""
    category = req.category.strip()
    message = req.message.strip()
    if category not in {"history", "product"}:
        raise HTTPException(status_code=422, detail="请选择史料纠错或功能建议。")
    if not message:
        raise HTTPException(status_code=422, detail="反馈内容不能为空。")
    if len(message) > 2000:
        raise HTTPException(status_code=422, detail="反馈内容请控制在 2000 字以内。")

    path = ROOT / "data" / "feedback.json"
    with _feedback_lock:
        records = []
        if path.exists():
            try:
                records = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=500, detail="本地反馈文件无法读取，请联系项目维护者。") from e
            if not isinstance(records, list):
                raise HTTPException(status_code=500, detail="本地反馈文件格式异常，请联系项目维护者。")
        records.append({
            "id": uuid.uuid4().hex[:12],
            "category": category,
            "message": message,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "new",
            "review_note": "",
        })
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    return {"ok": True}


def _require_local_admin(request: Request):
    host = request.client.host if request.client else ""
    if host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(status_code=403, detail="反馈审核仅允许本机访问。")


@app.get("/api/admin/feedback")
def feedback_list(request: Request):
    _require_local_admin(request)
    path = ROOT / "data" / "feedback.json"
    if not path.exists():
        return {"ok": True, "items": []}
    with _feedback_lock:
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise HTTPException(status_code=500, detail="反馈文件无法读取。") from e
    return {"ok": True, "items": sorted(items, key=lambda x: x.get("submitted_at", ""), reverse=True)}


class FeedbackReviewRequest(BaseModel):
    status: str
    review_note: str = ""


@app.patch("/api/admin/feedback/{feedback_id}")
def feedback_review(feedback_id: str, req: FeedbackReviewRequest, request: Request):
    _require_local_admin(request)
    if req.status not in {"new", "verified", "actioned", "rejected"}:
        raise HTTPException(status_code=422, detail="无效的处理状态。")
    path = ROOT / "data" / "feedback.json"
    with _feedback_lock:
        try:
            items = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except (json.JSONDecodeError, OSError) as e:
            raise HTTPException(status_code=500, detail="反馈文件无法读取。") from e
        item = next((x for x in items if x.get("id") == feedback_id), None)
        if item is None:
            raise HTTPException(status_code=404, detail="反馈不存在。")
        item["status"] = req.status
        item["review_note"] = req.review_note.strip()[:500]
        item["reviewed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    return {"ok": True, "item": item}


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
