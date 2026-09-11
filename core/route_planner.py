"""青年红色筑梦之旅：研学路线生成 + 任务打卡 + 实践成果生成

- RoutePlanner.generate(city)：从景点注册表选取该市景点，LLM 编排研学路线
  （路线名/站点意义/研学任务），高德路径规划 API 计算站间交通段。
- CheckinStore：data/checkins.json 记录打卡状态与心得。
- gen_report：基于路线 + 打卡记录生成实践报告（Markdown）。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from .amap import AmapClient, AmapUnavailable
from .llm import chat_json, chat_text, make_client
from .recommender import _haversine_m


class RouteError(RuntimeError):
    pass


class _JsonStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def read(self) -> dict:
        with self.lock:
            if not self.path.exists():
                return {}
            return json.loads(self.path.read_text(encoding="utf-8"))

    def write(self, data: dict):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


class RoutePlanner:
    def __init__(self, cfg: dict, rag, root: Path):
        self.cfg = cfg
        self.spots = cfg.get("spots", {})
        self.amap = AmapClient(cfg.get("amap", {}))
        self.rag = rag
        self.client, self.llm = make_client(cfg)
        self.routes_store = _JsonStore(root / "data" / "routes.json")
        self.checkins_store = _JsonStore(root / "data" / "checkins.json")
        self.reports_dir = root / "data" / "reports"

    # ---------- 路线生成 ----------
    def _candidates(self, city: str, limit: int = 6) -> list[tuple[str, dict]]:
        cands = [(n, m) for n, m in self.spots.items() if m.get("city") == city]
        cands.sort(key=lambda x: 0 if x[1].get("tag") == "red" else 1)
        return cands[:limit]

    def _kb_intro(self, name: str) -> tuple[str, str]:
        """知识库溯源：返回 (文档名, 摘要)"""
        try:
            hits = self.rag.retriever.retrieve(name, 2)
        except Exception:
            return "", ""
        for h in hits:
            if h.get("dense_score", 0) >= 0.4:
                return h["doc"], h["text"][:100]
        return "", ""

    def generate(self, city: str, days: int = 1) -> dict:
        cands = self._candidates(city)
        if not cands:
            raise RouteError(f"暂未收录 {city} 的景点，无法生成路线。")
        if not self.amap.ready:
            raise AmapUnavailable(
                "未配置高德地图 API key，无法计算研学路线的交通段。"
                "请在 .env 填入 AMAP_API_KEY 后重启服务。"
            )

        # 定位 + 知识库背景
        located = []
        for name, meta in cands:
            try:
                found = self.amap.find_spot(name, city)
            except AmapUnavailable:
                raise
            if found:
                doc, text = self._kb_intro(name)
                located.append({
                    "name": name, "location": found["location"],
                    "address": found["address"], "amap_name": found["name"],
                    "intro_doc": doc, "intro_text": text,
                })
        if not located:
            raise RouteError(f"{city} 的景点在高德地图上定位失败，无法编排路线。")

        # LLM 编排路线
        spots_for_llm = [
            {"name": s["name"], "intro": s["intro_text"] or "（知识库暂无详细介绍）"}
            for s in located
        ]
        system = (
            "你是「青年红色筑梦之旅」研学路线策划师，熟悉江淮红色历史。"
            "根据给定的候选景点，为指定城市编排一条研学路线：合理排序（兼顾主题逻辑与地理顺路）、"
            "为每个站点写一句选入理由和一项可现场完成的研学任务（如观察、提问、访谈、打卡答题）。"
            "涉及革命历史与英烈人物表述必须庄重准确。"
            f"路线为期 {days} 天。输出 JSON："
            '{"name": 路线名(15字内), "summary": 路线主题说明(80字内), '
            '"stops": [{"name": 必须从候选景点中选取, "why": 选入理由(50字内), "task": 研学任务(40字内)}]}'
        )
        user = json.dumps({"city": city, "candidates": spots_for_llm}, ensure_ascii=False)
        try:
            plan = chat_json(self.client, self.llm, system, user)
        except Exception as e:
            raise RouteError(f"路线生成失败：{e}")

        valid = {s["name"]: s for s in located}
        stops = []
        for st in plan.get("stops", []):
            base = valid.get(st.get("name", ""))
            if not base:
                continue
            stops.append({**base, "why": st.get("why", ""), "task": st.get("task", "")})
        if not stops:
            raise RouteError("路线编排结果异常，请重试。")

        # 站间交通段（高德路径规划；短途步行、长途驾车）
        legs = []
        for a, b in zip(stops, stops[1:]):
            straight = _haversine_m(*map(float, a["location"].split(",")),
                                    *map(float, b["location"].split(",")))
            mode = "walking" if straight < 1500 else "driving"
            d = self.amap.direction(a["location"], b["location"], mode)
            if d is None:
                d = {"mode": mode, "distance_m": round(straight), "duration_s": None}
            legs.append({"from": a["name"], "to": b["name"], **d})

        route = {
            "id": uuid.uuid4().hex[:8],
            "name": plan.get("name", f"{city}红色研学路线"),
            "city": city,
            "days": days,
            "summary": plan.get("summary", ""),
            "created": time.strftime("%Y-%m-%d %H:%M"),
            "stops": stops,
            "legs": legs,
        }
        routes = self.routes_store.read()
        routes[route["id"]] = route
        self.routes_store.write(routes)
        return route

    # ---------- 打卡 ----------
    def list_routes(self) -> list[dict]:
        routes = self.routes_store.read()
        return [
            {"id": r["id"], "name": r["name"], "city": r["city"], "days": r["days"],
             "summary": r["summary"], "created": r["created"],
             "stop_count": len(r["stops"])}
            for r in sorted(routes.values(), key=lambda x: -x["id"])
        ]

    def get_route(self, route_id: str) -> dict:
        route = self.routes_store.read().get(route_id)
        if not route:
            raise RouteError("路线不存在，请重新生成。")
        return route

    def checkin(self, route_id: str, spot: str, done: bool, note: str = "") -> dict:
        route = self.get_route(route_id)
        if spot not in {s["name"] for s in route["stops"]}:
            raise RouteError("打卡站点不在该路线中。")
        state = self.checkins_store.read()
        rv = state.setdefault(route_id, {})
        rv[spot] = {"done": bool(done), "note": note[:500], "time": time.strftime("%m-%d %H:%M")}
        self.checkins_store.write(state)
        return rv[spot]

    def get_checkins(self, route_id: str) -> dict:
        return self.checkins_store.read().get(route_id, {})

    # ---------- 实践报告 ----------
    def gen_report(self, route_id: str) -> str:
        route = self.get_route(route_id)
        checkins = self.get_checkins(route_id)
        done_count = sum(1 for s in route["stops"]
                         if checkins.get(s["name"], {}).get("done"))
        lines = []
        for s in route["stops"]:
            c = checkins.get(s["name"], {})
            note = c.get("note", "") or "（无心得记录）"
            lines.append(f"【{s['name']}】打卡：{'已完成' if c.get('done') else '未打卡'}；"
                         f"研学任务：{s['task']}；学生心得：{note}")
        system = (
            "你是「青年红色筑梦之旅」实践指导老师。请根据路线信息和学生的打卡记录、心得，"
            "生成一份庄重规范、格式工整的实践报告（Markdown）。结构：# 报告标题、"
            "## 一、实践概况（路线/时间/完成度）、## 二、各站点实践记录（逐站：站点介绍一句、"
            "任务完成情况、学生心得誊录并润色）、## 三、实践感悟（结合红色精神与乡村振兴，"
            "150字内）。引用英烈事迹时保持敬意，不要编造学生未记录的内容。"
        )
        user = json.dumps({
            "路线": f"{route['name']}（{route['city']}，{route['days']}天，共{len(route['stops'])}站，"
                    f"已打卡{done_count}站）",
            "路线简介": route["summary"],
            "打卡记录": lines,
        }, ensure_ascii=False)
        try:
            report = chat_text(self.client, self.llm, system, user)
        except Exception as e:
            raise RouteError(f"报告生成失败：{e}")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / f"{route_id}.md").write_text(report, encoding="utf-8")
        return report
