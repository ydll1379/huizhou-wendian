"""景点识别 + 周边推荐（美食/住宿）+ 文化溯源

- identify_spots：纯配置匹配，从问题文本识别已收录景点（无需 API key）
- Recommender.recommend：高德定位景点 -> 周边 POI -> 每条附知识库文化溯源
"""
from __future__ import annotations

import math

from .amap import AmapClient, AmapUnavailable


def identify_spots(text: str, spots_cfg: dict) -> list[str]:
    """返回问题中出现的规范景点名（最多 3 个）"""
    found = []
    for canonical, meta in spots_cfg.items():
        names = [canonical] + list(meta.get("aliases", []))
        if any(n in text for n in names):
            found.append(canonical)
    return found[:3]


def _resolve_spot(spot: str, spots_cfg: dict) -> tuple[str, dict] | None:
    """别名/规范名 -> (规范名, 注册表项)"""
    if spot in spots_cfg:
        return spot, spots_cfg[spot]
    for canonical, meta in spots_cfg.items():
        if spot in meta.get("aliases", []):
            return canonical, meta
    return None


def _haversine_m(lng1, lat1, lng2, lat2) -> float:
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class Recommender:
    def __init__(self, cfg: dict, rag):
        self.cfg = cfg
        self.spots = cfg.get("spots", {})
        self.amap = AmapClient(cfg.get("amap", {}))
        self.rag = rag  # RAGPipeline：用其检索器做文化溯源

    def kb_trace(self, query: str, k: int = 1, min_score: float = 0.5) -> list[dict]:
        """知识库溯源：检索与 query 相关的原文片段（余弦分过滤）"""
        try:
            hits = self.rag.retriever.retrieve(query, k)
        except Exception:
            return []
        out = []
        for h in hits:
            s = h.get("dense_score", h.get("score", 0))
            if s >= min_score:
                out.append({"doc": h["doc"], "text": h["text"], "score": round(s, 3)})
        return out

    def recommend(self, spot: str, kind: str = "food", limit: int | None = None) -> dict:
        resolved = _resolve_spot(spot, self.spots)
        if not resolved:
            raise ValueError(f"暂未收录景点「{spot}」，目前支持四城的 {len(self.spots)} 处景点。")
        name, meta = resolved

        if not self.amap.ready:
            raise AmapUnavailable(
                "未配置高德地图 API key。请在 lbs.amap.com 免费申请「Web服务」类型 key，"
                "填入 .env 的 AMAP_API_KEY 后重启服务。"
            )

        a_cfg = self.cfg["amap"]
        found = self.amap.find_spot(name, meta.get("city", ""))
        if not found:
            raise ValueError(f"高德地图未能定位「{name}」，请检查景点名称。")

        is_food = kind != "hotel"
        typecode = a_cfg["types"]["food" if is_food else "hotel"]
        limit = limit or (a_cfg["food_limit"] if is_food else a_cfg["hotel_limit"])
        radius = a_cfg["radius"]

        lng1, lat1 = map(float, found["location"].split(","))
        items = []
        for p in self.amap.around(found["location"], typecode, limit, radius):
            dist = None
            if "," in p["location"]:
                lng2, lat2 = map(float, p["location"].split(","))
                dist = round(_haversine_m(lng1, lat1, lng2, lat2))
            # 逐条溯源：对以菜品/地标命名的 POI（如"三河米饺"）能命中知识库
            trace = self.kb_trace(p["name"], k=1, min_score=0.5)
            items.append({
                **p,
                "distance_m": dist,
                "trace": trace,
                "nav_url": (f"https://uri.amap.com/marker?position={p['location']}&name={p['name']}"
                            if p["location"] else ""),
            })

        intro = self.kb_context_intro(name)
        return {
            "spot": name,
            "city": meta.get("city", ""),
            "location": found["location"],
            "amap_name": found["name"],
            "address": found["address"],
            "kind": "food" if is_food else "hotel",
            "map_url": self.amap.static_map_url(found["location"]),
            "intro": intro,
            "items": items,
        }

    def kb_context_intro(self, name: str) -> dict | None:
        """景点级文化背景（推荐列表的『为什么值得来』）"""
        hits = self.kb_trace(name, k=1, min_score=0.3)
        return hits[0] if hits else None
