"""高德地图 Web 服务客户端（REST API v3）

需要「Web服务」类型 key（lbs.amap.com 控制台申请），未配置时所有方法
抛出 AmapUnavailable，由上层转为友好提示。
"""
from __future__ import annotations

import os
import time
from urllib.parse import urlencode

import requests


class AmapUnavailable(RuntimeError):
    pass


class AmapClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.key = os.environ.get(cfg.get("api_key_env", "AMAP_API_KEY"), "")
        self.base = "https://restapi.amap.com/v3"
        self.session = requests.Session()
        self._last_call = 0.0
        self._min_interval = float(cfg.get("min_interval", 0.35))  # 个人 key QPS=3，节流防 10021

    @property
    def ready(self) -> bool:
        return bool(self.key)

    def _throttle(self):
        wait = self._min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def _get(self, path: str, **params) -> dict:
        if not self.ready:
            raise AmapUnavailable(
                "未配置高德地图 API key。请在 lbs.amap.com 免费申请「Web服务」类型 key，"
                "填入 .env 的 AMAP_API_KEY 后重启服务。"
            )
        params["key"] = self.key
        for attempt in range(2):
            self._throttle()
            try:
                r = self.session.get(f"{self.base}{path}", params=params, timeout=10)
                data = r.json()
            except Exception as e:
                raise AmapUnavailable(f"高德接口请求失败：{e}")
            if data.get("status") == "1":
                return data
            # QPS 超限：稍等后重试一次
            if data.get("infocode") == "10021" and attempt == 0:
                time.sleep(1.0)
                continue
            raise AmapUnavailable(
                f"高德接口返回错误：{data.get('info', '未知错误')}（{data.get('infocode', '')}）"
            )
        raise AmapUnavailable("高德接口 QPS 超限，请稍后重试。")

    def find_spot(self, name: str, city: str) -> dict | None:
        """按名称+城市检索 POI，返回 {'name','location'(lng,lat),'address'}"""
        data = self._get("/place/text", keywords=name, city=city or "",
                         citylimit="true", offset=5, page=1)
        for poi in data.get("pois", []):
            loc = poi.get("location", "")
            if "," in loc:
                return {
                    "name": poi.get("name") or name,
                    "location": loc,
                    "address": poi.get("address") or "",
                    "pname": poi.get("pname") or "",
                    "cityname": poi.get("cityname") or "",
                }
        return None

    def around(self, location: str, typecode: str, limit: int, radius: int) -> list[dict]:
        """周边搜索，带回评分、人均消费及餐饮特色菜等扩展字段。"""
        data = self._get("/place/around", location=location, types=typecode,
                         radius=radius, offset=max(limit, 10), page=1,
                         sortrule="distance", extensions="all")
        out = []
        for poi in data.get("pois", [])[:limit]:
            biz_ext = poi.get("biz_ext") or {}
            if not isinstance(biz_ext, dict):
                biz_ext = {}
            out.append({
                "name": poi.get("name", ""),
                "address": poi.get("address") or "",
                "location": poi.get("location", ""),
                "tel": poi.get("tel") or "",
                "rating": biz_ext.get("rating") or "",
                "cost": biz_ext.get("cost") or "",
                "tag": poi.get("tag") or "",
            })
        return out

    def static_map_url(self, location: str) -> str:
        """静态地图图片 URL（景点打红点）"""
        return (f"{self.base}/staticmap?location={location}&zoom=14&size=640*300"
                f"&scale=2&markers=mid,0xD8432C,:{location}&key={self.key}")

    def route_map_url(self, stops: list[dict], legs: list[dict]) -> str:
        """生成含顺序编号站点与真实站间轨迹的静态地图。"""
        if not self.ready or not stops:
            return ""
        locations = [s.get("location", "") for s in stops]
        valid_locations = [p for p in locations if "," in p]
        if not valid_locations:
            return ""
        markers = "|".join(
            f"mid,0xAD2930,{i + 1}:{location}"
            for i, location in enumerate(locations[:10]) if "," in location
        )
        route_points = []
        for i in range(len(stops) - 1):
            leg = legs[i] if i < len(legs) else {}
            polyline = leg.get("polyline", "")
            points = [p for p in polyline.split(";") if "," in p] if polyline else []
            if not points:
                points = [locations[i], locations[i + 1]]
            for point in points:
                if not route_points or point != route_points[-1]:
                    route_points.append(point)
        # 控制图片 URL 长度，同时保留路线形状与全部站点端点。
        if len(route_points) > 100:
            last = len(route_points) - 1
            route_points = [route_points[round(i * last / 99)] for i in range(100)]
        params = {
            "size": "1000*440",
            "scale": 2,
            "markers": markers,
            "key": self.key,
        }
        if len(route_points) > 1:
            params["paths"] = f"7,0xAD2930,0.9,,:{';'.join(route_points)}"
        return f"{self.base}/staticmap?{urlencode(params)}"

    def direction(self, origin: str, destination: str, mode: str = "driving") -> dict | None:
        """路径规划：返回 {'distance_m', 'duration_s', 'mode'}，失败返回 None"""
        path = "/direction/driving" if mode == "driving" else "/direction/walking"
        try:
            data = self._get(path, origin=origin, destination=destination, strategy=32 if mode == "driving" else 0)
        except AmapUnavailable:
            raise
        except Exception:
            return None
        routes = data.get("route", {}).get("paths", [])
        if not routes:
            return None
        p = routes[0]
        points = []
        for step in p.get("steps", []):
            points.extend(x for x in step.get("polyline", "").split(";") if "," in x)
        polyline = ";".join(x for i, x in enumerate(points) if i == 0 or x != points[i - 1])
        return {
            "distance_m": round(float(p.get("distance", 0))),
            "duration_s": round(float(p.get("duration", 0))),
            "mode": mode,
            "polyline": polyline,
        }
