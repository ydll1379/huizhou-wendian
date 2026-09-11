"""高德地图 Web 服务客户端（REST API v3）

需要「Web服务」类型 key（lbs.amap.com 控制台申请），未配置时所有方法
抛出 AmapUnavailable，由上层转为友好提示。
"""
from __future__ import annotations

import os

import requests


class AmapUnavailable(RuntimeError):
    pass


class AmapClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.key = os.environ.get(cfg.get("api_key_env", "AMAP_API_KEY"), "")
        self.base = "https://restapi.amap.com/v3"
        self.session = requests.Session()

    @property
    def ready(self) -> bool:
        return bool(self.key)

    def _get(self, path: str, **params) -> dict:
        if not self.ready:
            raise AmapUnavailable(
                "未配置高德地图 API key。请在 lbs.amap.com 免费申请「Web服务」类型 key，"
                "填入 .env 的 AMAP_API_KEY 后重启服务。"
            )
        params["key"] = self.key
        try:
            r = self.session.get(f"{self.base}{path}", params=params, timeout=10)
            data = r.json()
        except Exception as e:
            raise AmapUnavailable(f"高德接口请求失败：{e}")
        if data.get("status") != "1":
            raise AmapUnavailable(
                f"高德接口返回错误：{data.get('info', '未知错误')}（{data.get('infocode', '')}）"
            )
        return data

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
        """周边搜索：返回 [{'name','address','location','tel'}]"""
        data = self._get("/place/around", location=location, types=typecode,
                         radius=radius, offset=max(limit, 10), page=1,
                         sortrule="distance")
        out = []
        for poi in data.get("pois", [])[:limit]:
            out.append({
                "name": poi.get("name", ""),
                "address": poi.get("address") or "",
                "location": poi.get("location", ""),
                "tel": poi.get("tel") or "",
            })
        return out

    def static_map_url(self, location: str) -> str:
        """静态地图图片 URL（景点打红点）"""
        return (f"{self.base}/staticmap?location={location}&zoom=14&size=640*300"
                f"&scale=2&markers=mid,0xD8432C,:{location}&key={self.key}")
