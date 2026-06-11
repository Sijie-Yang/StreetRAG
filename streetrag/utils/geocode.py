"""Geocoding utilities."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd

USER_AGENT = "StreetRAG/0.3 (https://github.com/sijieyang/StreetRAG)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_LAST_CALL_AT = 0.0


def _cache_path(registry_path: str) -> Path:
    return Path(registry_path).parent / "place_cache.json"


def _load_cache(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(p: Path, cache: dict) -> None:
    p.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def target_bbox_wgs84(registry_path: str) -> Optional[list]:
    reg_p = Path(registry_path)
    try:
        reg = json.loads(reg_p.read_text(encoding="utf-8"))
    except Exception:
        return None
    t = reg.get("target_network", "")
    if not t:
        return None
    gpkg = Path(t)
    if not gpkg.is_absolute():
        gpkg = reg_p.parent / gpkg
    if not gpkg.exists():
        return None
    try:
        gdf = gpd.read_file(gpkg, layer=reg.get("target_layer", "edges"))
        if gdf.crs and not gdf.crs.is_geographic:
            gdf = gdf.to_crs("EPSG:4326")
        b = gdf.total_bounds
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    except Exception:
        return None


def _nominatim_request(params: Dict[str, str]) -> List[dict]:
    global _LAST_CALL_AT
    wait = 1.05 - (time.time() - _LAST_CALL_AT)
    if wait > 0:
        time.sleep(wait)
    url = NOMINATIM_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    _LAST_CALL_AT = time.time()
    return data if isinstance(data, list) else []


def _viewbox_from_bbox(bbox: List[float]) -> str:
    xmin, ymin, xmax, ymax = bbox
    return f"{xmin},{ymax},{xmax},{ymin}"


def _expand_bbox(bbox: List[float], factor: float) -> List[float]:
    xmin, ymin, xmax, ymax = bbox
    dx = (xmax - xmin) * factor
    dy = (ymax - ymin) * factor
    return [xmin - dx, ymin - dy, xmax + dx, ymax + dy]


def _result_from_nominatim(row: dict) -> Dict[str, Any]:
    bb = row.get("boundingbox")
    bbox_wgs84 = None
    if bb and len(bb) == 4:
        try:
            bbox_wgs84 = [float(bb[2]), float(bb[0]), float(bb[3]), float(bb[1])]
        except Exception:
            bbox_wgs84 = None
    return {
        "centroid_lonlat": [float(row["lon"]), float(row["lat"])],
        "bbox_wgs84": bbox_wgs84,
        "display_name": str(row.get("display_name") or ""),
        "osm_type": row.get("osm_type"),
        "osm_id": row.get("osm_id"),
    }


def geocode_place(query: str, registry_path: str) -> Dict[str, Any]:
    if not query or not query.strip():
        return {"query": query, "found": False, "error": "empty query"}
    cache_p = _cache_path(registry_path)
    cache = _load_cache(cache_p)
    key = query.strip().lower()
    if key in cache and cache[key].get("found"):
        out = dict(cache[key])
        out["source"] = "cache"
        return out
    bbox = target_bbox_wgs84(registry_path)
    base_params = {"q": query, "format": "json", "limit": "5", "addressdetails": "0"}
    candidates: List[dict] = []
    try:
        if bbox is not None:
            params = dict(base_params)
            params["viewbox"] = _viewbox_from_bbox(_expand_bbox(bbox, 0.1))
            params["bounded"] = "1"
            candidates = _nominatim_request(params)
        if not candidates:
            candidates = _nominatim_request(base_params)
            if bbox is not None and candidates:
                exp = _expand_bbox(bbox, 1.0)
                xmin, ymin, xmax, ymax = exp
                kept = [
                    r for r in candidates
                    if xmin <= float(r["lon"]) <= xmax and ymin <= float(r["lat"]) <= ymax
                ]
                candidates = kept or []
    except Exception as e:
        return {"query": query, "found": False, "error": f"nominatim error: {e}"}
    if not candidates:
        return {"query": query, "found": False, "error": "no results"}
    base = _result_from_nominatim(candidates[0])
    result = {"query": query, "found": True, **base}
    cache[key] = result
    _save_cache(cache_p, cache)
    result["source"] = "nominatim"
    return result
