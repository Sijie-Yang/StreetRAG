"""
StreetRAG web server.

Endpoints:
  GET  /api/health
  GET  /api/edges-geojson      (optional color_by=<column>, returns real value range)
  GET  /api/indices            (list saved indices with metadata for the UI drawer)
  GET  /api/index/{col}        (render a saved index without calling the LLM)
  GET  /api/feature/{name}     (per-edge contribution data for a feature, for hover)
  GET  /api/edge-info          (nearest edge near lon/lat, returns all attrs + index breakdown)
  POST /api/route              (shortest path; weight_col can be length or any index column)
  POST /api/chat               (one-shot natural-language chat)
  POST /api/chat-stream        (SSE progress + final result)

GPKG is read once and cached in memory (invalidated by mtime), so subsequent
requests are O(seconds) instead of re-reading the whole 194MB file.
"""

from __future__ import annotations

import asyncio
import json
import math
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import geopandas as gpd  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, File, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from streetrag.agent.runner import run_agent_query  # noqa: E402
from streetrag.core.feature_catalog import FeatureCatalog  # noqa: E402
from streetrag.core.spatial_utils import ensure_projected  # noqa: E402
from streetrag.core.street_network import StreetNetwork  # noqa: E402
from streetrag.ingest.integrators import INTEGRATORS  # noqa: E402
from streetrag.ingest.pipeline import integrate_source, scan_data_dir  # noqa: E402
from streetrag.ingest.readers import detect_geometry_type, list_layers, read_geodata  # noqa: E402
from streetrag.routing import street_route as _sr  # noqa: E402
from streetrag.utils.geocode import geocode_place  # noqa: E402

app = FastAPI(title="StreetRAG Web", version="0.3.0")
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r".*",
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Registry + cached GPKG loader
# ---------------------------------------------------------------------------

from streetrag.core.workspace import Workspace  # noqa: E402

_WORKSPACE = Workspace(ROOT / "data")
if _WORKSPACE.has_legacy_layout():
    _WORKSPACE.migrate_legacy()


def _default_catalog() -> FeatureCatalog:
    return _WORKSPACE.catalog()


def _default_registry() -> Path:
    return _default_catalog().path


def _resolve_target_gpkg() -> Path:
    catalog = _default_catalog()
    p = catalog.target_network_path
    if not p.exists():
        raise HTTPException(500, f"GPKG not found: {p}")
    return p


_GPKG_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_gpkg(gpkg_path: Path) -> Dict[str, Any]:
    """Cached GPKG loader; invalidated by mtime."""
    key = str(gpkg_path)
    mtime = gpkg_path.stat().st_mtime
    cached = _GPKG_CACHE.get(key)
    if cached and cached["mtime"] == mtime:
        return cached
    print(f"[server] loading GPKG {gpkg_path} (mtime={int(mtime)})")
    gdf_edges = gpd.read_file(gpkg_path, layer="edges")
    gdf_nodes = gpd.read_file(gpkg_path, layer="nodes")
    # The web tier always works in a metric CRS for accurate distances; we re-
    # project to WGS84 only at serialisation time.
    gdf_edges_proj, gdf_nodes_proj = ensure_projected(gdf_edges, gdf_nodes)

    # Pre-compute a 4326 simplified base layer for the basemap streets
    g0 = gdf_edges_proj.copy()
    g0["geometry"] = g0.geometry.simplify(18.0, preserve_topology=True)
    g0_wgs = g0.to_crs("EPSG:4326")
    base_geojson = json.loads(
        g0_wgs[["geometry"]].assign(v=0.0).to_json()
    )

    _GPKG_CACHE[key] = {
        "mtime": mtime,
        "edges": gdf_edges_proj,
        "nodes": gdf_nodes_proj,
        "edges_wgs84": gdf_edges.to_crs("EPSG:4326") if gdf_edges.crs and not gdf_edges.crs.is_geographic else gdf_edges,
        "base_geojson": base_geojson,
        "n_edges": int(len(gdf_edges_proj)),
    }
    return _GPKG_CACHE[key]


def _get_edges() -> gpd.GeoDataFrame:
    return _load_gpkg(_resolve_target_gpkg())["edges"]


# ---------------------------------------------------------------------------
# GeoJSON / colouring helpers
# ---------------------------------------------------------------------------

def _column_value_geojson(
    gdf_edges: gpd.GeoDataFrame,
    col: str,
    *,
    simplify_m: float = 10.0,
) -> Tuple[dict, dict]:
    """Return (GeoJSON FeatureCollection in WGS84, value summary).

    Unlike the previous implementation we KEEP the real values (`v`) in
    properties. The client decides how to map values to colours (it can use the
    summary's percentile cuts for a stable colormap across queries).
    """
    if col not in gdf_edges.columns:
        raise HTTPException(400, f"Unknown column: {col}")
    g = gdf_edges[["geometry", col]].copy()
    g = g[g[col].notna() & g.geometry.notna() & ~g.geometry.is_empty]
    if g.empty:
        return ({"type": "FeatureCollection", "features": []},
                {"col": col, "n": 0})
    if g.crs is None or g.crs.is_geographic:
        g_proj = ensure_projected(g)[0]
    else:
        g_proj = g
    g_proj["geometry"] = g_proj.geometry.simplify(simplify_m, preserve_topology=True)
    g_wgs = g_proj.to_crs("EPSG:4326")
    g_wgs = g_wgs.rename(columns={col: "v"})
    v = pd.to_numeric(g_wgs["v"], errors="coerce")
    g_wgs["v"] = v
    valid = v.dropna()
    summary = {
        "col": col,
        "n": int(len(valid)),
        "min": float(valid.min()) if len(valid) else None,
        "max": float(valid.max()) if len(valid) else None,
        "mean": float(valid.mean()) if len(valid) else None,
        "median": float(valid.median()) if len(valid) else None,
        "p05": float(valid.quantile(0.05)) if len(valid) else None,
        "p25": float(valid.quantile(0.25)) if len(valid) else None,
        "p75": float(valid.quantile(0.75)) if len(valid) else None,
        "p95": float(valid.quantile(0.95)) if len(valid) else None,
    }
    return json.loads(g_wgs.to_json()), summary


def _payload_from_index_pl(pl: dict) -> Dict[str, Any]:
    if not isinstance(pl, dict):
        return {"ok": True, "mode": "index", "no_payload": True, "reply": "完成。"}

    reply = pl.get("reply") or pl.get("explanation", "")
    base = {
        "ok": True,
        "kind": pl.get("kind"),
        "skill_name": pl.get("skill_name"),
        "index_col": pl.get("index_col"),
        "index_name": pl.get("index_name"),
        "reply": reply,
        "operator": pl.get("operator"),
        "normalization": pl.get("normalization"),
        "statistics": pl.get("statistics"),
        "morans_i": pl.get("morans_i"),
        "length_weighted_topk": pl.get("length_weighted_topk"),
        "narrative_evidence": pl.get("narrative_evidence"),
        "features_weights": pl.get("features_weights"),
        "spatial_target": pl.get("spatial_target"),
        "spatial_target_resolved": pl.get("spatial_target_resolved"),
        "spatial_filter_radius_m": pl.get("spatial_filter_radius_m"),
        "explanation": pl.get("explanation", ""),
    }

    # Analysis-only skills (correlate, etc.): text + evidence, no map layer.
    gdf: Optional[gpd.GeoDataFrame] = pl.get("gdf_edges")
    col = pl.get("index_col")
    renderable = (
        pl.get("render_map", bool(col))
        and gdf is not None
        and col
        and col in gdf.columns
    )
    if not renderable:
        return {**base, "mode": "analysis"}

    gj, summary = _column_value_geojson(gdf, col, simplify_m=8.0)
    return {**base, "mode": "index", "value_summary": summary, "geojson": gj}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RouteBody(BaseModel):
    from_lon: float
    from_lat: float
    to_lon: float
    to_lat: float
    weight_col: str = Field(
        "length",
        description="Edge cost column to minimise. 'length' or any index column.",
    )
    weight_mode: str = Field(
        "length_over_index",
        description=(
            "How to turn an index column into edge cost. "
            "'length_over_index' (default): cost = length / max(pctl, eps) — favours high-index streets. "
            "'length' is just length."
        ),
    )


class UnifiedChatBody(BaseModel):
    message: str
    run_route_on_coords: bool = True


class IndexChatBody(BaseModel):
    message: str


RE_COORDS4 = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*[,; ]\s*(-?\d+(?:\.\d+)?)\D+?(-?\d+(?:\.\d+)?)\s*[,; ]\s*(-?\d+(?:\.\d+)?)"
)


def parse_route_coords_in_text(msg: str) -> Optional[Tuple[float, float, float, float]]:
    m = RE_COORDS4.search(msg.strip())
    if not m:
        return None
    a, b, c, d = (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
    def pl(lon, la):
        return -180.0 <= lon <= 180.0 and -90.0 <= la <= 90.0
    if not (pl(a, b) and pl(c, d)):
        return None
    return a, b, c, d


# ---------------------------------------------------------------------------
# Route core
# ---------------------------------------------------------------------------

def _run_route_cached(
    from_lon: float,
    from_lat: float,
    to_lon: float,
    to_lat: float,
    weight_col: str,
    weight_mode: str,
) -> Tuple[dict, float, int, int, str]:
    cache = _load_gpkg(_resolve_target_gpkg())
    gdf_edges = cache["edges"]
    gdf_nodes = cache["nodes"]
    crs = gdf_edges.crs
    pta = gpd.GeoSeries([Point(from_lon, from_lat)], crs="EPSG:4326").to_crs(crs).iloc[0]
    ptb = gpd.GeoSeries([Point(to_lon, to_lat)], crs="EPSG:4326").to_crs(crs).iloc[0]
    s = _sr.nearest_osmid(gdf_nodes, pta)
    t = _sr.nearest_osmid(gdf_nodes, ptb)

    # Build a cost-shaped working frame so street_route can use whatever weight we want.
    gdf_work = gdf_edges.copy()
    cost_col = "_streetrag_cost"
    note = "length"
    if weight_col and weight_col != "length":
        if weight_col not in gdf_work.columns:
            raise HTTPException(400, f"weight_col not found: {weight_col}")
        if "length" not in gdf_work.columns:
            gdf_work["length"] = gdf_work.geometry.length
        v = pd.to_numeric(gdf_work[weight_col], errors="coerce")
        if weight_mode == "length_over_index":
            # Map the column to a non-negative attractiveness in (0, 1] using its
            # rank. Higher rank → lower cost. This works for ANY scale (positive,
            # negative, percentile, z-score) and is monotone in the original values.
            ranks = v.rank(pct=True, method="average")
            attr = ranks.fillna(0.0).clip(lower=1e-3, upper=1.0)
            gdf_work[cost_col] = gdf_work["length"] / attr
            note = f"length / pct_rank({weight_col})"
        else:
            gdf_work[cost_col] = v.fillna(v.median()).clip(lower=1e-6)
            note = weight_col
        use_col = cost_col
    else:
        use_col = "length"

    G = _sr.build_routing_graph(gdf_work, weight_col=use_col, respect_oneway=True)
    path = nx.shortest_path(G, s, t, weight="weight")
    geom, L = _sr.route_geometry_and_length(gdf_work, path)
    if geom is None:
        return {"type": "FeatureCollection", "features": []}, 0.0, s, t, note
    gr = gpd.GeoDataFrame(geometry=[geom], crs=gdf_work.crs)
    g4326 = gr.to_crs("EPSG:4326")
    gj = json.loads(g4326.to_json())
    return gj, float(L), s, t, note


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "root": str(ROOT), "version": app.version}


@app.get("/api/edges-geojson")
def edges_geojson(
    color_by: Optional[str] = Query(None),
    simplify: float = Query(10.0),
) -> dict:
    cache = _load_gpkg(_resolve_target_gpkg())
    if color_by:
        gj, summary = _column_value_geojson(cache["edges"], color_by, simplify_m=simplify)
        return {"geojson": gj, "value_summary": summary,
                "col": color_by, "n_edges": cache["n_edges"]}
    # No column → just the base streets (already precomputed at load time)
    return {
        "geojson": cache["base_geojson"],
        "value_summary": None,
        "col": "base",
        "n_edges": cache["n_edges"],
    }


@app.get("/api/indices")
def list_indices() -> dict:
    items = _default_catalog().list_indices()
    out = []
    for it in items:
        out.append({
            "index_col": it.get("index_col"),
            "index_name": it.get("index_name"),
            "original_query": it.get("original_query"),
            "timestamp": it.get("timestamp"),
            "operator": it.get("operator"),
            "normalization": it.get("normalization"),
            "summary": (it.get("summary") or "")[:280],
            "n_features": len(it.get("features_weights") or {}),
            "morans_i": (it.get("morans_i") or {}).get("I"),
            "statistics": {
                k: it.get("statistics", {}).get(k)
                for k in ("count", "min", "max", "mean", "median", "std")
            },
        })
    # Newest first
    out.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return {"indices": out}


@app.get("/api/index/{col}")
def get_index(col: str, simplify: float = Query(10.0)) -> dict:
    meta = _default_catalog().load_index_record(col) or {}
    gj, summary = _column_value_geojson(_get_edges(), col, simplify_m=simplify)
    return {
        "ok": True,
        "mode": "index",
        "kind": "existing_index",
        "index_col": col,
        "index_name": meta.get("index_name", col),
        "reply": meta.get("summary", ""),
        "explanation": meta.get("explanation", ""),
        "operator": meta.get("operator"),
        "normalization": meta.get("normalization"),
        "statistics": meta.get("statistics"),
        "morans_i": meta.get("morans_i"),
        "length_weighted_topk": meta.get("length_weighted_topk"),
        "features_weights": meta.get("features_weights"),
        "spatial_target": meta.get("spatial_target"),
        "spatial_target_resolved": meta.get("spatial_target_resolved"),
        "spatial_filter_radius_m": meta.get("spatial_filter_radius_m"),
        "value_summary": summary,
        "geojson": gj,
    }


@app.get("/api/geocode")
def geocode(q: str = Query(..., min_length=1)) -> dict:
    return geocode_place(q, str(_default_registry()))


class IntegrateBody(BaseModel):
    filename: str
    method_type: str = "snap_nearest"
    layer: Optional[str] = None
    columns: Optional[List[str]] = None
    method_params: Optional[dict] = None


@app.get("/api/integrators")
def list_integrators() -> dict:
    return {
        "integrators": [
            {"name": k, "class": v.__class__.__name__}
            for k, v in INTEGRATORS.items()
        ]
    }


@app.get("/api/cities")
def list_cities() -> dict:
    return {
        "active": _WORKSPACE.active_city(),
        "cities": _WORKSPACE.list_cities(),
    }


class CityBody(BaseModel):
    name: str


@app.post("/api/cities/activate")
def activate_city(body: CityBody) -> dict:
    try:
        _WORKSPACE.set_active(body.name)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"ok": True, "active": body.name}


@app.get("/api/datasets")
def list_datasets() -> dict:
    """Transparency endpoint: active city, network, source files grouped
    by status, and per-source integrated feature columns."""
    catalog = _default_catalog()
    data_dir = catalog.data_dir
    target_name = catalog.raw.get("target_network", "")

    integrated: Dict[str, dict] = {}
    for block in catalog.point_integrations:
        sf = block.get("source_file")
        if not sf:
            continue
        integrated[sf] = {
            "method": (block.get("integration_method") or {}).get("type"),
            "layer": block.get("source_layer"),
            "columns": list((block.get("columns") or {}).keys()),
        }

    fs = catalog.feature_statistics
    syntax_cols = sorted(
        c for c in fs
        if c.startswith(("integration_R", "angular_", "nain_", "choice_", "nach_"))
    )

    data_exts = {".gpkg", ".geojson", ".shp", ".csv", ".parquet"}
    networks: List[dict] = []
    sources: List[dict] = []
    seen = set()

    def classify(p: Path, in_sources: bool) -> None:
        seen.add(p.name)
        entry: dict = {"name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)}
        if p.name == target_name:
            entry["role"] = "network_active"
            networks.append(entry)
            return
        if not in_sources and re.search(r"_(drive|walk|bike|all|all_private)\.gpkg$", p.name):
            entry["role"] = "network_candidate"
            networks.append(entry)
            return
        if p.name in integrated:
            entry["role"] = "integrated"
            entry.update(integrated[p.name])
        elif (p.stem + ".gpkg") in integrated:
            entry["role"] = "converted"
            entry["note"] = f"已转换为 {p.stem}.gpkg 并整合"
        else:
            entry["role"] = "not_integrated"
        sources.append(entry)

    for p in sorted(data_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in data_exts and not p.name.startswith("."):
            classify(p, in_sources=False)
    src_dir = data_dir / "sources"
    if src_dir.is_dir():
        for p in sorted(src_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in data_exts and not p.name.startswith("."):
                classify(p, in_sources=True)

    for sf, info in integrated.items():
        if sf not in seen:
            sources.append({
                "name": sf, "size_mb": None, "role": "integrated",
                "missing": True, **info,
            })

    n_edges = None
    key = str(catalog.target_network_path)
    if key in _GPKG_CACHE:
        n_edges = _GPKG_CACHE[key]["n_edges"]

    return {
        "city": _WORKSPACE.active_city(),
        "cities": [c["name"] for c in _WORKSPACE.list_cities()],
        "network": {
            "file": target_name,
            "layer": catalog.target_layer,
            "n_edges": n_edges,
            "syntax_radii": catalog.syntax_radii,
            "syntax_columns": syntax_cols,
        },
        "networks": networks,
        "sources": sources,
        "feature_counts": {
            "total": len(fs),
            "space_syntax": len(syntax_cols),
            "integrated": sum(len(v["columns"]) for v in integrated.values()),
            "composite_index": len(catalog.composite_index_columns),
        },
        "multi_city_note": (
            "新增城市：streetrag download --city '<OSM名称>'（自动建目录并激活），"
            "然后 streetrag scan && streetrag integrate。"
        ),
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> dict:
    """Upload geodata file to data/ directory."""
    allowed = {".gpkg", ".geojson", ".json", ".shp", ".csv", ".parquet"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported format: {suffix}. Allowed: {allowed}")
    dest = _default_catalog().sources_dir() / (file.filename or "upload.gpkg")
    content = await file.read()
    dest.write_bytes(content)
    layers: List[str] = []
    gtype = "unknown"
    try:
        if suffix == ".gpkg":
            layers = list_layers(dest)
        gdf = read_geodata(dest, layer=layers[0] if layers else None)
        gtype = detect_geometry_type(gdf)
    except Exception as e:
        return {"ok": True, "filename": dest.name, "warning": str(e)}
    suggested = "snap_nearest"
    if gtype == "point":
        suggested = "buffer_density" if suffix == ".gpkg" else "snap_nearest"
    elif gtype in ("linestring", "multilinestring"):
        suggested = "line_overlay"
    elif gtype in ("polygon", "multipolygon"):
        suggested = "polygon_area_weighted"
    return {
        "ok": True,
        "filename": dest.name,
        "path": str(dest),
        "geometry_type": gtype,
        "layers": layers,
        "suggested_method": suggested,
        "integrators": list(INTEGRATORS.keys()),
    }


@app.post("/api/integrate")
def integrate_uploaded(body: IntegrateBody) -> dict:
    """Integrate an uploaded file onto the street network."""
    catalog = _default_catalog()
    source = catalog.resolve_path(body.filename)
    if not source.exists():
        raise HTTPException(404, f"File not found: {body.filename}")
    net = StreetNetwork.from_catalog(catalog)
    gdf = read_geodata(source, layer=body.layer)
    numeric_cols = body.columns or [
        c for c in gdf.columns
        if c != "geometry" and pd.api.types.is_numeric_dtype(gdf[c])
    ]
    columns = {c: f"Integrated from {body.filename}" for c in numeric_cols}
    net = integrate_source(
        net,
        source,
        method_type=body.method_type,
        columns=columns,
        layer=body.layer,
        method_params=body.method_params or {},
    )
    net.save()
    catalog.save()
    StreetNetwork._CACHE.clear()
    return {
        "ok": True,
        "columns_added": list(columns.keys()),
        "method": body.method_type,
    }


@app.post("/api/scan")
def rescan_data() -> dict:
    cat = _default_catalog()
    catalog = scan_data_dir(cat.data_dir, cat)
    return {"ok": True, "integrations": len(catalog.point_integrations)}


@app.get("/api/edge-info")
def edge_info(
    lon: float = Query(...),
    lat: float = Query(...),
    index_col: Optional[str] = Query(None),
) -> dict:
    """Return nearest edge attributes + index breakdown (top feature contributions)."""
    edges = _get_edges()
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(edges.crs).iloc[0]
    # Use bounding-box pre-filter then exact distance for speed
    try:
        sidx = edges.sindex
        candidates_idx = list(sidx.nearest(pt.bounds, num_results=8))
    except Exception:
        candidates_idx = list(range(min(2000, len(edges))))
    candidates = edges.iloc[candidates_idx] if candidates_idx else edges
    dists = candidates.geometry.distance(pt)
    i = int(dists.idxmin())
    row = edges.loc[i]
    out: Dict[str, Any] = {"edge_id": i, "distance_m": float(dists.min())}
    # Useful basic attributes
    for k in ("name", "highway", "lanes", "maxspeed", "oneway", "length", "u", "v", "osmid"):
        if k in edges.columns:
            val = row[k]
            try:
                if pd.isna(val):
                    val = None
                else:
                    val = (
                        float(val)
                        if isinstance(val, (np.floating, np.integer, float, int))
                        else str(val)
                    )
            except Exception:
                val = str(val)
            out[k] = val
    # Index breakdown
    if index_col:
        meta = _default_catalog().load_index_record(index_col) or {}
        fw = meta.get("features_weights") or {}
        breakdown: List[dict] = []
        for name, weight in fw.items():
            if name in edges.columns:
                raw = row[name]
                try:
                    raw = float(raw) if pd.notna(raw) else None
                except Exception:
                    raw = None
                breakdown.append({
                    "feature": name,
                    "weight": float(weight),
                    "raw_value": raw,
                    "contribution_abs": (abs(float(weight)) if weight is not None else 0.0)
                                          * (abs(raw) if raw is not None else 0.0),
                })
        breakdown.sort(key=lambda r: r["contribution_abs"], reverse=True)
        out["index_col"] = index_col
        out["index_value"] = (
            float(row[index_col]) if index_col in edges.columns and pd.notna(row[index_col])
            else None
        )
        out["top_features"] = breakdown[:5]
    return out


@app.post("/api/route")
def route_api(body: RouteBody) -> Any:
    try:
        gj, L, sn, tn, note = _run_route_cached(
            body.from_lon, body.from_lat, body.to_lon, body.to_lat,
            body.weight_col, body.weight_mode,
        )
    except HTTPException:
        raise
    except nx.NetworkXNoPath as e:
        raise HTTPException(400, f"no path: {e}") from e
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e
    return {
        "ok": True,
        "mode": "route",
        "length_m": L,
        "from_node": sn,
        "to_node": tn,
        "weight_note": note,
        "geojson": gj,
        "reply": f"路线 ~{L:.0f}m（{note}）",
    }


@app.post("/api/index-chat")
def index_chat(body: IndexChatBody) -> Any:
    try:
        pl = run_agent_query(
            _default_catalog(),
            body.message.strip(),
        )
    except SystemExit as e:
        raise HTTPException(500, f"Configuration error: {e}") from e
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e
    if not isinstance(pl, dict):
        pl = {}
    return _payload_from_index_pl(pl)


@app.post("/api/chat")
def unified_chat(body: UnifiedChatBody) -> Any:
    t = body.message or ""
    coords = parse_route_coords_in_text(t) if body.run_route_on_coords else None
    if coords is not None and body.run_route_on_coords:
        flon, flat, tlon, tlat = coords
        return route_api(RouteBody(from_lon=flon, from_lat=flat, to_lon=tlon, to_lat=tlat))
    out = index_chat(IndexChatBody(message=body.message))
    if isinstance(out, dict):
        out = dict(out)
        out["mode"] = "index"
    return out


@app.post("/api/chat-stream")
async def chat_stream(body: UnifiedChatBody) -> StreamingResponse:
    t = (body.message or "").strip()
    if not _default_registry().exists():
        raise HTTPException(500, "data/feature_registry.json not found")

    def sse(obj: dict) -> bytes:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

    coords = parse_route_coords_in_text(t) if body.run_route_on_coords else None
    if coords is not None:
        flon, flat, tlon, tlat = coords

        async def route_bytes():
            try:
                yield sse({
                    "type": "progress", "step": "route",
                    "detail": {"phase": "start", "message": "加载图并求最短路径"},
                })
                gj, L, sn, tn, note = _run_route_cached(flon, flat, tlon, tlat, "length", "length")
                yield sse({
                    "type": "progress", "step": "route",
                    "detail": {"phase": "done", "message": f"~{L:.0f}m"},
                })
                yield sse({
                    "type": "result",
                    "data": {
                        "ok": True, "mode": "route", "length_m": L,
                        "from_node": sn, "to_node": tn,
                        "weight_note": note,
                        "geojson": gj,
                        "reply": f"路线 ~{L:.0f}m（{sn} → {tn}）",
                    },
                })
            except nx.NetworkXNoPath as e:
                yield sse({"type": "error", "message": f"无可达路径: {e}"})
            except Exception as e:
                yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

        return StreamingResponse(route_bytes(), media_type="text/event-stream")

    out_q: queue.Queue = queue.Queue()
    done_sentinel = object()

    def progress_cb(step_id: str, detail: dict) -> None:
        out_q.put({"type": "progress", "step": step_id, "detail": detail})

    def worker() -> None:
        try:
            pl = run_agent_query(
                _default_catalog(),
                t,
                on_progress=progress_cb,
            )
            out_q.put({
                "type": "result",
                "data": _payload_from_index_pl(pl) if isinstance(pl, dict) else _payload_from_index_pl({}),
            })
        except SystemExit as e:
            out_q.put({"type": "error", "message": f"Config: {e}"})
        except Exception as e:
            out_q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            out_q.put(done_sentinel)

    threading.Thread(target=worker, daemon=True).start()

    async def event_iter():
        while True:
            item = await asyncio.to_thread(out_q.get)
            if item is done_sentinel:
                break
            yield sse(item)

    return StreamingResponse(event_iter(), media_type="text/event-stream")


@app.post("/api/chat-stream/", include_in_schema=False)
async def chat_stream_trailing_slash(body: UnifiedChatBody) -> StreamingResponse:
    return await chat_stream(body)


@app.get("/api/chat-stream", include_in_schema=False)
def chat_stream_get_info() -> dict:
    return {
        "error": "do_not_use_get",
        "hint": "Open http://127.0.0.1:8765/ in the browser and use the chat. /api/chat-stream is POST only.",
    }


@app.get("/")
def index_page() -> FileResponse:
    f = STATIC / "index.html"
    if not f.exists():
        raise HTTPException(404, f"Static missing: {f}")
    return FileResponse(f)


if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC), html=False), name="static")
