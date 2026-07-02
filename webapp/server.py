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
  POST /api/create-index       (create composite index from a confirmed proposal)

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

from streetrag.agent.loop import run_agent_loop  # noqa: E402
from streetrag.agent.runner import run_agent_query  # noqa: E402
from streetrag.skills.composite_index import CompositeIndexParams  # noqa: E402
from streetrag.skills.registry import register_all_skills, run_skill  # noqa: E402
from streetrag.core.feature_catalog import FeatureCatalog  # noqa: E402
from streetrag.core.feature_store import FeatureStore  # noqa: E402
from streetrag.core.network_gpkg import EDGE_ID_COL, list_network_column_names, read_network_gpkg  # noqa: E402
from streetrag.core.spatial_utils import ensure_projected  # noqa: E402
from streetrag.core.street_network import StreetNetwork  # noqa: E402
from streetrag.ingest.integrators import INTEGRATORS  # noqa: E402
from streetrag.ingest.columns import detect_text_columns, summarize_columns
from streetrag.ingest.pipeline import integrate_uploaded_file, scan_data_dir  # noqa: E402
from streetrag.ingest.readers import detect_geometry_type, list_layers, read_geodata  # noqa: E402
from streetrag.routing import street_route as _sr  # noqa: E402
from streetrag.utils.geocode import geocode_place  # noqa: E402
from streetrag.zones.layers import (  # noqa: E402
    aggregate_edges_to_zones,
    generate_hex_grid,
    generate_rect_grid,
    load_boundary_file,
    load_zones,
    save_zones,
)

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

from streetrag.core.bootstrap import ensure_default_city, setup_city_from_osm  # noqa: E402
from streetrag.core.settings_store import get_public_settings, save_local_settings  # noqa: E402
from streetrag.core.workspace import Workspace  # noqa: E402

_WORKSPACE = Workspace(ROOT / "data")
if _WORKSPACE.has_legacy_layout():
    _WORKSPACE.migrate_legacy()


@app.on_event("startup")
def _startup_bootstrap() -> None:
    """Ensure Singapore (or existing city) is ready when the UI opens."""
    try:
        name = ensure_default_city(_WORKSPACE)
        if name:
            print(f"[server] active city: {name}")
    except Exception as exc:
        print(f"[server] bootstrap warning: {exc}")


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


_NET_CACHE: Dict[str, Dict[str, Any]] = {}
_NET_LOAD_LOCK = threading.Lock()


def _invalidate_network_cache() -> None:
    _NET_CACHE.clear()
    StreetNetwork.clear_cache()


def _gpkg_column_names(gpkg_path: Path, *, layer: Optional[str] = None) -> set:
    """Lightweight column listing for metadata endpoints (no geometry load)."""
    try:
        return set(list_network_column_names(gpkg_path, edges_layer=layer))
    except Exception:
        return set()


def _catalog_edge_column_names(catalog: FeatureCatalog) -> set:
    if not catalog.target_network_path.exists():
        return set()
    cols = _gpkg_column_names(catalog.target_network_path, layer=catalog.target_layer)
    # Feature columns may live in parquet even before storage_layout flag is set.
    cols |= FeatureStore(catalog).list_feature_column_names()
    return cols


def _load_render_cache(
    *,
    color_by: Optional[str] = None,
    full_features: bool = False,
) -> Dict[str, Any]:
    """Unified StreetNetwork render cache (topology-only when possible)."""
    catalog = _default_catalog()
    gpkg = catalog.target_network_path
    key = str(gpkg.resolve())
    topology_only = (
        catalog.uses_split_storage()
        and not full_features
        and not color_by
    )
    feat_cols = None if full_features or topology_only else ([color_by] if color_by else None)
    token = (
        StreetNetwork._cache_token(catalog, gpkg),
        topology_only,
        full_features,
        tuple(feat_cols or ()),
    )
    cached = _NET_CACHE.get(key)
    if cached and cached.get("token") == token:
        return cached

    with _NET_LOAD_LOCK:
        cached = _NET_CACHE.get(key)
        if cached and cached.get("token") == token:
            return cached
        print(
            f"[server] loading network {gpkg.name}"
            + (" (topology only)" if topology_only else "")
            + (f" col={color_by}" if color_by else "")
        )
        if topology_only:
            edges, nodes = read_network_gpkg(gpkg, edges_layer=catalog.target_layer)
            edges, nodes = ensure_projected(edges, nodes)
        else:
            net = StreetNetwork.from_catalog(catalog, feature_columns=feat_cols)
            edges, nodes = net.edges, net.nodes

        g0 = edges.copy()
        g0["geometry"] = g0.geometry.simplify(18.0, preserve_topology=True)
        g0_wgs = g0.to_crs("EPSG:4326")
        base_geojson = json.loads(
            g0_wgs[["geometry"]].assign(v=0.0).to_json()
        )

        _NET_CACHE[key] = {
            "token": token,
            "edges": edges,
            "nodes": nodes,
            "base_geojson": base_geojson,
            "n_edges": int(len(edges)),
        }
        print(f"[server] network ready: {gpkg.name} ({_NET_CACHE[key]['n_edges']} edges)")
        return _NET_CACHE[key]


def _get_edges(
    *,
    color_by: Optional[str] = None,
    full_features: bool = False,
) -> gpd.GeoDataFrame:
    return _load_render_cache(color_by=color_by, full_features=full_features)["edges"]


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
        return {"ok": True, "mode": "index", "no_payload": True, "reply": "Done."}

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


class ChatContextItem(BaseModel):
    type: str  # feature | index | file
    name: str


class UnifiedChatBody(BaseModel):
    message: str
    run_route_on_coords: bool = True
    context: List[ChatContextItem] = Field(default_factory=list)
    history: List[dict] = Field(default_factory=list)


class IndexChatBody(BaseModel):
    message: str


class ProposalFeature(BaseModel):
    name: str
    weight: float
    rationale: str = ""


class CreateIndexBody(BaseModel):
    index_name: str
    index_col: Optional[str] = None
    features: List[ProposalFeature]
    operator: str = "weighted_sum"
    normalization: str = "robust"
    spatial_target: Optional[str] = None
    spatial_filter_radius_m: Optional[float] = None
    proximity_dominant: bool = False
    explanation: str = ""
    user_query: str = ""


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
    cache = _load_render_cache(
        color_by=weight_col if weight_col and weight_col != "length" else None,
        full_features=bool(weight_col and weight_col != "length"),
    )
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
    api = get_public_settings(_WORKSPACE)
    return {
        "ok": True,
        "root": str(ROOT),
        "version": app.version,
        "api_configured": api.get("configured", False),
    }


class SettingsUpdateBody(BaseModel):
    openai_api_key: Optional[str] = Field(
        None,
        description="OpenAI API key; empty string = do not change saved key",
    )
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    clear_api_key: bool = False


@app.get("/api/settings")
def get_settings() -> dict:
    return {"ok": True, **get_public_settings(_WORKSPACE)}


@app.post("/api/settings")
def update_settings(body: SettingsUpdateBody) -> dict:
    """Save API key and model prefs to data/RAG_setting.local.json (gitignored)."""
    updates: dict = {}
    if body.clear_api_key:
        updates["openai_api_key"] = ""
    elif body.openai_api_key and body.openai_api_key.strip():
        updates["openai_api_key"] = body.openai_api_key.strip()
    if body.llm_model and body.llm_model.strip():
        updates["llm_model"] = body.llm_model.strip()
    if body.embedding_model and body.embedding_model.strip():
        updates["embedding_model"] = body.embedding_model.strip()
    if not updates:
        return {"ok": True, **get_public_settings(_WORKSPACE), "unchanged": True}
    save_local_settings(_WORKSPACE.root, updates)
    return {"ok": True, **get_public_settings(_WORKSPACE), "saved": True}


@app.get("/api/edges-geojson")
def edges_geojson(
    color_by: Optional[str] = Query(None),
    simplify: float = Query(10.0),
) -> dict:
    cache = _load_render_cache(color_by=color_by)
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
    gj, summary = _column_value_geojson(
        _get_edges(color_by=col, full_features=True), col, simplify_m=simplify
    )
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


class AddCityBody(BaseModel):
    osm_query: str = Field(..., min_length=1, description="OSM place query, e.g. 'London, UK'")
    city_slug: Optional[str] = Field(None, description="Directory name; auto-derived if omitted")
    network_type: str = Field("drive", description="OSM network type")
    with_pois: bool = Field(False, description="Also download OSM POIs into sources/")
    run_syntax: bool = Field(False, description="Compute space syntax during integrate (slow)")
    syntax_radii: Optional[str] = Field(
        "500,1500,4500",
        description="Comma-separated syntax radii in meters",
    )


class SyntaxRunBody(BaseModel):
    radii: Optional[str] = Field(
        None,
        description="Comma-separated radii in meters; uses city default if omitted",
    )


@app.post("/api/cities/add-stream")
async def add_city_stream(body: AddCityBody) -> StreamingResponse:
    """Download a new city from OSM, scan, integrate, and activate (SSE progress)."""

    def sse(obj: dict) -> bytes:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

    out_q: queue.Queue = queue.Queue()
    done_sentinel = object()

    def worker() -> None:
        try:
            def prog(step: str, detail: dict) -> None:
                out_q.put({"type": "progress", "step": step, "detail": detail})

            slug = setup_city_from_osm(
                _WORKSPACE,
                body.osm_query.strip(),
                city_slug=body.city_slug,
                network_type=body.network_type,
                with_pois=body.with_pois,
                run_syntax=body.run_syntax,
                syntax_radii=body.syntax_radii,
                on_progress=prog,
            )
            _invalidate_network_cache()
            out_q.put({"type": "result", "data": {"ok": True, "city": slug}})
        except Exception as exc:
            out_q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
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


@app.post("/api/cities/activate")
def activate_city(body: CityBody) -> dict:
    try:
        _WORKSPACE.set_active(body.name)
    except ValueError as e:
        raise HTTPException(404, str(e))
    _invalidate_network_cache()
    return {"ok": True, "active": body.name}


@app.get("/api/syntax/config")
def syntax_config() -> dict:
    """Current city's syntax radii and computed column counts."""
    try:
        catalog = _default_catalog()
    except FileNotFoundError:
        return {
            "city": _WORKSPACE.active_city(),
            "radii": FeatureCatalog.DEFAULT_RADII,
            "default_radii": FeatureCatalog.DEFAULT_RADII,
            "syntax_columns": 0,
            "has_syntax": False,
            "ready": False,
        }
    fs = catalog.feature_statistics
    syntax_cols = sorted(
        c for c in fs
        if c.startswith(("integration_R", "angular_", "nain_", "choice_", "nach_"))
    )
    return {
        "city": _WORKSPACE.active_city(),
        "radii": catalog.syntax_radii,
        "default_radii": FeatureCatalog.DEFAULT_RADII,
        "syntax_columns": len(syntax_cols),
        "has_syntax": len(syntax_cols) > 0,
        "ready": bool(catalog.raw.get("target_network")),
    }


@app.post("/api/syntax/run-stream")
async def run_syntax_stream(body: SyntaxRunBody) -> StreamingResponse:
    """Update radii (optional) and compute space syntax for the active city (SSE)."""
    from streetrag.syntax.runner import run_syntax_for_catalog

    def sse(obj: dict) -> bytes:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

    out_q: queue.Queue = queue.Queue()
    done_sentinel = object()

    def worker() -> None:
        try:
            catalog = _default_catalog()

            def prog(step: str, detail: dict) -> None:
                out_q.put({"type": "progress", "step": step, "detail": detail})

            result = run_syntax_for_catalog(
                catalog,
                radii=body.radii,
                on_progress=prog,
            )
            _invalidate_network_cache()
            out_q.put({"type": "result", "data": {"ok": True, **result}})
        except Exception as exc:
            out_q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
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


@app.get("/api/datasets")
def list_datasets() -> dict:
    """Transparency endpoint: active city, network, source files grouped
    by status, and per-source integrated feature columns."""
    try:
        catalog = _default_catalog()
    except FileNotFoundError:
        return {
            "city": _WORKSPACE.active_city(),
            "cities": [c["name"] for c in _WORKSPACE.list_cities()],
            "data_dir": None,
            "network": {},
            "networks": [],
            "sources": [],
            "feature_counts": {},
            "ready": False,
            "multi_city_note": "No city data yet. Use + City to download a street network.",
        }

    data_dir = catalog.data_dir
    target_name = catalog.raw.get("target_network", "")

    stored_cols: set = set()
    if target_name:
        try:
            stored_cols = _catalog_edge_column_names(catalog)
        except Exception:
            pass

    integrated: Dict[str, dict] = {}
    for block in catalog.point_integrations:
        sf = block.get("source_file")
        if not sf:
            continue
        reg_cols = list((block.get("columns") or {}).keys())
        present_cols = [c for c in reg_cols if c in stored_cols]
        integrated[sf] = {
            "method": (block.get("integration_method") or {}).get("type"),
            "layer": block.get("source_layer"),
            "columns": present_cols,
            "registered_columns": reg_cols,
            "columns_in_gpkg": len(present_cols),
            "columns_pending": max(0, len(reg_cols) - len(present_cols)),
            "text_columns": block.get("text_columns") or [],
        }

    text_indexed: Dict[str, dict] = {}
    for block in catalog.text_integrations:
        sf = block.get("source_file")
        if not sf:
            continue
        text_indexed[sf] = {
            "text_columns": block.get("text_columns") or [],
            "n_chunks": block.get("n_chunks", 0),
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
            info = integrated[p.name]
            pending = info.get("columns_pending", 0)
            if pending and not info.get("columns_in_gpkg"):
                entry["role"] = "pending_integration"
            elif pending:
                entry["role"] = "partial_integration"
            else:
                entry["role"] = "integrated"
            entry.update(info)
            if p.name in text_indexed:
                entry["text_index"] = text_indexed[p.name]
        elif (p.stem + ".gpkg") in integrated:
            entry["role"] = "converted"
            entry["note"] = f"Converted to {p.stem}.gpkg and integrated"
        else:
            entry["role"] = "not_integrated"
            if p.name in text_indexed:
                entry["text_index"] = text_indexed[p.name]
                entry["role"] = "text_only"
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
    key = str(catalog.target_network_path.resolve())
    if key in _NET_CACHE:
        n_edges = _NET_CACHE[key]["n_edges"]
    elif catalog.target_network_path.exists():
        try:
            import fiona

            with fiona.open(
                catalog.target_network_path, layer=catalog.target_layer
            ) as src:
                n_edges = len(src)
        except Exception:
            pass

    return {
        "city": _WORKSPACE.active_city(),
        "cities": [c["name"] for c in _WORKSPACE.list_cities()],
        "city_details": _WORKSPACE.list_cities(),
        "data_dir": str(data_dir),
        "ready": bool(target_name),
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
            "integrated": sum(v.get("columns_in_gpkg", len(v["columns"])) for v in integrated.values()),
            "composite_index": len(catalog.composite_index_columns),
            "review_chunks": sum(v.get("n_chunks", 0) for v in text_indexed.values()),
        },
        "multi_city_note": (
            "Default example city: Singapore. Use + City to download from OSM, "
            "or switch cities in the dropdown above."
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
    column_summary = {}
    text_columns: List[str] = []
    if gtype != "unknown":
        try:
            column_summary = summarize_columns(gdf)
            text_columns = column_summary.get("text_columns") or []
            if gtype == "point" and column_summary.get("category_column"):
                suggested = "poi_category_density_rating"
            elif gtype == "point":
                suggested = "buffer_density"
            elif gtype in ("linestring", "multilinestring"):
                suggested = "line_overlay"
            elif gtype in ("polygon", "multipolygon"):
                suggested = "polygon_area_weighted"
        except Exception:
            pass
    return {
        "ok": True,
        "filename": dest.name,
        "path": str(dest),
        "geometry_type": gtype,
        "layers": layers,
        "suggested_method": suggested,
        "integrators": list(INTEGRATORS.keys()),
        "numeric_columns": column_summary.get("numeric_columns") or [],
        "text_columns": text_columns,
        "category_column": column_summary.get("category_column"),
        "rating_column": column_summary.get("rating_column"),
        "text_detected_message": (
            f"Detected {len(text_columns)} text column(s); a review index will be built on integrate."
            if text_columns
            else None
        ),
    }


@app.post("/api/integrate")
def integrate_uploaded(body: IntegrateBody) -> dict:
    """Integrate an uploaded file onto the street network."""
    catalog = _default_catalog()
    try:
        result = integrate_uploaded_file(
            catalog,
            filename=body.filename,
            method_type=body.method_type,
            layer=body.layer,
            columns=(
                {c: f"Integrated from {body.filename}" for c in body.columns}
                if body.columns
                else None
            ),
            method_params=body.method_params or {},
            index_reviews=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    _invalidate_network_cache()
    return {"ok": True, **result}


@app.post("/api/scan")
def rescan_data() -> dict:
    cat = _default_catalog()
    catalog = scan_data_dir(cat.data_dir, cat)
    return {"ok": True, "integrations": len(catalog.point_integrations)}


def _drop_gpkg_columns(catalog: FeatureCatalog, columns: List[str]) -> None:
    if catalog.uses_split_storage():
        store = FeatureStore(catalog)
        store.drop_columns(columns)
    else:
        net = StreetNetwork.from_catalog(catalog, use_cache=False)
        drop = [c for c in columns if c in net.edges.columns and c != "geometry"]
        if drop:
            net.edges = net.edges.drop(columns=drop)
            net.save()
    catalog.save()
    _invalidate_network_cache()


@app.get("/api/features")
def list_features_api() -> dict:
    catalog = _default_catalog()
    stored_cols = _catalog_edge_column_names(catalog)
    feats = catalog.list_all_features()
    for f in feats:
        f["in_gpkg"] = f["name"] in stored_cols
        f["in_storage"] = f["in_gpkg"]
    return {"ok": True, "features": feats, "n": len(feats)}


@app.delete("/api/features/{col}")
def delete_feature_api(col: str) -> dict:
    catalog = _default_catalog()
    removed = catalog.remove_feature_stats(col)
    _drop_gpkg_columns(catalog, removed)
    return {"ok": True, "removed": removed}


@app.delete("/api/indices/{col}")
def delete_index_api(col: str) -> dict:
    catalog = _default_catalog()
    existed = catalog.remove_index(col)
    removed = catalog.related_columns(col)
    _drop_gpkg_columns(catalog, removed)
    return {"ok": True, "existed": existed, "removed": removed}


@app.delete("/api/files/{name}")
def delete_file_api(name: str) -> dict:
    catalog = _default_catalog()
    path = catalog.resolve_path(name)
    if not path.exists():
        path = catalog.sources_dir() / name
    if not path.exists():
        raise HTTPException(404, f"File not found: {name}")
    block = catalog.remove_point_integration(name)
    if path.name == catalog.raw.get("target_network"):
        raise HTTPException(400, "Cannot delete active network GPKG")
    path.unlink()
    catalog.save()
    return {"ok": True, "deleted": name, "integration_removed": block is not None}


class FileIntegrateBody(BaseModel):
    method_type: Optional[str] = None
    layer: Optional[str] = None
    method_params: Optional[dict] = None


@app.post("/api/files/{name}/integrate-stream")
async def integrate_file_stream(name: str, body: FileIntegrateBody) -> StreamingResponse:
    catalog = _default_catalog()

    def sse(obj: dict) -> bytes:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

    out_q: queue.Queue = queue.Queue()
    done_sentinel = object()

    def worker() -> None:
        try:
            out_q.put({
                "type": "progress",
                "step": "integrate",
                "detail": {"phase": "start", "pct": 0, "message": f"Starting {name}", "file": name},
            })
            block = next(
                (b for b in catalog.point_integrations if b.get("source_file") == name),
                None,
            )
            method = body.method_type or (block or {}).get("integration_method", {}).get("type", "snap_nearest")
            layer = body.layer or (block or {}).get("source_layer")
            params = body.method_params or {
                k: v for k, v in ((block or {}).get("integration_method") or {}).items() if k != "type"
            }

            def prog(step: str, detail: dict) -> None:
                out_q.put({"type": "progress", "step": step, "detail": detail})

            result = integrate_uploaded_file(
                catalog,
                filename=name,
                method_type=method,
                layer=layer,
                method_params=params,
                index_reviews=True,
                on_progress=prog,
            )
            _invalidate_network_cache()
            out_q.put({"type": "result", "data": {"ok": True, **result}})
        except Exception as exc:
            out_q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
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


@app.post("/api/files/{name}/reintegrate-stream")
async def reintegrate_file_stream(name: str, body: FileIntegrateBody) -> StreamingResponse:
    catalog = _default_catalog()
    block = catalog.remove_point_integration(name)

    def sse(obj: dict) -> bytes:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

    out_q: queue.Queue = queue.Queue()
    done_sentinel = object()

    def worker() -> None:
        try:
            if block:
                cols = list((block.get("columns") or {}).keys())
                out_q.put({
                    "type": "progress",
                    "step": "reintegrate",
                    "detail": {"phase": "clear", "pct": 10, "message": "Clearing old columns…"},
                })
                _drop_gpkg_columns(catalog, cols)
            out_q.put({
                "type": "progress",
                "step": "reintegrate",
                "detail": {"phase": "integrate", "pct": 15, "message": f"Reintegrating {name}…"},
            })
            method = body.method_type or (block or {}).get("integration_method", {}).get("type", "snap_nearest")
            layer = body.layer or (block or {}).get("source_layer")
            params = body.method_params or {
                k: v for k, v in ((block or {}).get("integration_method") or {}).items() if k != "type"
            }

            def prog(step: str, detail: dict) -> None:
                out_q.put({"type": "progress", "step": step, "detail": detail})

            result = integrate_uploaded_file(
                catalog,
                filename=name,
                method_type=method,
                layer=layer,
                method_params=params,
                index_reviews=True,
                on_progress=prog,
            )
            _invalidate_network_cache()
            out_q.put({"type": "result", "data": {"ok": True, **result}})
        except Exception as exc:
            out_q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
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


class ZoneCreateBody(BaseModel):
    kind: str  # hex | rect | boundary
    radius_m: Optional[float] = 500.0
    cell_m: Optional[float] = 500.0
    rotation_deg: Optional[float] = 0.0
    boundary_file: Optional[str] = None
    aggregate_columns: List[str] = Field(default_factory=list)


@app.get("/api/zones")
def get_zones() -> dict:
    catalog = _default_catalog()
    zones, meta = load_zones(catalog.data_dir)
    if zones is None:
        return {"ok": True, "zones": None, "meta": meta}
    gj = json.loads(zones.to_crs("EPSG:4326").to_json())
    return {"ok": True, "geojson": gj, "meta": meta, "n": len(zones)}


@app.post("/api/zones/create")
def create_zones(body: ZoneCreateBody) -> dict:
    catalog = _default_catalog()
    net = StreetNetwork.from_catalog(catalog)
    edges = net.edges
    if edges.crs is None or edges.crs.is_geographic:
        edges = edges.to_crs(epsg=3857)
    bounds = edges.total_bounds

    if body.kind == "hex":
        zones = generate_hex_grid(bounds, radius_m=float(body.radius_m or 500), crs=str(edges.crs))
    elif body.kind == "rect":
        zones = generate_rect_grid(
            bounds,
            cell_m=float(body.cell_m or 500),
            rotation_deg=float(body.rotation_deg or 0),
            crs=str(edges.crs),
        )
    elif body.kind == "boundary":
        if not body.boundary_file:
            raise HTTPException(400, "boundary_file required")
        zones = load_boundary_file(catalog.resolve_path(body.boundary_file))
        zones = zones.to_crs(edges.crs)
        if "zone_id" not in zones.columns:
            zones["zone_id"] = range(len(zones))
    else:
        raise HTTPException(400, f"Unknown zone kind: {body.kind}")

    cols = body.aggregate_columns or []
    if cols:
        zones = aggregate_edges_to_zones(edges, zones, cols)
    meta = {"kind": body.kind, "columns": cols, "n_zones": len(zones)}
    save_zones(catalog.data_dir, zones, meta)
    gj = json.loads(zones.to_crs("EPSG:4326").to_json())
    return {"ok": True, "geojson": gj, "meta": meta}


@app.delete("/api/zones")
def delete_zones() -> dict:
    catalog = _default_catalog()
    for name in ("zones.gpkg", "zones_meta.json"):
        p = catalog.data_dir / name
        if p.exists():
            p.unlink()
    return {"ok": True}


@app.get("/api/edge-info")
def edge_info(
    lon: float = Query(...),
    lat: float = Query(...),
    index_col: Optional[str] = Query(None),
) -> dict:
    """Return nearest edge attributes + index breakdown (top feature contributions)."""
    edges = _get_edges(full_features=bool(index_col))
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
    edge_id_val = int(row[EDGE_ID_COL]) if EDGE_ID_COL in row.index else int(i)
    out: Dict[str, Any] = {"edge_id": edge_id_val, "distance_m": float(dists.min())}
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
        "reply": f"Route ~{L:.0f}m ({note})",
    }


@app.post("/api/create-index")
def create_index_api(body: CreateIndexBody) -> Any:
    """Create a composite index directly from a user-confirmed proposal."""
    try:
        register_all_skills()
        catalog = _default_catalog()
        net = StreetNetwork.from_catalog(catalog)
        index_name = (body.index_name or body.index_col or "composite_index").strip()
        params = CompositeIndexParams(
            intent="create_new",
            index_name=index_name,
            features=[f.model_dump() for f in body.features],
            operator=body.operator,
            normalization=body.normalization,
            spatial_target=body.spatial_target,
            spatial_filter_radius_m=body.spatial_filter_radius_m,
            proximity_dominant=body.proximity_dominant,
            explanation=body.explanation,
            user_query=body.user_query or f"Create index {index_name}",
        )
        result = run_skill("composite_index", net, params.model_dump())
        _invalidate_network_cache()
        pl = {
            "kind": "new_index",
            "render_map": result.render_map,
            "index_col": result.index_col,
            "index_name": result.index_name,
            "reply": result.reply,
            "statistics": result.stats,
            "morans_i": (result.narrative_evidence or {}).get("morans_i"),
            "length_weighted_topk": (result.narrative_evidence or {}).get("length_weighted_topk"),
            "narrative_evidence": result.narrative_evidence,
            "skill_name": result.skill_name,
            "explanation": result.explanation,
            "operator": result.operator,
            "normalization": result.normalization,
            "features_weights": result.features_weights,
            "spatial_target": result.spatial_target,
            "spatial_target_resolved": result.spatial_target_resolved,
            "spatial_filter_radius_m": result.spatial_filter_radius_m,
            "gdf_edges": net.edges,
        }
        return _payload_from_index_pl(pl)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e


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
                    "detail": {"phase": "start", "message": "Loading graph and shortest path"},
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
                        "reply": f"Route ~{L:.0f}m ({sn} → {tn})",
                    },
                })
            except nx.NetworkXNoPath as e:
                yield sse({"type": "error", "message": f"No path: {e}"})
            except Exception as e:
                yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

        return StreamingResponse(route_bytes(), media_type="text/event-stream")

    out_q: queue.Queue = queue.Queue()
    done_sentinel = object()
    ctx = [c.model_dump() for c in body.context] if body.context else []

    def worker() -> None:
        try:
            final = None
            for ev in run_agent_loop(
                _default_catalog(),
                t,
                context=ctx,
                history=body.history,
            ):
                out_q.put(ev)
                if ev.get("type") == "done":
                    final = ev.get("data")
            if final is not None:
                # Back-compat result envelope for map rendering
                out_q.put({
                    "type": "result",
                    "data": {
                        "ok": True,
                        "mode": "agent",
                        "reply": final.get("reply", ""),
                        "index_col": final.get("index_col"),
                        "render_map": final.get("render_map", False),
                        "skill_name": final.get("skill_name"),
                    },
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
            if item.get("type") == "done":
                continue  # already forwarded as result
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
