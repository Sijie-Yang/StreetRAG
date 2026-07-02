"""Zone layers: hex, rect, boundaries, aggregation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

ZONES_GPKG = "zones.gpkg"
ZONES_LAYER = "zones"
ZONES_META = "zones_meta.json"


def _hexagon(center_x: float, center_y: float, radius: float) -> Polygon:
    """Flat-top hexagon."""
    coords = []
    for i in range(6):
        ang = math.pi / 3 * i
        coords.append((center_x + radius * math.cos(ang), center_y + radius * math.sin(ang)))
    coords.append(coords[0])
    return Polygon(coords)


def generate_hex_grid(
    bounds: Tuple[float, float, float, float],
    *,
    radius_m: float = 500.0,
    crs: str = "EPSG:3857",
) -> gpd.GeoDataFrame:
    """Generate hex grid over bounds (projected CRS)."""
    minx, miny, maxx, maxy = bounds
    w = radius_m * 2
    h = radius_m * math.sqrt(3)
    polys = []
    row = 0
    y = miny
    while y <= maxy + h:
      x_off = (w * 0.75) if row % 2 else 0
      x = minx + x_off
      while x <= maxx + w:
        polys.append(_hexagon(x, y, radius_m))
        x += w * 0.75
      y += h * 0.5
      row += 1
    gdf = gpd.GeoDataFrame({"zone_id": range(len(polys))}, geometry=polys, crs=crs)
    return gdf


def generate_rect_grid(
    bounds: Tuple[float, float, float, float],
    *,
    cell_m: float = 500.0,
    rotation_deg: float = 0.0,
    crs: str = "EPSG:3857",
) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bounds
    polys = []
    y = miny
    zid = 0
    while y < maxy:
      x = minx
      while x < maxx:
        polys.append(box(x, y, x + cell_m, y + cell_m))
        zid += 1
        x += cell_m
      y += cell_m
    gdf = gpd.GeoDataFrame({"zone_id": range(len(polys))}, geometry=polys, crs=crs)
    if rotation_deg:
      cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
      gdf["geometry"] = gdf.geometry.rotate(rotation_deg, origin=(cx, cy))
    return gdf


def load_boundary_file(path: Path, *, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def aggregate_edges_to_zones(
    edges: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    columns: List[str],
    *,
    method: str = "length_weighted_mean",
) -> gpd.GeoDataFrame:
    """Aggregate edge columns to zone polygons."""
    if edges.crs != zones.crs:
        edges = edges.to_crs(zones.crs)
    zones = zones.copy()
    for col in columns:
        if col not in edges.columns:
            continue
        vals = pd.to_numeric(edges[col], errors="coerce")
        work = edges[["geometry"]].copy()
        work[col] = vals
        work["length"] = work.geometry.length
        joined = gpd.sjoin(work, zones[["geometry", "zone_id"]], how="inner", predicate="intersects")
        if joined.empty:
            zones[col] = np.nan
            continue
        if method == "length_weighted_mean":
            def _agg(g):
                w = g["length"].fillna(0)
                v = g[col]
                m = w > 0
                if not m.any():
                    return np.nan
                return np.average(v[m], weights=w[m])
            zones[col] = joined.groupby("zone_id").apply(_agg).reindex(zones["zone_id"]).values
        else:
            zones[col] = joined.groupby("zone_id")[col].mean().reindex(zones["zone_id"]).values
    return zones


def save_zones(data_dir: Path, zones: gpd.GeoDataFrame, meta: dict) -> Path:
    out = data_dir / ZONES_GPKG
    zones.to_file(out, layer=ZONES_LAYER, driver="GPKG")
    with open(data_dir / ZONES_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out


def load_zones(data_dir: Path) -> Tuple[Optional[gpd.GeoDataFrame], dict]:
    p = data_dir / ZONES_GPKG
    meta_path = data_dir / ZONES_META
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not p.exists():
        return None, meta
    return gpd.read_file(p, layer=ZONES_LAYER), meta
