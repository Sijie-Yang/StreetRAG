"""Column detection helpers for integration and review indexing."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd

REVIEW_NAME_HINTS = (
    "review",
    "comment",
    "text",
    "content",
    "description",
    "feedback",
    "body",
    "snippet",
)
CATEGORY_NAME_HINTS = (
    "category",
    "type",
    "l1_types",
    "amenity",
    "class",
    "poi_type",
    "place_type",
)
RATING_NAME_HINTS = ("rating", "score", "stars", "star_rating")
NAME_NAME_HINTS = ("name", "title", "place_name", "poi_name")


def _col_lower(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def is_geometry_col(name: str) -> bool:
    return name.lower() in {"geometry", "geom"}


def detect_numeric_columns(gdf: gpd.GeoDataFrame) -> List[str]:
    out: List[str] = []
    for col in gdf.columns:
        if is_geometry_col(col):
            continue
        if pd.api.types.is_numeric_dtype(gdf[col]):
            out.append(col)
    return out


def detect_text_columns(gdf: gpd.GeoDataFrame, *, min_nonempty: int = 1) -> List[str]:
    """Non-numeric string/object columns suitable for review indexing."""
    out: List[str] = []
    for col in gdf.columns:
        if is_geometry_col(col):
            continue
        if pd.api.types.is_numeric_dtype(gdf[col]):
            continue
        series = gdf[col].dropna().astype(str).str.strip()
        series = series[series != ""]
        if len(series) < min_nonempty:
            continue
        low = _col_lower(col)
        hinted = any(h in low for h in REVIEW_NAME_HINTS)
        avg_len = float(series.str.len().mean()) if len(series) else 0.0
        if hinted or avg_len >= 20:
            out.append(col)
    return out


def infer_category_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    for col in gdf.columns:
        if is_geometry_col(col):
            continue
        low = _col_lower(col)
        if any(h in low for h in CATEGORY_NAME_HINTS):
            return col
    return None


def infer_rating_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    for col in gdf.columns:
        if is_geometry_col(col):
            continue
        low = _col_lower(col)
        if any(h == low or low.endswith(f"_{h}") for h in RATING_NAME_HINTS):
            if pd.api.types.is_numeric_dtype(gdf[col]):
                return col
    return None


def infer_name_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    for col in gdf.columns:
        if is_geometry_col(col):
            continue
        low = _col_lower(col)
        if any(h == low or low.endswith(f"_{h}") for h in NAME_NAME_HINTS):
            return col
    return None


def plan_poi_category_columns(
    gdf: gpd.GeoDataFrame,
    *,
    category_col: str,
    radius: float,
    rating_col: Optional[str] = None,
) -> Dict[str, str]:
    """Predict POI category density/rating columns for registry metadata."""
    if category_col not in gdf.columns:
        return {}
    out: Dict[str, str] = {}
    for category in gdf[category_col].dropna().unique():
        safe = str(category).replace(" ", "_").replace("&", "and").replace("/", "_")
        density_col = f"POI_L1_{safe}_density_{int(radius)}m"
        out[density_col] = f"POI count for {category} within {int(radius)}m"
        if rating_col and rating_col in gdf.columns:
            rating_name = f"POI_L1_{safe}_avg_rating_{int(radius)}m"
            out[rating_name] = f"Mean {rating_col} for {category} within {int(radius)}m"
    return out


def summarize_columns(gdf: gpd.GeoDataFrame) -> dict:
    numeric = detect_numeric_columns(gdf)
    text = detect_text_columns(gdf)
    return {
        "numeric_columns": numeric,
        "text_columns": text,
        "category_column": infer_category_column(gdf),
        "rating_column": infer_rating_column(gdf),
        "name_column": infer_name_column(gdf),
    }


def pick_integration_method(
    gtype: str,
    gdf: gpd.GeoDataFrame,
) -> Tuple[str, dict]:
    """Return (method_type, method_params)."""
    if gtype == "point":
        cat = infer_category_column(gdf)
        if cat:
            rating = infer_rating_column(gdf)
            return "poi_category_density_rating", {
                "category_column": cat,
                "rating_column": rating,
                "radius": 500,
            }
        return "snap_nearest", {"k": 5}
    if gtype in ("linestring", "multilinestring"):
        return "line_overlay", {"buffer_m": 20}
    if gtype in ("polygon", "multipolygon"):
        return "polygon_area_weighted", {"buffer_m": 30}
    raise ValueError(f"Unsupported geometry type: {gtype}")
