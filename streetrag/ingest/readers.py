"""Read geodata from multiple formats."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd


LON_COLS = ["Longitude", "longitude", "lon", "lng", "x"]
LAT_COLS = ["Latitude", "latitude", "lat", "y"]


def _find_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def csv_to_gdf(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    lon = _find_col(list(df.columns), LON_COLS)
    lat = _find_col(list(df.columns), LAT_COLS)
    if not lon or not lat:
        raise ValueError(f"No lon/lat columns in {path.name}")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon], df[lat]),
        crs="EPSG:4326",
    )
    return gdf


def read_geodata(path: str | Path, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return csv_to_gdf(path)
    if layer:
        return gpd.read_file(path, layer=layer)
    return gpd.read_file(path)


def detect_geometry_type(gdf: gpd.GeoDataFrame) -> str:
    if gdf.empty or gdf.geometry.is_empty.all():
        return "unknown"
    geom = gdf.geometry.iloc[0]
    return geom.geom_type.lower()


def list_layers(path: Path) -> list[str]:
    try:
        import fiona
        return list(fiona.listlayers(str(path)))
    except Exception:
        return []
