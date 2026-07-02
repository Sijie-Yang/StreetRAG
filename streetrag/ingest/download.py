"""OSMnx network downloader."""

from __future__ import annotations

import re
from pathlib import Path


def _safe_name(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "", text)
    return text or "city"


def download_network(
    city: str,
    *,
    network_type: str = "drive",
    data_dir: str | Path = "data",
) -> Path:
    """Download a city street network and save as a single ``network`` layer GPKG."""
    import geopandas as gpd
    import osmnx as ox

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ox.settings.http_user_agent = "StreetRAG/0.3 (set OSMNX_USER_AGENT env var)"
    ox.settings.use_cache = True

    G = ox.graph_from_place(city, network_type=network_type, simplify=True)
    _, gdf_edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    out_path = data_dir / f"{_safe_name(city)}_{_safe_name(network_type)}.gpkg"
    if out_path.exists():
        out_path.unlink()
    gdf_network = gpd.GeoDataFrame(geometry=gdf_edges.geometry, crs=gdf_edges.crs)
    gdf_network.to_file(out_path, layer="network", driver="GPKG")
    print(f"Saved: {out_path} (layer: network, {len(gdf_network)} segments)")
    return out_path


def download_pois(
    city: str,
    *,
    data_dir: str | Path = "data",
) -> Path:
    """Download OSM POI points into sources/ for integration."""
    import geopandas as gpd
    import osmnx as ox
    import pandas as pd

    data_dir = Path(data_dir)
    sources = data_dir / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    ox.settings.http_user_agent = "StreetRAG/0.3 (set OSMNX_USER_AGENT env var)"
    ox.settings.use_cache = True

    tags = {"amenity": True, "shop": True, "tourism": True, "leisure": True}
    gdf = ox.features_from_place(city, tags=tags)
    if gdf.empty:
        raise RuntimeError(f"No OSM POIs found for {city!r}")

    gdf = gdf.reset_index(drop=True)
    if "geometry" not in gdf.columns:
        raise RuntimeError("OSM POI download returned no geometry column")

    point_mask = gdf.geometry.geom_type.isin(["Point", "MultiPoint"])
    gdf = gdf[point_mask].copy()
    if gdf.empty:
        raise RuntimeError("No point POIs in OSM download")

    def _primary_category(row: pd.Series) -> str:
        for key in ("amenity", "shop", "tourism", "leisure"):
            val = row.get(key)
            if pd.notna(val) and str(val).strip():
                return str(val).strip()
        return "poi"

    gdf["L1_types"] = gdf.apply(_primary_category, axis=1)
    if "name" not in gdf.columns:
        gdf["name"] = ""
    keep = ["L1_types", "name", "geometry"]
    for extra in ("amenity", "shop", "tourism", "leisure"):
        if extra in gdf.columns:
            keep.append(extra)
    out = sources / f"{_safe_name(city)}_osm_pois.gpkg"
    if out.exists():
        out.unlink()
    gdf[keep].to_file(out, driver="GPKG")
    print(f"Saved POIs: {out} ({len(gdf)} points, layer=L1_types/name)")
    return out
