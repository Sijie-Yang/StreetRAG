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
    """Download a city street network and save as nodes/edges GPKG."""
    import osmnx as ox

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ox.settings.http_user_agent = "StreetRAG/0.3 (set OSMNX_USER_AGENT env var)"
    ox.settings.use_cache = True

    G = ox.graph_from_place(city, network_type=network_type, simplify=True)
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    out_path = data_dir / f"{_safe_name(city)}_{_safe_name(network_type)}.gpkg"
    if out_path.exists():
        out_path.unlink()
    gdf_nodes.to_file(out_path, layer="nodes", driver="GPKG")
    gdf_edges.to_file(out_path, layer="edges", driver="GPKG")
    print(f"Saved: {out_path} (layers: nodes, edges)")
    return out_path
