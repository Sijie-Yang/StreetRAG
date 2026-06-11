"""Atomic GeoPackage I/O."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import geopandas as gpd


def atomic_write_gpkg(
    path: str | Path,
    *,
    edges: gpd.GeoDataFrame,
    nodes: Optional[gpd.GeoDataFrame] = None,
    edges_layer: str = "edges",
    nodes_layer: str = "nodes",
) -> None:
    """Write edges (+ optional nodes) to GPKG atomically via temp file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".gpkg", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        if nodes is not None:
            nodes.to_file(tmp_path, layer=nodes_layer, driver="GPKG")
        edges.to_file(tmp_path, layer=edges_layer, driver="GPKG")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def read_network_gpkg(
    path: str | Path,
    *,
    edges_layer: str = "edges",
    nodes_layer: str = "nodes",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    path = Path(path)
    edges = gpd.read_file(path, layer=edges_layer)
    nodes = gpd.read_file(path, layer=nodes_layer)
    return edges, nodes
