"""Atomic GeoPackage I/O."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import geopandas as gpd

from streetrag.core.network_gpkg import NETWORK_LAYER, read_network_gpkg as _read_network_gpkg


def atomic_write_gpkg(
    path: str | Path,
    *,
    edges: gpd.GeoDataFrame,
    nodes: Optional[gpd.GeoDataFrame] = None,
    edges_layer: str = NETWORK_LAYER,
    nodes_layer: str = "nodes",
) -> None:
    """Write street lines to GPKG atomically via temp file.

    Default: single ``network`` layer. Legacy nodes+edges when *nodes* is given
    and *edges_layer* is ``edges``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".gpkg", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        if nodes is not None and edges_layer == "edges":
            nodes.to_file(tmp_path, layer=nodes_layer, driver="GPKG")
            edges.to_file(tmp_path, layer=edges_layer, driver="GPKG")
        else:
            edges.to_file(tmp_path, layer=edges_layer, driver="GPKG")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def read_network_gpkg(
    path: str | Path,
    *,
    edges_layer: str = NETWORK_LAYER,
    nodes_layer: str = "nodes",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    return _read_network_gpkg(path, edges_layer=edges_layer, nodes_layer=nodes_layer)
