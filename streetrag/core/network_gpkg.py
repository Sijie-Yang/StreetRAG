"""Street network GeoPackage I/O — single ``network`` layer (geometry lines)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

NETWORK_LAYER = "network"
LEGACY_EDGES_LAYER = "edges"
LEGACY_NODES_LAYER = "nodes"
EDGE_ID_COL = "edge_id"


def _endpoint_key(x: float, y: float, snap: float = 0.5) -> Tuple[int, int]:
    return (int(round(x / snap)), int(round(y / snap)))


def graph_from_lines(
    lines: gpd.GeoDataFrame,
    *,
    snap_m: float = 0.5,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build routing ``u``/``v`` and node points from line geometries."""
    edges = lines.copy()
    endpoint_to_node: Dict[Tuple[int, int], int] = {}
    node_coords: Dict[int, Tuple[float, float]] = {}
    next_id = 0
    u_list: List[int] = []
    v_list: List[int] = []

    for geom in edges.geometry:
        if geom is None or geom.is_empty:
            u_list.append(-1)
            v_list.append(-1)
            continue
        coords = list(geom.coords)
        keys = (
            _endpoint_key(coords[0][0], coords[0][1], snap_m),
            _endpoint_key(coords[-1][0], coords[-1][1], snap_m),
        )
        for pt, key in zip((coords[0], coords[-1]), keys):
            if key not in endpoint_to_node:
                endpoint_to_node[key] = next_id
                node_coords[next_id] = pt
                next_id += 1
        u_list.append(endpoint_to_node[keys[0]])
        v_list.append(endpoint_to_node[keys[1]])

    edges["u"] = u_list
    edges["v"] = v_list
    if "length" not in edges.columns:
        edges["length"] = edges.geometry.length
    if "mm_len" not in edges.columns:
        edges["mm_len"] = edges["length"]
    if EDGE_ID_COL not in edges.columns:
        edges[EDGE_ID_COL] = np.arange(len(edges), dtype=np.int64)
    if "osmid" not in edges.columns:
        edges["osmid"] = edges[EDGE_ID_COL].values

    node_ids = sorted(node_coords)
    nodes = gpd.GeoDataFrame(
        {"osmid": node_ids},
        geometry=[Point(node_coords[i]) for i in node_ids],
        crs=edges.crs,
    )
    return edges, nodes


def ensure_edge_ids(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign stable ``edge_id`` values when missing (0..n-1, never regenerated)."""
    if EDGE_ID_COL in edges.columns and edges[EDGE_ID_COL].notna().all():
        return edges
    out = edges.copy()
    out[EDGE_ID_COL] = np.arange(len(out), dtype=np.int64)
    return out


def persist_edge_ids_if_missing(
    path: str | Path,
    edges: gpd.GeoDataFrame,
    *,
    layer: str = NETWORK_LAYER,
) -> bool:
    """Write *edges* back when ``edge_id`` was just assigned. Returns True if written."""
    if EDGE_ID_COL in edges.columns and path.exists():
        try:
            existing = list_network_column_names(path, edges_layer=layer)
            if EDGE_ID_COL in existing:
                return False
        except Exception:
            pass
    write_network_gpkg(path, ensure_edge_ids(edges), layer=layer)
    return True


def detect_network_format(path: str | Path) -> dict:
    """Return ``{format, layer}`` where format is ``network`` or ``nodes_edges``."""
    from streetrag.ingest.readers import list_layers

    layers = set(list_layers(path) or [])
    if LEGACY_NODES_LAYER in layers and LEGACY_EDGES_LAYER in layers:
        return {"format": "nodes_edges", "edges_layer": LEGACY_EDGES_LAYER, "nodes_layer": LEGACY_NODES_LAYER}
    if NETWORK_LAYER in layers:
        return {"format": "network", "layer": NETWORK_LAYER}
    raise ValueError(
        f"No street network in {path}: expected layer {NETWORK_LAYER!r} "
        f"or {LEGACY_NODES_LAYER!r}+{LEGACY_EDGES_LAYER!r}"
    )


def is_street_network_gpkg(path: str | Path) -> bool:
    try:
        detect_network_format(path)
        return True
    except ValueError:
        return False


def list_network_column_names(
    path: str | Path,
    *,
    edges_layer: Optional[str] = None,
) -> List[str]:
    """Return attribute column names without loading geometries (fast for large GPKGs)."""
    import fiona

    path = Path(path)
    info = detect_network_format(path)
    if info["format"] == "nodes_edges":
        layer = edges_layer or info["edges_layer"]
    else:
        layer = edges_layer or info["layer"]
    with fiona.open(path, layer=layer) as src:
        return list(src.schema.get("properties", {}).keys())


def read_network_gpkg(
    path: str | Path,
    *,
    edges_layer: Optional[str] = None,
    nodes_layer: str = LEGACY_NODES_LAYER,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load edges + nodes from a GPKG (``network`` layer or legacy nodes/edges)."""
    path = Path(path)
    info = detect_network_format(path)

    if info["format"] == "nodes_edges":
        el = edges_layer or info["edges_layer"]
        edges = gpd.read_file(path, layer=el)
        nodes = gpd.read_file(path, layer=info["nodes_layer"])
        edges = ensure_edge_ids(edges)
        return edges, nodes

    layer = edges_layer or info["layer"]
    edges = gpd.read_file(path, layer=layer)
    if "u" in edges.columns and "v" in edges.columns:
        edges = ensure_edge_ids(edges)
        if "osmid" not in edges.columns:
            edges = edges.copy()
            edges["osmid"] = edges[EDGE_ID_COL].values
        node_ids = pd.unique(edges[["u", "v"]].values.ravel())
        # Rebuild node geometry from edge endpoints when nodes layer is absent.
        coords: Dict[int, Tuple[float, float]] = {}
        for _, row in edges.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            c0, c1 = geom.coords[0], geom.coords[-1]
            coords.setdefault(int(row["u"]), c0)
            coords.setdefault(int(row["v"]), c1)
        nodes = gpd.GeoDataFrame(
            {"osmid": [int(i) for i in node_ids if int(i) in coords]},
            geometry=[Point(coords[int(i)]) for i in node_ids if int(i) in coords],
            crs=edges.crs,
        )
        return edges, nodes

    edges, nodes = graph_from_lines(edges)
    return ensure_edge_ids(edges), nodes


def write_network_gpkg(
    path: str | Path,
    edges: gpd.GeoDataFrame,
    *,
    layer: str = NETWORK_LAYER,
) -> None:
    """Write street lines to a single ``network`` layer."""
    from streetrag.core.io import atomic_write_gpkg

    atomic_write_gpkg(path, edges=edges, edges_layer=layer, nodes=None)
