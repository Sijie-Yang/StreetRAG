"""Shortest-path routing on street networks."""

from __future__ import annotations

import math

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Point
from shapely.ops import linemerge


def build_routing_graph(
    gdf_edges: gpd.GeoDataFrame,
    weight_col: str = "length",
    *,
    respect_oneway: bool = True,
) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for i, (_, row) in enumerate(gdf_edges.iterrows()):
        u, v = int(row["u"]), int(row["v"])
        if weight_col in gdf_edges.columns and pd.notna(row[weight_col]):
            try:
                w = float(row[weight_col])
            except (TypeError, ValueError):
                w = float(row.geometry.length)
        else:
            w = float(row.geometry.length)
        if not math.isfinite(w) or w <= 0:
            w = max(float(row.geometry.length), 1e-6)
        G.add_edge(u, v, key=i, weight=w)
        if not respect_oneway:
            G.add_edge(v, u, key=10_000_000 + i, weight=w)
            continue
        ow = row.get("oneway")
        if ow is True:
            continue
        G.add_edge(v, u, key=10_000_000 + i, weight=w)
    return G


def nearest_osmid(gdf_nodes: gpd.GeoDataFrame, point: Point) -> int:
    dist = gdf_nodes.geometry.distance(point)
    return int(gdf_nodes.loc[dist.idxmin(), "osmid"])


def route_geometry_and_length(gdf_edges: gpd.GeoDataFrame, node_path: list[int]) -> tuple:
    parts = []
    total = 0.0
    for a, b in zip(node_path[:-1], node_path[1:]):
        fwd = gdf_edges[(gdf_edges["u"] == a) & (gdf_edges["v"] == b)]
        bak = gdf_edges[(gdf_edges["u"] == b) & (gdf_edges["v"] == a)]
        seg = fwd.iloc[0] if len(fwd) > 0 else (bak.iloc[0] if len(bak) > 0 else None)
        if seg is None:
            continue
        parts.append(seg.geometry)
        L = seg.get("length", None)
        total += float(L) if L is not None and pd.notna(L) else float(seg.geometry.length)
    if not parts:
        return None, 0.0
    return linemerge(parts), total
