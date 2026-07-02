"""Space syntax engine: metric + angular segment analysis.

Angular segment analysis follows the standard formulation
(Turner 2001; Hillier, Yang & Turner 2012):

- Each street segment is a node in the dual graph; two segments are
  adjacent when they share an endpoint.
- The cost of moving between adjacent segments is the TURN ANGLE
  (deviation from straight continuation), normalized so that a 90° turn
  costs 1.0 and a straight continuation costs 0.
- The analysis radius is METRIC (meters along the network), while
  shortest paths are computed on ANGULAR cost — i.e. "how little do I
  have to turn to reach everything within R meters".

Per-segment measures at radius R:
- node_count NC      — segments reachable within R
- total_depth TD     — sum of angular shortest-path costs
- angular integration = NC / TD (higher = better integrated)
- NAIN = NC^1.2 / TD                       (Hillier et al. 2012)
- choice CH          — angular betweenness (Brandes accumulation)
- NACH = log10(CH + 1) / log10(TD + 3)     (Hillier et al. 2012)
"""

from __future__ import annotations

import heapq
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import momepy
import numpy as np
import pandas as pd

from streetrag.core.street_network import StreetNetwork

# A 90-degree turn costs 1.0 angular unit (depthmapX convention).
_ANGLE_UNIT = math.pi / 2.0


def _endpoint_key(x: float, y: float, snap: float = 0.5) -> Tuple[int, int]:
    """Quantize coordinates so nearly-coincident endpoints share a junction."""
    return (int(round(x / snap)), int(round(y / snap)))


def _direction_at_endpoint(coords: List[Tuple[float, float]], at_start: bool) -> Optional[Tuple[float, float]]:
    """Unit vector pointing AWAY from the endpoint along the segment."""
    if len(coords) < 2:
        return None
    if at_start:
        (x0, y0), (x1, y1) = coords[0], coords[1]
    else:
        (x0, y0), (x1, y1) = coords[-1], coords[-2]
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return None
    return (dx / norm, dy / norm)


def build_segment_graph(
    edges: gpd.GeoDataFrame,
) -> Dict[int, List[Tuple[int, float, float]]]:
    """Build dual-graph adjacency in O(n) via endpoint hashing.

    Returns {seg_idx: [(nbr_idx, angular_cost, metric_step), ...]} where
    metric_step is the center-to-center distance (half of each segment).
    """
    # junction -> list of (seg_pos, direction away from junction, half-length)
    junctions: Dict[Tuple[int, int], List[Tuple[int, Tuple[float, float], float]]] = defaultdict(list)

    positions = list(range(len(edges)))
    geoms = edges.geometry.values
    for pos in positions:
        geom = geoms[pos]
        if geom is None or geom.is_empty:
            continue
        try:
            coords = list(geom.coords)
        except NotImplementedError:  # MultiLineString: merge or take longest part
            parts = list(geom.geoms)
            longest = max(parts, key=lambda p: p.length)
            coords = list(longest.coords)
        if len(coords) < 2:
            continue
        half = geom.length / 2.0
        d_start = _direction_at_endpoint(coords, at_start=True)
        d_end = _direction_at_endpoint(coords, at_start=False)
        if d_start:
            junctions[_endpoint_key(*coords[0])].append((pos, d_start, half))
        if d_end:
            junctions[_endpoint_key(*coords[-1])].append((pos, d_end, half))

    adjacency: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for incident in junctions.values():
        k = len(incident)
        if k < 2:
            continue
        for i in range(k):
            pos_i, dir_i, half_i = incident[i]
            for j in range(i + 1, k):
                pos_j, dir_j, half_j = incident[j]
                if pos_i == pos_j:
                    continue
                # Turn angle = deviation from straight continuation.
                # dir_i and dir_j both point AWAY from the junction, so going
                # straight through means dir_j ≈ -dir_i (dot = -1, deviation 0).
                dot = max(-1.0, min(1.0, dir_i[0] * dir_j[0] + dir_i[1] * dir_j[1]))
                turn = math.pi - math.acos(dot)
                cost = turn / _ANGLE_UNIT  # 90° turn = 1.0
                step = half_i + half_j
                adjacency[pos_i].append((pos_j, cost, step))
                adjacency[pos_j].append((pos_i, cost, step))
    return adjacency


def angular_segment_analysis(
    edges: gpd.GeoDataFrame,
    radius_m: float,
    *,
    compute_choice: bool = True,
) -> pd.DataFrame:
    """Radius-restricted angular analysis (Dijkstra on angular cost,
    cutoff on cumulative metric distance) with Brandes choice accumulation.

    Complexity is O(n · E log V); on large networks (>30k segments) expect
    minutes per radius. See experiments/benchmark_syntax.py for timings.
    """
    adjacency = build_segment_graph(edges)
    n = len(edges)
    node_count = np.zeros(n, dtype=np.int64)
    total_depth = np.zeros(n, dtype=np.float64)
    choice = np.zeros(n, dtype=np.float64)

    INF = float("inf")
    for source in range(n):
        if source not in adjacency:
            continue
        # Dijkstra on angular cost with metric-radius restriction.
        ang = {source: 0.0}
        metric = {source: 0.0}
        sigma = {source: 1.0}      # shortest-path counts (Brandes)
        preds: Dict[int, List[int]] = defaultdict(list)
        finalized: List[int] = []
        visited = set()
        heap: List[Tuple[float, int]] = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            finalized.append(u)
            for v, cost, step in adjacency.get(u, ()):
                new_metric = metric[u] + step
                if new_metric > radius_m:
                    continue
                new_ang = d + cost
                old = ang.get(v, INF)
                if new_ang < old - 1e-12:
                    ang[v] = new_ang
                    metric[v] = new_metric
                    sigma[v] = sigma[u]
                    preds[v] = [u]
                    heapq.heappush(heap, (new_ang, v))
                elif abs(new_ang - old) <= 1e-12 and v not in visited:
                    sigma[v] = sigma.get(v, 0.0) + sigma[u]
                    preds[v].append(u)
                    if new_metric < metric[v]:
                        metric[v] = new_metric

        reached = len(finalized) - 1
        node_count[source] = reached
        total_depth[source] = sum(ang[v] for v in finalized if v != source)

        if compute_choice and reached > 0:
            # Brandes dependency accumulation (reverse finalization order).
            delta = defaultdict(float)
            for w in reversed(finalized):
                for p in preds[w]:
                    delta[p] += (sigma[p] / sigma[w]) * (1.0 + delta[w])
                if w != source:
                    choice[w] += delta[w]

    if compute_choice:
        choice /= 2.0  # undirected double counting

    with np.errstate(divide="ignore", invalid="ignore"):
        integration = np.where(total_depth > 0, node_count / total_depth, np.nan)
        nain = np.where(total_depth > 0, np.power(node_count, 1.2) / total_depth, np.nan)
        nach = np.log10(choice + 1.0) / np.log10(total_depth + 3.0)

    return pd.DataFrame(
        {
            "node_count": node_count,
            "total_depth": total_depth,
            "angular_integration": integration,
            "nain": nain,
            "choice": choice,
            "nach": nach,
        },
        index=edges.index,
    )


def _metric_integration(edges: gpd.GeoDataFrame, radius: int, col_name: str) -> gpd.GeoDataFrame:
    from streetrag.core.network_gpkg import EDGE_ID_COL, ensure_edge_ids

    edges = ensure_edge_ids(edges)
    preserved = edges.drop(columns=["geometry"]).copy()
    gdf = edges.copy()
    if "length" not in gdf.columns:
        gdf["length"] = gdf.geometry.length
    gdf["mm_len"] = gdf["length"]
    primal = momepy.gdf_to_nx(gdf, approach="primal")
    primal = momepy.closeness_centrality(
        primal,
        radius=radius,
        name=col_name,
        mode="edges",
        distance="mm_len",
        weight="mm_len",
    )
    momepy.mean_nodes(primal, col_name)
    result = momepy.nx_to_gdf(primal, points=False)
    if "mm_len" in result.columns and "mm_len" not in edges.columns:
        result = result.drop(columns=["mm_len"])
    # Reattach stable edge_id and topology columns after momepy reordering.
    if EDGE_ID_COL in preserved.columns:
        if len(result) == len(preserved):
            for col in preserved.columns:
                if col not in result.columns:
                    result[col] = preserved[col].values
        else:
            result = result.merge(
                preserved[[EDGE_ID_COL] + [c for c in preserved.columns if c != EDGE_ID_COL]],
                left_index=True,
                right_index=True,
                how="left",
                suffixes=("", "_old"),
            )
            for col in list(result.columns):
                if col.endswith("_old"):
                    base = col[:-4]
                    if base in result.columns:
                        result[base] = result[base].combine_first(result[col])
                    result = result.drop(columns=[col])
    return result


def _timings_path(data_dir: Path) -> Path:
    d = data_dir / "syntax_cache"
    d.mkdir(exist_ok=True)
    return d / "timings.json"


def _log_timing(data_dir: Path, record: dict) -> None:
    p = _timings_path(data_dir)
    rows = []
    if p.exists():
        try:
            rows = json.loads(p.read_text())
        except Exception:
            rows = []
    rows.append(record)
    p.write_text(json.dumps(rows, indent=2))


def compute_syntax(
    net: StreetNetwork,
    *,
    radii: Optional[List[int]] = None,
    measures: Optional[List[str]] = None,
    compute_choice: bool = True,
) -> StreetNetwork:
    """Compute metric integration and angular segment measures per radius.

    Columns already present on the edges (persisted in the GPKG) are skipped,
    so the GPKG itself acts as the cache. Timings are logged to
    data/syntax_cache/timings.json for benchmarking.
    """
    radii = radii or net.catalog.syntax_radii
    measures = measures or ["metric_integration", "angular"]

    for radius in radii:
        if "metric_integration" in measures:
            col = f"integration_R{radius}"
            if col not in net.edges.columns:
                t0 = time.time()
                net.edges = _metric_integration(net.edges, radius, col)
                _log_timing(net.catalog.data_dir, {
                    "measure": "metric_integration", "radius": radius,
                    "n_edges": len(net.edges), "seconds": round(time.time() - t0, 2),
                })
                net.compute_normalizations(col)
                net.catalog.set_description(
                    col,
                    f"Metric closeness integration (momepy) at radius {radius}m. "
                    f"Higher = more spatially accessible within {radius}m.",
                )

        if "angular" in measures:
            ang_col = f"angular_integration_R{radius}"
            if ang_col not in net.edges.columns:
                t0 = time.time()
                df = angular_segment_analysis(
                    net.edges, float(radius), compute_choice=compute_choice
                )
                _log_timing(net.catalog.data_dir, {
                    "measure": "angular", "radius": radius,
                    "n_edges": len(net.edges), "seconds": round(time.time() - t0, 2),
                })
                col_map = {
                    ang_col: ("angular_integration",
                              f"Angular integration (NC/total angular depth) within {radius}m metric radius."),
                    f"nain_R{radius}": ("nain",
                                        f"NAIN = NC^1.2 / total angular depth at R{radius} (Hillier et al. 2012)."),
                    f"choice_R{radius}": ("choice",
                                          f"Angular choice (betweenness) within {radius}m metric radius."),
                    f"nach_R{radius}": ("nach",
                                        f"NACH = log10(choice+1)/log10(total depth+3) at R{radius} (Hillier et al. 2012)."),
                }
                if not compute_choice:
                    col_map.pop(f"choice_R{radius}")
                    col_map.pop(f"nach_R{radius}")
                for out_col, (src, desc) in col_map.items():
                    net.edges[out_col] = df[src]
                    net.compute_normalizations(out_col)
                    net.catalog.set_description(out_col, desc)

    net.catalog.raw.setdefault("space_syntax_integration", {})["radii"] = radii
    net.catalog.save()
    return net
