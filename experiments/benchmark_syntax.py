#!/usr/bin/env python3
"""Benchmark space syntax runtime scaling on synthetic grid networks.

Produces the timing table for the paper's scalability section:
network size (edges) x measure (metric / angular) x radius -> seconds.

Usage:
    python experiments/benchmark_syntax.py --sizes 1000 5000 20000 --radii 500 1500
    python experiments/benchmark_syntax.py --gpkg data/Singapore_drive.gpkg --radii 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from streetrag.syntax.engine import angular_segment_analysis


def make_grid(n_edges_target: int, spacing: float = 100.0) -> gpd.GeoDataFrame:
    """Square grid with roughly n_edges_target edges."""
    # k x k grid has 2*k*(k+1) edges
    k = max(2, int((-1 + np.sqrt(1 + 2 * n_edges_target)) / 2))
    lines = []
    for i in range(k + 1):
        for j in range(k):
            lines.append(LineString([(i * spacing, j * spacing), (i * spacing, (j + 1) * spacing)]))
            lines.append(LineString([(j * spacing, i * spacing), ((j + 1) * spacing, i * spacing)]))
    return gpd.GeoDataFrame(geometry=lines, crs="EPSG:32648")


def bench_angular(edges: gpd.GeoDataFrame, radius: float, compute_choice: bool) -> float:
    t0 = time.time()
    angular_segment_analysis(edges, radius, compute_choice=compute_choice)
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description="StreetRAG syntax benchmark")
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 2000, 8000])
    ap.add_argument("--radii", type=float, nargs="+", default=[500, 1500])
    ap.add_argument("--gpkg", help="Benchmark a real network instead of synthetic grids")
    ap.add_argument("--no-choice", action="store_true")
    ap.add_argument("--output", default="experiments/results/benchmark_syntax.json")
    args = ap.parse_args()

    rows = []
    if args.gpkg:
        edges = gpd.read_file(args.gpkg, layer="edges")
        if edges.crs is None or edges.crs.is_geographic:
            from streetrag.core.spatial_utils import ensure_projected
            edges = ensure_projected(edges)[0]
        datasets = [(f"real:{Path(args.gpkg).name}", edges)]
    else:
        datasets = [(f"grid:{s}", make_grid(s)) for s in args.sizes]

    for name, edges in datasets:
        for radius in args.radii:
            seconds = bench_angular(edges, radius, compute_choice=not args.no_choice)
            row = {
                "dataset": name,
                "n_edges": len(edges),
                "radius_m": radius,
                "measure": "angular" + ("" if not args.no_choice else "_no_choice"),
                "seconds": round(seconds, 2),
                "edges_per_second": round(len(edges) / seconds, 1) if seconds > 0 else None,
            }
            rows.append(row)
            print(f"{name:>16s}  R{radius:>6.0f}  {len(edges):>7d} edges  {seconds:8.2f}s")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
