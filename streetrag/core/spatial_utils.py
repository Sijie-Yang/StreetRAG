"""Spatial helpers: CRS, length-weighted stats, Moran's I."""

from __future__ import annotations

from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def pick_local_utm(gdf: gpd.GeoDataFrame) -> int:
    if gdf is None or gdf.empty:
        return 3857
    g = gdf
    if g.crs is None:
        g = g.set_crs("EPSG:4326", allow_override=True)
    cen = g.geometry.iloc[: min(2000, len(g))].representative_point()
    cen = gpd.GeoSeries(cen, crs=g.crs).to_crs("EPSG:4326")
    lon = float(cen.x.mean())
    lat = float(cen.y.mean())
    zone = int((lon + 180) // 6) + 1
    zone = max(1, min(60, zone))
    return (32600 if lat >= 0 else 32700) + zone


def ensure_projected(
    *gdfs: gpd.GeoDataFrame, preferred_epsg: Optional[int] = None
) -> Tuple[gpd.GeoDataFrame, ...]:
    if not gdfs:
        return tuple()
    target: Optional[int] = preferred_epsg
    if target is None:
        for g in gdfs:
            if g is not None and g.crs is not None and not g.crs.is_geographic:
                target = int(g.crs.to_epsg() or 0) or None
                if target:
                    break
    if target is None:
        target = pick_local_utm(gdfs[0])
    out = []
    for g in gdfs:
        if g is None:
            out.append(g)
            continue
        if g.crs is None:
            g = g.set_crs("EPSG:4326", allow_override=True)
        if g.crs.to_epsg() != target:
            g = g.to_crs(epsg=target)
        out.append(g)
    return tuple(out)


def length_weighted_stats(values: pd.Series, lengths: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce")
    L = pd.to_numeric(lengths, errors="coerce")
    mask = v.notna() & L.notna() & (L > 0)
    if mask.sum() == 0:
        return {
            "weighted_mean": None,
            "weighted_median": None,
            "weighted_std": None,
            "total_length_km": 0.0,
            "n_edges": 0,
        }
    v_ = v[mask].to_numpy(dtype=np.float64)
    w_ = L[mask].to_numpy(dtype=np.float64)
    total = float(w_.sum())
    wm = float(np.average(v_, weights=w_))
    wv = float(np.average((v_ - wm) ** 2, weights=w_))
    wsd = float(np.sqrt(max(wv, 0.0)))
    order = np.argsort(v_)
    v_sorted = v_[order]
    w_sorted = w_[order]
    cw = np.cumsum(w_sorted)
    half = total / 2.0
    wmed = float(v_sorted[int(np.searchsorted(cw, half))])
    return {
        "weighted_mean": wm,
        "weighted_median": wmed,
        "weighted_std": wsd,
        "total_length_km": total / 1000.0,
        "n_edges": int(mask.sum()),
    }


def morans_i_on_edges(
    gdf_edges: gpd.GeoDataFrame,
    col: str,
    *,
    k_neighbors: int = 6,
    max_samples: int = 20000,
    seed: int = 42,
) -> Optional[dict]:
    if col not in gdf_edges.columns:
        return None
    g = gdf_edges[[col, "geometry"]].copy()
    g = g[g[col].notna() & g.geometry.notna() & ~g.geometry.is_empty]
    if len(g) < max(k_neighbors + 1, 10):
        return None
    if g.crs is None or g.crs.is_geographic:
        g = ensure_projected(g)[0]
    sampled = len(g) > max_samples
    if sampled:
        rng = np.random.default_rng(seed)
        g = g.iloc[rng.choice(len(g), size=max_samples, replace=False)].copy()
    centroids = np.asarray(
        [(p.x, p.y) for p in g.geometry.representative_point()],
        dtype=np.float64,
    )
    vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=np.float64)
    n = len(vals)
    if n < k_neighbors + 1:
        return None
    tree = cKDTree(centroids)
    _, idx = tree.query(centroids, k=k_neighbors + 1)
    nbr = idx[:, 1:]
    rows = np.repeat(np.arange(n), k_neighbors)
    cols = nbr.reshape(-1)
    w = 1.0 / k_neighbors
    S0 = n * 1.0
    mean = float(vals.mean())
    dev = vals - mean
    var = float((dev * dev).sum())
    if var <= 0:
        return None
    cross = float((dev[rows] * dev[cols] * w).sum())
    moran = (n / S0) * (cross / var)
    return {
        "I": float(moran),
        "expected": float(-1.0 / (n - 1)),
        "n": int(n),
        "k_neighbors": int(k_neighbors),
        "sampled": sampled,
    }
