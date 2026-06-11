"""Composite index statistics helpers."""

from __future__ import annotations

from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from streetrag.core.spatial_utils import length_weighted_stats


def calculate_statistics(values: pd.Series) -> dict:
    valid = values[values.notna()]
    if len(valid) == 0:
        return {
            "count": 0, "min": None, "max": None, "mean": None,
            "std": None, "median": None, "q25": None, "q75": None,
        }
    return {
        "count": int(len(valid)),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "median": float(valid.median()),
        "q25": float(valid.quantile(0.25)),
        "q75": float(valid.quantile(0.75)),
    }


def _find_col_ci(gdf: gpd.GeoDataFrame, name: str) -> Optional[str]:
    n = name.lower()
    for c in gdf.columns:
        if c.lower() == n:
            return c
    return None


def resolve_normalized_series(
    gdf: gpd.GeoDataFrame,
    feature_name: str,
    normalization: str,
    *,
    percentile_suffix: str = "_pctl",
    registry: Optional[dict] = None,
) -> pd.Series:
    raw_col = _find_col_ci(gdf, feature_name)
    if raw_col is None:
        return pd.Series(np.nan, index=gdf.index)
    if normalization == "raw":
        return pd.to_numeric(gdf[raw_col], errors="coerce")
    norm_col: Optional[str] = None
    fs = (registry or {}).get("feature_statistics", {}).get(feature_name, {})
    norm_cols = fs.get("normalization_columns", {}) or {}
    if normalization in norm_cols:
        norm_col = norm_cols[normalization]
    if not norm_col:
        suffix = {
            "percentile": percentile_suffix,
            "zscore": "_zscore",
            "minmax": "_minmax",
            "robust": "_robust",
        }.get(normalization)
        if suffix:
            norm_col = f"{feature_name}{suffix}"
    if norm_col:
        actual = _find_col_ci(gdf, norm_col)
        if actual:
            s = pd.to_numeric(gdf[actual], errors="coerce")
            if normalization == "percentile":
                s = s / 100.0
            return s
    raw = pd.to_numeric(gdf[raw_col], errors="coerce")
    if normalization == "percentile":
        return raw.rank(pct=True, method="average")
    if normalization == "zscore":
        m, sd = raw.mean(skipna=True), raw.std(skipna=True)
        return (raw - m) / sd if sd and not np.isnan(sd) else raw * 0.0
    if normalization == "minmax":
        lo, hi = raw.min(skipna=True), raw.max(skipna=True)
        return (raw - lo) / (hi - lo) if hi > lo else raw * 0.0
    if normalization == "robust":
        med = raw.median(skipna=True)
        q25, q75 = raw.quantile(0.25), raw.quantile(0.75)
        iqr = q75 - q25
        return (raw - med) / iqr if iqr and iqr > 0 else raw * 0.0
    return raw


def combine_features(
    gdf_edges: gpd.GeoDataFrame,
    plan,
    *,
    percentile_suffix: str,
    registry: dict,
) -> pd.Series:
    from streetrag.llm.retrieval import IndexPlan

    if not plan.features:
        return pd.Series(0.0, index=gdf_edges.index)
    op = plan.operator
    if op == "weighted_sum":
        out = pd.Series(0.0, index=gdf_edges.index)
        for fw in plan.features:
            z = resolve_normalized_series(
                gdf_edges, fw.name, plan.normalization,
                percentile_suffix=percentile_suffix, registry=registry,
            ).fillna(0.0)
            out = out + (fw.weight * z)
        return out
    if op == "geometric_mean":
        eps = 1e-6
        log_sum = pd.Series(0.0, index=gdf_edges.index)
        w_total = 0.0
        for fw in plan.features:
            p = resolve_normalized_series(
                gdf_edges, fw.name, "percentile",
                percentile_suffix=percentile_suffix, registry=registry,
            ).clip(eps, 1.0)
            if fw.weight < 0:
                p = (1.0 - p).clip(eps, 1.0)
            w = abs(fw.weight)
            if w == 0:
                continue
            log_sum = log_sum + w * np.log(p)
            w_total += w
        if w_total <= 0:
            return pd.Series(0.0, index=gdf_edges.index)
        return np.exp(log_sum / w_total)
    if op in ("owa_top", "owa_bottom"):
        cols, weights = [], []
        for fw in plan.features:
            z = resolve_normalized_series(
                gdf_edges, fw.name, plan.normalization,
                percentile_suffix=percentile_suffix, registry=registry,
            ).fillna(0.0)
            if fw.weight < 0:
                z = -z
            cols.append(z.to_numpy())
            weights.append(abs(fw.weight))
        if not cols:
            return pd.Series(0.0, index=gdf_edges.index)
        M = np.vstack(cols).T
        if op == "owa_top":
            M_sorted = -np.sort(-M, axis=1)
            w = np.array(sorted(weights, reverse=True), dtype=np.float64)
        else:
            M_sorted = np.sort(M, axis=1)
            w = np.array(sorted(weights), dtype=np.float64)
        if w.sum() <= 0:
            return pd.Series(0.0, index=gdf_edges.index)
        w = w / w.sum()
        return pd.Series((M_sorted * w).sum(axis=1), index=gdf_edges.index)
    raise ValueError(f"Unknown operator: {op}")


def length_weighted_topk_by_integration(
    gdf_edges: gpd.GeoDataFrame,
    index_values: pd.Series,
    registry: dict,
    *,
    top_k: int,
) -> dict:
    radii = (registry.get("space_syntax_integration") or {}).get("radii") or []
    out: dict = {}
    lengths = (
        pd.to_numeric(gdf_edges["length"], errors="coerce")
        if "length" in gdf_edges.columns
        else gdf_edges.geometry.length
    )
    for r in radii:
        col = f"integration_R{r}"
        actual = _find_col_ci(gdf_edges, col)
        if not actual:
            continue
        s = pd.to_numeric(gdf_edges[actual], errors="coerce")
        mask = s.notna() & index_values.notna()
        if mask.sum() == 0:
            continue
        k = min(top_k, int(mask.sum()))
        idx = s[mask].nlargest(k).index
        lw = length_weighted_stats(index_values.loc[idx], lengths.loc[idx])
        out[f"integration_R{r}_top{top_k}"] = {
            "length_weighted": lw,
            "index_unweighted": calculate_statistics(index_values.loc[idx]),
        }
    return out
