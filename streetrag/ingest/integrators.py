"""Integrators: map external geodata onto street edges."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point
from tqdm import tqdm


class Integrator(ABC):
    name: str = "base"

    @abstractmethod
    def integrate(
        self,
        edges: gpd.GeoDataFrame,
        source: gpd.GeoDataFrame,
        columns: Dict[str, str],
        **params,
    ) -> gpd.GeoDataFrame:
        ...


class SnapNearestIntegrator(Integrator):
    name = "snap_nearest"

    def integrate(
        self,
        edges: gpd.GeoDataFrame,
        source: gpd.GeoDataFrame,
        columns: Dict[str, str],
        **params,
    ) -> gpd.GeoDataFrame:
        k = int(params.get("k", 1))
        edge_centroids = edges.geometry.centroid
        valid = source[~source.geometry.is_empty & source.geometry.notna()]
        if valid.empty:
            return edges
        tree = cKDTree(np.array([(p.x, p.y) for p in valid.geometry]))
        centroid_coords = np.array([(c.x, c.y) for c in edge_centroids])
        _, indices = tree.query(centroid_coords, k=min(k, len(valid)))
        if k == 1:
            indices = indices.reshape(-1, 1)
        for col_name in columns:
            if col_name not in valid.columns:
                continue
            if not pd.api.types.is_numeric_dtype(valid[col_name]):
                continue
            values = valid[col_name].values
            out = np.zeros(len(edges))
            for i in range(len(edges)):
                nearest = values[indices[i]]
                valid_vals = nearest[pd.notna(nearest)]
                out[i] = np.mean(valid_vals) if len(valid_vals) else np.nan
            edges[col_name] = out
        return edges


class BufferDensityIntegrator(Integrator):
    name = "buffer_density"

    def integrate(
        self,
        edges: gpd.GeoDataFrame,
        source: gpd.GeoDataFrame,
        columns: Dict[str, str],
        **params,
    ) -> gpd.GeoDataFrame:
        radius = float(params.get("radius", 500))
        category_col = params.get("category_column")
        rating_col = params.get("rating_column")
        valid = source[~source.geometry.is_empty & source.geometry.notna()]
        if valid.empty:
            return edges
        edge_centroids = edges.geometry.centroid
        centroid_coords = np.array([(c.x, c.y) for c in edge_centroids])

        if category_col and category_col in valid.columns:
            for category in valid[category_col].dropna().unique():
                cat_mask = valid[category_col] == category
                cat_pts = valid[cat_mask]
                safe = str(category).replace(" ", "_").replace("&", "and").replace("/", "_")
                density_col = f"POI_L1_{safe}_density_{int(radius)}m"
                if density_col not in columns:
                    continue
                tree = cKDTree(np.array([(p.x, p.y) for p in cat_pts.geometry]))
                density = np.zeros(len(edges))
                for i in range(len(edges)):
                    idx = tree.query_ball_point(centroid_coords[i], radius)
                    density[i] = len(idx)
                edges[density_col] = density
                if rating_col and rating_col in cat_pts.columns:
                    rating_col_name = f"POI_L1_{safe}_avg_rating_{int(radius)}m"
                    if rating_col_name in columns:
                        avg = np.zeros(len(edges))
                        for i in range(len(edges)):
                            idx = tree.query_ball_point(centroid_coords[i], radius)
                            if idx:
                                ratings = cat_pts.iloc[idx][rating_col].dropna()
                                avg[i] = ratings.mean() if len(ratings) else np.nan
                        edges[rating_col_name] = avg
            return edges

        tree = cKDTree(np.array([(p.x, p.y) for p in valid.geometry]))
        for col_name in columns:
            if col_name in valid.columns and pd.api.types.is_numeric_dtype(valid[col_name]):
                out = np.zeros(len(edges))
                for i in range(len(edges)):
                    idx = tree.query_ball_point(centroid_coords[i], radius)
                    if idx:
                        vals = valid.iloc[idx][col_name].dropna()
                        out[i] = vals.mean() if len(vals) else np.nan
                edges[col_name] = out
            else:
                out = np.zeros(len(edges))
                for i in range(len(edges)):
                    idx = tree.query_ball_point(centroid_coords[i], radius)
                    out[i] = len(idx)
                edges[col_name] = out
        return edges


class LineOverlayIntegrator(Integrator):
    name = "line_overlay"

    def integrate(
        self,
        edges: gpd.GeoDataFrame,
        source: gpd.GeoDataFrame,
        columns: Dict[str, str],
        **params,
    ) -> gpd.GeoDataFrame:
        buffer_m = float(params.get("buffer_m", 20))
        valid = source[~source.geometry.is_empty & source.geometry.notna()]
        if valid.empty:
            return edges
        for col_name in columns:
            if col_name not in valid.columns:
                continue
            if not pd.api.types.is_numeric_dtype(valid[col_name]):
                continue
            out = np.full(len(edges), np.nan)
            edge_buf = edges.copy()
            edge_buf["geometry"] = edge_buf.geometry.buffer(buffer_m, cap_style=2)
            joined = gpd.sjoin(valid[[col_name, "geometry"]], edge_buf[["geometry"]], how="inner", predicate="intersects")
            if joined.empty:
                edges[col_name] = out
                continue
            for idx, grp in joined.groupby("index_right"):
                out[int(idx)] = grp[col_name].mean()
            edges[col_name] = out
        return edges


class PolygonAreaWeightedIntegrator(Integrator):
    name = "polygon_area_weighted"

    def integrate(
        self,
        edges: gpd.GeoDataFrame,
        source: gpd.GeoDataFrame,
        columns: Dict[str, str],
        **params,
    ) -> gpd.GeoDataFrame:
        buffer_m = float(params.get("buffer_m", 30))
        valid = source[~source.geometry.is_empty & source.geometry.notna()]
        if valid.empty:
            return edges
        edge_buf = edges.copy()
        edge_buf["geometry"] = edge_buf.geometry.buffer(buffer_m, cap_style=2)
        for col_name in columns:
            if col_name not in valid.columns:
                continue
            if not pd.api.types.is_numeric_dtype(valid[col_name]):
                continue
            out = np.full(len(edges), np.nan)
            for i in tqdm(range(len(edges)), desc=f"polygon→edge {col_name}"):
                buf = edge_buf.geometry.iloc[i]
                inter = valid[valid.geometry.intersects(buf)]
                if inter.empty:
                    continue
                areas = inter.geometry.intersection(buf).area
                vals = inter[col_name]
                mask = areas > 0
                if mask.sum() == 0:
                    continue
                out[i] = np.average(vals[mask], weights=areas[mask])
            edges[col_name] = out
        return edges


INTEGRATORS: Dict[str, Integrator] = {
    "snap_nearest": SnapNearestIntegrator(),
    "nearest_points_average": SnapNearestIntegrator(),
    "buffer_density": BufferDensityIntegrator(),
    "poi_category_density_rating": BufferDensityIntegrator(),
    "line_overlay": LineOverlayIntegrator(),
    "polygon_area_weighted": PolygonAreaWeightedIntegrator(),
}


def get_integrator(method_type: str) -> Integrator:
    if method_type not in INTEGRATORS:
        raise ValueError(f"Unknown integrator: {method_type}. Available: {list(INTEGRATORS)}")
    return INTEGRATORS[method_type]
