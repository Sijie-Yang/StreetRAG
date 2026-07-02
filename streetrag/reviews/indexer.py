"""Index POI/review text onto edges and into ReviewStore."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from streetrag.core.feature_catalog import FeatureCatalog
from streetrag.core.street_network import StreetNetwork
from streetrag.ingest.columns import (
    infer_category_column,
    infer_name_column,
    infer_rating_column,
    plan_poi_category_columns,
)
from streetrag.ingest.integrators import BufferDensityIntegrator
from streetrag.ingest.readers import read_geodata
from streetrag.reviews.store import ReviewStore


from streetrag.core.network_gpkg import EDGE_ID_COL


def _snap_points_to_edges(
    edges: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
) -> np.ndarray:
    centroids = edges.geometry.centroid
    coords = np.array([(p.x, p.y) for p in points.geometry])
    tree = cKDTree(np.array([(c.x, c.y) for c in centroids]))
    _, idx = tree.query(coords, k=1)
    pos = idx.astype(int)
    if EDGE_ID_COL in edges.columns:
        return edges.iloc[pos][EDGE_ID_COL].astype(np.int64).values
    return pos


def index_reviews_from_source(
    catalog: FeatureCatalog,
    net: StreetNetwork,
    source_path: str | Path,
    *,
    text_columns: List[str],
    layer: Optional[str] = None,
    create_edge_aggregates: bool = True,
    agg_radius: float = 100.0,
) -> dict:
    source_path = Path(source_path)
    gdf = read_geodata(source_path, layer=layer)
    if gdf.crs != net.edges.crs:
        gdf = gdf.to_crs(net.edges.crs)
    valid = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    if valid.empty or not text_columns:
        return {"n_chunks": 0, "text_columns": text_columns}

    edge_ids = _snap_points_to_edges(net.edges, valid)
    category_col = infer_category_column(valid)
    rating_col = infer_rating_column(valid)
    name_col = infer_name_column(valid)

    records = []
    for i, row in valid.iterrows():
        pos = valid.index.get_loc(i)
        edge_id = int(edge_ids[pos])
        poi_id = str(row.get("poi_id", row.get("id", f"{source_path.stem}_{pos}")))
        category = str(row[category_col]) if category_col and pd.notna(row.get(category_col)) else ""
        rating = float(row[rating_col]) if rating_col and pd.notna(row.get(rating_col)) else None
        poi_name = str(row[name_col]) if name_col and pd.notna(row.get(name_col)) else ""
        for col in text_columns:
            text = row.get(col)
            if pd.isna(text) or not str(text).strip():
                continue
            records.append(
                {
                    "poi_id": poi_id,
                    "edge_id": edge_id,
                    "text": str(text).strip()[:4000],
                    "text_column": col,
                    "category": category,
                    "rating": rating,
                    "poi_name": poi_name,
                    "source_file": source_path.name,
                }
            )

    store = ReviewStore(catalog)
    n_added = store.upsert_records(records)

    agg_cols: List[str] = []
    if create_edge_aggregates and category_col and rating_col:
        method_params = {
            "category_column": category_col,
            "rating_column": rating_col,
            "radius": agg_radius,
        }
        columns = plan_poi_category_columns(
            valid,
            category_col=category_col,
            radius=agg_radius,
            rating_col=rating_col,
        )
        integrator = BufferDensityIntegrator()
        net.edges = integrator.integrate(
            net.edges,
            valid,
            columns,
            **method_params,
        )
        for col in columns:
            if col in net.edges.columns:
                net.compute_normalizations(col)
                catalog.set_description(col, columns[col])
                agg_cols.append(col)

    return {
        "n_chunks": n_added,
        "n_review_records": len(records),
        "text_columns": text_columns,
        "edge_aggregate_columns": agg_cols,
    }
