"""Pytest fixtures: synthetic street network."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from streetrag.core.feature_catalog import FeatureCatalog
from streetrag.core.street_network import StreetNetwork


@pytest.fixture
def synthetic_edges() -> gpd.GeoDataFrame:
    """Simple grid-like network (~20 edges)."""
    lines = []
    lengths = []
    for i in range(5):
        for j in range(4):
            x0, y0 = i * 100, j * 100
            x1, y1 = (i + 1) * 100, j * 100
            lines.append(LineString([(x0, y0), (x1, y1)]))
            lengths.append(100.0)
    for i in range(4):
        for j in range(5):
            x0, y0 = i * 100, j * 100
            x1, y1 = i * 100, (j + 1) * 100
            lines.append(LineString([(x0, y0), (x1, y1)]))
            lengths.append(100.0)
    gdf = gpd.GeoDataFrame(
        {"length": lengths, "feature_a": np.random.rand(len(lines))},
        geometry=lines,
        crs="EPSG:32648",
    )
    gdf["mm_len"] = gdf["length"]
    return gdf


@pytest.fixture
def synthetic_nodes(synthetic_edges) -> gpd.GeoDataFrame:
    pts = []
    for i in range(6):
        for j in range(6):
            pts.append(Point(i * 100, j * 100))
    return gpd.GeoDataFrame(geometry=pts, crs=synthetic_edges.crs)


@pytest.fixture
def tmp_catalog(tmp_path, synthetic_edges, synthetic_nodes) -> FeatureCatalog:
    gpkg = tmp_path / "test_network.gpkg"
    synthetic_nodes.to_file(gpkg, layer="nodes", driver="GPKG")
    synthetic_edges.to_file(gpkg, layer="edges", driver="GPKG")
    catalog = FeatureCatalog(tmp_path / "feature_registry.json")
    catalog._data = {
        "target_network": gpkg.name,
        "target_layer": "edges",
        "percentile_column_suffix": "_pctl",
        "normalization_methods": ["percentile", "zscore", "minmax", "robust"],
        "space_syntax_integration": {"radii": [500, 1500]},
        "feature_statistics": {},
        "composite_index_columns": [],
        "point_integrations": [],
    }
    catalog.save()
    return catalog


@pytest.fixture
def synthetic_network(tmp_catalog) -> StreetNetwork:
    return StreetNetwork.from_catalog(tmp_catalog, use_cache=False)
