"""Tests for column detection and POI integration."""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from streetrag.ingest.columns import (
    detect_text_columns,
    plan_poi_category_columns,
    summarize_columns,
)
from streetrag.ingest.integrators import BufferDensityIntegrator
from streetrag.ingest.pipeline import build_integration_block, register_integration_block


def test_detect_text_columns():
    gdf = gpd.GeoDataFrame(
        {
            "rating": [4.5, 3.0],
            "review_text": ["Great street food and lively atmosphere here", "Too noisy at night"],
            "name": ["A", "B"],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:32648",
    )
    text_cols = detect_text_columns(gdf)
    assert "review_text" in text_cols


def test_poi_category_integrator_dynamic_columns(synthetic_network):
    pts = gpd.GeoDataFrame(
        {
            "L1_types": ["restaurant", "restaurant", "cafe"],
            "rating": [4.5, 3.0, 5.0],
        },
        geometry=[Point(100, 100), Point(105, 105), Point(200, 200)],
        crs=synthetic_network.edges.crs,
    )
    integrator = BufferDensityIntegrator()
    columns = {}
    result = integrator.integrate(
        synthetic_network.edges.copy(),
        pts,
        columns,
        radius=150,
        category_column="L1_types",
        rating_column="rating",
    )
    assert "POI_L1_restaurant_density_150m" in result.columns
    assert "POI_L1_restaurant_avg_rating_150m" in result.columns
    assert "POI_L1_cafe_density_150m" in result.columns


def test_plan_poi_category_columns():
    gdf = gpd.GeoDataFrame(
        {"L1_types": ["restaurant", "cafe"], "rating": [4.0, 5.0]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:32648",
    )
    cols = plan_poi_category_columns(gdf, category_col="L1_types", radius=500, rating_col="rating")
    assert "POI_L1_restaurant_density_500m" in cols
    assert "POI_L1_restaurant_avg_rating_500m" in cols


def test_build_integration_block_with_text(tmp_path):
    gdf = gpd.GeoDataFrame(
        {
            "L1_types": ["shop"],
            "rating": [4.0],
            "comment": ["Nice shop on this corner with friendly staff"],
        },
        geometry=[Point(0, 0)],
        crs="EPSG:32648",
    )
    gpkg = tmp_path / "poi.gpkg"
    gdf.to_file(gpkg, driver="GPKG")
    gdf2 = gpd.read_file(gpkg)
    block = build_integration_block("poi.gpkg", gdf2)
    assert block is not None
    assert block["integration_method"]["type"] == "poi_category_density_rating"
    assert "comment" in block["text_columns"]
