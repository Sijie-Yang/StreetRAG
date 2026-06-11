"""Tests for integrators."""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon

from streetrag.ingest.integrators import BufferDensityIntegrator, PolygonAreaWeightedIntegrator, SnapNearestIntegrator


def test_snap_nearest(synthetic_network):
    pts = gpd.GeoDataFrame(
        {"val": [10.0, 20.0, 30.0]},
        geometry=[Point(50, 50), Point(150, 50), Point(250, 50)],
        crs=synthetic_network.edges.crs,
    )
    integrator = SnapNearestIntegrator()
    result = integrator.integrate(
        synthetic_network.edges.copy(),
        pts,
        {"val": "nearest value"},
        k=1,
    )
    assert "val" in result.columns
    assert result["val"].notna().any()


def test_buffer_density(synthetic_network):
    pts = gpd.GeoDataFrame(
        {"count": [1.0, 2.0]},
        geometry=[Point(100, 100), Point(105, 105)],
        crs=synthetic_network.edges.crs,
    )
    integrator = BufferDensityIntegrator()
    result = integrator.integrate(
        synthetic_network.edges.copy(),
        pts,
        {"poi_density": "POI count"},
        radius=150,
    )
    assert "poi_density" in result.columns


def test_polygon_area_weighted(synthetic_network):
    poly = gpd.GeoDataFrame(
        {"pop": [1000.0]},
        geometry=[Polygon([(0, 0), (300, 0), (300, 300), (0, 300)])],
        crs=synthetic_network.edges.crs,
    )
    integrator = PolygonAreaWeightedIntegrator()
    result = integrator.integrate(
        synthetic_network.edges.copy(),
        poly,
        {"pop": "population"},
        buffer_m=50,
    )
    assert "pop" in result.columns
