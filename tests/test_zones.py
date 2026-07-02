"""Tests for zone grid generation."""

import geopandas as gpd
from shapely.geometry import LineString

from streetrag.zones.layers import generate_hex_grid, generate_rect_grid


def test_generate_rect_grid() -> None:
    gdf = generate_rect_grid((0, 0, 1000, 1000), cell_m=500, crs="EPSG:3857")
    assert len(gdf) >= 4
    assert "zone_id" in gdf.columns


def test_generate_hex_grid() -> None:
    gdf = generate_hex_grid((0, 0, 2000, 2000), radius_m=300, crs="EPSG:3857")
    assert len(gdf) > 0
    assert all(gdf.geometry.geom_type == "Polygon")
