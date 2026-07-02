"""Tests for single-layer network GPKG format."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from streetrag.core.network_gpkg import (
    EDGE_ID_COL,
    NETWORK_LAYER,
    graph_from_lines,
    is_street_network_gpkg,
    list_network_column_names,
    read_network_gpkg,
)
from streetrag.core.street_network import StreetNetwork


@pytest.fixture
def network_only_gpkg(tmp_path: Path) -> Path:
    lines = [
        LineString([(0, 0), (100, 0)]),
        LineString([(100, 0), (100, 100)]),
        LineString([(0, 0), (0, 100)]),
    ]
    gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:32648")
    path = tmp_path / "city_drive.gpkg"
    gdf.to_file(path, layer=NETWORK_LAYER, driver="GPKG")
    return path


def test_graph_from_lines_assigns_uv(network_only_gpkg: Path) -> None:
    edges, nodes = read_network_gpkg(network_only_gpkg)
    assert len(edges) == 3
    assert "u" in edges.columns and "v" in edges.columns
    assert EDGE_ID_COL in edges.columns
    assert len(nodes) >= 3
    assert "osmid" in nodes.columns


def test_is_street_network_gpkg(network_only_gpkg: Path) -> None:
    assert is_street_network_gpkg(network_only_gpkg)


def test_street_network_roundtrip(tmp_path: Path, network_only_gpkg: Path) -> None:
    from streetrag.core.feature_catalog import FeatureCatalog

    catalog = FeatureCatalog(tmp_path / "feature_registry.json")
    catalog._data = {
        "target_network": network_only_gpkg.name,
        "target_layer": NETWORK_LAYER,
        "feature_statistics": {},
        "composite_index_columns": [],
        "point_integrations": [],
    }
    catalog.path.parent.mkdir(parents=True, exist_ok=True)
    # symlink or copy gpkg into catalog dir
    dest = catalog.data_dir / network_only_gpkg.name
    dest.write_bytes(network_only_gpkg.read_bytes())

    net = StreetNetwork.from_catalog(catalog, use_cache=False)
    assert len(net.edges) == 3
    net.edges["test_col"] = 1.0
    net.save()
    reloaded, _ = read_network_gpkg(dest)
    assert "test_col" not in reloaded.columns
    net2 = StreetNetwork.from_catalog(catalog, use_cache=False)
    assert "test_col" in net2.edges.columns


def test_graph_from_lines_helper() -> None:
    gdf = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (50, 0)]), LineString([(50, 0), (100, 0)])],
        crs="EPSG:32648",
    )
    edges, nodes = graph_from_lines(gdf)
    assert edges["length"].tolist() == pytest.approx([50.0, 50.0])
    assert len(nodes) == 3


def test_list_network_column_names(network_only_gpkg: Path) -> None:
    cols = list_network_column_names(network_only_gpkg)
    assert cols == []
    edges, _ = read_network_gpkg(network_only_gpkg)
    edges["foo"] = 1
    edges.to_file(network_only_gpkg, layer=NETWORK_LAYER, driver="GPKG")
    assert "foo" in list_network_column_names(network_only_gpkg)
