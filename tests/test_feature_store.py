"""Tests for split feature storage (parquet) and edge_id."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from streetrag.core.feature_catalog import FeatureCatalog
from streetrag.core.feature_store import (
    FeatureStore,
    migrate_wide_gpkg_to_split,
)
from streetrag.core.network_gpkg import (
    EDGE_ID_COL,
    NETWORK_LAYER,
    ensure_edge_ids,
    graph_from_lines,
    read_network_gpkg,
)
from streetrag.core.street_network import StreetNetwork


@pytest.fixture
def wide_network_gpkg(tmp_path: Path) -> tuple[Path, FeatureCatalog]:
    lines = [
        LineString([(0, 0), (100, 0)]),
        LineString([(100, 0), (100, 100)]),
        LineString([(0, 0), (0, 100)]),
    ]
    gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:32648")
    edges, _nodes = graph_from_lines(gdf)
    edges["poi_a"] = [1.0, 2.0, 3.0]
    edges["poi_a_pctl"] = [10.0, 50.0, 90.0]
    path = tmp_path / "city_drive.gpkg"
    edges.to_file(path, layer=NETWORK_LAYER, driver="GPKG")

    catalog = FeatureCatalog(tmp_path / "feature_registry.json")
    catalog._data = {
        "target_network": path.name,
        "target_layer": NETWORK_LAYER,
        "feature_statistics": {
            "poi_a": {
                "min": 1,
                "max": 3,
                "mean": 2,
                "std": 1,
                "median": 2,
                "q25": 1.5,
                "q75": 2.5,
                "iqr": 1,
                "normalization_columns": {"percentile": "poi_a_pctl"},
            }
        },
        "composite_index_columns": [],
        "point_integrations": [
            {
                "source_file": "poi_source.gpkg",
                "columns": {"poi_a": "test"},
            }
        ],
    }
    catalog.save()
    return path, catalog


def test_ensure_edge_ids_stable() -> None:
    gdf = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])],
        crs="EPSG:32648",
    )
    edges, _ = graph_from_lines(gdf)
    assert EDGE_ID_COL in edges.columns
    assert edges[EDGE_ID_COL].tolist() == [0, 1]
    again = ensure_edge_ids(edges)
    assert again[EDGE_ID_COL].tolist() == [0, 1]


def test_migrate_and_reload_roundtrip(wide_network_gpkg) -> None:
    _path, catalog = wide_network_gpkg
    summary = migrate_wide_gpkg_to_split(catalog, verbose=False)
    assert summary["feature_files"] >= 1
    assert catalog.uses_split_storage()

    net = StreetNetwork.from_catalog(catalog, use_cache=False)
    assert "poi_a" in net.edges.columns
    assert net.edges["poi_a"].tolist() == pytest.approx([1.0, 2.0, 3.0])

    topo_cols = set(pd.read_parquet(catalog.data_dir / "features" / "poi_source.parquet").columns)
    assert "poi_a" in topo_cols

    gpkg_cols = set(
        __import__("streetrag.core.network_gpkg", fromlist=["list_network_column_names"]).list_network_column_names(
            catalog.target_network_path, edges_layer=NETWORK_LAYER
        )
    )
    assert "poi_a" not in gpkg_cols
    assert EDGE_ID_COL in gpkg_cols


def test_feature_store_drop_columns(wide_network_gpkg) -> None:
    _path, catalog = wide_network_gpkg
    migrate_wide_gpkg_to_split(catalog, verbose=False)
    store = FeatureStore(catalog)
    removed = store.drop_columns(["poi_a", "poi_a_pctl"])
    assert "poi_a" in removed
    net = StreetNetwork.from_catalog(catalog, use_cache=False)
    assert "poi_a" not in net.edges.columns


def test_save_splits_without_prior_migrate(tmp_path: Path) -> None:
    """Integrate/save with feature columns goes to parquet, not GPKG."""
    lines = [
        LineString([(0, 0), (100, 0)]),
        LineString([(100, 0), (100, 100)]),
    ]
    gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:32648")
    edges, _ = graph_from_lines(gdf)
    path = tmp_path / "city_drive.gpkg"
    edges.to_file(path, layer=NETWORK_LAYER, driver="GPKG")

    catalog = FeatureCatalog(tmp_path / "feature_registry.json")
    catalog._data = {
        "target_network": path.name,
        "target_layer": NETWORK_LAYER,
        "feature_statistics": {},
        "composite_index_columns": [],
        "point_integrations": [
            {"source_file": "poi.gpkg", "columns": {"poi_a": "test"}},
        ],
    }
    catalog.save()

    net = StreetNetwork.from_catalog(catalog, use_cache=False)
    net.edges["poi_a"] = [1.0, 2.0]
    net.save()

    gpkg_cols = set(
        __import__(
            "streetrag.core.network_gpkg",
            fromlist=["list_network_column_names"],
        ).list_network_column_names(path, edges_layer=NETWORK_LAYER)
    )
    assert "poi_a" not in gpkg_cols
    assert (tmp_path / "features" / "poi.parquet").exists()
    reloaded = StreetNetwork.from_catalog(catalog, use_cache=False)
    assert reloaded.edges["poi_a"].tolist() == pytest.approx([1.0, 2.0])


def test_street_network_save_split(wide_network_gpkg) -> None:
    _path, catalog = wide_network_gpkg
    migrate_wide_gpkg_to_split(catalog, verbose=False)
    net = StreetNetwork.from_catalog(catalog, use_cache=False)
    catalog.composite_index_columns.append("new_idx")
    net.edges["new_idx"] = np.linspace(0, 1, len(net.edges))
    net.compute_normalizations("new_idx")
    net.save()

    reloaded = StreetNetwork.from_catalog(catalog, use_cache=False)
    assert "new_idx" in reloaded.edges.columns
    store = FeatureStore(catalog)
    assert "new_idx" in store.list_feature_column_names()
