"""Tests for catalog delete operations."""

from pathlib import Path

from streetrag.core.feature_catalog import FeatureCatalog


def test_remove_feature_stats(tmp_path: Path) -> None:
    cat = FeatureCatalog(tmp_path / "feature_registry.json")
    cat._data = {
        "feature_statistics": {
            "foo": {
                "min": 0, "max": 1, "mean": 0.5, "std": 0.1, "median": 0.5,
                "q25": 0.25, "q75": 0.75, "iqr": 0.5,
                "normalization_columns": {"percentile": "foo_pctl"},
            },
        },
        "composite_index_columns": [],
        "target_network_features": {"foo": "test", "foo_pctl": "pct"},
    }
    removed = cat.remove_feature_stats("foo")
    assert "foo" in removed
    assert "foo_pctl" in removed
    assert "foo" not in cat.feature_statistics


def test_remove_point_integration(tmp_path: Path) -> None:
    cat = FeatureCatalog(tmp_path / "feature_registry.json")
    cat._data = {
        "point_integrations": [{
            "source_file": "test.gpkg",
            "columns": {"col_a": "desc"},
        }],
        "feature_statistics": {"col_a": {"min": 0, "max": 1}},
        "text_integrations": [{"source_file": "test.gpkg", "n_chunks": 5}],
    }
    block = cat.remove_point_integration("test.gpkg")
    assert block is not None
    assert len(cat.point_integrations) == 0
    assert "col_a" not in cat.feature_statistics
