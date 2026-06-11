"""Tests for core math and normalization."""

import numpy as np
import pandas as pd

from streetrag.skills.stats import combine_features, resolve_normalized_series
from streetrag.llm.retrieval import FeatureWeight, IndexPlan


def test_resolve_normalized_percentile(synthetic_network):
    gdf = synthetic_network.edges
    gdf["test_feat"] = np.linspace(0, 100, len(gdf))
    s = resolve_normalized_series(gdf, "test_feat", "percentile")
    assert s.min() >= 0
    assert s.max() <= 1


def test_combine_weighted_sum(synthetic_network):
    gdf = synthetic_network.edges
    gdf["f1"] = np.random.rand(len(gdf))
    gdf["f2"] = np.random.rand(len(gdf))
    plan = IndexPlan(
        intent="create_new",
        index_name="test_index",
        features=[
            FeatureWeight(name="f1", weight=0.6, rationale="a"),
            FeatureWeight(name="f2", weight=0.4, rationale="b"),
        ],
        operator="weighted_sum",
        normalization="percentile",
        explanation="test",
    )
    result = combine_features(gdf, plan, percentile_suffix="_pctl", registry={})
    assert len(result) == len(gdf)
    assert result.notna().all()


def test_compute_normalizations(synthetic_network):
    gdf = synthetic_network.edges
    gdf["new_col"] = np.random.rand(len(gdf)) * 100
    synthetic_network.edges = gdf
    stats = synthetic_network.compute_normalizations("new_col")
    assert "min" in stats
    assert f"new_col_pctl" in synthetic_network.edges.columns


def test_length_weighted_stats():
    from streetrag.core.spatial_utils import length_weighted_stats

    v = pd.Series([1.0, 2.0, 3.0])
    L = pd.Series([100.0, 100.0, 100.0])
    s = length_weighted_stats(v, L)
    assert s["weighted_mean"] == 2.0
    assert s["n_edges"] == 3
