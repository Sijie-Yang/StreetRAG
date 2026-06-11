"""Tests for skills."""

import pytest

from streetrag.skills.correlate import CorrelateSkill, CorrelateParams
from streetrag.skills.registry import register_all_skills, list_skills


def test_skill_registry():
    register_all_skills()
    names = [s["name"] for s in list_skills()]
    assert "composite_index" in names
    assert "multiscale_profile" in names
    assert "correlate" in names


def test_correlate_skill(synthetic_network):
    import numpy as np

    gdf = synthetic_network.edges
    gdf["x"] = np.arange(len(gdf))
    gdf["y"] = np.arange(len(gdf)) * 2 + 1
    synthetic_network.edges = gdf
    skill = CorrelateSkill()
    result = skill.run(
        synthetic_network,
        CorrelateParams(feature_a="x", feature_b="y", user_query="test"),
    )
    assert result.stats["pearson"] > 0.99
