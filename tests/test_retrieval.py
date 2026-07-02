"""Tests for source-stratified feature retrieval."""

from __future__ import annotations

from streetrag.llm.retrieval import (
    _group_by_source,
    _group_features_for_planner,
    rank_features_for_query,
)


def _feat(name: str, source: str, desc: str = "") -> dict:
    return {
        "name": name,
        "source": source,
        "description": desc or name,
        "range": "[0, 1]",
    }


def test_group_by_source():
    feats = [
        _feat("a", "small_src"),
        _feat("b", "small_src"),
        _feat("c", "big_src"),
    ]
    groups = _group_by_source(feats)
    assert len(groups["small_src"]) == 2
    assert len(groups["big_src"]) == 1


def test_stratified_includes_entire_small_source():
    feats = [_feat(f"poi_{i}", "big_src") for i in range(50)]
    feats += [_feat(f"perc_{i}", "small_src", "thermal comfort perception") for i in range(8)]
    feats += [_feat("integration_R500", "syntax", "local integration")]

    ranked = rank_features_for_query(
        "thermal comfort perception",
        feats,
        top_m=30,
        method="stratified",
    )
    names = {f["name"] for f in ranked}
    assert all(f"perc_{i}" in names for i in range(8))
    assert "integration_R500" in names
    poi_names = [f["name"] for f in ranked if f["name"].startswith("poi_")]
    assert len(poi_names) <= 12  # 30 * 0.4 cap


def test_stratified_caps_large_source_share():
    feats = [_feat(f"poi_{i}", "big_src") for i in range(100)]
    feats += [_feat("alpha", "tiny_a")]
    feats += [_feat("beta", "tiny_b")]

    ranked = rank_features_for_query(
        "poi cafe restaurant",
        feats,
        top_m=20,
        method="stratified",
    )
    by_source = _group_by_source(ranked)
    assert "alpha" in {f["name"] for f in ranked}
    assert "beta" in {f["name"] for f in ranked}
    assert len(by_source.get("big_src", [])) <= 8


def test_group_features_for_planner_structure():
    feats = [
        _feat("x", "src_a"),
        _feat("y", "src_b"),
    ]
    grouped = _group_features_for_planner(feats)
    assert len(grouped) == 2
    assert grouped[0]["source"] in {"src_a", "src_b"}
    assert "features" in grouped[0]


def test_full_method_returns_all_when_small_catalog():
    feats = [_feat(f"f{i}", "src") for i in range(12)]
    ranked = rank_features_for_query("anything", feats, top_m=5, method="full")
    assert len(ranked) == 12
