"""Tests for agent tools and context."""

import json

from streetrag.agent.tools import action_tools_openai, build_context_prompt, execute_action_tool
from streetrag.core.feature_catalog import FeatureCatalog


def test_action_tools_schema() -> None:
    tools = action_tools_openai()
    names = {t["function"]["name"] for t in tools}
    assert "visualize_feature" in names
    assert "propose_index_options" in names
    assert "show_table" in names
    assert "top_edges_table" in names


def test_top_edges_table(synthetic_network) -> None:
    gdf = synthetic_network.edges
    gdf["heat"] = range(len(gdf))
    gdf["integration_R500"] = [float(i) / len(gdf) for i in range(len(gdf))]
    synthetic_network.edges = gdf
    events: list = []
    result = execute_action_tool(
        "top_edges_table",
        {"column": "heat", "n": 5, "title": "Top heat"},
        catalog=synthetic_network.catalog,
        net=synthetic_network,
        emit=events.append,
    )
    payload = json.loads(result)
    assert payload["ok"] is True
    assert payload["rows"] == 5
    assert events and events[0]["type"] == "table"
    assert events[0]["rows"][0][2] == 39
    assert events[0]["rows"][-1][2] == 35


def test_build_context_prompt(tmp_path) -> None:
    cat = FeatureCatalog(tmp_path / "feature_registry.json")
    cat._data = {
        "feature_statistics": {
            "greenery": {"min": 0, "max": 5, "mean": 3},
        },
    }
    cat.set_description("greenery", "Green view rate")
    ctx = build_context_prompt(cat, [{"type": "feature", "name": "greenery"}])
    assert "greenery" in ctx
    assert "Green view rate" in ctx
