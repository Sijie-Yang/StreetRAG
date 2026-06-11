"""Tests for the angular segment analysis engine."""

import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from streetrag.syntax.engine import angular_segment_analysis, build_segment_graph


@pytest.fixture
def straight_line_edges() -> gpd.GeoDataFrame:
    """Three collinear segments: A--B--C, no turns anywhere."""
    return gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
            LineString([(200, 0), (300, 0)]),
        ],
        crs="EPSG:32648",
    )


@pytest.fixture
def t_junction_edges() -> gpd.GeoDataFrame:
    """A straight street plus a perpendicular branch at the middle."""
    return gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
            LineString([(100, 0), (100, 100)]),
        ],
        crs="EPSG:32648",
    )


def test_dual_graph_adjacency(straight_line_edges):
    adj = build_segment_graph(straight_line_edges)
    # middle segment touches both ends
    assert len(adj[1]) == 2
    # straight continuation costs ~0 angular units
    for _, cost, _ in adj[1]:
        assert cost == pytest.approx(0.0, abs=1e-6)


def test_dual_graph_right_angle(t_junction_edges):
    adj = build_segment_graph(t_junction_edges)
    # branch (idx 2) connects to both street halves with a 90° turn = 1.0
    costs = sorted(cost for _, cost, _ in adj[2])
    assert costs == pytest.approx([1.0, 1.0], abs=1e-6)


def test_angular_analysis_straight(straight_line_edges):
    df = angular_segment_analysis(straight_line_edges, radius_m=1000)
    # all three reachable from each other, zero angular depth
    assert (df["node_count"] == 2).all()
    assert df["total_depth"].max() == pytest.approx(0.0, abs=1e-9)
    # middle segment lies on the only A→C path → positive choice
    assert df.loc[1, "choice"] > 0
    assert df.loc[0, "choice"] == 0


def test_angular_analysis_radius_restriction(straight_line_edges):
    # center-to-center step is 100m; with a 50m radius nothing is reachable
    df = angular_segment_analysis(straight_line_edges, radius_m=50)
    assert (df["node_count"] == 0).all()


def test_nain_nach_formulas(t_junction_edges):
    df = angular_segment_analysis(t_junction_edges, radius_m=1000)
    for idx in df.index:
        nc, td = df.loc[idx, "node_count"], df.loc[idx, "total_depth"]
        if td > 0:
            assert df.loc[idx, "nain"] == pytest.approx(nc ** 1.2 / td)
        ch = df.loc[idx, "choice"]
        assert df.loc[idx, "nach"] == pytest.approx(
            math.log10(ch + 1) / math.log10(td + 3)
        )


def test_compute_syntax_columns(synthetic_network):
    from streetrag.syntax.engine import compute_syntax

    net = compute_syntax(synthetic_network, radii=[500], measures=["angular"])
    for col in (
        "angular_integration_R500",
        "nain_R500",
        "choice_R500",
        "nach_R500",
    ):
        assert col in net.edges.columns
    # grid is fully connected at R500 (steps are 100m)
    assert net.edges["angular_integration_R500"].notna().any()
