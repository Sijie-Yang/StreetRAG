"""LISA / Moran cluster analysis skill."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from streetrag.core.spatial_utils import morans_i_on_edges
from streetrag.skills.base import Skill, SkillResult, skill
from streetrag.skills.stats import calculate_statistics


class ClusterLisaParams(BaseModel):
    col: str = Field(..., description="Column for spatial cluster analysis")
    k_neighbors: int = Field(6, description="KNN neighbors")
    user_query: str = ""


@skill
class ClusterLisaSkill(Skill):
    name = "cluster_lisa"
    description = (
        "Compute spatial autocorrelation (Moran's I) and optional LISA clusters "
        "for a street attribute."
    )
    params_model = ClusterLisaParams

    def run(self, net, params: ClusterLisaParams) -> SkillResult:
        col = params.col
        if col not in net.edges.columns:
            raise ValueError(f"Column {col} not found")

        morans = morans_i_on_edges(
            net.edges, col, k_neighbors=params.k_neighbors,
        )
        lisa_summary = None
        lisa_col = f"{col}_lisa_cluster"

        try:
            from esda.moran import Moran_Local
            from libpysal.weights import KNN

            g = net.edges[[col, "geometry"]].copy()
            g = g[g[col].notna() & g.geometry.notna()]
            if len(g) >= params.k_neighbors + 1:
                coords = np.array([(p.x, p.y) for p in g.geometry.centroid])
                w = KNN.from_array(coords, k=params.k_neighbors)
                w.transform = "r"
                vals = pd.to_numeric(g[col], errors="coerce").values
                ml = Moran_Local(vals, w, seed=42)
                quadrant_labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
                clusters = pd.Series(pd.NA, index=net.edges.index, dtype="object")
                significant = ml.p_sim < 0.05
                for i, orig_idx in enumerate(g.index):
                    if significant[i]:
                        clusters.loc[orig_idx] = quadrant_labels.get(int(ml.q[i]))
                net.edges[lisa_col] = clusters
                lisa_summary = clusters.dropna().value_counts().to_dict()
                lisa_summary["not_significant"] = int((~significant).sum())
        except ImportError:
            lisa_summary = {"note": "Install esda/libpysal for full LISA (pip install streetrag[lisa])"}

        stats = calculate_statistics(net.edges[col])
        reply = (
            f"Moran's I for '{col}': {morans.get('I') if morans else 'N/A'}. "
            f"LISA clusters: {lisa_summary or 'not computed'}."
        )
        cols_written = [lisa_col] if lisa_col in net.edges.columns else []
        if cols_written:
            net.save()

        return SkillResult(
            skill_name=self.name,
            columns_written=cols_written,
            stats=stats,
            narrative_evidence={"morans_i": morans, "lisa_summary": lisa_summary},
            reply=reply,
            index_col=col,
            render_map=True,
        )
