"""Feature correlation skill."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from streetrag.skills.base import Skill, SkillResult, skill


class CorrelateParams(BaseModel):
    feature_a: str = Field(..., description="First feature column")
    feature_b: str = Field(..., description="Second feature column")
    user_query: str = ""


@skill
class CorrelateSkill(Skill):
    name = "correlate"
    description = "Compute Pearson/Spearman correlation between two street features."
    params_model = CorrelateParams

    def run(self, net, params: CorrelateParams) -> SkillResult:
        a, b = params.feature_a, params.feature_b
        for col in (a, b):
            if col not in net.edges.columns:
                raise ValueError(f"Column {col} not found")
        sub = net.edges[[a, b]].dropna()
        if len(sub) < 3:
            raise ValueError("Not enough valid pairs for correlation")
        pearson = float(sub[a].corr(sub[b], method="pearson"))
        spearman = float(sub[a].corr(sub[b], method="spearman"))
        reply = (
            f"Correlation between '{a}' and '{b}' (n={len(sub)}): "
            f"Pearson={pearson:.4f}, Spearman={spearman:.4f}."
        )
        return SkillResult(
            skill_name=self.name,
            stats={
                "n": len(sub),
                "pearson": pearson,
                "spearman": spearman,
                "feature_a_mean": float(sub[a].mean()),
                "feature_b_mean": float(sub[b].mean()),
            },
            narrative_evidence={
                "feature_a": a,
                "feature_b": b,
                "pearson": pearson,
                "spearman": spearman,
            },
            reply=reply,
        )
