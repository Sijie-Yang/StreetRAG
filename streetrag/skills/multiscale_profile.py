"""Multi-scale profile skill."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from streetrag.core.spatial_utils import length_weighted_stats
from streetrag.skills.base import Skill, SkillResult, skill
from streetrag.skills.stats import calculate_statistics


class MultiscaleProfileParams(BaseModel):
    index_col: str = Field(..., description="Column to profile across integration scales")
    top_k: int = Field(10, description="Top-k edges per scale")
    user_query: str = ""


@skill
class MultiscaleProfileSkill(Skill):
    name = "multiscale_profile"
    description = (
        "Analyze how an index behaves across LOCAL/MEDIUM/GLOBAL integration scales. "
        "Returns length-weighted top-k stats and cross-scale correlation matrix."
    )
    params_model = MultiscaleProfileParams

    def run(self, net, params: MultiscaleProfileParams) -> SkillResult:
        if params.index_col not in net.edges.columns:
            raise ValueError(f"Column {params.index_col} not found")
        registry = net.registry_dict()
        radii = (registry.get("space_syntax_integration") or {}).get("radii") or []
        index_values = pd.to_numeric(net.edges[params.index_col], errors="coerce")
        lengths = (
            pd.to_numeric(net.edges["length"], errors="coerce")
            if "length" in net.edges.columns
            else net.edges.geometry.length
        )

        scale_profiles = {}
        integration_cols = []
        for r in radii:
            col = f"integration_R{r}"
            if col not in net.edges.columns:
                continue
            integration_cols.append(col)
            s = pd.to_numeric(net.edges[col], errors="coerce")
            mask = s.notna() & index_values.notna()
            if mask.sum() == 0:
                continue
            k = min(params.top_k, int(mask.sum()))
            idx = s[mask].nlargest(k).index
            lw = length_weighted_stats(index_values.loc[idx], lengths.loc[idx])
            scale_profiles[f"R{r}"] = {
                "radius_m": r,
                "length_weighted": lw,
                "index_stats": calculate_statistics(index_values.loc[idx]),
            }

        corr_matrix = {}
        if len(integration_cols) >= 2:
            sub = net.edges[integration_cols + [params.index_col]].dropna()
            if len(sub) > 10:
                corr = sub.corr()
                for c1 in integration_cols + [params.index_col]:
                    corr_matrix[c1] = {
                        c2: float(corr.loc[c1, c2]) if c1 in corr.index and c2 in corr.columns else None
                        for c2 in integration_cols + [params.index_col]
                    }

        overall = calculate_statistics(index_values)
        reply = (
            f"Multi-scale profile for '{params.index_col}': "
            f"{len(scale_profiles)} scales analyzed. "
            f"Overall mean={overall.get('mean')!r}."
        )
        if params.user_query:
            try:
                import json

                from streetrag.llm.client import chat_text, load_rag_settings

                registry_path = str(net.catalog.path)
                settings = load_rag_settings(registry_path)
                reply = chat_text(
                    settings=settings,
                    registry_path=registry_path,
                    system=(
                        "You are an urban analyst interpreting multi-scale "
                        "space-syntax evidence. Ground every claim in the numbers."
                    ),
                    prompt=(
                        f"User question: {params.user_query!r}\n"
                        f"Index profiled: {params.index_col}\n"
                        f"Overall stats: {json.dumps(overall, ensure_ascii=False)}\n"
                        f"Per-scale top-{params.top_k} profiles (length-weighted):\n"
                        f"{json.dumps(scale_profiles, ensure_ascii=False, indent=2)}\n"
                        f"Cross-scale correlation matrix:\n"
                        f"{json.dumps(corr_matrix, ensure_ascii=False, indent=2)}\n\n"
                        "Explain how this index behaves at LOCAL vs MEDIUM vs GLOBAL "
                        "scales, citing the numbers (km, means, correlations). "
                        "Answer in the same language as the user question. "
                        "Plain text, no markdown."
                    ),
                )
            except Exception:
                pass  # keep template reply if LLM unavailable
        return SkillResult(
            skill_name=self.name,
            columns_written=[],
            stats=overall,
            narrative_evidence={
                "scale_profiles": scale_profiles,
                "correlation_matrix": corr_matrix,
                "radii": radii,
            },
            reply=reply,
            index_col=params.index_col,
            render_map=True,
        )
