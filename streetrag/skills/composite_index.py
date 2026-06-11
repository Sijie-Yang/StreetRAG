"""Composite index skill."""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Literal, Optional

import geopandas as gpd
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from shapely.geometry import Point

from streetrag.core.spatial_utils import morans_i_on_edges
from streetrag.llm.client import chat_text, load_rag_settings
from streetrag.llm.retrieval import FeatureWeight, IndexPlan
from streetrag.skills.base import Skill, SkillResult, skill
from streetrag.skills.stats import (
    calculate_statistics,
    combine_features,
    length_weighted_topk_by_integration,
)
from streetrag.utils.geocode import geocode_place

_ANALYST_SYSTEM = (
    "You are an urban analyst. Use the supplied evidence to answer "
    "the user's question grounded in numbers."
)


def _format_answer_prompt(
    user_query: str,
    index_name: str,
    statistics: dict,
    lw_topk: dict,
    morans_i: Optional[dict],
    plan_summary: Optional[dict],
) -> str:
    return (
        f"User question: {user_query!r}\n\n"
        f"Index name: {index_name}\n\n"
        f"Plan summary:\n{json.dumps(plan_summary or {}, ensure_ascii=False, indent=2)}\n\n"
        f"Overall statistics (entire network):\n{json.dumps(statistics, ensure_ascii=False, indent=2)}\n\n"
        f"Spatial autocorrelation (KNN Moran's I):\n{json.dumps(morans_i, ensure_ascii=False, indent=2)}\n\n"
        f"Length-weighted top-k by integration (multi-scale):\n"
        f"{json.dumps(lw_topk, ensure_ascii=False, indent=2)}\n\n"
        "Answer the user's question in three paragraphs:\n"
        "1. Overall distribution and how it answers the question.\n"
        "2. Multi-scale comparison: how does the index behave on top integration "
        "streets at LOCAL vs MEDIUM vs GLOBAL scale? Cite the length-weighted "
        "numbers (km of network) so the reader knows the magnitude.\n"
        "3. Spatial pattern: comment briefly on Moran's I (clustered / dispersed / "
        "random) and note any caveats (sampling, scale).\n"
        "Answer in the same language as the user question. Plain text, no markdown, no JSON."
    )


class CompositeIndexParams(BaseModel):
    intent: Literal["create_new", "analyze_existing"] = "create_new"
    target_index_col: Optional[str] = None
    index_name: str = Field(..., description="Snake_case index name")
    features: List[FeatureWeight] = Field(
        ...,
        description="List of {name, weight, rationale} objects; weight is signed.",
    )

    @field_validator("features", mode="before")
    @classmethod
    def _coerce_features(cls, v):
        # Tolerate LLM outputs that pass bare feature names instead of
        # {name, weight} objects: assign equal positive weights.
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            w = round(1.0 / len(v), 4)
            return [{"name": x, "weight": w} for x in v]
        return v

    @model_validator(mode="after")
    def _check_consistency(self):
        if self.proximity_dominant and not self.spatial_target:
            raise ValueError(
                "proximity_dominant=true REQUIRES spatial_target to be set to the "
                "named place the query wants to be close to (e.g. 'Changi Airport'). "
                "Set spatial_target, or set proximity_dominant=false."
            )
        if self.spatial_filter_radius_m and not self.spatial_target:
            raise ValueError(
                "spatial_filter_radius_m requires spatial_target to be set."
            )
        if self.intent == "analyze_existing" and not self.target_index_col:
            raise ValueError(
                "intent='analyze_existing' requires target_index_col "
                "(the column name of the existing index to analyze)."
            )
        return self
    operator: Literal["weighted_sum", "geometric_mean", "owa_top", "owa_bottom"] = "weighted_sum"
    normalization: Literal["percentile", "zscore", "minmax", "robust", "raw"] = "robust"
    spatial_target: Optional[str] = None
    spatial_filter_radius_m: Optional[float] = None
    proximity_dominant: bool = Field(
        False,
        description=(
            "True when the query is PRIMARILY about being near the spatial target "
            "('closest to X'); the distance feature will then dominate the index."
        ),
    )
    explanation: str = ""
    user_query: str = ""


@skill
class CompositeIndexSkill(Skill):
    name = "composite_index"
    description = (
        "Build or analyze a weighted composite street index from multiple features. "
        "Use for walkability, comfort, vibrancy, or proximity queries."
    )
    params_model = CompositeIndexParams

    def run(self, net, params: CompositeIndexParams) -> SkillResult:
        registry = net.registry_dict()
        settings = net.catalog.load_settings()
        registry_path = str(net.catalog.path)
        top_k = int(settings.get("analysis_top_k", settings.get("top_k", 10)))
        suffix = net.catalog.percentile_suffix
        plan = IndexPlan(
            intent=params.intent,
            target_index_col=params.target_index_col,
            index_name=params.index_name,
            features=params.features,
            operator=params.operator,
            normalization=params.normalization,
            spatial_target=params.spatial_target,
            spatial_filter_radius_m=params.spatial_filter_radius_m,
            proximity_dominant=params.proximity_dominant,
            explanation=params.explanation,
        )

        if plan.intent == "analyze_existing" and plan.target_index_col:
            col = plan.target_index_col
            if col not in net.edges.columns:
                raise ValueError(f"Column {col} not found")
            index_values = net.edges[col]
            stats = calculate_statistics(index_values)
            morans = morans_i_on_edges(net.edges, col)
            lw = length_weighted_topk_by_integration(net.edges, index_values, registry, top_k=top_k)
            meta = net.catalog.load_index_record(col) or {}
            reply = chat_text(
                settings=settings,
                registry_path=registry_path,
                system=_ANALYST_SYSTEM,
                prompt=_format_answer_prompt(
                    user_query=params.user_query,
                    index_name=meta.get("index_name") or col,
                    statistics=stats,
                    lw_topk=lw,
                    morans_i=morans,
                    plan_summary={
                        "intent": "analyze_existing",
                        "original_plan": {
                            "explanation": meta.get("explanation"),
                            "operator": meta.get("operator"),
                            "normalization": meta.get("normalization"),
                            "features_weights": meta.get("features_weights"),
                        },
                    },
                ),
            )
            return SkillResult(
                skill_name=self.name,
                columns_written=[col],
                stats=stats,
                narrative_evidence={"morans_i": morans, "length_weighted_topk": lw},
                reply=reply,
                index_col=col,
                index_name=meta.get("index_name", col),
                render_map=True,
                explanation=meta.get("explanation", ""),
                operator=meta.get("operator"),
                normalization=meta.get("normalization"),
                features_weights=meta.get("features_weights") or {},
                spatial_target=meta.get("spatial_target"),
                spatial_target_resolved=meta.get("spatial_target_resolved"),
                spatial_filter_radius_m=meta.get("spatial_filter_radius_m"),
            )

        # Drop hallucinated feature names (e.g. the LLM echoing the new index's
        # own name) so they don't silently dilute the real weights via fillna(0).
        from streetrag.skills.stats import _find_col_ci

        known, dropped = [], []
        for fw in plan.features:
            if _find_col_ci(net.edges, fw.name) is not None:
                known.append(fw)
            else:
                dropped.append(fw.name)
        if dropped and known:
            plan.features = known
        if dropped:
            plan.explanation = (
                plan.explanation + f" [dropped unknown features: {', '.join(dropped)}]"
            ).strip()

        spatial_resolved = None
        if plan.spatial_target:
            spatial_resolved = geocode_place(plan.spatial_target, registry_path)
            if spatial_resolved.get("found"):
                lon, lat = spatial_resolved["centroid_lonlat"]
                target_pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(net.edges.crs).iloc[0]
                net.edges["__distance_to_target_m"] = net.edges.geometry.distance(target_pt)
                other = [abs(fw.weight) for fw in plan.features]
                if plan.proximity_dominant:
                    # 'closest to X' queries: distance carries ~2/3 of the
                    # index; other features only break ties among nearby streets.
                    inj_w = -2.0 * (sum(other) or 1.0)
                else:
                    inj_w = -max(0.3, float(np.median(other))) if other else -0.4
                plan.features.append(FeatureWeight(
                    name="__distance_to_target_m",
                    weight=inj_w,
                    rationale=f"Proximity to {plan.spatial_target}",
                ))

        index_col = "".join(
            ch if ch.isalnum() or ch == "_" else "_"
            for ch in (plan.index_name or "composite_index").lower()
        )
        net.edges[index_col] = combine_features(
            net.edges, plan, percentile_suffix=suffix, registry=registry,
        )

        if (
            spatial_resolved and spatial_resolved.get("found")
            and plan.spatial_filter_radius_m
            and "__distance_to_target_m" in net.edges.columns
        ):
            outside = net.edges["__distance_to_target_m"] > float(plan.spatial_filter_radius_m)
            net.edges.loc[outside, index_col] = np.nan

        index_values = net.edges[index_col]
        stats = calculate_statistics(index_values)
        morans = morans_i_on_edges(net.edges, index_col)
        lw = length_weighted_topk_by_integration(net.edges, index_values, registry, top_k=top_k)
        net.compute_normalizations(index_col)
        net.catalog.register_feature_stats(
            index_col, stats, description=plan.explanation, composite=True,
        )
        net.save()
        net.catalog.save()

        reply = chat_text(
            settings=settings,
            registry_path=registry_path,
            system=_ANALYST_SYSTEM,
            prompt=_format_answer_prompt(
                user_query=params.user_query,
                index_name=plan.index_name,
                statistics=stats,
                lw_topk=lw,
                morans_i=morans,
                plan_summary=plan.model_dump(),
            ),
        )

        settings_meta = {
            "llm_model": settings.get("llm_model"),
            "llm_seed": settings.get("llm_seed"),
            "llm_temperature": settings.get("llm_temperature"),
        }
        record = {
            "index_name": plan.index_name,
            "index_col": index_col,
            "original_query": params.user_query,
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "length_weighted_topk": lw,
            "morans_i": morans,
            "summary": reply,
            "features_weights": {fw.name: fw.weight for fw in plan.features},
            "feature_rationales": [
                {"name": fw.name, "weight": fw.weight, "rationale": fw.rationale}
                for fw in plan.features
            ],
            "explanation": plan.explanation,
            "operator": plan.operator,
            "normalization": plan.normalization,
            "spatial_target": plan.spatial_target,
            "spatial_target_resolved": spatial_resolved,
            "spatial_filter_radius_m": plan.spatial_filter_radius_m,
            **settings_meta,
            "schema_version": 3,
        }
        net.catalog.save_index_record(record)

        return SkillResult(
            skill_name=self.name,
            columns_written=[index_col],
            stats=stats,
            narrative_evidence={"morans_i": morans, "length_weighted_topk": lw},
            reply=reply,
            index_col=index_col,
            index_name=plan.index_name,
            render_map=True,
            explanation=plan.explanation,
            operator=plan.operator,
            normalization=plan.normalization,
            features_weights={fw.name: fw.weight for fw in plan.features},
            spatial_target=plan.spatial_target,
            spatial_target_resolved=spatial_resolved,
            spatial_filter_radius_m=plan.spatial_filter_radius_m,
        )
