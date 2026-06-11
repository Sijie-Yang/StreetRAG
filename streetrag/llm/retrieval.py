"""Feature retrieval and IndexPlan schema."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator

from streetrag.core.feature_catalog import FeatureCatalog
from streetrag.llm.client import (
    chat_structured,
    cosine_topk,
    embed_texts,
    embeddings_paths,
    resolve_api_key,
)


OperatorName = Literal["weighted_sum", "geometric_mean", "owa_top", "owa_bottom"]
NormalizationName = Literal["percentile", "zscore", "minmax", "robust", "raw"]


class FeatureWeight(BaseModel):
    name: str = Field(..., description="Exact feature name from the catalog")
    weight: float = Field(..., description="Signed weight; use negative for inverse meaning")
    rationale: str = Field("", description="Why this feature and this sign")


class IndexPlan(BaseModel):
    intent: Literal["create_new", "analyze_existing"] = Field(
        ..., description="Whether to build a fresh composite index or analyse an existing one"
    )
    target_index_col: Optional[str] = Field(
        None,
        description="When intent='analyze_existing', the column name of the matching existing index",
    )
    index_name: str = Field(..., description="Snake_case name for the composite index")
    features: List[FeatureWeight] = Field(
        ..., description="Selected features with signed weights (5-10 entries recommended)"
    )
    operator: OperatorName = Field("weighted_sum")
    normalization: NormalizationName = Field("robust")
    spatial_target: Optional[str] = Field(None)
    spatial_filter_radius_m: Optional[float] = Field(None)
    proximity_dominant: bool = Field(
        False,
        description=(
            "Set true when the query is PRIMARILY about being near the spatial "
            "target ('closest to X', '离X最近'); the injected distance feature "
            "will then dominate the index and other features act as tie-breakers."
        ),
    )
    explanation: str = Field(...)

    @field_validator("features")
    @classmethod
    def _non_empty(cls, v: List[FeatureWeight]) -> List[FeatureWeight]:
        if not v:
            raise ValueError("features must contain at least one entry")
        return v


def effective_street_rag_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "analysis_top_k": int(settings.get("analysis_top_k", settings.get("top_k", 10))),
        "feature_retrieval_top_m": int(settings.get("feature_retrieval_top_m", 60)),
        "feature_retrieval_method": settings.get("feature_retrieval_method", "embedding"),
        "preferred_crs_epsg": settings.get("preferred_crs_epsg"),
    }


def list_features_info(catalog: FeatureCatalog | dict) -> List[dict]:
    if isinstance(catalog, FeatureCatalog):
        return catalog.list_features_info()
    fc = FeatureCatalog.__new__(FeatureCatalog)
    fc._data = catalog
    fc.path = Path(".")
    fc.data_dir = Path(".")
    return fc.list_features_info()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _token_rank(user_query: str, features_info: List[dict], top_m: int) -> List[dict]:
    integ = [f for f in features_info if f.get("name", "").startswith("integration_R")]
    rest = [f for f in features_info if not f.get("name", "").startswith("integration_R")]
    q_toks = set(_tokenize(user_query))
    if not q_toks:
        return integ + rest[: max(0, top_m - len(integ))]

    def overlap(f: dict) -> int:
        t = f"{f.get('name', '')} {f.get('description', '')}"
        return len(q_toks & set(_tokenize(t)))

    rest.sort(key=overlap, reverse=True)
    budget = max(0, top_m - len(integ))
    return (integ + rest[:budget])[:top_m]


def _feature_doc(f: dict) -> str:
    return (
        f"name: {f.get('name','')}. "
        f"source: {f.get('source','')}. "
        f"range: {f.get('range','')}. "
        f"description: {f.get('description','')}"
    )


def _embedding_index_payload(features_info: List[dict]) -> dict:
    return {
        "version": 2,
        "names": [f.get("name", "") for f in features_info],
        "docs": [_feature_doc(f) for f in features_info],
    }


def _load_embedding_index(registry_path: str, features_info: List[dict]) -> Optional[np.ndarray]:
    npz_path, meta_path = embeddings_paths(registry_path)
    if not (npz_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cur = _embedding_index_payload(features_info)
        if meta.get("docs") != cur["docs"]:
            return None
        with np.load(npz_path) as z:
            return z["features"].astype(np.float32)
    except Exception:
        return None


def _build_embedding_index(
    registry_path: str, settings: dict, features_info: List[dict]
) -> Optional[np.ndarray]:
    if not resolve_api_key(settings):
        return None
    docs = [_feature_doc(f) for f in features_info]
    if not docs:
        return None
    mat = embed_texts(settings=settings, registry_path=registry_path, texts=docs)
    npz_path, meta_path = embeddings_paths(registry_path)
    np.savez_compressed(npz_path, features=mat)
    meta_path.write_text(
        json.dumps(
            {
                **_embedding_index_payload(features_info),
                "embedding_model": settings.get("embedding_model", "text-embedding-3-small"),
                "ts": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return mat


def rank_features_for_query(
    user_query: str,
    features_info: List[dict],
    top_m: int,
    *,
    registry_path: Optional[str] = None,
    settings: Optional[dict] = None,
    method: str = "embedding",
) -> List[dict]:
    if top_m <= 0 or len(features_info) <= top_m:
        return list(features_info)
    method = method or "embedding"
    if method == "embedding" and registry_path and settings:
        mat = _load_embedding_index(registry_path, features_info)
        if mat is None:
            mat = _build_embedding_index(registry_path, settings, features_info)
        if mat is not None and mat.size:
            q_vec = embed_texts(
                settings=settings, registry_path=registry_path, texts=[user_query]
            )[0]
            order = cosine_topk(q_vec, mat, k=len(features_info))
            ranked = [features_info[i] for i in order]
            integ = [f for f in ranked if f.get("name", "").startswith("integration_R")]
            rest = [f for f in ranked if not f.get("name", "").startswith("integration_R")]
            budget = max(0, top_m - len(integ))
            return (integ + rest[:budget])[:top_m]
    return _token_rank(user_query, features_info, top_m)


def _scale_hints(registry: dict) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    radii = (registry.get("space_syntax_integration") or {}).get("radii") or []
    radii_sorted = sorted(int(r) for r in radii)
    if not radii_sorted:
        return None, None, None
    local_r = radii_sorted[0]
    global_r = radii_sorted[-1]
    medium_r = radii_sorted[len(radii_sorted) // 2] if len(radii_sorted) >= 2 else radii_sorted[0]
    return local_r, medium_r, global_r


def build_planner_context(
    user_query: str,
    registry: dict,
    features_info: List[dict],
    existing_indices: List[dict],
) -> str:
    """Shared evidence + rules block used by both the structured planner
    and the function-calling agent. Contains the full feature catalog
    excerpt, scale hints, and spatial-targeting / operator guidance."""
    local_r, medium_r, global_r = _scale_hints(registry)

    catalog_excerpt = [
        {
            "name": f.get("name", ""),
            "source": f.get("source", ""),
            "range": f.get("range", ""),
            "description": (f.get("description") or "")[:240],
        }
        for f in features_info
    ]
    existing_excerpt = [
        {
            "index_col": e.get("index_col"),
            "index_name": e.get("index_name"),
            "original_query": e.get("original_query"),
            "explanation": (e.get("explanation") or "")[:160],
        }
        for e in existing_indices[-12:]
    ]

    return (
        f"User query:\n{user_query!r}\n\n"
        "STEP 0 — Spatial targeting (always check first).\n"
        "Does the query reference a SPECIFIC NAMED place? Examples that ARE named places:\n"
        "  'NUS', 'Marina Bay Sands', 'Changi Airport', 'Sentosa', 'Orchard Road', 'Little India'.\n"
        "Examples that are NOT (do not set spatial_target):\n"
        "  'the city centre', 'commercial areas', 'university streets', 'food streets'.\n"
        "If YES — set spatial_target to the canonical name. The system will auto-inject a\n"
        "signed distance-to-target feature, so the whole map keeps a smooth proximity\n"
        "gradient. This is what you want for almost all 'near X / around X / 靠近 X /\n"
        "在 X 附近' queries. For those, LEAVE spatial_filter_radius_m null.\n"
        "If the query is PRIMARILY about proximity itself ('closest to X', '离X最近的\n"
        "地方', 'nearest to X') — set BOTH spatial_target='X' AND proximity_dominant=true,\n"
        "and pick only 2-4 secondary features with small weights (they break ties among\n"
        "equally-near streets); the distance gradient will dominate the index.\n"
        "RULE: proximity_dominant=true is INVALID without spatial_target — always set both.\n"
        "Only set spatial_filter_radius_m (1000-3000m) when the query has HARD EXCLUSION\n"
        "semantics ('inside the campus only', 'within 1km of X', '步行5分钟内').\n\n"
        "STEP 1 — Decide intent.\n"
        "If the query is conceptually equivalent to an existing index (possibly\n"
        "paraphrased), set intent='analyze_existing' and put that index's column name\n"
        "in target_index_col. Otherwise intent='create_new'.\n\n"
        f"Existing indices ({len(existing_excerpt)}):\n"
        f"{json.dumps(existing_excerpt, ensure_ascii=False, indent=2)}\n\n"
        "STEP 2 — If create_new, choose features from the catalog below\n"
        "(pre-filtered by semantic retrieval; multi-scale syntax measures always included):\n"
        f"{json.dumps(catalog_excerpt, ensure_ascii=False, indent=2)}\n\n"
        "Space-syntax scale hints (radii present in this dataset, meters):\n"
        f"  local={local_r}, medium={medium_r}, global={global_r}\n"
        "  - 'walkable / neighbourhood / daily / nearby' → prefer the smallest radius\n"
        "  - 'district / commercial corridors / regional' → prefer the middle radius\n"
        "  - 'city-wide / strategic / major destinations' → prefer the largest radius\n"
        "  - integration_R* = metric closeness; nain_R* / angular_integration_R* = angular\n"
        "    to-movement potential; choice_R* / nach_R* = through-movement potential.\n\n"
        "STEP 3 — Assign signed weights.\n"
        "  HIGHER feature ⇒ query MORE true → POSITIVE weight\n"
        "  HIGHER feature ⇒ query LESS true → NEGATIVE weight\n"
        "  Sum of |weights| ≈ 1.0; pick 5–10 features (2-4 if proximity_dominant).\n"
        "  DO NOT add a __distance_to_target_m feature yourself — the system will.\n"
        "  WARNING: integration_R*/nain/choice measure NETWORK CENTRALITY, not distance\n"
        "  to a named place. NEVER use them (or POI densities) as a proxy for 'being\n"
        "  close to X' — that is what spatial_target is for. Only include such features\n"
        "  when the query also asks for accessibility / vibrancy / activity themselves.\n"
        "  Only pick features whose meaning DIRECTLY maps to a concept in the query;\n"
        "  when unsure whether a feature is relevant, leave it out.\n\n"
        "STEP 4 — Operator and normalization.\n"
        "  - geometric_mean only when features are non-negative AND query has AND semantics.\n"
        "  - owa_top for OR / 'any of these'; owa_bottom for 'worst-feature-driven'.\n"
        "  - weighted_sum is the default linear combination.\n"
        "  - Use 'robust' normalization unless the query asks for percentile rankings.\n"
    )


def build_planner_prompt(
    user_query: str,
    registry: dict,
    features_info: List[dict],
    existing_indices: List[dict],
) -> Tuple[str, str]:
    system = (
        "You are an urban-network analyst that turns natural-language questions "
        "into a Composite Street Index. You must reason about MULTI-SCALE space "
        "syntax integration (local vs district vs city-wide) and pick the right "
        "features, signed weights, aggregation operator, and per-feature "
        "normalization. Return only the structured plan."
    )
    user = build_planner_context(user_query, registry, features_info, existing_indices) + (
        "\nReturn the IndexPlan."
    )
    return system, user


def plan_index(
    *,
    user_query: str,
    settings: dict,
    registry_path: str,
    registry: dict,
    features_info: List[dict],
    existing_indices: List[dict],
) -> IndexPlan:
    system, user = build_planner_prompt(
        user_query, registry, features_info, existing_indices
    )
    return chat_structured(
        settings=settings,
        registry_path=registry_path,
        system=system,
        prompt=user,
        schema_model=IndexPlan,
    )