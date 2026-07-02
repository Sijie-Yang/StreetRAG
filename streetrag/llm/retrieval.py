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
from streetrag.llm.language import language_lock_instruction
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
        "feature_retrieval_method": settings.get("feature_retrieval_method", "stratified"),
        "feature_retrieval_small_source_max": int(
            settings.get("feature_retrieval_small_source_max", 20)
        ),
        "feature_retrieval_min_per_source": int(
            settings.get("feature_retrieval_min_per_source", 2)
        ),
        "feature_retrieval_max_share": float(
            settings.get("feature_retrieval_max_share", 0.4)
        ),
        "feature_retrieval_full_catalog_max": int(
            settings.get("feature_retrieval_full_catalog_max", 500)
        ),
        "preferred_crs_epsg": settings.get("preferred_crs_epsg"),
    }


def _is_syntax_feature(feature: dict) -> bool:
    return (feature.get("name") or "").startswith("integration_R")


def _group_by_source(features_info: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for f in features_info:
        src = f.get("source") or "unknown"
        groups.setdefault(src, []).append(f)
    return groups


def _feature_catalog_excerpt(features_info: List[dict]) -> List[dict]:
    return [
        {
            "name": f.get("name", ""),
            "source": f.get("source", ""),
            "range": f.get("range", ""),
            "description": (f.get("description") or "")[:240],
        }
        for f in features_info
    ]


def _group_features_for_planner(features_info: List[dict]) -> List[dict]:
    """Group retrieved features by source for the LLM planner menu."""
    groups = _group_by_source(features_info)
    out: List[dict] = []
    for source in sorted(groups, key=lambda s: (-len(groups[s]), s)):
        feats = groups[source]
        out.append({
            "source": source,
            "n_total": len(feats),
            "features": _feature_catalog_excerpt(feats),
        })
    return out


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


def _load_query_embedding(
    user_query: str,
    *,
    registry_path: str,
    settings: dict,
) -> Optional[np.ndarray]:
    try:
        return embed_texts(
            settings=settings, registry_path=registry_path, texts=[user_query]
        )[0]
    except Exception:
        return None


def _order_features_by_relevance(
    user_query: str,
    features_info: List[dict],
    *,
    all_features: List[dict],
    mat: Optional[np.ndarray],
    q_vec: Optional[np.ndarray],
    method: str,
) -> List[dict]:
    """Return *features_info* ordered best-first for *user_query*."""
    if not features_info:
        return []
    if method == "token" or mat is None or q_vec is None:
        return _token_rank(user_query, features_info, len(features_info))

    name_to_idx = {f.get("name", ""): i for i, f in enumerate(all_features)}
    subset: List[dict] = []
    row_indices: List[int] = []
    for f in features_info:
        idx = name_to_idx.get(f.get("name", ""))
        if idx is not None:
            subset.append(f)
            row_indices.append(idx)
    if not subset:
        return list(features_info)
    sub_mat = mat[np.asarray(row_indices, dtype=np.intp)]
    local_order = cosine_topk(q_vec, sub_mat, k=len(subset))
    return [subset[i] for i in local_order]


def _flat_embedding_rank(
    user_query: str,
    features_info: List[dict],
    top_m: int,
    *,
    registry_path: str,
    settings: dict,
) -> List[dict]:
    mat = _load_embedding_index(registry_path, features_info)
    if mat is None:
        mat = _build_embedding_index(registry_path, settings, features_info)
    if mat is None or not mat.size:
        return _token_rank(user_query, features_info, top_m)
    q_vec = _load_query_embedding(user_query, registry_path=registry_path, settings=settings)
    if q_vec is None:
        return _token_rank(user_query, features_info, top_m)
    order = cosine_topk(q_vec, mat, k=len(features_info))
    ranked = [features_info[i] for i in order]
    integ = [f for f in ranked if _is_syntax_feature(f)]
    rest = [f for f in ranked if not _is_syntax_feature(f)]
    budget = max(0, top_m - len(integ))
    return (integ + rest[:budget])[:top_m]


def _stratified_rank(
    user_query: str,
    features_info: List[dict],
    top_m: int,
    *,
    registry_path: Optional[str],
    settings: Optional[dict],
    rank_method: str = "embedding",
) -> List[dict]:
    """Source-balanced retrieval: small families in full, large families capped."""
    eff = effective_street_rag_settings(settings or {})
    t_small = eff["feature_retrieval_small_source_max"]
    k_min = max(1, eff["feature_retrieval_min_per_source"])
    max_share = max(0.1, min(1.0, eff["feature_retrieval_max_share"]))
    per_source_cap = max(k_min, int(top_m * max_share))

    groups = _group_by_source(features_info)
    pinned = [f for f in features_info if _is_syntax_feature(f)]
    pinned_names = {f.get("name", "") for f in pinned}

    mat: Optional[np.ndarray] = None
    q_vec: Optional[np.ndarray] = None
    if rank_method == "embedding" and registry_path and settings:
        mat = _load_embedding_index(registry_path, features_info)
        if mat is None:
            mat = _build_embedding_index(registry_path, settings, features_info)
        if mat is not None and mat.size:
            q_vec = _load_query_embedding(user_query, registry_path=registry_path, settings=settings)

    mandatory: List[dict] = list(pinned)
    mandatory_names = set(pinned_names)
    large_ranked: Dict[str, List[dict]] = {}

    for source, feats in groups.items():
        feats = [f for f in feats if f.get("name", "") not in pinned_names]
        if not feats:
            continue
        if len(feats) <= t_small:
            for f in feats:
                if f.get("name", "") not in mandatory_names:
                    mandatory.append(f)
                    mandatory_names.add(f.get("name", ""))
        else:
            large_ranked[source] = _order_features_by_relevance(
                user_query,
                feats,
                all_features=features_info,
                mat=mat,
                q_vec=q_vec,
                method=rank_method if q_vec is not None else "token",
            )

    selected: List[dict] = list(mandatory)
    selected_names = set(mandatory_names)

    for source, ordered in large_ranked.items():
        n_take = min(per_source_cap, k_min, len(ordered))
        for f in ordered[:n_take]:
            name = f.get("name", "")
            if name not in selected_names:
                selected.append(f)
                selected_names.add(name)

    if len(selected) < top_m:
        filler: List[dict] = []
        for ordered in large_ranked.values():
            for f in ordered:
                name = f.get("name", "")
                if name not in selected_names:
                    filler.append(f)
        filler = _order_features_by_relevance(
            user_query,
            filler,
            all_features=features_info,
            mat=mat,
            q_vec=q_vec,
            method=rank_method if q_vec is not None else "token",
        )
        for f in filler:
            if len(selected) >= top_m:
                break
            source = f.get("source") or "unknown"
            n_from_source = sum(1 for x in selected if (x.get("source") or "unknown") == source)
            if n_from_source >= per_source_cap:
                continue
            name = f.get("name", "")
            if name not in selected_names:
                selected.append(f)
                selected_names.add(name)

    if len(selected) > top_m:
        mandatory_set = set(mandatory_names)
        trimmed = [f for f in selected if f.get("name", "") in mandatory_set]
        trimmed_names = {f.get("name", "") for f in trimmed}
        extras = [f for f in selected if f.get("name", "") not in mandatory_set]
        extras = _order_features_by_relevance(
            user_query,
            extras,
            all_features=features_info,
            mat=mat,
            q_vec=q_vec,
            method=rank_method if q_vec is not None else "token",
        )
        for f in extras:
            if len(trimmed) >= top_m:
                break
            trimmed.append(f)
            trimmed_names.add(f.get("name", ""))
        selected = trimmed

    return selected


def rank_features_for_query(
    user_query: str,
    features_info: List[dict],
    top_m: int,
    *,
    registry_path: Optional[str] = None,
    settings: Optional[dict] = None,
    method: str = "stratified",
) -> List[dict]:
    if top_m <= 0 or len(features_info) <= top_m:
        return list(features_info)
    eff = effective_street_rag_settings(settings or {})
    method = method or eff["feature_retrieval_method"]

    if method == "full":
        full_max = eff["feature_retrieval_full_catalog_max"]
        if len(features_info) <= full_max:
            return list(features_info)

    if method == "stratified":
        return _stratified_rank(
            user_query,
            features_info,
            top_m,
            registry_path=registry_path,
            settings=settings,
            rank_method="embedding",
        )

    if method == "embedding" and registry_path and settings:
        return _flat_embedding_rank(
            user_query, features_info, top_m, registry_path=registry_path, settings=settings
        )
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

    catalog_by_source = _group_features_for_planner(features_info)
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
        f"{language_lock_instruction(user_query)}\n\n"
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
        "(pre-filtered by source-balanced retrieval: entire small families, capped\n"
        "samples from large families; multi-scale syntax measures always included):\n"
        f"{json.dumps(catalog_by_source, ensure_ascii=False, indent=2)}\n\n"
        "When several sources could answer the query, prefer features whose source\n"
        "semantics DIRECTLY measure the concept; use other sources only as proxies\n"
        "when no direct measure exists in the menu.\n\n"
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