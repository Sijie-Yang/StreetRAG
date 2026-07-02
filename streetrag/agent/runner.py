"""LLM agent: skill selection via function calling."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

from pydantic import ValidationError as PydanticValidationError

from streetrag.core.feature_catalog import FeatureCatalog
from streetrag.core.street_network import StreetNetwork
from streetrag.llm.client import chat_text, chat_tools, load_rag_settings, resolve_api_key
from streetrag.llm.language import language_lock_instruction
from streetrag.llm.retrieval import (
    IndexPlan,
    build_planner_context,
    list_features_info,
    plan_index,
    rank_features_for_query,
)
from streetrag.skills.base import SkillResult
from streetrag.skills.composite_index import CompositeIndexParams, CompositeIndexSkill
from streetrag.skills.registry import (
    get_skill,
    list_skills,
    register_all_skills,
    run_skill,
    skills_to_openai_tools,
)


def _ensure_skills_registered() -> None:
    register_all_skills()


def _build_agent_system(user_query: str) -> str:
    # Routing guide is assembled from skill manifests so new skills are
    # picked up without editing this prompt.
    lines = [
        "You are StreetRAG, an urban street analysis agent.",
        language_lock_instruction(user_query),
        "Choose exactly ONE skill (tool) that best answers the user's question,",
        "then fill its parameters carefully following the supplied rules.",
        "Available skills:",
    ]
    for m in list_skills():
        lines.append(f"- {m['name']}: {m['description']}")
    lines.append(
        "Default to composite_index when the user asks WHERE something is "
        "true on the map (walkability, comfort, vibrancy, proximity)."
    )
    lines.append(
        "Use poi_review_search when the user asks WHY or what people say in reviews/comments."
    )
    lines.append(
        "Use answer_directly for meta questions about data, features, or usage "
        "when no analysis skill applies."
    )
    return "\n".join(lines)


def run_agent_query(
    catalog: FeatureCatalog,
    user_query: str,
    *,
    on_progress: Optional[Callable[[str, dict], None]] = None,
    use_function_calling: bool = True,
) -> Dict[str, Any]:
    """Run a natural-language query through the agent pipeline."""
    _ensure_skills_registered()
    settings = catalog.load_settings()
    registry_path = str(catalog.path)
    registry = catalog.to_legacy_registry()

    def emit(step: str, **kwargs) -> None:
        if on_progress:
            try:
                on_progress(step, kwargs)
            except Exception:
                pass

    if not resolve_api_key(settings):
        raise SystemExit(
            "OpenAI API key not configured. Set OPENAI_API_KEY or RAG_setting.local.json."
        )

    emit("load_settings", phase="done")
    net = StreetNetwork.from_catalog(catalog)
    emit("load_gpkg", phase="done", n_edges=len(net.edges))

    features_info_all = list_features_info(catalog)
    eff = {
        "feature_retrieval_top_m": int(settings.get("feature_retrieval_top_m", 60)),
        "feature_retrieval_method": settings.get("feature_retrieval_method", "embedding"),
    }
    features_info = rank_features_for_query(
        user_query,
        features_info_all,
        eff["feature_retrieval_top_m"],
        registry_path=registry_path,
        settings=settings,
        method=eff["feature_retrieval_method"],
    )
    emit("retrieve_features", n=len(features_info))

    existing = catalog.list_indices()

    if use_function_calling:
        emit("agent", phase="tool_select")
        tools = skills_to_openai_tools()
        base_prompt = build_planner_context(
            user_query, registry, features_info, existing
        ) + "\nSelect the best skill and fill its parameters."

        # Validate-and-repair loop: if the tool call fails Pydantic
        # validation (including cross-field consistency rules), feed the
        # error back to the LLM and let it correct itself.
        max_repair = 3
        prompt = base_prompt
        skill_name, params_dict, last_error = None, None, None
        for attempt in range(max_repair):
            tool_result = chat_tools(
                settings=settings,
                registry_path=registry_path,
                system=_build_agent_system(user_query),
                prompt=prompt,
                tools=tools,
            )
            skill_name = tool_result["tool_name"]
            params_dict = dict(tool_result["arguments"])
            params_dict.setdefault("user_query", user_query)
            try:
                get_skill(skill_name).params_model(**params_dict)
                last_error = None
                break
            except (PydanticValidationError, ValueError, KeyError) as exc:
                last_error = exc
                emit("agent", phase="repair", attempt=attempt + 1, error=str(exc)[:300])
                prompt = (
                    base_prompt
                    + "\n\nYour previous tool call was:\n"
                    + json.dumps(tool_result, ensure_ascii=False)
                    + "\n\nIt FAILED validation with this error:\n"
                    + str(exc)
                    + "\n\nFix the parameters and call the tool again."
                )
        if last_error is not None:
            emit("agent", phase="fallback", error=str(last_error)[:300])
            fallback_reply = (
                "I could not run a street analysis for that question with the "
                "available skills and data. "
                f"Details: {last_error}. "
                "Try uploading POI/review data, running streetrag scan && integrate, "
                "or ask about a specific map metric or review theme."
            )
            from streetrag.skills.answer_directly import AnswerDirectlyParams, AnswerDirectlySkill

            result = AnswerDirectlySkill().run(
                net,
                AnswerDirectlyParams(reply=fallback_reply, user_query=user_query),
            )
        else:
            emit("agent", phase="run_skill", skill=skill_name)
            result = run_skill(skill_name, net, params_dict)
    else:
        emit("plan", phase="llm_start")
        plan = plan_index(
            user_query=user_query,
            settings=settings,
            registry_path=registry_path,
            registry=registry,
            features_info=features_info,
            existing_indices=existing,
        )
        emit("plan", phase="llm_done", intent=plan.intent)
        params = CompositeIndexParams(
            intent=plan.intent,
            target_index_col=plan.target_index_col,
            index_name=plan.index_name,
            features=plan.features,
            operator=plan.operator,
            normalization=plan.normalization,
            spatial_target=plan.spatial_target,
            spatial_filter_radius_m=plan.spatial_filter_radius_m,
            proximity_dominant=plan.proximity_dominant,
            explanation=plan.explanation,
            user_query=user_query,
        )
        result = CompositeIndexSkill().run(net, params)

    emit("complete", index_col=result.index_col, skill=result.skill_name)
    return {
        "kind": "new_index" if result.render_map else "analysis",
        "render_map": result.render_map,
        "index_col": result.index_col,
        "index_name": result.index_name,
        "reply": result.reply,
        "statistics": result.stats,
        "morans_i": (result.narrative_evidence or {}).get("morans_i"),
        "length_weighted_topk": (result.narrative_evidence or {}).get("length_weighted_topk"),
        "narrative_evidence": result.narrative_evidence,
        "skill_name": result.skill_name,
        "explanation": result.explanation,
        "operator": result.operator,
        "normalization": result.normalization,
        "features_weights": result.features_weights,
        "spatial_target": result.spatial_target,
        "spatial_target_resolved": result.spatial_target_resolved,
        "spatial_filter_radius_m": result.spatial_filter_radius_m,
        "gdf_edges": net.edges,
    }
