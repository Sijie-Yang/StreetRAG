"""Multi-turn agent loop with streaming events (Claude Code style)."""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Dict, Generator, List, Optional

from streetrag.agent.tools import (
    ACTION_TOOL_NAMES,
    all_agent_tools,
    build_context_prompt,
    execute_action_tool,
)
from streetrag.core.feature_catalog import FeatureCatalog
from streetrag.core.street_network import StreetNetwork
from streetrag.llm.client import chat_tools_stream, load_rag_settings, resolve_api_key
from streetrag.llm.language import language_lock_instruction
from streetrag.llm.retrieval import build_planner_context, list_features_info, rank_features_for_query
from streetrag.skills.registry import register_all_skills


def _build_system(catalog: FeatureCatalog, context: Optional[List[dict]], user_query: str) -> str:
    register_all_skills()
    lines = [
        "You are StreetRAG, an interactive urban street analysis agent.",
        "You can use tools to visualize features, show tables, propose index options,",
        "and run analysis skills. Think step by step.",
        "",
        language_lock_instruction(user_query),
        "",
        "Rules:",
        "- Use propose_index_options when the user might want a NEW composite index and has NOT confirmed one yet.",
        "- When you call propose_index_options, write at most ONE short sentence in your message",
        "  (e.g. 'Choose an option below.'). Do NOT list or describe the proposals in prose —",
        "  the UI shows interactive proposal cards with names and rationales.",
        "- When the user confirms a specific proposal (names index_col/features or asks to create it), call composite_index directly — do NOT propose again.",
        "- Use visualize_feature to show data on the map when discussing a column.",
        "- Use show_table for small summary tables (not per-edge rankings).",
        "- Use top_edges_table for top-N streets by any numeric column (real network query).",
        "- Do NOT ask the user to confirm before running top_edges_table — run it immediately.",
        "- Short confirmations (yes / ok / proceed / 好 / 是) continue the previous request.",
        "- Use list_features_stats / get_feature_detail to explore existing data.",
        "- For map questions (where is X best?), use composite_index or other skills.",
        "- For review/comment questions, use poi_review_search.",
    ]
    ctx = build_context_prompt(catalog, context)
    if ctx:
        lines.append("")
        lines.append(ctx)
    return "\n".join(lines)


def run_agent_loop(
    catalog: FeatureCatalog,
    user_query: str,
    *,
    context: Optional[List[dict]] = None,
    history: Optional[List[dict]] = None,
    max_turns: int = 8,
) -> Generator[dict, None, Dict[str, Any]]:
    """Yield typed SSE events; return final payload dict."""
    settings = catalog.load_settings()
    registry_path = str(catalog.path)
    if not resolve_api_key(settings):
        yield {"type": "error", "message": "OpenAI API key not configured"}
        return {"ok": False, "error": "api_key"}

    yield {"type": "progress", "step": "load_settings", "detail": {"phase": "done"}}

    net = StreetNetwork.from_catalog(catalog)
    yield {"type": "progress", "step": "load_gpkg", "detail": {"phase": "done", "n_edges": len(net.edges)}}

    features_info_all = list_features_info(catalog)
    eff = {
        "feature_retrieval_top_m": int(settings.get("feature_retrieval_top_m", 60)),
        "feature_retrieval_method": settings.get("feature_retrieval_method", "stratified"),
    }
    features_info = rank_features_for_query(
        user_query,
        features_info_all,
        eff["feature_retrieval_top_m"],
        registry_path=registry_path,
        settings=settings,
        method=eff["feature_retrieval_method"],
    )
    yield {"type": "progress", "step": "retrieve_features", "detail": {"n": len(features_info)}}

    registry = catalog.to_legacy_registry()
    existing = catalog.list_indices()
    planner_ctx = build_planner_context(user_query, registry, features_info, existing)

    system = _build_system(catalog, context, user_query)
    tools = all_agent_tools()

    messages: List[dict] = [{"role": "system", "content": system}]
    if history:
        for h in history[-12:]:
            role = h.get("role")
            content = h.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({
        "role": "user",
        "content": planner_ctx + "\n\nUser question:\n" + user_query,
    })

    final_payload: Dict[str, Any] = {"ok": True, "reply": "", "mode": "agent"}
    collected_events: List[dict] = []

    def emit(ev: dict) -> None:
        collected_events.append(ev)

    for turn in range(max_turns):
        yield {"type": "progress", "step": "agent", "detail": {"phase": "llm_start", "turn": turn + 1}}

        assistant_message: Optional[dict] = None
        for ev in chat_tools_stream(
            settings=settings,
            registry_path=registry_path,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        ):
            if ev["type"] == "done":
                assistant_message = ev["message"]
            else:
                yield ev

        if not assistant_message:
            yield {"type": "error", "message": "Empty LLM response"}
            return {"ok": False, "error": "empty_response"}

        messages.append(assistant_message)
        tool_calls = assistant_message.get("tool_calls") or []
        text = assistant_message.get("content") or ""
        if text:
            final_payload["reply"] = text

        if not tool_calls:
            break

        for tc in tool_calls:
            tc_id = tc.get("id") or str(uuid.uuid4())
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            yield {
                "type": "tool_use",
                "id": tc_id,
                "name": name,
                "arguments": args,
            }

            try:
                if name in ACTION_TOOL_NAMES or name in {t["function"]["name"] for t in tools if "function" in t}:
                    result_str = execute_action_tool(
                        name,
                        args,
                        catalog=catalog,
                        net=net,
                        emit=lambda e: collected_events.append(e),
                        user_query=user_query,
                    )
                else:
                    result_str = json.dumps({"ok": False, "error": f"Unknown tool: {name}"})
            except Exception as exc:
                result_str = json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

            yield {"type": "tool_result", "id": tc_id, "name": name, "result": result_str}

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result_str,
            })

            try:
                parsed = json.loads(result_str)
                if parsed.get("index_col") and parsed.get("render_map"):
                    final_payload.update({
                        "index_col": parsed["index_col"],
                        "render_map": True,
                        "skill_name": parsed.get("skill"),
                    })
                if parsed.get("reply"):
                    final_payload["reply"] = parsed["reply"]
            except Exception:
                pass
        else:
            continue
        # loop continues if tool_calls were processed

    # Re-emit side-effect events collected during tool execution
    for ev in collected_events:
        if ev.get("type") in ("visualize", "table", "proposals", "skill_result"):
            yield ev

    yield {"type": "progress", "step": "complete", "detail": {"phase": "done"}}
    yield {"type": "done", "data": final_payload}
    return final_payload
