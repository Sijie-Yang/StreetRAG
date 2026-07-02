"""Agent action tools (visualize, table, proposals, feature queries)."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from streetrag.core.feature_catalog import FeatureCatalog
from streetrag.core.street_network import StreetNetwork
from streetrag.skills.registry import register_all_skills, run_skill, skills_to_openai_tools
from streetrag.skills.stats import _find_col_ci

EmitFn = Callable[[dict], None]


def action_tools_openai() -> List[dict]:
    """OpenAI tool definitions for agent action tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "visualize_feature",
                "description": "Highlight a street feature column on the map.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string", "description": "Edge column name"},
                        "colormap": {"type": "string", "description": "Optional colormap name"},
                    },
                    "required": ["column"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "show_table",
                "description": (
                    "Show a small summary table in the side panel (aggregates, correlations). "
                    "Do NOT use for per-street top-N rankings — use top_edges_table instead."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "rows": {
                            "type": "array",
                            "items": {"type": "array", "items": {}},
                        },
                    },
                    "required": ["title", "columns", "rows"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "top_edges_table",
                "description": (
                    "Query the street network and show the top-N edges by a numeric column "
                    "in the side table panel. Automatically includes edge_id and the smallest "
                    "available integration_R* column when syntax has been computed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "Numeric edge column to rank (exact catalog name)",
                        },
                        "n": {"type": "integer", "default": 20, "description": "How many rows (max 100)"},
                        "extra_columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional additional columns to include per row",
                        },
                        "include_syntax": {
                            "type": "boolean",
                            "default": True,
                            "description": "Attach smallest-radius integration_R* if present",
                        },
                        "title": {"type": "string"},
                    },
                    "required": ["column"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "propose_index_options",
                "description": (
                    "Propose 2-4 composite index options for the user to choose. "
                    "Do NOT create an index directly — wait for user confirmation. "
                    "Write each rationale in the same language as the user's question "
                    "(see LANGUAGE LOCK in the system prompt; default English if unclear)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "proposals": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "index_col": {"type": "string"},
                                    "features": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "weight": {"type": "number"},
                                            },
                                            "required": ["name", "weight"],
                                        },
                                    },
                                    "operator": {"type": "string"},
                                    "normalization": {"type": "string"},
                                    "rationale": {"type": "string"},
                                },
                                "required": ["name", "index_col", "features", "rationale"],
                            },
                        },
                    },
                    "required": ["proposals"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_features_stats",
                "description": "List available street features with optional name filter.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Substring filter"},
                        "limit": {"type": "integer", "default": 40},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_feature_detail",
                "description": "Get statistics and metadata for one feature column.",
                "parameters": {
                    "type": "object",
                    "properties": {"column": {"type": "string"}},
                    "required": ["column"],
                },
            },
        },
    ]


def all_agent_tools() -> List[dict]:
    register_all_skills()
    return action_tools_openai() + skills_to_openai_tools()


ACTION_TOOL_NAMES = {
    "visualize_feature",
    "show_table",
    "top_edges_table",
    "propose_index_options",
    "list_features_stats",
    "get_feature_detail",
}


def _resolve_column(net: StreetNetwork, name: str) -> Optional[str]:
    if not name:
        return None
    if name in net.edges.columns:
        return name
    return _find_col_ci(net.edges, name)


def _top_edges_table(
    net: StreetNetwork,
    arguments: dict,
    emit: EmitFn,
) -> str:
    raw_col = arguments.get("column", "")
    col = _resolve_column(net, raw_col)
    if not col:
        return json.dumps({"ok": False, "error": f"Column not found: {raw_col}"})

    n = max(1, min(100, int(arguments.get("n") or 20)))
    extra = arguments.get("extra_columns") or []
    include_syntax = arguments.get("include_syntax", True)

    data_cols: List[str] = [col]
    missing: List[str] = []
    for ec in extra:
        resolved = _resolve_column(net, str(ec))
        if resolved and resolved not in data_cols:
            data_cols.append(resolved)
        else:
            missing.append(str(ec))

    syntax_col: Optional[str] = None
    registry = net.registry_dict()
    radii = (registry.get("space_syntax_integration") or {}).get("radii") or []
    if include_syntax and radii:
        for r in sorted(int(x) for x in radii):
            ic = f"integration_R{r}"
            if ic in net.edges.columns:
                syntax_col = ic
                if ic not in data_cols:
                    data_cols.append(ic)
                break
        if syntax_col is None:
            missing.append("integration (run Syntax in the UI to compute)")

    sub = net.edges.copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    top = sub.nlargest(n, col)

    headers = ["rank", "edge_id", col]
    for c in data_cols:
        if c not in headers:
            headers.append(c)

    rows: List[list] = []
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        out_row: List[Any] = [rank]
        eid = row.get("edge_id")
        out_row.append(int(eid) if pd.notna(eid) else "")
        for h in headers[2:]:
            v = row.get(h)
            try:
                fv = float(v)
                out_row.append(round(fv, 4) if pd.notna(fv) else "")
            except (TypeError, ValueError):
                out_row.append("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        rows.append(out_row)

    title = arguments.get("title") or f"Top {n} edges by {col}"
    emit({"type": "table", "title": title, "columns": headers, "rows": rows})
    return json.dumps({
        "ok": True,
        "rows": len(rows),
        "column": col,
        "syntax_column": syntax_col,
        "missing_columns": missing,
    })


def execute_action_tool(
    name: str,
    arguments: dict,
    *,
    catalog: FeatureCatalog,
    net: StreetNetwork,
    emit: EmitFn,
    user_query: str = "",
) -> str:
    """Run an action tool; return JSON string for tool_result."""
    if name == "visualize_feature":
        col = arguments.get("column", "")
        if col not in net.edges.columns:
            return json.dumps({"ok": False, "error": f"Column not found: {col}"})
        emit({"type": "visualize", "column": col, "colormap": arguments.get("colormap")})
        return json.dumps({"ok": True, "column": col, "message": f"Visualizing {col} on map"})

    if name == "show_table":
        emit({
            "type": "table",
            "title": arguments.get("title", "Table"),
            "columns": arguments.get("columns", []),
            "rows": arguments.get("rows", []),
        })
        return json.dumps({"ok": True, "rows": len(arguments.get("rows", []))})

    if name == "top_edges_table":
        return _top_edges_table(net, arguments, emit)

    if name == "propose_index_options":
        proposals = arguments.get("proposals") or []
        emit({"type": "proposals", "proposals": proposals})
        return json.dumps({"ok": True, "n_proposals": len(proposals)})

    if name == "list_features_stats":
        pattern = (arguments.get("pattern") or "").lower()
        limit = int(arguments.get("limit") or 40)
        feats = catalog.list_all_features()
        if pattern:
            feats = [f for f in feats if pattern in f["name"].lower() or pattern in (f.get("description") or "").lower()]
        feats = feats[:limit]
        emit({"type": "progress", "step": "features", "detail": {"n": len(feats)}})
        return json.dumps({"ok": True, "features": feats})

    if name == "get_feature_detail":
        col = arguments.get("column", "")
        if col not in net.edges.columns:
            return json.dumps({"ok": False, "error": f"Column not found: {col}"})
        info = next((f for f in catalog.list_all_features() if f["name"] == col), None)
        if not info:
            info = {"name": col, "source": catalog.infer_source(col)}
        return json.dumps({"ok": True, "feature": info})

    # Skill execution
    register_all_skills()
    args = dict(arguments)
    args.setdefault("user_query", user_query)
    result = run_skill(name, net, args)
    payload = {
        "ok": True,
        "skill": result.skill_name,
        "reply": result.reply,
        "index_col": result.index_col,
        "render_map": result.render_map,
    }
    if result.render_map and result.index_col:
        emit({"type": "visualize", "column": result.index_col, "index_name": result.index_name})
    emit({"type": "skill_result", "data": payload})
    return json.dumps(payload, ensure_ascii=False)


def build_context_prompt(catalog: FeatureCatalog, context: Optional[List[dict]]) -> str:
    """Inject user-selected context chips into the system prompt."""
    if not context:
        return ""
    lines = ["User has added the following items to the conversation context:"]
    for item in context:
        typ = item.get("type", "")
        name = item.get("name", "")
        if typ == "feature":
            stats = catalog.feature_statistics.get(name, {})
            desc = catalog.get_description(name)
            lines.append(f"- Feature `{name}` ({catalog.infer_source(name)}): {desc}")
            if stats:
                lines.append(f"  stats: min={stats.get('min')}, max={stats.get('max')}, mean={stats.get('mean')}")
        elif typ == "index":
            rec = catalog.load_index_record(name) or {}
            lines.append(f"- Index `{name}`: {rec.get('index_name', name)}")
            fw = rec.get("features_weights") or {}
            if fw:
                lines.append(f"  weights: {fw}")
        elif typ == "file":
            lines.append(f"- Data file `{name}` (see datasets panel for integration status)")
    lines.append("Prefer discussing and visualizing these context items when relevant.")
    return "\n".join(lines)
