"""Workspace-level RAG / API settings (read + write local secrets)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from streetrag.core.feature_catalog import FeatureCatalog
from streetrag.core.workspace import Workspace
from streetrag.llm.client import resolve_api_key


def mask_api_key(key: str) -> str:
    key = (key or "").strip()
    if not key:
        return ""
    if len(key) <= 12:
        return key[:3] + "…"
    return f"{key[:7]}…{key[-4:]}"


def api_key_source(data_dir: Path) -> str:
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "env"
    local = data_dir / "RAG_setting.local.json"
    if local.exists():
        try:
            data = json.loads(local.read_text(encoding="utf-8"))
            k = (data.get("openai_api_key") or "").strip()
            if k and not k.startswith("your_"):
                return "local"
        except Exception:
            pass
    return "none"


def get_public_settings(workspace: Workspace) -> dict:
    """Settings safe to expose to the web UI (no raw API key)."""
    data_dir = workspace.root
    try:
        merged = workspace.catalog().load_settings()
    except FileNotFoundError:
        merged = {}
        base = data_dir / "RAG_setting.json"
        if base.exists():
            merged = json.loads(base.read_text(encoding="utf-8"))

    key = resolve_api_key(merged)
    source = api_key_source(data_dir)
    configured = bool(key and not key.startswith("your_"))
    return {
        "configured": configured,
        "source": source,
        "masked_key": mask_api_key(key) if configured else "",
        "llm_model": merged.get("llm_model", "gpt-4o"),
        "embedding_model": merged.get("embedding_model", "text-embedding-3-small"),
        "llm_temperature": merged.get("llm_temperature", 0.0),
        "feature_retrieval_top_m": merged.get("feature_retrieval_top_m", 60),
        "local_settings_path": str(data_dir / "RAG_setting.local.json"),
        "note_env_overrides_local": source == "env",
    }


def save_local_settings(data_dir: Path, updates: Dict[str, Any]) -> dict:
    """Merge updates into data/RAG_setting.local.json."""
    path = data_dir / "RAG_setting.local.json"
    data_dir.mkdir(parents=True, exist_ok=True)
    current: dict = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current.update(updates)
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    return current
