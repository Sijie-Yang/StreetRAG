"""Skill registry and OpenAI tool conversion."""

from __future__ import annotations

from typing import Dict, List, Type

from pydantic import BaseModel

from streetrag.core.street_network import StreetNetwork
from streetrag.skills.base import Skill, SkillResult, _SKILL_REGISTRY


def get_skill(name: str) -> Type[Skill]:
    if name not in _SKILL_REGISTRY:
        raise KeyError(f"Unknown skill: {name}. Available: {list(_SKILL_REGISTRY)}")
    return _SKILL_REGISTRY[name]


def list_skills() -> List[dict]:
    return [cls.manifest() for cls in _SKILL_REGISTRY.values()]


def skills_to_openai_tools() -> List[dict]:
    tools = []
    for cls in _SKILL_REGISTRY.values():
        schema = cls.params_model.model_json_schema()
        # Keep the full schema (especially $defs): nested models like
        # FeatureWeight are referenced via $ref and the LLM cannot fill
        # them correctly if the definitions are stripped.
        parameters = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }
        if "$defs" in schema:
            parameters["$defs"] = schema["$defs"]
        tools.append({
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": parameters,
            },
        })
    return tools


def run_skill(
    name: str,
    net: StreetNetwork,
    params_dict: dict,
) -> SkillResult:
    cls = get_skill(name)
    params = cls.params_model(**params_dict)
    instance = cls()
    return instance.run(net, params)


def register_all_skills() -> None:
    """Auto-import every module in streetrag.skills so @skill decorators run.

    Adding a new skill = dropping a new .py file in this package; no manual
    import list to maintain.
    """
    import importlib
    import pkgutil

    import streetrag.skills as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name in ("base", "registry", "stats"):
            continue
        importlib.import_module(f"streetrag.skills.{mod.name}")
