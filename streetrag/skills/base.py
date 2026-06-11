"""Skill base class and decorator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from streetrag.core.street_network import StreetNetwork


class SkillResult(BaseModel):
    skill_name: str
    columns_written: List[str] = []
    stats: Dict[str, Any] = {}
    narrative_evidence: Dict[str, Any] = {}
    reply: str = ""
    index_col: Optional[str] = None
    index_name: Optional[str] = None
    # Set when the result should be rendered as a colored map layer.
    render_map: bool = False
    spatial_target: Optional[str] = None
    spatial_target_resolved: Optional[Dict[str, Any]] = None
    spatial_filter_radius_m: Optional[float] = None
    explanation: str = ""
    operator: Optional[str] = None
    normalization: Optional[str] = None
    features_weights: Dict[str, float] = {}


class Skill(ABC):
    name: str = "base"
    description: str = ""
    params_model: Type[BaseModel] = BaseModel

    @classmethod
    def manifest(cls) -> dict:
        schema = cls.params_model.model_json_schema()
        return {
            "name": cls.name,
            "description": cls.description,
            "parameters": schema,
        }

    @abstractmethod
    def run(self, net: StreetNetwork, params: BaseModel) -> SkillResult:
        ...


_SKILL_REGISTRY: Dict[str, Type[Skill]] = {}


def skill(cls: Type[Skill]) -> Type[Skill]:
    _SKILL_REGISTRY[cls.name] = cls
    return cls
