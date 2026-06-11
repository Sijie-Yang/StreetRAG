"""Skill plugin system."""

from streetrag.skills.base import Skill, SkillResult, skill
from streetrag.skills.registry import get_skill, list_skills, skills_to_openai_tools

__all__ = ["Skill", "SkillResult", "skill", "get_skill", "list_skills", "skills_to_openai_tools"]
