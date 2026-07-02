"""Direct answer skill for meta / unsupported questions."""

from __future__ import annotations

from pydantic import BaseModel, Field

from streetrag.skills.base import Skill, SkillResult, skill


class AnswerDirectlyParams(BaseModel):
    reply: str = Field(..., description="Helpful plain-text answer for the user")
    user_query: str = ""


@skill
class AnswerDirectlySkill(Skill):
    name = "answer_directly"
    description = (
        "Answer general questions about StreetRAG, available data/features, "
        "integration status, or questions that cannot be answered by map analysis skills."
    )
    params_model = AnswerDirectlyParams

    def run(self, net, params: AnswerDirectlyParams) -> SkillResult:
        return SkillResult(
            skill_name=self.name,
            reply=params.reply,
            render_map=False,
        )
