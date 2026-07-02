"""Infer which language the user wrote in; lock LLM output to that language."""

from __future__ import annotations

import re


def infer_response_language(text: str) -> str:
    """Conservative language tag for LLM replies. Defaults to English."""
    q = (text or "").strip()
    if not q:
        return "English"
    if re.search(r"[\u4e00-\u9fff]", q):
        return "Chinese"
    if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", q):
        return "Japanese"
    if re.search(r"[\uac00-\ud7a3]", q):
        return "Korean"
    if re.search(r"[\u0400-\u04ff]", q):
        return "Russian"

    lower = q.lower()
    if re.search(r"[áéíóúñ¿¡]", lower) or re.search(
        r"\b(el|la|los|las|dónde|cómo|índice|habitabilidad|por favor)\b", lower
    ):
        return "Spanish"
    if re.search(r"[àâçéèêëïîôùûüœ]", lower) or re.search(
        r"\b(où|des|les|une|pour|comment)\b", lower
    ):
        return "French"
    if re.search(r"[äöüß]", lower) or re.search(r"\b(und|der|die|das|wo)\b", lower):
        return "German"

    return "English"


def language_lock_instruction(user_query: str) -> str:
    """Prompt block: respond only in the language detected from the user message."""
    lang = infer_response_language(user_query)
    return (
        f"LANGUAGE LOCK: The user's message is in {lang}. "
        f"Write ALL user-visible text strictly in {lang} — assistant replies, "
        f"proposal titles, rationales, and explanations. "
        f"Never switch to another language (e.g. do not use Spanish if the user wrote in English). "
        f"If the message language is unclear, use English."
    )
