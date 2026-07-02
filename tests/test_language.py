"""Tests for response language inference."""

from streetrag.llm.language import infer_response_language, language_lock_instruction


def test_english_by_default():
    assert infer_response_language("Which streets have the worst thermal comfort?") == "English"
    assert infer_response_language("create urban liveability index") == "English"


def test_chinese():
    assert infer_response_language("哪里热舒适最差？") == "Chinese"


def test_spanish_requires_markers():
    assert infer_response_language("¿Dónde está la mejor habitabilidad?") == "Spanish"
    assert infer_response_language("urban vibrancy near NUS") == "English"


def test_language_lock_mentions_detected_language():
    text = language_lock_instruction("thermal comfort index")
    assert "English" in text
    assert "LANGUAGE LOCK" in text
