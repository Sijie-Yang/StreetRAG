"""Tests for settings store."""

from streetrag.core.settings_store import mask_api_key


def test_mask_api_key():
    assert mask_api_key("sk-abcdefghijklmnop") == "sk-abcd…mnop"
    assert mask_api_key("") == ""
    assert mask_api_key("short") == "sho…"
