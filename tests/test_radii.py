"""Tests for radii parsing."""

import pytest

from streetrag.core.feature_catalog import FeatureCatalog
from streetrag.utils.radii import format_radii, parse_radii


def test_parse_radii_string():
    assert parse_radii("500,1500,4500") == [500, 1500, 4500]


def test_parse_radii_list():
    assert parse_radii([800, 2400]) == [800, 2400]


def test_parse_radii_default():
    assert parse_radii(None) == FeatureCatalog.DEFAULT_RADII


def test_parse_radii_invalid():
    with pytest.raises(ValueError):
        parse_radii("abc")


def test_format_radii():
    assert format_radii([500, 1500]) == "500,1500"
