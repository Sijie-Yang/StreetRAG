"""Tests for city bootstrap."""

from streetrag.core.bootstrap import ensure_default_city
from streetrag.core.workspace import Workspace


def test_ensure_default_city_activates_existing(tmp_path):
    ws = Workspace(tmp_path / "data")
    city_dir = ws.create_city("singapore")
    gpkg = city_dir / "Singapore_drive.gpkg"
    # minimal fake: bootstrap only checks registry; create empty registry target
    from streetrag.core.feature_catalog import FeatureCatalog
    cat = FeatureCatalog(city_dir / "feature_registry.json")
    cat._data = {"target_network": "missing.gpkg"}
    cat.save()
    ws.set_active("singapore")
    name = ensure_default_city(ws)
    assert name == "singapore"
