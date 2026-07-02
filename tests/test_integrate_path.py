"""Integration path resolution tests."""

from pathlib import Path

from streetrag.core.feature_catalog import FeatureCatalog


def test_resolve_path_does_not_double_prefix(tmp_path: Path) -> None:
    city = tmp_path / "singapore"
    sources = city / "sources"
    sources.mkdir(parents=True)
    src = sources / "points.gpkg"
    src.write_bytes(b"")

    cat = FeatureCatalog(city / "feature_registry.json")
    once = cat.resolve_path("points.gpkg")
    assert once == src
    assert once.exists()

    # Simulates integrate_source guard: already-resolved relative paths must stay valid.
    again = Path(once)
    if not again.exists():
        again = cat.resolve_path(again.name)
    assert again == src
    assert again.exists()
