"""Bootstrap default city and download new cities from the web UI."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

from streetrag.core.feature_catalog import FeatureCatalog
from streetrag.core.network_gpkg import is_street_network_gpkg
from streetrag.core.workspace import Workspace, slugify_city
from streetrag.utils.radii import RadiiInput, parse_radii

DEFAULT_CITY_SLUG = "singapore"
DEFAULT_OSM_QUERY = "Singapore"

ProgressFn = Optional[Callable[[str, dict], None]]


def _emit(cb: ProgressFn, step: str, **detail) -> None:
    if cb:
        try:
            cb(step, detail)
        except Exception:
            pass


def setup_city_from_osm(
    workspace: Workspace,
    osm_query: str,
    *,
    city_slug: Optional[str] = None,
    network_type: str = "drive",
    with_pois: bool = False,
    run_syntax: bool = False,
    syntax_radii: RadiiInput = None,
    on_progress: ProgressFn = None,
) -> str:
    """Download OSM network (+ optional POIs), scan, integrate, activate. Returns slug."""
    from streetrag.ingest.download import download_network, download_pois
    from streetrag.ingest.pipeline import run_integration, scan_data_dir

    slug = city_slug or slugify_city(osm_query.split(",")[0])
    dest = workspace.create_city(slug)

    _emit(on_progress, "download", phase="start", message=f"Downloading network: {osm_query}")
    download_network(osm_query, network_type=network_type, data_dir=dest)
    _emit(on_progress, "download", phase="done", message="Network download complete")

    if with_pois:
        _emit(on_progress, "pois", phase="start", message="Downloading OSM POIs…")
        try:
            download_pois(osm_query, data_dir=dest)
            _emit(on_progress, "pois", phase="done", message="POI download complete")
        except Exception as exc:
            _emit(on_progress, "pois", phase="warn", message=f"POI skipped: {exc}")

    catalog = workspace.catalog(slug)
    _emit(on_progress, "scan", phase="start", message="Scanning data directory…")
    scan_data_dir(dest, catalog, verbose=False)
    radii = parse_radii(syntax_radii, default=FeatureCatalog.DEFAULT_RADII)
    catalog.set_syntax_radii(radii)
    catalog.save()
    _emit(on_progress, "scan", phase="done", message=f"Scan complete · syntax radii {radii} m")

    _emit(on_progress, "integrate", phase="start", message="Integrating onto street network…")
    run_integration(
        catalog,
        compute_syntax_metrics=run_syntax,
        index_reviews=True,
        verbose=False,
    )
    _emit(on_progress, "integrate", phase="done", message="Integration complete")

    workspace.set_active(slug)
    _emit(on_progress, "activate", phase="done", city=slug, message=f"Switched to {slug}")
    return slug


def ensure_default_city(workspace: Workspace, on_progress: ProgressFn = None) -> Optional[str]:
    """Ensure Singapore (or any existing city) is ready on first open."""
    cities = workspace.list_cities()
    active = workspace.active_city()

    if active:
        city_dir = workspace.city_dir(active)
        reg_path = city_dir / "feature_registry.json"
        if not reg_path.exists():
            gpkgs = list(city_dir.glob("*_drive.gpkg")) + list(city_dir.glob("*.gpkg"))
            has_network = any(is_street_network_gpkg(g) for g in gpkgs) if gpkgs else False
            if has_network:
                from streetrag.ingest.pipeline import scan_data_dir

                _emit(on_progress, "scan", phase="start", message=f"Initializing {active}…")
                scan_data_dir(city_dir, workspace.catalog(active), verbose=False)
                _emit(on_progress, "scan", phase="done")
        return active

    if cities:
        name = cities[0]["name"]
        workspace.set_active(name)
        return name

    _emit(on_progress, "bootstrap", phase="start", message="First launch — downloading Singapore sample data…")
    slug = setup_city_from_osm(
        workspace,
        DEFAULT_OSM_QUERY,
        city_slug=DEFAULT_CITY_SLUG,
        with_pois=True,
        run_syntax=False,
        syntax_radii=FeatureCatalog.DEFAULT_RADII,
        on_progress=on_progress,
    )
    _emit(on_progress, "bootstrap", phase="done", city=slug)
    return slug


def __layers(gpkg: Path) -> set:
    try:
        from streetrag.ingest.readers import list_layers
        return set(list_layers(gpkg))
    except Exception:
        return set()
