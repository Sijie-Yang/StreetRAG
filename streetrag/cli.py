"""StreetRAG CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _root_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def _workspace():
    from streetrag.core.workspace import Workspace

    ws = Workspace(_root_data_dir())
    if ws.has_legacy_layout():
        ws.migrate_legacy()
    return ws


def _city_dir(args: argparse.Namespace) -> Path:
    """Resolve the working city dir: --city flag > active city."""
    ws = _workspace()
    name = getattr(args, "city_name", None)
    return ws.city_dir(name)


def _registry_path(args: argparse.Namespace) -> Path:
    explicit = getattr(args, "registry", None)
    if explicit:
        return Path(explicit)
    return _city_dir(args) / "feature_registry.json"


def cmd_city(args: argparse.Namespace) -> None:
    ws = _workspace()
    if args.action == "list":
        active = ws.active_city()
        cities = ws.list_cities()
        if not cities:
            print("No cities yet. Create one: streetrag city new <name>")
        for c in cities:
            mark = "*" if c["name"] == active else " "
            print(f" {mark} {c['name']:<20s} network={c['network'] or '—':<30s} "
                  f"features={c['n_features']} indices={c['n_indices']}")
    elif args.action == "new":
        d = ws.create_city(args.name)
        print(f"Created city dir: {d}")
        print(f"Next: streetrag download --city '{args.name}' --city-name {d.name}")
    elif args.action == "use":
        ws.set_active(args.name)
        print(f"Active city: {args.name}")


def cmd_download(args: argparse.Namespace) -> None:
    from streetrag.core.workspace import slugify_city
    from streetrag.ingest.download import download_network, download_pois

    ws = _workspace()
    name = args.city_name or slugify_city(args.city.split(",")[0])
    dest = ws.create_city(name)
    download_network(args.city, network_type=args.network_type, data_dir=dest)
    if getattr(args, "with_pois", False):
        download_pois(args.city, data_dir=dest)
        print("POIs saved under sources/. Next: streetrag scan && streetrag integrate")
    ws.set_active(name)
    print(f"Active city: {name}. Next: streetrag scan && streetrag integrate")


def cmd_clean(args: argparse.Namespace) -> None:
    from streetrag.core.feature_catalog import FeatureCatalog

    catalog = FeatureCatalog(_registry_path(args))
    indices = catalog.list_indices()
    if args.list or not (args.index_col or args.all):
        if not indices:
            print("No saved indices.")
        for it in indices:
            print(f"  {it.get('index_col')}  ←  {it.get('original_query', '')!r}")
        return
    targets = [args.index_col] if args.index_col else [it.get("index_col") for it in indices]
    for col in targets:
        if not col:
            continue
        p = catalog.index_path(col)
        if p.exists():
            p.unlink()
            print(f"Deleted index record: {col}")
        catalog.feature_statistics.pop(col, None)
        if col in catalog.composite_index_columns:
            catalog.composite_index_columns.remove(col)
        catalog.raw.get("target_network_features", {}).pop(col, None)
    catalog.save()
    print("Registry updated. (GPKG columns retained; re-run integrate to rebuild stats.)")


def cmd_scan(args: argparse.Namespace) -> None:
    from streetrag.core.feature_catalog import FeatureCatalog
    from streetrag.ingest.pipeline import scan_data_dir

    data_dir = Path(args.data_dir) if args.data_dir else _city_dir(args)
    catalog = FeatureCatalog(data_dir / "feature_registry.json")
    scan_data_dir(data_dir, catalog)


def cmd_integrate(args: argparse.Namespace) -> None:
    from streetrag.core.feature_catalog import FeatureCatalog
    from streetrag.ingest.pipeline import run_integration

    catalog = FeatureCatalog(_registry_path(args))
    run_integration(
        catalog,
        compute_syntax_metrics=not args.no_syntax,
        index_reviews=not getattr(args, "no_reviews", False),
        verbose=True,
    )


def cmd_syntax(args: argparse.Namespace) -> None:
    from streetrag.core.feature_catalog import FeatureCatalog
    from streetrag.core.street_network import StreetNetwork
    from streetrag.syntax.engine import compute_syntax

    catalog = FeatureCatalog(_registry_path(args))
    net = StreetNetwork.from_catalog(catalog)
    radii = [int(r) for r in args.radii.split(",")] if args.radii else None
    net = compute_syntax(net, radii=radii)
    net.save()
    catalog.save()
    print(f"Syntax computed for radii: {catalog.syntax_radii}")


def cmd_ask(args: argparse.Namespace) -> None:
    from streetrag.agent.runner import run_agent_query
    from streetrag.core.feature_catalog import FeatureCatalog

    catalog = FeatureCatalog(_registry_path(args))
    result = run_agent_query(
        catalog,
        args.query,
        use_function_calling=not args.legacy_plan,
    )
    print(f"\nIndex: {result.get('index_col')}")
    print(f"Reply:\n{result.get('reply', '')}")


def cmd_migrate_storage(args: argparse.Namespace) -> None:
    from streetrag.core.feature_catalog import FeatureCatalog
    from streetrag.core.feature_store import migrate_wide_gpkg_to_split

    catalog = FeatureCatalog(_registry_path(args))
    summary = migrate_wide_gpkg_to_split(catalog, verbose=True)
    print(f"Done: {summary}")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    _workspace()  # trigger legacy migration before the app imports
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    uvicorn.run(
        "webapp.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def _add_city_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--city-name", dest="city_name", default=None,
                   help="City directory name (default: active city)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="streetrag", description="StreetRAG urban analysis framework")
    sub = parser.add_subparsers(dest="command", required=True)

    p_city = sub.add_parser("city", help="Manage cities (list/new/use)")
    p_city.add_argument("action", choices=["list", "new", "use"])
    p_city.add_argument("name", nargs="?")
    p_city.set_defaults(func=cmd_city)

    p_dl = sub.add_parser("download", help="Download a city street network via OSMnx")
    p_dl.add_argument("--city", required=True, help="OSM place query, e.g. 'London, UK'")
    p_dl.add_argument("--network-type", default="drive",
                      choices=["drive", "walk", "bike", "all", "all_private"])
    p_dl.add_argument(
        "--with-pois",
        action="store_true",
        help="Also download OSM POIs (amenity/shop/tourism) into sources/",
    )
    _add_city_arg(p_dl)
    p_dl.set_defaults(func=cmd_download)

    p_clean = sub.add_parser("clean", help="List or delete saved indices")
    p_clean.add_argument("registry", nargs="?", default=None)
    p_clean.add_argument("--index-col")
    p_clean.add_argument("--all", action="store_true")
    p_clean.add_argument("--list", action="store_true")
    _add_city_arg(p_clean)
    p_clean.set_defaults(func=cmd_clean)

    p_scan = sub.add_parser("scan", help="Scan city data directory and update registry")
    p_scan.add_argument("--data-dir", default=None)
    _add_city_arg(p_scan)
    p_scan.set_defaults(func=cmd_scan)

    p_int = sub.add_parser("integrate", help="Integrate features onto street network")
    p_int.add_argument("registry", nargs="?", default=None)
    p_int.add_argument("--no-syntax", action="store_true")
    p_int.add_argument("--no-reviews", action="store_true", help="Skip review text indexing")
    _add_city_arg(p_int)
    p_int.set_defaults(func=cmd_integrate)

    p_syn = sub.add_parser("syntax", help="Compute space syntax metrics")
    p_syn.add_argument("registry", nargs="?", default=None)
    p_syn.add_argument("--radii", help="Comma-separated radii in meters, e.g. 500,1500,4500")
    _add_city_arg(p_syn)
    p_syn.set_defaults(func=cmd_syntax)

    p_mig = sub.add_parser(
        "migrate-storage",
        help="Split wide network GPKG into topology GPKG + features/*.parquet",
    )
    p_mig.add_argument("registry", nargs="?", default=None)
    _add_city_arg(p_mig)
    p_mig.set_defaults(func=cmd_migrate_storage)

    p_ask = sub.add_parser("ask", help="Ask a natural-language question")
    p_ask.add_argument("query", help="Natural language query")
    p_ask.add_argument("--registry", default=None)
    p_ask.add_argument("--legacy-plan", action="store_true", help="Use IndexPlan instead of function-calling agent")
    _add_city_arg(p_ask)
    p_ask.set_defaults(func=cmd_ask)

    p_srv = sub.add_parser("serve", help="Launch web UI")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8765)
    p_srv.add_argument("--reload", action="store_true")
    p_srv.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
