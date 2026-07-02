"""Data scan and integration pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd

from streetrag.core.feature_catalog import FeatureCatalog
from streetrag.core.network_gpkg import (
    NETWORK_LAYER,
    detect_network_format,
    is_street_network_gpkg,
    read_network_gpkg,
)
from streetrag.core.street_network import StreetNetwork
from streetrag.ingest.columns import (
    detect_numeric_columns,
    detect_text_columns,
    infer_category_column,
    infer_rating_column,
    pick_integration_method,
    plan_poi_category_columns,
    summarize_columns,
)
from streetrag.ingest.integrators import get_integrator
from streetrag.ingest.readers import csv_to_gdf, detect_geometry_type, list_layers, read_geodata
from streetrag.syntax.engine import compute_syntax

SCAN_EXTENSIONS = {".gpkg", ".geojson", ".json", ".shp", ".csv", ".parquet"}
CONVERT_TO_GPKG = {".csv", ".geojson", ".json", ".parquet"}


def _scan_dirs(data_dir: Path) -> List[Path]:
    sources_dir = data_dir / "sources"
    dirs = [data_dir]
    if sources_dir.is_dir():
        dirs.append(sources_dir)
    return dirs


def _convert_sidecar_to_gpkg(path: Path) -> Optional[Path]:
    """Convert CSV/GeoJSON/Parquet to sibling GPKG when missing."""
    gpkg_out = path.with_suffix(".gpkg")
    if gpkg_out.exists():
        return gpkg_out
    try:
        gdf = read_geodata(path)
        if gdf.empty:
            print(f"Skip convert {path.name}: empty dataset")
            return None
        gdf.to_file(gpkg_out, driver="GPKG")
        print(f"Converted {path.name} → {gpkg_out.name}")
        return gpkg_out
    except Exception as exc:
        print(f"Skip convert {path.name}: {exc}")
        return None


def _collect_data_files(data_dir: Path) -> List[Path]:
    files: List[Path] = []
    seen = set()
    for d in _scan_dirs(data_dir):
        for path in sorted(d.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            suffix = path.suffix.lower()
            if suffix not in SCAN_EXTENSIONS:
                continue
            if suffix in CONVERT_TO_GPKG:
                converted = _convert_sidecar_to_gpkg(path)
                if converted is not None:
                    key = str(converted.resolve())
                    if key not in seen:
                        seen.add(key)
                        files.append(converted)
                continue
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                files.append(path)
    return files


def _build_integration_columns(
    gdf: gpd.GeoDataFrame,
    gtype: str,
    method_type: str,
    method_params: dict,
    source_name: str,
) -> Dict[str, str]:
    numeric_cols = {
        c: f"{gtype} feature from {source_name}"
        for c in detect_numeric_columns(gdf)
    }
    if method_type == "poi_category_density_rating":
        cat = method_params.get("category_column") or infer_category_column(gdf)
        if cat:
            numeric_cols.update(
                plan_poi_category_columns(
                    gdf,
                    category_col=cat,
                    radius=float(method_params.get("radius", 500)),
                    rating_col=method_params.get("rating_column") or infer_rating_column(gdf),
                )
            )
    return numeric_cols


def build_integration_block(
    source_file: str,
    gdf: gpd.GeoDataFrame,
    *,
    layer: Optional[str] = None,
) -> Optional[dict]:
    gtype = detect_geometry_type(gdf)
    if gtype == "unknown":
        return None
    method_type, method_params = pick_integration_method(gtype, gdf)
    columns = _build_integration_columns(gdf, gtype, method_type, method_params, source_file)
    text_cols = detect_text_columns(gdf)
    if not columns and not text_cols:
        return None
    return {
        "source_file": source_file,
        "source_layer": layer,
        "geometry_type": gtype,
        "integration_method": {"type": method_type, **method_params},
        "columns": columns,
        "text_columns": text_cols,
    }


def register_integration_block(catalog: FeatureCatalog, block: dict) -> None:
    catalog.upsert_point_integration(block)
    text_cols = block.get("text_columns") or []
    if text_cols:
        catalog.upsert_text_integration(
            {
                "source_file": block.get("source_file"),
                "source_layer": block.get("source_layer"),
                "text_columns": text_cols,
                "n_chunks": block.get("n_review_chunks", 0),
            }
        )


def scan_data_dir(
    data_dir: str | Path,
    catalog: Optional[FeatureCatalog] = None,
    *,
    verbose: bool = True,
) -> FeatureCatalog:
    """Scan a city data directory, convert sidecar formats, update registry."""
    data_dir = Path(data_dir)
    if catalog is None:
        catalog = FeatureCatalog(data_dir / "feature_registry.json")

    gpkg_files = _collect_data_files(data_dir)
    target = None
    target_layer = NETWORK_LAYER
    for gpkg in gpkg_files:
        if is_street_network_gpkg(gpkg):
            target = gpkg
            info = detect_network_format(gpkg)
            target_layer = (
                info["edges_layer"] if info["format"] == "nodes_edges" else info["layer"]
            )
            break

    if target is None:
        raise FileNotFoundError(
            f"No street network GPKG found (layer {NETWORK_LAYER!r} or nodes+edges)"
        )

    catalog._data["target_network"] = target.name
    catalog._data["target_layer"] = target_layer
    catalog._data.setdefault("percentile_column_suffix", "_pctl")
    catalog._data.setdefault("default_normalization", "percentile")
    catalog._data.setdefault("normalization_methods", ["percentile", "zscore", "minmax", "robust"])
    catalog.set_syntax_radii(catalog.syntax_radii or FeatureCatalog.DEFAULT_RADII)

    edges, _nodes = read_network_gpkg(target, edges_layer=target_layer)
    exclude = {"geometry", "fid", "id"}
    tn_feats = {}
    for col in edges.columns:
        if col.lower() not in exclude:
            tn_feats[col] = catalog.get_description(col) or f"Network column: {col}"
    catalog._data["target_network_features"] = tn_feats

    point_integrations: List[dict] = []
    for path in gpkg_files:
        if path == target:
            continue
        layers = list_layers(path) or [None]
        for layer in layers:
            try:
                gdf = read_geodata(path, layer=layer)
            except Exception as exc:
                if verbose:
                    print(f"Skip {path.name} layer={layer}: {exc}")
                continue
            block = build_integration_block(path.name, gdf, layer=layer)
            if block is None:
                continue
            point_integrations.append(block)
            if verbose:
                cols = block.get("columns") or {}
                text_cols = block.get("text_columns") or []
                print(
                    f"  {path.name}"
                    + (f" [{layer}]" if layer else "")
                    + f" → {block['integration_method']['type']}"
                    + f" ({len(cols)} numeric"
                    + (f", {len(text_cols)} text" if text_cols else "")
                    + ")"
                )
                if cols:
                    col_names = list(cols.keys())
                    preview = ", ".join(col_names[:6])
                    if len(col_names) > 6:
                        preview += ", …"
                    print(f"      columns: {preview}")

    catalog._data["point_integrations"] = point_integrations
    catalog.save()
    if verbose:
        print(f"Scanned {data_dir}: target={target.name}, integrations={len(point_integrations)}")
    return catalog


def integrate_source(
    net: StreetNetwork,
    source_path: str | Path,
    *,
    method_type: str,
    columns: Dict[str, str],
    layer: Optional[str] = None,
    method_params: Optional[dict] = None,
    on_progress: Optional[Callable[[str, dict], None]] = None,
) -> Tuple[StreetNetwork, Dict[str, str]]:
    """Integrate one external dataset onto street edges."""
    source_path = Path(source_path)
    if not source_path.exists():
        # Resolve basename only — avoid doubling paths like data/cities/x/sources/foo.gpkg
        source_path = net.catalog.resolve_path(source_path.name)
    source = read_geodata(source_path, layer=layer)
    if source.crs != net.edges.crs:
        source = source.to_crs(net.edges.crs)

    before_cols = set(net.edges.columns)
    integrator = get_integrator(method_type)
    params = dict(method_params or {})
    if on_progress:
        params["_on_progress"] = on_progress
    net.edges = integrator.integrate(net.edges, source, dict(columns), **params)

    new_cols = [c for c in net.edges.columns if c not in before_cols]
    written: Dict[str, str] = {}
    total_norm = len(new_cols)
    for i, col in enumerate(new_cols):
        if on_progress:
            on_progress(
                "integrate",
                {
                    "phase": "normalize",
                    "pct": 75 + int(20 * (i + 1) / max(total_norm, 1)),
                    "current": i + 1,
                    "total": total_norm,
                    "message": f"Normalizing column {i + 1}/{total_norm}",
                },
            )
        desc = columns.get(col, f"Integrated from {source_path.name}")
        written[col] = desc
        net.compute_normalizations(col)
        if desc:
            net.catalog.set_description(col, desc)
    return net, written


def run_integration(
    catalog: FeatureCatalog,
    *,
    compute_syntax_metrics: bool = True,
    index_reviews: bool = True,
    verbose: bool = True,
) -> Tuple[StreetNetwork, List[dict]]:
    """Full integration: syntax + all point/line/polygon sources."""
    net = StreetNetwork.from_catalog(catalog)
    results: List[dict] = []

    if compute_syntax_metrics:
        if verbose:
            print("Computing space syntax metrics…")
        net = compute_syntax(net, radii=catalog.syntax_radii)

    for block in catalog.point_integrations:
        source_name = block.get("source_file", "?")
        layer = block.get("source_layer")
        method = block.get("integration_method", {})
        method_type = method.get("type", "snap_nearest")
        source = catalog.resolve_path(source_name)
        try:
            net, written = integrate_source(
                net,
                source,
                method_type=method_type,
                columns=dict(block.get("columns") or {}),
                layer=layer,
                method_params={k: v for k, v in method.items() if k != "type"},
            )
            block["columns"] = {**dict(block.get("columns") or {}), **written}
            review_info = {}
            if index_reviews and block.get("text_columns"):
                review_info = _maybe_index_reviews(
                    catalog,
                    net,
                    source,
                    layer=layer,
                    text_columns=block.get("text_columns"),
                )
            results.append(
                {
                    "source_file": source_name,
                    "layer": layer,
                    "method": method_type,
                    "columns_added": list(written.keys()),
                    "ok": True,
                    **review_info,
                }
            )
            if verbose:
                print(
                    f"Integrated {source_name}"
                    + (f" [{layer}]" if layer else "")
                    + f": {len(written)} columns"
                    + (
                        f", {review_info.get('n_review_chunks', 0)} review chunks"
                        if review_info.get("n_review_chunks")
                        else ""
                    )
                )
                if written:
                    print(f"  → {', '.join(written.keys())}")
        except Exception as exc:
            results.append(
                {
                    "source_file": source_name,
                    "layer": layer,
                    "method": method_type,
                    "ok": False,
                    "error": str(exc),
                }
            )
            if verbose:
                print(f"FAILED {source_name}: {exc}")

    net.save()
    catalog.save()
    if verbose:
        ok = sum(1 for r in results if r.get("ok"))
        print(f"Integration complete: {ok}/{len(results)} sources OK")
    return net, results


def _maybe_index_reviews(
    catalog: FeatureCatalog,
    net: StreetNetwork,
    source_path: Path,
    *,
    layer: Optional[str],
    text_columns: Optional[List[str]] = None,
) -> dict:
    from streetrag.reviews.indexer import index_reviews_from_source

    gdf = read_geodata(source_path, layer=layer)
    cols = text_columns or detect_text_columns(gdf)
    if not cols:
        return {}
    summary = index_reviews_from_source(
        catalog,
        net,
        source_path,
        text_columns=cols,
        layer=layer,
        create_edge_aggregates=True,
    )
    catalog.upsert_text_integration(
        {
            "source_file": source_path.name,
            "source_layer": layer,
            "text_columns": cols,
            "n_chunks": summary.get("n_chunks", 0),
        }
    )
    return summary


def integrate_uploaded_file(
    catalog: FeatureCatalog,
    *,
    filename: str,
    method_type: str,
    layer: Optional[str] = None,
    columns: Optional[Dict[str, str]] = None,
    method_params: Optional[dict] = None,
    index_reviews: bool = True,
    on_progress: Optional[Callable[[str, dict], None]] = None,
) -> dict:
    """Integrate one uploaded file, update registry, optionally index reviews."""
    def prog(phase: str, pct: int, message: str, **extra) -> None:
        if on_progress:
            on_progress("integrate", {"phase": phase, "pct": pct, "message": message, **extra})

    prog("load", 2, f"Loading {filename}…")
    source = catalog.resolve_path(filename)
    if not source.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    gdf = read_geodata(source, layer=layer)
    gtype = detect_geometry_type(gdf)
    params = dict(method_params or {})
    if method_type == "poi_category_density_rating" and "category_column" not in params:
        cat = infer_category_column(gdf)
        if cat:
            params["category_column"] = cat
            params.setdefault("rating_column", infer_rating_column(gdf))
            params.setdefault("radius", 500)

    numeric_cols = columns or {
        c: f"Integrated from {filename}"
        for c in detect_numeric_columns(gdf)
    }
    if method_type == "poi_category_density_rating":
        numeric_cols = {
            **numeric_cols,
            **_build_integration_columns(gdf, gtype, method_type, params, filename),
        }

    prog("network", 5, "Loading street network…")
    net = StreetNetwork.from_catalog(catalog)
    prog("integrate", 8, f"Integrating onto {len(net.edges):,} edges…")
    net, written = integrate_source(
        net,
        source,
        method_type=method_type,
        columns=numeric_cols,
        layer=layer,
        method_params=params,
        on_progress=on_progress,
    )
    text_cols = detect_text_columns(gdf)
    block = {
        "source_file": filename,
        "source_layer": layer,
        "geometry_type": gtype,
        "integration_method": {"type": method_type, **params},
        "columns": {**numeric_cols, **written},
        "text_columns": text_cols,
    }
    register_integration_block(catalog, block)

    review_info: dict = {}
    if index_reviews and text_cols:
        prog("reviews", 96, "Indexing review text…")
        review_info = _maybe_index_reviews(
            catalog,
            net,
            source,
            layer=layer,
            text_columns=text_cols,
        )

    prog("save", 98, "Saving features…")
    net.save()
    catalog.save()
    StreetNetwork.clear_cache()
    prog("done", 100, f"Done — {len(written)} columns added")
    return {
        "columns_added": list(written.keys()),
        "method": method_type,
        "text_columns": text_cols,
        **review_info,
    }
