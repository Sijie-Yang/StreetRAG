"""Data scan and integration pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import fiona
import geopandas as gpd
import pandas as pd

from streetrag.core.feature_catalog import FeatureCatalog
from streetrag.core.street_network import StreetNetwork
from streetrag.ingest.integrators import get_integrator
from streetrag.ingest.readers import csv_to_gdf, detect_geometry_type, list_layers, read_geodata
from streetrag.syntax.engine import compute_syntax


def scan_data_dir(data_dir: str | Path, catalog: Optional[FeatureCatalog] = None) -> FeatureCatalog:
    """Scan a city data directory (root + sources/), convert CSVs,
    identify the network, update the catalog."""
    data_dir = Path(data_dir)
    if catalog is None:
        catalog = FeatureCatalog(data_dir / "feature_registry.json")
    sources_dir = data_dir / "sources"
    scan_dirs = [data_dir] + ([sources_dir] if sources_dir.is_dir() else [])

    for d in scan_dirs:
        for csv_path in d.glob("*.csv"):
            gpkg_out = csv_path.with_suffix(".gpkg")
            if gpkg_out.exists():
                continue
            try:
                gdf = csv_to_gdf(csv_path)
                gdf.to_file(gpkg_out, driver="GPKG")
                print(f"Converted {csv_path.name} → {gpkg_out.name}")
            except ValueError as e:
                print(f"Skip CSV {csv_path.name}: {e}")

    gpkg_files = [p for d in scan_dirs for p in sorted(d.glob("*.gpkg"))]
    target = None
    for gpkg in gpkg_files:
        layers = list_layers(gpkg)
        if "nodes" in layers and "edges" in layers:
            target = gpkg
            break

    if target is None:
        raise FileNotFoundError("No GPKG with nodes/edges layers found")

    catalog._data["target_network"] = target.name
    catalog._data.setdefault("target_layer", "edges")
    catalog._data.setdefault("percentile_column_suffix", "_pctl")
    catalog._data.setdefault("default_normalization", "percentile")
    catalog._data.setdefault("normalization_methods", ["percentile", "zscore", "minmax", "robust"])
    catalog.set_syntax_radii(catalog.syntax_radii or FeatureCatalog.DEFAULT_RADII)

    edges = gpd.read_file(target, layer="edges")
    exclude = {"geometry", "fid", "id"}
    tn_feats = {}
    for col in edges.columns:
        if col.lower() not in exclude:
            tn_feats[col] = catalog.get_description(col) or f"Network column: {col}"
    catalog._data["target_network_features"] = tn_feats

    point_integrations: List[dict] = []
    for gpkg in gpkg_files:
        if gpkg == target:
            continue
        layers = list_layers(gpkg) or [None]
        for layer in layers:
            try:
                gdf = read_geodata(gpkg, layer=layer)
            except Exception:
                continue
            gtype = detect_geometry_type(gdf)
            if gtype == "unknown":
                continue
            numeric_cols = {
                c: f"{gtype} feature from {gpkg.name}"
                for c in gdf.columns
                if c != "geometry" and pd.api.types.is_numeric_dtype(gdf[c])
            }
            if not numeric_cols:
                continue
            if gtype == "point":
                if "L1_types" in gdf.columns:
                    method = {
                        "type": "poi_category_density_rating",
                        "category_column": "L1_types",
                        "rating_column": "rating" if "rating" in gdf.columns else None,
                        "radius": 500,
                    }
                else:
                    method = {"type": "snap_nearest", "k": 5}
            elif gtype in ("linestring", "multilinestring"):
                method = {"type": "line_overlay", "buffer_m": 20}
            elif gtype in ("polygon", "multipolygon"):
                method = {"type": "polygon_area_weighted", "buffer_m": 30}
            else:
                continue
            point_integrations.append({
                "source_file": gpkg.name,
                "source_layer": layer,
                "geometry_type": gtype,
                "integration_method": method,
                "columns": numeric_cols,
            })

    catalog._data["point_integrations"] = point_integrations
    catalog.save()
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
) -> StreetNetwork:
    """Integrate one external dataset onto street edges."""
    source_path = Path(source_path)
    if not source_path.is_absolute():
        source_path = net.catalog.data_dir / source_path
    source = read_geodata(source_path, layer=layer)
    if source.crs != net.edges.crs:
        source = source.to_crs(net.edges.crs)
    integrator = get_integrator(method_type)
    net.edges = integrator.integrate(
        net.edges, source, columns, **(method_params or {})
    )
    for col in columns:
        if col in net.edges.columns:
            net.compute_normalizations(col)
    return net


def run_integration(catalog: FeatureCatalog, *, compute_syntax_metrics: bool = True) -> StreetNetwork:
    """Full integration: syntax + all point/line/polygon sources."""
    net = StreetNetwork.from_catalog(catalog)
    if compute_syntax_metrics:
        net = compute_syntax(net, radii=catalog.syntax_radii)

    for block in catalog.point_integrations:
        source = catalog.resolve_path(block["source_file"])
        method = block.get("integration_method", {})
        method_type = method.get("type", "snap_nearest")
        net = integrate_source(
            net,
            source,
            method_type=method_type,
            columns=block.get("columns", {}),
            layer=block.get("source_layer"),
            method_params={k: v for k, v in method.items() if k != "type"},
        )

    net.save()
    catalog.save()
    return net
