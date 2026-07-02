"""Feature column storage in Parquet files (split from network GPKG)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import geopandas as gpd
import pandas as pd

from streetrag.core.feature_catalog import FeatureCatalog
from streetrag.core.network_gpkg import EDGE_ID_COL

FEATURES_DIR = "features"
SYNTAX_SOURCE_KEY = "syntax"
INDICES_SOURCE_KEY = "indices"
STORAGE_LAYOUT_SPLIT = "split"

NETWORK_TOPOLOGY_COLS = frozenset(
    {
        EDGE_ID_COL,
        "u",
        "v",
        "length",
        "mm_len",
        "osmid",
        "oneway",
        "name",
        "highway",
        "ref",
        "lanes",
        "maxspeed",
        "junction",
        "bridge",
        "tunnel",
        "access",
        "width",
        "reversed",
    }
)


def is_topology_column(col: str) -> bool:
    return col in NETWORK_TOPOLOGY_COLS


def features_dir(catalog: FeatureCatalog) -> Path:
    d = catalog.data_dir / FEATURES_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def source_key_from_filename(filename: str) -> str:
    return Path(filename).stem


def parquet_path(catalog: FeatureCatalog, source_key: str) -> Path:
    return features_dir(catalog) / f"{source_key}.parquet"


class FeatureStore:
    """Read/write per-source feature Parquet files keyed by ``edge_id``."""

    def __init__(self, catalog: FeatureCatalog):
        self.catalog = catalog

    def uses_split_storage(self) -> bool:
        if self.catalog.raw.get("storage_layout") == STORAGE_LAYOUT_SPLIT:
            return True
        d = self.catalog.data_dir / FEATURES_DIR
        return d.is_dir() and any(d.glob("*.parquet"))

    def has_feature_columns(self, edges: gpd.GeoDataFrame) -> bool:
        return any(
            c not in ("geometry",) and not is_topology_column(c)
            for c in edges.columns
        )

    def list_parquet_files(self) -> List[Path]:
        d = features_dir(self.catalog)
        if not d.is_dir():
            return []
        return sorted(d.glob("*.parquet"))

    def list_feature_column_names(self) -> Set[str]:
        cols: Set[str] = set()
        for p in self.list_parquet_files():
            try:
                import pyarrow.parquet as pq

                names = pq.read_schema(p).names
            except Exception:
                names = list(pd.read_parquet(p, columns=[EDGE_ID_COL]).columns)
            cols.update(c for c in names if c != EDGE_ID_COL)
        return cols

    def column_source(self, col: str) -> str:
        mapping = self.catalog.raw.get("feature_column_sources") or {}
        if col in mapping:
            return mapping[col]
        if col.startswith(("integration_R", "angular_", "choice_", "nain_", "nach_")):
            return SYNTAX_SOURCE_KEY
        if col in self.catalog.composite_index_columns:
            return INDICES_SOURCE_KEY
        for block in self.catalog.point_integrations:
            cols = block.get("columns") or {}
            if col in cols:
                return source_key_from_filename(block.get("source_file", "misc"))
            for base, _desc in cols.items():
                if col == base or col.startswith(base + "_"):
                    return source_key_from_filename(block.get("source_file", "misc"))
        stats = self.catalog.feature_statistics.get(col) or {}
        for derived in (stats.get("normalization_columns") or {}).values():
            if derived:
                return self.column_source(derived)
        return "misc"

    def register_column_sources(self, columns: Iterable[str]) -> None:
        mapping = self.catalog.raw.setdefault("feature_column_sources", {})
        for col in columns:
            if col == EDGE_ID_COL or is_topology_column(col):
                continue
            mapping[col] = self.column_source(col)

    def join_features(
        self,
        edges: gpd.GeoDataFrame,
        *,
        columns: Optional[Iterable[str]] = None,
    ) -> gpd.GeoDataFrame:
        """Left-join feature columns onto topology *edges*."""
        if EDGE_ID_COL not in edges.columns:
            from streetrag.core.network_gpkg import ensure_edge_ids

            edges = ensure_edge_ids(edges)

        wanted = set(columns) if columns is not None else None
        out = edges
        for p in self.list_parquet_files():
            try:
                import pyarrow.parquet as pq

                file_cols = set(pq.read_schema(p).names) - {EDGE_ID_COL}
            except Exception:
                file_cols = set(pd.read_parquet(p).columns) - {EDGE_ID_COL}
            if wanted is not None:
                load_cols = sorted(file_cols & wanted)
                if not load_cols:
                    continue
                feat = pd.read_parquet(p, columns=[EDGE_ID_COL] + load_cols)
            else:
                feat = pd.read_parquet(p)
            if EDGE_ID_COL not in feat.columns:
                continue
            feat_cols = [c for c in feat.columns if c != EDGE_ID_COL]
            if not feat_cols:
                continue
            out = out.merge(feat, on=EDGE_ID_COL, how="left", suffixes=("", "_dup"))
            dup_cols = [c for c in out.columns if c.endswith("_dup")]
            if dup_cols:
                out = out.drop(columns=dup_cols)
        return out

    def split_edges(self, edges: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, Dict[str, pd.DataFrame]]:
        """Split a wide edges table into topology GDF and per-source feature frames."""
        from streetrag.core.network_gpkg import ensure_edge_ids

        edges = ensure_edge_ids(edges)
        topo_cols = ["geometry"] + [
            c for c in edges.columns if c != "geometry" and is_topology_column(c)
        ]
        topo = edges[topo_cols].copy()
        feature_cols = [c for c in edges.columns if c not in topo_cols]
        groups: Dict[str, List[str]] = {}
        for col in feature_cols:
            key = self.column_source(col)
            groups.setdefault(key, []).append(col)
        frames: Dict[str, pd.DataFrame] = {}
        for key, cols in groups.items():
            frames[key] = edges[[EDGE_ID_COL] + cols].copy()
        return topo, frames

    def write_source(self, source_key: str, frame: pd.DataFrame) -> None:
        """Write or merge columns for one source Parquet file."""
        if EDGE_ID_COL not in frame.columns:
            raise ValueError(f"{EDGE_ID_COL} required in feature frame")
        path = parquet_path(self.catalog, source_key)
        feat_cols = [c for c in frame.columns if c != EDGE_ID_COL]
        if not feat_cols:
            return
        new = frame[[EDGE_ID_COL] + feat_cols].copy()
        if path.exists():
            existing = pd.read_parquet(path)
            drop = [c for c in feat_cols if c in existing.columns]
            if drop:
                existing = existing.drop(columns=drop)
            merged = existing.merge(new, on=EDGE_ID_COL, how="outer")
        else:
            merged = new
        merged = merged[[EDGE_ID_COL] + sorted(c for c in merged.columns if c != EDGE_ID_COL)]
        merged.to_parquet(path, index=False)
        self.register_column_sources(feat_cols)

    def write_from_edges(self, edges: gpd.GeoDataFrame) -> None:
        """Persist all feature columns from a wide edges table."""
        _topo, groups = self.split_edges(edges)
        for key, frame in groups.items():
            self.write_source(key, frame)

    def drop_columns(self, columns: Iterable[str]) -> List[str]:
        """Remove columns from their source Parquet files."""
        removed: List[str] = []
        cols = [c for c in columns if c != EDGE_ID_COL and not is_topology_column(c)]
        if not cols:
            return removed
        by_source: Dict[str, List[str]] = {}
        for col in cols:
            by_source.setdefault(self.column_source(col), []).append(col)
        mapping = self.catalog.raw.get("feature_column_sources") or {}
        for source_key, drop_cols in by_source.items():
            path = parquet_path(self.catalog, source_key)
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            present = [c for c in drop_cols if c in df.columns]
            if not present:
                continue
            df = df.drop(columns=present)
            feat_cols = [c for c in df.columns if c != EDGE_ID_COL]
            if feat_cols:
                df.to_parquet(path, index=False)
            else:
                path.unlink(missing_ok=True)
            removed.extend(present)
            for c in present:
                mapping.pop(c, None)
        self.catalog.raw["feature_column_sources"] = mapping
        return removed

    def storage_mtime(self) -> float:
        mtimes = [p.stat().st_mtime for p in self.list_parquet_files()]
        return max(mtimes) if mtimes else 0.0


def migrate_wide_gpkg_to_split(catalog: FeatureCatalog, *, verbose: bool = True) -> dict:
    """One-shot migration: wide network GPKG -> topology GPKG + features/*.parquet."""
    from streetrag.core.io import atomic_write_gpkg
    from streetrag.core.network_gpkg import read_network_gpkg, ensure_edge_ids
    from streetrag.core.street_network import StreetNetwork

    gpkg = catalog.target_network_path
    if not gpkg.exists():
        raise FileNotFoundError(f"Network not found: {gpkg}")

    edges, nodes = read_network_gpkg(gpkg, edges_layer=catalog.target_layer)
    edges = ensure_edge_ids(edges)
    store = FeatureStore(catalog)
    topo, groups = store.split_edges(edges)

    for key, frame in groups.items():
        store.write_source(key, frame)
        if verbose:
            print(f"  wrote features/{key}.parquet ({len(frame.columns) - 1} cols)")

    layer = catalog.target_layer or "network"
    atomic_write_gpkg(gpkg, edges=topo, nodes=None, edges_layer=layer)
    catalog.raw["storage_layout"] = STORAGE_LAYOUT_SPLIT
    store.register_column_sources(
        c for g in groups.values() for c in g.columns if c != EDGE_ID_COL
    )
    catalog.save()
    StreetNetwork._CACHE.clear()
    if verbose:
        print(
            f"Migrated {gpkg.name}: topology {len(topo.columns) - 1} attrs, "
            f"{len(groups)} feature file(s)"
        )
    return {
        "topology_columns": len(topo.columns) - 1,
        "feature_files": len(groups),
        "n_edges": len(topo),
    }
