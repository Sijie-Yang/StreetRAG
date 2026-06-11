"""StreetNetwork: central data structure for edges, nodes, and catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import pandas as pd

from streetrag.core.feature_catalog import FeatureCatalog
from streetrag.core.io import atomic_write_gpkg, read_network_gpkg
from streetrag.core.spatial_utils import ensure_projected


class StreetNetwork:
    """Edges + nodes GeoDataFrames with feature catalog and caching."""

    _CACHE: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        edges: gpd.GeoDataFrame,
        nodes: gpd.GeoDataFrame,
        catalog: FeatureCatalog,
        *,
        gpkg_path: Optional[Path] = None,
    ):
        self.edges = edges
        self.nodes = nodes
        self.catalog = catalog
        self.gpkg_path = Path(gpkg_path) if gpkg_path else catalog.target_network_path
        self._ensure_lengths()

    def _ensure_lengths(self) -> None:
        if "length" not in self.edges.columns:
            self.edges = self.edges.copy()
            self.edges["length"] = self.edges.geometry.length
        if "mm_len" not in self.edges.columns:
            self.edges["mm_len"] = self.edges["length"]

    @classmethod
    def from_catalog(
        cls,
        catalog: FeatureCatalog,
        *,
        preferred_epsg: Optional[int] = None,
        use_cache: bool = True,
    ) -> "StreetNetwork":
        gpkg = catalog.target_network_path
        if not gpkg.exists():
            raise FileNotFoundError(f"Target network not found: {gpkg}")

        key = str(gpkg.resolve())
        mtime = gpkg.stat().st_mtime
        if use_cache:
            cached = cls._CACHE.get(key)
            if cached and cached["mtime"] == mtime:
                return cached["network"]

        edges, nodes = read_network_gpkg(
            gpkg,
            edges_layer=catalog.target_layer,
        )
        settings = catalog.load_settings()
        epsg = preferred_epsg or settings.get("preferred_crs_epsg")
        edges, nodes = ensure_projected(
            edges, nodes,
            preferred_epsg=int(epsg) if epsg else None,
        )
        net = cls(edges, nodes, catalog, gpkg_path=gpkg)
        cls._CACHE[key] = {"mtime": mtime, "network": net}
        return net

    @classmethod
    def from_gpkg(
        cls,
        gpkg_path: str | Path,
        catalog: Optional[FeatureCatalog] = None,
        *,
        preferred_epsg: Optional[int] = None,
    ) -> "StreetNetwork":
        gpkg_path = Path(gpkg_path)
        if catalog is None:
            catalog = FeatureCatalog.from_data_dir(gpkg_path.parent)
        edges, nodes = read_network_gpkg(gpkg_path)
        edges, nodes = ensure_projected(edges, nodes, preferred_epsg=preferred_epsg)
        return cls(edges, nodes, catalog, gpkg_path=gpkg_path)

    def invalidate_cache(self) -> None:
        if self.gpkg_path:
            StreetNetwork._CACHE.pop(str(self.gpkg_path.resolve()), None)

    def save(self, path: Optional[str | Path] = None) -> Path:
        out = Path(path) if path else self.gpkg_path
        atomic_write_gpkg(
            out,
            edges=self.edges,
            nodes=self.nodes,
            edges_layer=self.catalog.target_layer,
        )
        self.gpkg_path = out
        self.invalidate_cache()
        return out

    def add_column(self, name: str, values: pd.Series) -> None:
        self.edges[name] = values

    def column_names(self) -> list[str]:
        return [c for c in self.edges.columns if c != "geometry"]

    def registry_dict(self) -> dict:
        return self.catalog.to_legacy_registry()

    def compute_normalizations(
        self,
        col_name: str,
        *,
        methods: Optional[list[str]] = None,
    ) -> dict:
        """Compute normalization columns and update catalog statistics."""
        if col_name not in self.edges.columns:
            return {}
        values = self.edges[col_name]
        valid = values[values.notna()]
        if len(valid) == 0:
            return {}

        suffix = self.catalog.percentile_suffix
        methods = methods or self.catalog.raw.get(
            "normalization_methods", ["percentile", "zscore", "minmax", "robust"]
        )
        col_min = float(valid.min())
        col_max = float(valid.max())
        col_mean = float(valid.mean())
        col_std = float(valid.std())
        col_median = float(valid.median())
        col_q25 = float(valid.quantile(0.25))
        col_q75 = float(valid.quantile(0.75))
        col_iqr = col_q75 - col_q25
        norm_cols: dict = {}

        if "percentile" in methods:
            pcol = f"{col_name}{suffix}"
            self.edges[pcol] = values.rank(pct=True, method="average") * 100
            norm_cols["percentile"] = pcol
        if "zscore" in methods:
            zcol = f"{col_name}_zscore"
            self.edges[zcol] = (values - col_mean) / col_std if col_std > 0 else 0.0
            norm_cols["zscore"] = zcol
        if "minmax" in methods:
            mcol = f"{col_name}_minmax"
            self.edges[mcol] = (values - col_min) / (col_max - col_min) if col_max > col_min else 0.0
            norm_cols["minmax"] = mcol
        if "robust" in methods:
            rcol = f"{col_name}_robust"
            self.edges[rcol] = (values - col_median) / col_iqr if col_iqr > 0 else 0.0
            norm_cols["robust"] = rcol

        stats = {
            "min": col_min,
            "max": col_max,
            "mean": col_mean,
            "std": col_std,
            "median": col_median,
            "q25": col_q25,
            "q75": col_q75,
            "iqr": col_iqr,
            "normalization_columns": norm_cols,
        }
        if "percentile" in norm_cols:
            stats["percentile_column"] = norm_cols["percentile"]
        self.catalog.register_feature_stats(col_name, stats)
        return stats

    def to_wgs84_edges(self) -> gpd.GeoDataFrame:
        if self.edges.crs and not self.edges.crs.is_geographic:
            return self.edges.to_crs("EPSG:4326")
        return self.edges
