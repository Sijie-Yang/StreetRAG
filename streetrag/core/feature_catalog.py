"""Feature catalog: registry metadata split from LLM descriptions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class FeatureCatalog:
    """Manages feature_registry.json and feature descriptions."""

    DEFAULT_RADII = [500, 1500, 4500]

    def __init__(self, registry_path: str | Path):
        self.path = Path(registry_path)
        self.data_dir = self.path.parent
        self._data: dict = {}
        self.reload()

    @classmethod
    def from_data_dir(cls, data_dir: str | Path) -> "FeatureCatalog":
        return cls(Path(data_dir) / "feature_registry.json")

    def reload(self) -> None:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    @property
    def raw(self) -> dict:
        return self._data

    def resolve_path(self, rel: str) -> Path:
        p = Path(rel)
        if p.is_absolute():
            return p
        direct = self.data_dir / p
        if direct.exists():
            return direct
        # external source files live in sources/ in the multi-city layout
        in_sources = self.data_dir / "sources" / p
        if in_sources.exists():
            return in_sources
        return direct

    def sources_dir(self) -> Path:
        d = self.data_dir / "sources"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def global_dir(self) -> Optional[Path]:
        """Workspace root (data/) when this catalog lives in data/cities/<name>/."""
        if self.data_dir.parent.name == "cities":
            return self.data_dir.parent.parent
        return None

    @property
    def target_network_path(self) -> Path:
        return self.resolve_path(self._data.get("target_network", ""))

    @property
    def target_layer(self) -> str:
        return self._data.get("target_layer", "edges")

    @property
    def percentile_suffix(self) -> str:
        return self._data.get("percentile_column_suffix", "_pctl")

    @property
    def syntax_radii(self) -> List[int]:
        cfg = self._data.get("space_syntax_integration") or {}
        return [int(r) for r in cfg.get("radii", self.DEFAULT_RADII)]

    def set_syntax_radii(self, radii: List[int]) -> None:
        self._data.setdefault("space_syntax_integration", {})["radii"] = radii

    @property
    def feature_statistics(self) -> dict:
        return self._data.setdefault("feature_statistics", {})

    @property
    def composite_index_columns(self) -> List[str]:
        return self._data.setdefault("composite_index_columns", [])

    @property
    def point_integrations(self) -> List[dict]:
        return self._data.setdefault("point_integrations", [])

    def descriptions_path(self) -> Path:
        return self.data_dir / "feature_descriptions.json"

    def load_descriptions(self) -> dict:
        p = self.descriptions_path()
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_descriptions(self, descriptions: dict) -> None:
        with open(self.descriptions_path(), "w", encoding="utf-8") as f:
            json.dump(descriptions, f, indent=2, ensure_ascii=False)

    def get_description(self, feature_name: str) -> str:
        descs = self.load_descriptions()
        if feature_name in descs:
            return descs[feature_name]
        tn = self._data.get("target_network_features") or {}
        if feature_name in tn:
            return tn[feature_name]
        for block in self.point_integrations:
            cols = block.get("columns") or {}
            if feature_name in cols:
                return cols[feature_name]
        if feature_name.startswith("integration_R"):
            cfg = self._data.get("space_syntax_integration") or {}
            return cfg.get("description", "Space syntax integration value")
        if feature_name.startswith("angular_"):
            return "Angular space syntax measure"
        return ""

    def set_description(self, feature_name: str, description: str) -> None:
        descs = self.load_descriptions()
        descs[feature_name] = description
        self.save_descriptions(descs)

    def infer_source(self, feature_name: str) -> str:
        if feature_name.startswith(("integration_R", "angular_", "choice_", "nain_", "nach_")):
            return "space_syntax"
        if feature_name in self.composite_index_columns:
            return "composite_index"
        tn = self._data.get("target_network_features") or {}
        if feature_name in tn:
            return "target_network"
        for block in self.point_integrations:
            if feature_name in (block.get("columns") or {}):
                return "point_integration"
        return "derived"

    def list_features_info(self) -> List[dict]:
        out: List[dict] = []
        for feature_name, stats in self.feature_statistics.items():
            desc = self.get_description(feature_name)
            try:
                rlo = "?" if stats.get("min") is None else f"{float(stats['min']):.6e}"
                rhi = "?" if stats.get("max") is None else f"{float(stats['max']):.6e}"
            except (KeyError, TypeError, ValueError):
                rlo, rhi = "?", "?"
            info: dict = {
                "name": feature_name,
                "description": desc,
                "source": self.infer_source(feature_name),
                "range": f"[{rlo}, {rhi}]",
            }
            if "normalization_columns" in stats:
                info["normalization_columns"] = stats["normalization_columns"]
            else:
                pcol = stats.get("percentile_column", f"{feature_name}{self.percentile_suffix}")
                info["normalization_columns"] = {"percentile": pcol}
            out.append(info)
        return out

    def register_feature_stats(
        self,
        col_name: str,
        statistics: dict,
        *,
        description: str = "",
        composite: bool = False,
    ) -> None:
        q25 = statistics.get("q25")
        q75 = statistics.get("q75")
        iqr = 0.0 if (q25 is None or q75 is None) else float(q75) - float(q25)
        self.feature_statistics[col_name] = {
            "min": statistics.get("min"),
            "max": statistics.get("max"),
            "mean": statistics.get("mean"),
            "std": statistics.get("std"),
            "median": statistics.get("median"),
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
            "normalization_columns": statistics.get("normalization_columns", {}),
        }
        if description:
            self.set_description(col_name, description[:2000])
        if composite and col_name not in self.composite_index_columns:
            self.composite_index_columns.append(col_name)
        self._data.setdefault("target_network_features", {})[col_name] = (
            description or f"Composite index: {col_name}"
        )

    def add_point_integration(self, config: dict) -> None:
        self.point_integrations.append(config)

    def rag_settings_path(self) -> Path:
        local = self.data_dir / "RAG_setting.local.json"
        if local.exists():
            return local
        return self.data_dir / "RAG_setting.json"

    def load_settings(self) -> dict:
        """Merge settings: global workspace first, then per-city overrides."""
        settings: dict = {}
        dirs = []
        g = self.global_dir()
        if g is not None:
            dirs.append(g)
        dirs.append(self.data_dir)
        for d in dirs:
            for fname in ("RAG_setting.json", "RAG_setting.local.json"):
                p = d / fname
                if p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        settings.update(json.load(f))
        return settings

    def indices_dir(self) -> Path:
        d = self.data_dir / "indices"
        d.mkdir(exist_ok=True)
        return d

    def index_path(self, col: str) -> Path:
        return self.indices_dir() / f"{col}.json"

    def save_index_record(self, record: dict) -> None:
        col = record.get("index_col")
        if not col:
            raise ValueError("index_col required")
        with open(self.index_path(col), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    def load_index_record(self, col: str) -> Optional[dict]:
        p = self.index_path(col)
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_indices(self) -> List[dict]:
        out = []
        for p in self.indices_dir().glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    out.append(json.load(f))
            except Exception:
                pass
        return out

    def to_legacy_registry(self) -> dict:
        """Return registry dict compatible with legacy scripts."""
        return self._data
