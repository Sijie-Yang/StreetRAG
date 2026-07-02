"""Multi-city workspace.

Directory layout:

    data/
    ├── RAG_setting.json        global LLM settings (shared by all cities)
    ├── RAG_setting.local.json  global secrets (gitignored)
    ├── active_city.json        {"active": "singapore"}
    └── cities/
        └── singapore/
            ├── Singapore_drive.gpkg     street network (nodes+edges)
            ├── feature_registry.json    per-city catalog
            ├── sources/                 external data files (gpkg/csv/geojson/…)
            ├── indices/  llm_cache/  llm_logs/  syntax_cache/
            └── embeddings.npz …

One city = one self-contained data directory; the workspace only manages
which city is active and migrates the legacy single-city layout.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import List, Optional

from streetrag.core.feature_catalog import FeatureCatalog

_GLOBAL_FILES = {"RAG_setting.json", "RAG_setting.local.json", "active_city.json", ".gitkeep"}
_SOURCE_EXTS = {".gpkg", ".geojson", ".shp", ".csv", ".parquet", ".shx", ".dbf", ".prj", ".cpg"}


def slugify_city(name: str) -> str:
    s = re.sub(r"\s+", "_", name.strip().lower())
    s = re.sub(r"[^a-z0-9._-]+", "", s)
    return s or "city"


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    @property
    def cities_dir(self) -> Path:
        return self.root / "cities"

    @property
    def _active_file(self) -> Path:
        return self.root / "active_city.json"

    # -- city listing / activation -----------------------------------------

    def list_cities(self) -> List[dict]:
        out = []
        if not self.cities_dir.exists():
            return out
        for d in sorted(self.cities_dir.iterdir()):
            if not d.is_dir():
                continue
            reg = d / "feature_registry.json"
            info = {"name": d.name, "has_registry": reg.exists(), "network": None,
                    "n_features": 0, "n_indices": 0}
            if reg.exists():
                try:
                    cat = FeatureCatalog(reg)
                    info["network"] = cat.raw.get("target_network")
                    info["n_features"] = len(cat.feature_statistics)
                    info["syntax_radii"] = cat.syntax_radii
                    info["n_syntax"] = sum(
                        1 for c in cat.feature_statistics
                        if c.startswith(("integration_R", "angular_", "nain_", "choice_", "nach_"))
                    )
                    info["n_indices"] = len(list((d / "indices").glob("*.json"))) if (d / "indices").exists() else 0
                except Exception:
                    pass
            else:
                # network downloaded but not scanned yet
                gpkgs = list(d.glob("*.gpkg"))
                if gpkgs:
                    info["network"] = gpkgs[0].name
            out.append(info)
        return out

    def active_city(self) -> Optional[str]:
        if self._active_file.exists():
            try:
                name = json.loads(self._active_file.read_text()).get("active")
                if name and (self.cities_dir / name).is_dir():
                    return name
            except Exception:
                pass
        cities = self.list_cities()
        return cities[0]["name"] if cities else None

    def set_active(self, name: str) -> None:
        if not (self.cities_dir / name).is_dir():
            raise ValueError(f"Unknown city: {name}")
        self._active_file.write_text(json.dumps({"active": name}, ensure_ascii=False))

    def city_dir(self, name: Optional[str] = None) -> Path:
        name = name or self.active_city()
        if not name:
            raise FileNotFoundError(
                "No city found. Run `streetrag city new <name>` then download a network."
            )
        return self.cities_dir / name

    def create_city(self, name: str) -> Path:
        slug = slugify_city(name)
        d = self.cities_dir / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "sources").mkdir(exist_ok=True)
        if not self._active_file.exists():
            self.set_active(slug)
        return d

    def catalog(self, name: Optional[str] = None) -> FeatureCatalog:
        return FeatureCatalog(self.city_dir(name) / "feature_registry.json")

    # -- legacy migration ----------------------------------------------------

    def has_legacy_layout(self) -> bool:
        return (self.root / "feature_registry.json").exists()

    def migrate_legacy(self, city_name: Optional[str] = None) -> Optional[str]:
        """Move legacy flat data/ layout into data/cities/<name>/.
        Non-network geodata files go into sources/. Returns city name or None."""
        if not self.has_legacy_layout():
            return None
        legacy_reg = self.root / "feature_registry.json"
        try:
            reg = json.loads(legacy_reg.read_text())
        except Exception:
            reg = {}
        network = reg.get("target_network", "")
        if not city_name:
            m = re.match(r"([A-Za-z_]+?)_(drive|walk|bike|all|all_private)\.gpkg", network or "")
            city_name = slugify_city(m.group(1)) if m else "default"

        dest = self.create_city(city_name)
        sources = dest / "sources"
        for item in sorted(self.root.iterdir()):
            if item.name in _GLOBAL_FILES or item == self.cities_dir:
                continue
            if item.is_file() and item.suffix.lower() in _SOURCE_EXTS and item.name != network:
                shutil.move(str(item), str(sources / item.name))
            else:
                shutil.move(str(item), str(dest / item.name))
        self.set_active(city_name)
        print(f"[workspace] migrated legacy data layout → cities/{city_name}/")
        return city_name
