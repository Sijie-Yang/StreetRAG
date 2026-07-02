"""Review vector index (LanceDB with numpy fallback)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from streetrag.core.feature_catalog import FeatureCatalog
from streetrag.llm.client import cosine_topk, embed_texts, load_rag_settings


class ReviewStore:
    """Per-city review/POI text index."""

    META_NAME = "reviews.meta.json"

    def __init__(self, catalog: FeatureCatalog):
        self.catalog = catalog
        self.data_dir = catalog.data_dir
        self.lance_path = catalog.review_index_path()
        self.meta_path = self.data_dir / self.META_NAME
        self._use_lance = False
        self._table = None
        try:
            import lancedb

            self._db = lancedb.connect(str(self.lance_path))
            self._use_lance = True
        except ImportError:
            self._db = None

    def _load_meta(self) -> dict:
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"records": [], "embedding_model": None}

    def _save_meta(self, meta: dict) -> None:
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def count(self) -> int:
        if self._use_lance:
            try:
                tbl = self._db.open_table("reviews")
                return tbl.count_rows()
            except Exception:
                return 0
        return len(self._load_meta().get("records", []))

    def upsert_records(self, records: List[dict], *, settings: Optional[dict] = None) -> int:
        if not records:
            return 0
        settings = settings or self.catalog.load_settings()
        registry_path = str(self.catalog.path)
        texts = [r.get("text", "") for r in records]
        embeddings = embed_texts(
            settings=settings,
            registry_path=registry_path,
            texts=texts,
        )
        for rec, vec in zip(records, embeddings):
            rec["vector"] = vec.tolist()

        if self._use_lance:
            import pyarrow as pa

            table_name = "reviews"
            if table_name in self._db.table_names():
                tbl = self._db.open_table(table_name)
                tbl.add(records)
            else:
                self._db.create_table(table_name, data=records)
            self._save_meta(
                {
                    "backend": "lancedb",
                    "n_records": self.count(),
                    "embedding_model": settings.get("embedding_model", "text-embedding-3-small"),
                }
            )
            return len(records)

        meta = self._load_meta()
        existing = meta.setdefault("records", [])
        existing.extend(records)
        meta["backend"] = "numpy"
        meta["embedding_model"] = settings.get("embedding_model", "text-embedding-3-small")
        meta["n_records"] = len(existing)
        self._save_meta(meta)
        np.savez_compressed(
            self.data_dir / "reviews.npz",
            vectors=np.asarray([r["vector"] for r in existing], dtype=np.float32),
        )
        return len(records)

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        category: Optional[str] = None,
        edge_ids: Optional[List[int]] = None,
        settings: Optional[dict] = None,
    ) -> List[dict]:
        settings = settings or self.catalog.load_settings()
        registry_path = str(self.catalog.path)
        if self.count() == 0:
            return []

        qvec = embed_texts(
            settings=settings,
            registry_path=registry_path,
            texts=[query],
        )[0]

        if self._use_lance:
            try:
                tbl = self._db.open_table("reviews")
                rows = tbl.search(qvec.tolist()).limit(max(top_k * 4, top_k)).to_list()
            except Exception:
                rows = []
            out = []
            for row in rows:
                if category and row.get("category") and row["category"] != category:
                    continue
                if edge_ids is not None and row.get("edge_id") not in edge_ids:
                    continue
                out.append(dict(row))
                if len(out) >= top_k:
                    break
            return out

        meta = self._load_meta()
        records = meta.get("records", [])
        if not records:
            return []
        npz_path = self.data_dir / "reviews.npz"
        if not npz_path.exists():
            return []
        mat = np.load(npz_path)["vectors"]
        idxs = cosine_topk(qvec, mat, min(top_k * 4, len(records)))
        out = []
        for i in idxs:
            rec = records[int(i)]
            if category and rec.get("category") and rec["category"] != category:
                continue
            if edge_ids is not None and rec.get("edge_id") not in edge_ids:
                continue
            out.append(rec)
            if len(out) >= top_k:
                break
        return out
