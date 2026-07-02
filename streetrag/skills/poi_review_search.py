"""Semantic search over indexed POI/review text."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from streetrag.llm.client import chat_text, load_rag_settings
from streetrag.reviews.store import ReviewStore
from streetrag.skills.base import Skill, SkillResult, skill
from streetrag.utils.geocode import geocode_place

_HIGHLIGHT_COL = "_review_search_hits"


class PoiReviewSearchParams(BaseModel):
    query: str = Field(..., description="Semantic search query over review/POI text")
    category: Optional[str] = Field(None, description="Optional POI category filter")
    spatial_target: Optional[str] = Field(
        None, description="Optional place name to restrict search near"
    )
    spatial_filter_radius_m: Optional[float] = Field(
        800.0, description="Radius around spatial_target in meters"
    )
    top_k: int = Field(8, ge=1, le=20)
    user_query: str = ""


@skill
class PoiReviewSearchSkill(Skill):
    name = "poi_review_search"
    description = (
        "Search indexed POI/review text and explain WHY streets or places are "
        "rated highly or poorly, citing review snippets. Use when the user asks "
        "what people say, reviews, comments, or qualitative reasons."
    )
    params_model = PoiReviewSearchParams

    def run(self, net, params: PoiReviewSearchParams) -> SkillResult:
        store = ReviewStore(net.catalog)
        if store.count() == 0:
            return SkillResult(
                skill_name=self.name,
                reply=(
                    "No review text is indexed for this city yet. "
                    "Upload a POI/review CSV or GPKG with a text column via the web UI "
                    "or place it in data/sources/ and run: streetrag scan && streetrag integrate."
                ),
                render_map=False,
            )

        edge_ids: Optional[List[int]] = None
        spatial_meta = None
        if params.spatial_target:
            import geopandas as gpd
            from shapely.geometry import Point

            geo = geocode_place(params.spatial_target, str(net.catalog.path))
            spatial_meta = geo
            if geo.get("lon") is not None and geo.get("lat") is not None:
                pt = Point(float(geo["lon"]), float(geo["lat"]))
                gpd_point = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs(net.edges.crs).iloc[0]
                radius = float(params.spatial_filter_radius_m or 800.0)
                dists = net.edges.geometry.centroid.distance(gpd_point)
                edge_ids = [int(i) for i, d in dists.items() if d <= radius]

        hits = store.search(
            params.query or params.user_query,
            top_k=params.top_k,
            category=params.category,
            edge_ids=edge_ids,
            settings=net.catalog.load_settings(),
        )
        if not hits:
            return SkillResult(
                skill_name=self.name,
                reply="No matching review snippets found for that query.",
                render_map=False,
            )

        hit_counts = {}
        for h in hits:
            eid = h.get("edge_id")
            if eid is not None:
                hit_counts[int(eid)] = hit_counts.get(int(eid), 0) + 1

        net.edges[_HIGHLIGHT_COL] = 0.0
        for eid, cnt in hit_counts.items():
            if eid in net.edges.index:
                net.edges.at[eid, _HIGHLIGHT_COL] = float(cnt)

        snippets = []
        for i, h in enumerate(hits[: params.top_k], 1):
            name = h.get("poi_name") or h.get("poi_id") or "POI"
            cat = h.get("category") or ""
            rating = h.get("rating")
            rating_s = f" ({rating}★)" if rating is not None else ""
            text = (h.get("text") or "")[:280]
            snippets.append(f"{i}. {name}{rating_s}{' [' + cat + ']' if cat else ''}: \"{text}\"")

        settings = load_rag_settings(str(net.catalog.path))
        prompt = (
            f"User question: {params.user_query or params.query!r}\n\n"
            f"Retrieved review snippets:\n" + "\n".join(snippets) + "\n\n"
            "Write a concise answer citing specific review quotes and street/place patterns. "
            "Same language as the user question. Plain text."
        )
        try:
            reply = chat_text(
                settings=settings,
                registry_path=str(net.catalog.path),
                system="You are an urban analyst summarizing POI reviews grounded in evidence.",
                prompt=prompt,
            )
        except Exception:
            reply = "Review evidence:\n" + "\n".join(snippets)

        top_edges = sorted(hit_counts.items(), key=lambda x: -x[1])[:5]
        return SkillResult(
            skill_name=self.name,
            columns_written=[_HIGHLIGHT_COL],
            stats={"n_hits": len(hits), "n_edges": len(hit_counts)},
            narrative_evidence={
                "review_snippets": hits[: params.top_k],
                "top_edges": [{"edge_id": e, "hits": c} for e, c in top_edges],
            },
            reply=reply,
            index_col=_HIGHLIGHT_COL,
            index_name="Review search hits",
            render_map=True,
            spatial_target=params.spatial_target,
            spatial_target_resolved=spatial_meta,
            spatial_filter_radius_m=params.spatial_filter_radius_m,
        )
