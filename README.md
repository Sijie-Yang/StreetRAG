# StreetRAG

> Ask a city questions in natural language — get a colour-mapped street network
> plus a multi-scale, evidence-grounded answer.

StreetRAG is an open framework for **LLM-agent-driven, multi-scale street
network analysis**. The core idea fits in one sentence:

> *Any geodata becomes a column on street edges; any analysis is a Skill;
> the LLM selects tools and skills across a multi-turn agent loop.*

```
question
  ▸ source-balanced feature retrieval (small families in full, large families capped)
  ▸ multi-turn agent loop (stream text · tools · skills · proposals)
  ▸ skills: composite index · multiscale profile · LISA · correlate · review search …
  ▸ action tools: map visualize · top_edges_table · show_table · propose_index_options
  ▸ evidence: length-weighted multi-scale top-k, Moran's I, review snippets
  ▸ render: MapLibre + colorbar + table panel + hover breakdown
  ▸ persist: slim GPKG (topology) + features/*.parquet + indices/{col}.json
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the interactive agent loop, SSE event protocol, split storage layout, frontend modules, and API design.

## Dual-layer architecture

| Layer | What it stores | Used for |
|---|---|---|
| **Layer A** | Numeric columns on street edges (parquet per source + slim GPKG) | *Where* streets rank — maps, composite indices, LISA |
| **Layer B** | POI/review text chunks + embeddings (`reviews.lance` or `reviews.npz`) | *Why* — semantic search over review text |

Layer B is optional. POI **ratings** and **densities** still land in Layer A during integrate.

## Repository layout

```
streetrag/
├── core/      StreetNetwork, FeatureCatalog, FeatureStore, network GPKG IO
├── ingest/    integrators, pipeline, OSMnx download
├── reviews/   POI/review text index (LanceDB or numpy fallback)
├── syntax/    metric + angular space syntax
├── skills/    plugin system — one file per Skill (@skill decorator)
├── agent/     multi-turn loop + action tools (visualize, table, proposals)
├── llm/       OpenAI client, source-stratified retrieval, language lock
└── cli.py     download · scan · integrate · syntax · ask · serve · migrate-storage
```

## Quick start (with bundled Singapore sample)

This repo ships a **Singapore demo case** (~31 MB):

| Path | Contents |
|---|---|
| `data/cities/singapore/Singapore_Street_Network_drive.gpkg` | Drive street network (topology) |
| `data/cities/singapore/sources/Singapore_osm_pois.gpkg` | OSM POI points |
| `data/cities/singapore/sources/singapore_VATA_perception_points.gpkg` | VATA street-perception points |
| `data/cities/singapore/feature_registry.json` | Catalog + integration metadata |
| `data/active_city.json` | Points at `singapore` |

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev,reviews]"

export OPENAI_API_KEY=sk-...   # or data/RAG_setting.local.json (gitignored)

# Integrate sample sources onto edges (POI densities + perception columns)
.venv/bin/streetrag integrate

# Optional: compute space syntax (integration_R500, …)
.venv/bin/streetrag syntax --radii 500,1500,4500

# Web UI
.venv/bin/streetrag serve          # → http://127.0.0.1:8765/
```

CLI one-shot query:

```bash
.venv/bin/streetrag ask "Which streets have the highest Temperature Intensity?"
```

### Fresh city (no sample data)

```bash
.venv/bin/streetrag download --city "London, UK" --network-type drive
.venv/bin/streetrag scan
.venv/bin/streetrag integrate
.venv/bin/streetrag serve
```

## Multi-city workspace

```
data/
├── RAG_setting.json
├── active_city.json
└── cities/
    └── singapore/          # committed sample (network + sources)
        ├── Singapore_Street_Network_drive.gpkg
        ├── sources/
        ├── features/         # local after integrate (gitignored)
        ├── indices/          # saved composite indices (local)
        └── feature_registry.json
```

```bash
streetrag city list
streetrag download --city "Tokyo, Japan"
streetrag city use singapore
```

> Do not expose the server publicly without auth; it uses your API key and can mutate city data.

## Storage: geometry vs features

After integrate, StreetNetwork **saves topology to GPKG** and **feature columns to
`features/<source>.parquet`**, joined at load time by stable `edge_id`.
One-shot migration for old wide GPKGs:

```bash
streetrag migrate-storage --city singapore
```

## Feature retrieval (for index planning)

Default: **`stratified`** (`data/RAG_setting.json`)

- Small source families (≤20 columns) → always shown to the LLM
- Large families (e.g. 352 POI columns) → capped per source (~40% of menu)
- Space-syntax `integration_R*` always included when present

Alternatives: `embedding`, `token`, `full` (see `feature_retrieval_method`).

## Data integration

| Geometry | Integrator | Method |
|---|---|---|
| Point | `snap_nearest` | mean of k nearest points per edge |
| Point | `buffer_density` | count / mean within radius |
| Point | `poi_category_density_rating` | per-category density + mean rating |
| Line | `line_overlay` | mean over intersecting lines |
| Polygon | `polygon_area_weighted` | area-weighted mean |

Formats: GPKG, GeoJSON, Shapefile, CSV (lon/lat), GeoParquet.

## Space syntax

Per radius (`500,1500,4500` m by default):

* **Metric integration** `integration_R{r}` — momepy closeness (metric `mm_len`)
* **Angular** `nain_R{r}`, `nach_R{r}`, `choice_R{r}`, … — segment turn-angle graph

Run from CLI or the web UI **Syntax** button.

## Skills

| Skill | What it does |
|---|---|
| `composite_index` | Weighted composite index; spatial targeting; map render |
| `multiscale_profile` | Index behaviour across LOCAL / MEDIUM / GLOBAL scales |
| `cluster_lisa` | Moran's I + LISA clusters (`esda` / `libpysal`) |
| `correlate` | Pearson / Spearman between two edge columns |
| `poi_review_search` | Semantic search over review text (Layer B) |
| `answer_directly` | Meta / help when no analysis skill applies |

## Agent action tools (web chat)

| Tool | Purpose |
|---|---|
| `propose_index_options` | 2–4 index proposals as clickable cards (user confirms → create) |
| `top_edges_table` | **Real** top-N edge query → right-hand table panel + CSV export |
| `show_table` | Small summary tables composed by the LLM |
| `visualize_feature` | Colour map by column |
| `list_features_stats` / `get_feature_detail` | Explore catalog |

Example chat prompts:

```text
Show a table of the top 20 streets with the highest Temperature Intensity.
Create an urban comfort index using greenery, shade, and temperature perception.
Correlate Greenery Rate and Temperature Intensity.
Map Shading Area on the network.
```

## Web UI

Open http://127.0.0.1:8765/ after `streetrag serve`.

| Area | Description |
|---|---|
| **Data** | File list by source, integrate / reload / delete, upload |
| **Features** | All edge columns; Map · + Chat · Del |
| **Indices** | Saved composite indices; click to render |
| **Chat** | Multi-turn agent with streaming progress |
| **Table panel** | Opens on `top_edges_table` / `show_table`; export CSV |
| **Syntax** | Configure radii and run space syntax |
| **+ City** | Download a new city from OSM |

On first launch without any city data, the server auto-downloads Singapore from OSM.
With the **bundled sample**, it activates `singapore` immediately.

## Web API (selected)

Read: `GET /api/edges-geojson`, `/api/indices`, `/api/index/{col}`, `/api/datasets`, `/api/features`, …

Actions: `POST /api/chat-stream` (SSE), `/api/create-index`, `/api/integrate-stream`,
`/api/upload`, `/api/syntax/run-stream`, `/api/route`, …

## Reproducibility

LLM calls cached under `data/cities/<city>/llm_cache/`. Saved indices in
`indices/{col}.json` record weights, stats, model, and seed.

## Experiments

* `experiments/queries.json` — evaluation queries
* `experiments/run_eval.py` — retrieval baselines (`token` / `embedding` / `stratified`)

## Tests

```bash
.venv/bin/pytest tests/ -q
```

## License

See repository license file. Sample geodata is provided for research/demo use; check source attributions (OSM, VATA) before redistribution.
