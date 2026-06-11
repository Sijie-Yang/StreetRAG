# StreetRAG

> Ask a city questions in natural language — get a colour-mapped street network
> plus a multi-scale, evidence-grounded answer.

StreetRAG is an open framework for **LLM-agent-driven, multi-scale street
network analysis**. The core idea fits in one sentence:

> *Any geodata becomes a column on street edges; any analysis is a Skill;
> the LLM only selects a Skill and fills its parameters.*

```
question
  ▸ semantic-retrieve a focused feature catalog (embedding cosine, cached)
  ▸ ONE function-calling LLM step → pick Skill + fill Pydantic params
  ▸ Skill runs on the StreetNetwork (composite index · multiscale profile ·
    LISA clusters · correlation · ...)
  ▸ evidence: length-weighted multi-scale top-k, Moran's I, NAIN/NACH
  ▸ render: MapLibre + colorbar + hover feature contributions
  ▸ persist: GPKG columns + data/indices/{col}.json (plan, seed, model, stats)
```

## Architecture

```
streetrag/
├── core/      StreetNetwork (edges+nodes+catalog), FeatureCatalog, atomic GPKG IO
├── ingest/    point/line/polygon integrators, multi-format readers, OSMnx download
├── syntax/    metric integration (momepy) + angular segment analysis
│              (turn-angle dual graph, metric radius, NAIN / NACH / choice)
├── skills/    plugin system — one file per Skill, auto-discovered
├── agent/     function-calling agent: skill manifests = OpenAI tool schemas
├── llm/       OpenAI client (cache, seed, logs), feature retrieval, IndexPlan
└── cli.py     streetrag download|scan|integrate|syntax|ask|clean|serve
```

**Design rule:** `StreetNetwork` is the single data structure. Every capability
is a `Skill` — a Pydantic params model plus a `run(net, params) -> SkillResult`.
The agent converts skill manifests to OpenAI tool schemas automatically, so a
new skill = one new file under `streetrag/skills/`, nothing else.

## Quick start

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"

export OPENAI_API_KEY=sk-...   # or data/RAG_setting.local.json (gitignored)

# 1. Get a network (creates data/cities/singapore/ and activates it)
.venv/bin/streetrag download --city "Singapore" --network-type drive

# 2. Discover data files (city root + sources/; CSV auto-converted)
.venv/bin/streetrag scan

# 3. Integrate: space syntax + all external sources onto edges
.venv/bin/streetrag integrate

# 4. Ask
.venv/bin/streetrag ask "find the most walkable streets"

# 5. Web UI (map + chat + upload + city switcher)
.venv/bin/streetrag serve          # → http://127.0.0.1:8765/
```

## Multi-city workspace

Each city is a self-contained directory; the active one is switchable from
the CLI or the web UI's city dropdown:

```
data/
├── RAG_setting.json          global LLM settings (shared)
├── active_city.json          which city is active
└── cities/
    ├── singapore/
    │   ├── Singapore_drive.gpkg     street network
    │   ├── feature_registry.json    per-city feature catalog
    │   ├── sources/                 external geodata (gpkg/csv/geojson/…)
    │   └── indices/ llm_cache/ …    per-city artifacts
    └── london/ …
```

```bash
streetrag city list                      # show cities (* = active)
streetrag download --city "London, UK"   # add a new city + activate
streetrag city use singapore             # switch back
streetrag ask "..." --city-name london   # one-off query against another city
```

Legacy flat `data/` layouts are migrated automatically on first run.

> Do not expose the server publicly without auth; it uses your API key and can
> mutate the GPKG.

## Data integration ("any geodata → street edges")

`streetrag scan` classifies every file in `data/` by geometry type and picks an
integrator; you can also upload via the web UI (`POST /api/upload` →
`POST /api/integrate`):

| Geometry | Integrator | Method |
|---|---|---|
| Point | `snap_nearest` | mean of k nearest points per edge centroid |
| Point | `buffer_density` | count / mean rating within radius (per POI category) |
| Line | `line_overlay` | mean over intersecting source lines (edge buffer) |
| Polygon | `polygon_area_weighted` | area-weighted mean over intersecting polygons |

Formats: GPKG, GeoJSON, Shapefile, CSV (lon/lat), GeoParquet. CRS is
auto-projected to a local UTM zone (override: `preferred_crs_epsg`).

## Space syntax

Two families of measures, computed per radius (`--radii 500,1500,4500`):

* **Metric integration** `integration_R{r}` — momepy closeness centrality with
  metric distance (`mm_len`).
* **Angular segment analysis** `angular_integration_R{r}`, `nain_R{r}`,
  `choice_R{r}`, `nach_R{r}` — standard formulation (Turner 2001; Hillier,
  Yang & Turner 2012): segments are dual-graph nodes, the cost between adjacent
  segments is the turn angle (90° = 1.0), shortest paths are angular while the
  radius is metric. NAIN = NC^1.2 / total angular depth;
  NACH = log10(choice+1) / log10(total depth+3).

Computed columns persist in the GPKG (which acts as the cache); timings are
logged to `data/syntax_cache/timings.json`. Angular analysis is O(n·E log V) —
see `experiments/benchmark_syntax.py` for measured scaling.

## Skills

| Skill | What it does |
|---|---|
| `composite_index` | weighted composite street index (4 operators, 5 normalizations, spatial targeting) |
| `multiscale_profile` | how an index behaves across LOCAL/MEDIUM/GLOBAL radii + cross-scale correlations |
| `cluster_lisa` | Moran's I + local LISA clusters (HH/LL/HL/LH, needs `pip install esda libpysal`) |
| `correlate` | Pearson/Spearman between any two edge features |

Adding a skill:

```python
# streetrag/skills/my_skill.py — that's the whole integration
from pydantic import BaseModel
from streetrag.skills.base import Skill, SkillResult, skill

class MyParams(BaseModel):
    col: str
    user_query: str = ""

@skill
class MySkill(Skill):
    name = "my_skill"
    description = "One sentence the agent uses for routing."
    params_model = MyParams

    def run(self, net, params) -> SkillResult:
        ...
        return SkillResult(skill_name=self.name, reply="...", render_map=False)
```

## Reproducibility

Every LLM call is cached on disk keyed by `(model, seed, temperature, prompt)`
sha and fully logged to `data/llm_logs/`. Saved indices
(`data/indices/{col}.json`) record the plan, weights, rationales, model, seed,
and all spatial statistics needed to re-run them.

## Experiments (`experiments/`)

* `queries.json` — bilingual evaluation query set
* `cities.json` — multi-city study plan
* `run_eval.py` — retrieval baselines (token vs embedding), plan reproducibility
* `benchmark_syntax.py` — angular/metric syntax runtime scaling

## Web API surface

`GET /api/edges-geojson · /api/indices · /api/index/{col} · /api/edge-info ·
/api/geocode · /api/integrators` — read-only.
`POST /api/chat-stream (SSE) · /api/chat · /api/index-chat · /api/route ·
/api/upload · /api/integrate · /api/scan` — actions.
