# StreetRAG Architecture

## Overview

StreetRAG is an interactive urban street analysis agent. Users chat with a multi-turn LLM agent that can visualize features on a MapLibre map, propose composite indices, show tables, integrate geodata, and run spatial analysis skills.

## Agent loop (`streetrag/agent/loop.py`)

Inspired by Claude Code's `queryLoop`:

```
User message + context chips
    → LLM stream (text/thinking/tool_call deltas)
    → Execute tools (visualize, table, proposals, skills)
    → Append tool_result to messages
    → Repeat until no tool calls (max 8 turns)
    → Emit done event
```

### SSE event protocol

| Event | Purpose |
|-------|---------|
| `text_delta` | Streaming assistant text |
| `thinking_delta` | Reasoning model output |
| `tool_use` | Tool invocation started |
| `tool_result` | Tool JSON result |
| `visualize` | Map should color by column |
| `table` | Show table in side panel |
| `proposals` | Index proposal cards for user |
| `progress` | Step progress with optional `pct` |
| `result` | Final payload (back-compat) |
| `error` | Error message |
| `done` | Agent loop complete |

## Tools

**Action tools** (`streetrag/agent/tools.py`):
- `visualize_feature` — map coloring
- `show_table` — data table panel
- `propose_index_options` — user must confirm before creating index
- `list_features_stats` / `get_feature_detail` — explore registry

**Skills** (`streetrag/skills/`): analysis skills registered via `@skill` decorator, exposed as OpenAI function tools.

## Data model

Split storage layout (after `streetrag migrate-storage`):

```
network.gpkg              # geometry + edge_id + u/v/length (topology only)
features/
  ├── <source>.parquet    # edge_id + integrated columns per data source
  ├── syntax.parquet      # space syntax columns
  └── indices.parquet     # composite index columns
feature_registry.json     # stats, integrations, column → parquet mapping
```

- **Stable `edge_id`**: persisted integer key on every edge; reviews and parquet joins use it (not row position).
- **StreetNetwork**: loads topology GPKG + joins parquet into a wide `net.edges` view; `save()` writes topology to GPKG and features to parquet.
- **Legacy mode**: wide single GPKG still supported until migration.
- **Feature registry**: `feature_registry.json` tracks integrations, statistics, indices
- **Zones** (`streetrag/zones/`): hex/rect grids, admin boundaries, edge→zone aggregation

## Frontend modules (`webapp/static/js/`)

| Module | Role |
|--------|------|
| `state.js` | Pub/sub + context chips state |
| `sse.js` | SSE parsing |
| `chat.js` | (via app.js patches) streaming chat UI |
| `data-panel.js` | File tree integrate/delete/reload |
| `features-panel.js` | Feature list CRUD + visualize |
| `indices-panel.js` | Index list + delete + add to chat |
| `table-panel.js` | Sortable table + CSV export |
| `context-chips.js` | Context injection UI |
| `progress-bar.js` | Global progress bar |
| `main.js` | Bootstrap |

Legacy map rendering remains in `app.js`; new modules extend it via `window.StreetRAG` hooks.

## API surface (management)

- `GET /api/features` — feature list
- `DELETE /api/features/{col}` — remove column from GPKG + registry
- `DELETE /api/indices/{col}` — remove saved index
- `DELETE /api/files/{name}` — delete source file
- `POST /api/files/{name}/integrate-stream` — SSE integrate with progress
- `POST /api/files/{name}/reintegrate-stream` — clear + re-integrate
- `GET/POST/DELETE /api/zones` — zone layers

## References

- Claude Code `queryLoop` — multi-turn tool-use with streaming
- auto-urban-science-app — SSE event protocol, ThinkingBlock, ToolUseCard
