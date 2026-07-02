/* global maplibregl */
(function () {
  // -------------------------------------------------------------------
  // Endpoint helper (supports file:// fallback for local debugging)
  // -------------------------------------------------------------------
  function apiPath(path) {
    try {
      if (typeof location !== "undefined" && location.protocol === "file:") {
        return (window.__STREETRAG_API__ || "http://127.0.0.1:8765") + path;
      }
    } catch (_) {}
    return path;
  }

  // -------------------------------------------------------------------
  // DOM
  // -------------------------------------------------------------------
  const log = document.getElementById("log");
  const activity = document.getElementById("activity");
  const form = document.getElementById("f");
  const input = document.getElementById("msg");
  const goBtn = document.getElementById("go");
  const basemapSelect = document.getElementById("basemapSelect");
  const colormapSelect = document.getElementById("colormapSelect");
  const stretchSelect = document.getElementById("stretchSelect");
  const routeModeBtn = document.getElementById("routeModeBtn");
  const exportBtn = document.getElementById("exportBtn");
  const drawer = document.getElementById("drawer");
  const drawerToggle = document.getElementById("drawerToggle");
  const indicesList = document.getElementById("indicesList");
  const refreshIndicesBtn = document.getElementById("refreshIndicesBtn");
  const routeBanner = document.getElementById("routeBanner");
  const routeWeightSelect = document.getElementById("routeWeightSelect");
  const routeCancelBtn = document.getElementById("routeCancelBtn");
  const cb = {
    root: document.getElementById("colorbar"),
    title: document.getElementById("cbTitle"),
    bar: document.getElementById("cbBar"),
    min: document.getElementById("cbMin"),
    mid: document.getElementById("cbMid"),
    max: document.getElementById("cbMax"),
    stats: document.getElementById("cbStats"),
  };

  // -------------------------------------------------------------------
  // Persisted UI state
  // -------------------------------------------------------------------
  const LS = {
    BASE: "streetRag.basemap",
    COLOR: "streetRag.colormap",
    STRETCH: "streetRag.stretch",
    DRAWER: "streetRag.drawerCollapsed",
  };

  const BASEMAPS = {
    osm: { tiles: ["https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"], attribution: "© OpenStreetMap" },
    "carto-light": { tiles: ["https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"], attribution: "© CARTO" },
    "carto-dark": { tiles: ["https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"], attribution: "© CARTO", dark: true },
    "carto-voyager": { tiles: ["https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"], attribution: "© CARTO" },
    opentopomap: { tiles: ["https://a.tile.opentopomap.org/{z}/{x}/{y}.png"], attribution: "© OpenStreetMap · OpenTopoMap" },
    "esri-sat": { tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"], attribution: "© Esri" },
  };

  // 6-stop ramps — vmin → vmax. We sample 5 colours linearly.
  const RAMPS = {
    cividis: ["#00224e", "#2b446a", "#5b6a76", "#8b8675", "#bba75e", "#fee838"],
    viridis: ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"],
    magma: ["#000004", "#3b0f70", "#8c2981", "#de4968", "#fea16e", "#fcfdbf"],
    diverging: ["#8e0152", "#c994c7", "#f7f7f7", "#a1d99b", "#41ab5d", "#276419"],
    rdbu: ["#b2182b", "#ef8a62", "#f7f7f7", "#a6cee3", "#5288bd", "#2166ac"],
    default: ["#001a4d", "#0a4ad9", "#0094ff", "#00b8a9", "#ffcc00", "#ff2b00"],
  };

  function getMode() {
    const m = (colormapSelect && colormapSelect.value) || "cividis";
    return RAMPS[m] ? m : "cividis";
  }
  function getStretch() {
    return (stretchSelect && stretchSelect.value) || "p5_p95";
  }
  function isBasemapDark() {
    const id = (basemapSelect && basemapSelect.value) || "carto-voyager";
    return !!(BASEMAPS[id] && BASEMAPS[id].dark);
  }
  function contextStreetPaint() {
    return {
      "line-color": isBasemapDark() ? "#94a3b8" : "#475569",
      "line-width": 1.0,
      "line-opacity": isBasemapDark() ? 0.45 : 0.35,
    };
  }
  const INDEX_LINE_BASE = { "line-width": 3.2, "line-opacity": 0.92 };

  // -------------------------------------------------------------------
  // Index render state
  // -------------------------------------------------------------------
  /** {col, summary, gj, vmin, vmax, vmid, n}, see setIndexLayer */
  let lastIndex = null;

  function stretchRange(summary) {
    if (!summary) return [0, 1, 0.5];
    const mode = getStretch();
    const fallback = [summary.min ?? 0, summary.max ?? 1];
    let lo = fallback[0];
    let hi = fallback[1];
    if (mode === "p5_p95" && summary.p05 != null && summary.p95 != null) {
      lo = summary.p05;
      hi = summary.p95;
    } else if (mode === "iqr" && summary.p25 != null && summary.p75 != null) {
      lo = summary.p25;
      hi = summary.p75;
    }
    if (!(hi > lo)) { hi = lo + 1e-9; }
    const mid = summary.median != null ? summary.median : (lo + hi) / 2;
    return [lo, hi, mid];
  }

  function paintForIndex(v0, v1, mode) {
    const ramp = RAMPS[mode] || RAMPS.cividis;
    if (v1 - v0 < 1e-12) {
      return { "line-color": ramp[ramp.length - 1], ...INDEX_LINE_BASE };
    }
    const step = (v1 - v0) / (ramp.length - 1);
    const stops = [];
    for (let i = 0; i < ramp.length; i++) {
      stops.push(v0 + step * i, ramp[i]);
    }
    return {
      "line-color": [
        "interpolate", ["linear"],
        ["max", v0, ["min", v1, ["to-number", ["get", "v"], v0]]],
        ...stops,
      ],
      ...INDEX_LINE_BASE,
    };
  }

  function colorbarBackground(mode) {
    const ramp = RAMPS[mode] || RAMPS.cividis;
    const stops = ramp.map((c, i) => `${c} ${Math.round((i * 100) / (ramp.length - 1))}%`).join(", ");
    return `linear-gradient(90deg, ${stops})`;
  }

  function fmtNum(x) {
    if (x == null || !isFinite(x)) return "—";
    const a = Math.abs(x);
    if (a !== 0 && (a < 1e-3 || a >= 1e4)) return x.toExponential(2);
    return Number(x).toPrecision(4).replace(/\.?0+$/, "");
  }

  function showColorbar(col, summary, indexName) {
    if (!summary || summary.n === 0 || summary.min == null) {
      cb.root.classList.add("hidden");
      return;
    }
    const [lo, hi, mid] = stretchRange(summary);
    cb.root.classList.remove("hidden");
    cb.title.textContent = indexName ? `${indexName} (${col})` : col;
    cb.bar.style.background = colorbarBackground(getMode());
    cb.min.textContent = fmtNum(lo);
    cb.mid.textContent = `mid ${fmtNum(mid)}`;
    cb.max.textContent = fmtNum(hi);
    cb.stats.textContent =
      `n=${summary.n.toLocaleString()} · ` +
      `min=${fmtNum(summary.min)} · median=${fmtNum(summary.median)} · max=${fmtNum(summary.max)} · ` +
      `[${getStretch()}]`;
  }

  function hideColorbar() {
    cb.root.classList.add("hidden");
    lastIndex = null;
  }

  // -------------------------------------------------------------------
  // Persistence
  // -------------------------------------------------------------------
  function loadSavedSelects() {
    const sB = localStorage.getItem(LS.BASE);
    if (sB && BASEMAPS[sB]) basemapSelect.value = sB;
    const sC = localStorage.getItem(LS.COLOR);
    if (sC && Array.from(colormapSelect.options).some((o) => o.value === sC)) {
      colormapSelect.value = sC;
    } else {
      colormapSelect.value = "cividis";
    }
    const sS = localStorage.getItem(LS.STRETCH);
    if (sS && Array.from(stretchSelect.options).some((o) => o.value === sS)) {
      stretchSelect.value = sS;
    }
    const dr = localStorage.getItem(LS.DRAWER);
    if (dr === "1") drawer.classList.add("collapsed");
    updateDrawerToggle();
  }
  function updateDrawerToggle() {
    drawerToggle.textContent = drawer.classList.contains("collapsed") ? "›" : "‹";
  }

  // URL hash state: index=<col>&base=<basemap>&color=<colormap>&stretch=<stretch>
  function readHashState() {
    const h = (location.hash || "").replace(/^#/, "");
    const out = {};
    h.split("&").forEach((kv) => {
      const [k, v] = kv.split("=");
      if (k && v) out[decodeURIComponent(k)] = decodeURIComponent(v);
    });
    return out;
  }
  function writeHashState(state) {
    const parts = [];
    for (const k in state) {
      if (state[k] != null && state[k] !== "") {
        parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(state[k])}`);
      }
    }
    history.replaceState(null, "", "#" + parts.join("&"));
  }
  function updateHashIndex(col) {
    const s = readHashState();
    if (col) s.index = col; else delete s.index;
    s.base = basemapSelect.value;
    s.color = colormapSelect.value;
    s.stretch = stretchSelect.value;
    writeHashState(s);
  }

  // -------------------------------------------------------------------
  // Activity / log
  // -------------------------------------------------------------------
  const STEP_LABELS = {
    load_settings: "Loading settings",
    load_registry: "Reading feature registry",
    load_gpkg: "Loading street GeoPackage",
    retrieve_features: "Embedding retrieval · feature recall",
    plan: "Structured planning",
    agent: "Agent selects skill",
    geocode: "Resolving spatial target",
    compute_index: "Computing composite index",
    multiscale_topk: "Multi-scale top-k stats",
    spatial_diag: "Spatial autocorrelation (Moran's I)",
    llm: "LLM narrative",
    save_gpkg: "Writing GPKG",
    update_registry: "Updating registry",
    complete: "Complete",
    route: "Routing",
    syntax: "Space Syntax",
  };
  function formatProgress(o) {
    if (o.type !== "progress" || !o.step) return JSON.stringify(o);
    const name = STEP_LABELS[o.step] || o.step;
    const d = o.detail || {};
    const parts = [name];
    if (d.phase) parts.push(`[${d.phase}]`);
    if (d.method) parts.push(`method=${d.method}`);
    if (d.top_m) parts.push(`top_m=${d.top_m}`);
    if (d.total) parts.push(`/${d.total}`);
    if (d.intent) parts.push(`intent=${d.intent}`);
    if (d.operator) parts.push(`op=${d.operator}`);
    if (d.normalization) parts.push(`norm=${d.normalization}`);
    if (d.n_edges != null) parts.push(`${d.n_edges} edges`);
    if (d.n_features) parts.push(`feat=${d.n_features}`);
    if (d.crs) parts.push(d.crs);
    if (d.index_col) parts.push("col " + d.index_col);
    if (d.morans_i && d.morans_i.I != null) parts.push(`I=${Number(d.morans_i.I).toFixed(3)}`);
    if (d.query) parts.push(`q="${d.query}"`);
    if (d.display_name) parts.push(`→ ${d.display_name}`);
    if (d.filter_radius_m) parts.push(`radius ${Number(d.filter_radius_m).toFixed(0)}m`);
    if (d.kept != null && d.total != null) parts.push(`kept ${d.kept}/${d.total}`);
    if (d.message) parts.push(d.message);
    return parts.join(" · ");
  }
  function clearActivity() {
    if (activity) activity.innerHTML = "";
    streamBubble = null;
    thinkDetails = null;
  }
  let streamBubble = null;
  let thinkDetails = null;
  function appendStreamText(delta) {
    if (!streamBubble) {
      streamBubble = document.createElement("div");
      streamBubble.className = "log-line sys stream-text";
      log.appendChild(streamBubble);
    }
    streamBubble.textContent += delta;
    log.scrollTop = log.scrollHeight;
  }
  function appendThinking(delta) {
    if (!thinkDetails) {
      thinkDetails = document.createElement("details");
      thinkDetails.className = "thinking-block";
      thinkDetails.open = true;
      const sum = document.createElement("summary");
      sum.textContent = "Thinking…";
      const pre = document.createElement("pre");
      pre.className = "think-pre";
      thinkDetails.appendChild(sum);
      thinkDetails.appendChild(pre);
      activity.appendChild(thinkDetails);
    }
    const pre = thinkDetails.querySelector(".think-pre");
    if (pre) pre.textContent += delta;
  }
  function appendToolCard(name, data, status) {
    const card = document.createElement("div");
    card.className = "tool-card " + status;
    const body = typeof data === "string" ? data : JSON.stringify(data, null, 2);
    card.innerHTML = "<strong>" + escapeHtml(name) + "</strong> <span class=\"tool-status\">" + status + "</span><pre>" + escapeHtml(body) + "</pre>";
    activity.appendChild(card);
  }
  function proposalHintText() {
    const q = window.__lastUserQuery || "";
    if (/[\u4e00-\u9fff]/.test(q)) return "选择一个要创建的指数：";
    return "Choose an index to create:";
  }

  function appendProposals(proposals) {
    if (!proposals.length) return;
    const wrap = document.createElement("div");
    wrap.className = "proposal-wrap";
    const hint = document.createElement("div");
    hint.className = "proposal-hint";
    hint.textContent = proposalHintText();
    wrap.appendChild(hint);
    proposals.forEach((p) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "proposal-card";
      const title = p.name || p.index_col || "Index";
      const col = p.index_col && p.index_col !== p.name ? p.index_col : "";
      btn.innerHTML =
        "<strong>" + escapeHtml(title) + "</strong>" +
        (col ? "<div class=\"proposal-col\">" + escapeHtml(col) + "</div>" : "") +
        "<p>" + escapeHtml(p.rationale || "") + "</p>";
      btn.addEventListener("click", () => createIndexFromProposal(p, btn, wrap));
      wrap.appendChild(btn);
    });
    log.appendChild(wrap);
  }

  function buildCreateIndexUserQuery(indexName) {
    const orig = (window.__lastUserQuery || "").trim();
    if (orig) {
      return `${orig}\n\n(User confirmed index proposal: ${indexName})`;
    }
    return `Create index ${indexName}`;
  }

  async function createIndexFromProposal(p, btn, wrap) {
    const features = (p.features || []).map((f) => ({
      name: f.name,
      weight: f.weight != null ? f.weight : 1,
      rationale: f.rationale || "",
    }));
    if (!features.length) {
      appendLog("Error", "Proposal has no features.", "err");
      return;
    }
    const indexName = p.index_col || p.name;
    if (!indexName) {
      appendLog("Error", "Proposal is missing index name.", "err");
      return;
    }
    wrap.querySelectorAll(".proposal-card").forEach((b) => { b.disabled = true; });
    btn.textContent = "Creating…";
    appendLog("You", `Create index: ${indexName}`, "user");
    try {
      window.StreetRAG?.showProgress?.(5, "Creating index…");
      const r = await fetch(apiPath("/api/create-index"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          index_name: p.name || indexName,
          index_col: indexName,
          features,
          operator: p.operator || "weighted_sum",
          normalization: p.normalization || "robust",
          spatial_target: p.spatial_target || null,
          spatial_filter_radius_m: p.spatial_filter_radius_m || null,
          proximity_dominant: !!p.proximity_dominant,
          explanation: p.rationale || "",
          user_query: buildCreateIndexUserQuery(indexName),
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      const final = await r.json();
      btn.classList.add("created");
      btn.innerHTML = "<strong>" + escapeHtml(indexName) + "</strong><p>Created</p>";
      if (final.reply) appendLog("System", final.reply, "sys");
      if (final.geojson && final.value_summary) {
        setIndexLayer(final.index_col, final.value_summary, final.geojson, final.index_name);
        setSpatialTarget(final.spatial_target_resolved, final.spatial_filter_radius_m, final.spatial_target);
      } else if (final.index_col) {
        await loadExistingIndex(final.index_col);
      }
      await refreshIndices();
      window.StreetRAG?.showProgress?.(100, "Index created");
    } catch (e) {
      appendLog("Error", String(e.message || e), "err");
      wrap.querySelectorAll(".proposal-card").forEach((b) => { b.disabled = false; });
      btn.innerHTML = "<strong>" + escapeHtml(p.name || indexName) + "</strong><p>" + escapeHtml(p.rationale || "") + "</p>";
    }
  }
  function appendActivity(line) {
    if (!activity) return;
    const div = document.createElement("div");
    div.className = "act-line";
    div.textContent = line;
    activity.appendChild(div);
    activity.scrollTop = activity.scrollHeight;
  }
  function appendLog(role, text, cssClass) {
    const div = document.createElement("div");
    div.className = "msg";
    const r = document.createElement("div");
    r.className = "role " + (cssClass || "");
    r.textContent = role;
    const t = document.createElement("div");
    t.textContent = text || "";
    t.style.whiteSpace = "pre-wrap";
    div.appendChild(r);
    div.appendChild(t);
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  // -------------------------------------------------------------------
  // Map
  // -------------------------------------------------------------------
  const map = new maplibregl.Map({
    container: "map",
    style: {
      version: 8,
      sources: {
        basemap: {
          type: "raster",
          tiles: ["https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"],
          tileSize: 256,
          attribution: "© CARTO",
        },
      },
      layers: [{ id: "basemap-raster", type: "raster", source: "basemap" }],
    },
    center: [103.85, 1.29],
    zoom: 11.2,
  });
  map.addControl(new maplibregl.NavigationControl(), "top-left");
  loadSavedSelects();

  map.on("load", async () => {
    applyBasemap((basemapSelect && basemapSelect.value) || "carto-voyager");
    await loadBaseStreets();
    await refreshIndices();
    // Apply any hash state (index=…)
    const st = readHashState();
    if (st.index) {
      loadExistingIndex(st.index).catch(() => {});
    }
  });

  function applyBasemap(id) {
    const spec = BASEMAPS[id] || BASEMAPS["carto-voyager"];
    if (!map.isStyleLoaded()) return;
    const src = map.getSource("basemap");
    if (src && typeof src.setTiles === "function") {
      src.setTiles(spec.tiles);
    } else {
      if (map.getLayer("basemap-raster")) map.removeLayer("basemap-raster");
      if (map.getSource("basemap")) map.removeSource("basemap");
      map.addSource("basemap", { type: "raster", tiles: spec.tiles, tileSize: 256, attribution: spec.attribution });
      const def = { id: "basemap-raster", type: "raster", source: "basemap" };
      if (map.getLayer("street-lines")) map.addLayer(def, "street-lines");
      else map.addLayer(def);
    }
    if (map.getLayer("street-lines")) {
      const p = contextStreetPaint();
      map.setPaintProperty("street-lines", "line-color", p["line-color"]);
      map.setPaintProperty("street-lines", "line-opacity", p["line-opacity"]);
    }
  }

  async function loadBaseStreets() {
    try {
      const r = await fetch(apiPath("/api/edges-geojson?simplify=18"));
      if (!r.ok) throw new Error(await r.text());
      const d = await r.json();
      if (map.getLayer("street-lines")) map.removeLayer("street-lines");
      if (map.getSource("streets")) map.removeSource("streets");
      map.addSource("streets", { type: "geojson", data: d.geojson });
      const ctx = contextStreetPaint();
      map.addLayer({
        id: "street-lines", type: "line", source: "streets",
        paint: { "line-color": ctx["line-color"], "line-width": ctx["line-width"], "line-opacity": ctx["line-opacity"] },
      });
      if (d.geojson && d.geojson.features && d.geojson.features.length) {
        const b = computeBounds(d.geojson);
        if (b) map.fitBounds(b, { padding: 40, maxZoom: 12 });
      }
    } catch (e) {
      appendLog("Error", "Failed to load network: " + e, "err");
    }
  }

  function computeBounds(fc) {
    let minX = 180, minY = 90, maxX = -180, maxY = -90;
    (fc.features || []).forEach((f) => {
      const g = f.geometry;
      if (!g || !g.coordinates) return;
      const it = g.type === "MultiLineString" ? [].concat(...g.coordinates) : g.coordinates;
      it.forEach(([x, y]) => {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      });
    });
    if (minX > maxX) return null;
    return [[minX, minY], [maxX, maxY]];
  }

  function setIndexLayer(col, summary, geojson, indexName) {
    if (!geojson || !geojson.features || !geojson.features.length) {
      hideColorbar();
      appendLog("Error", `No map data for column: ${col}`, "err");
      return;
    }
    const [lo, hi] = stretchRange(summary);
    lastIndex = { col, summary, vmin: lo, vmax: hi, indexName: indexName || col };
    const paint = paintForIndex(lo, hi, getMode());
    if (map.getSource("index")) {
      map.getSource("index").setData(geojson);
    } else {
      map.addSource("index", { type: "geojson", data: geojson });
    }
    if (!map.getLayer("index-lines")) {
      map.addLayer({ id: "index-lines", type: "line", source: "index", paint });
    } else {
      map.setPaintProperty("index-lines", "line-color", paint["line-color"]);
      map.setPaintProperty("index-lines", "line-width", paint["line-width"]);
      map.setPaintProperty("index-lines", "line-opacity", paint["line-opacity"]);
    }
    showColorbar(col, summary, indexName || col);
    updateHashIndex(col);
  }

  async function visualizeColumn(col, indexName) {
    if (!col) return;
    try {
      appendLog("System", `Loading map layer: ${col}…`, "sys");
      const r = await fetch(
        apiPath("/api/edges-geojson?color_by=" + encodeURIComponent(col) + "&simplify=8")
      );
      if (!r.ok) {
        const err = await r.text();
        appendLog("Error", `Cannot map ${col}: ${err}`, "err");
        return;
      }
      const data = await r.json();
      setIndexLayer(col, data.value_summary || {}, data.geojson, indexName || col);
    } catch (e) {
      appendLog("Error", `Map failed: ${e.message || e}`, "err");
    }
  }

  // -------------------------------------------------------------------
  // Spatial target overlay (named POI + optional filter radius)
  // -------------------------------------------------------------------
  let targetMarker = null;
  function clearSpatialTarget() {
    if (targetMarker) { targetMarker.remove(); targetMarker = null; }
    if (map.getLayer("target-buffer-fill")) map.removeLayer("target-buffer-fill");
    if (map.getLayer("target-buffer-line")) map.removeLayer("target-buffer-line");
    if (map.getSource("target-buffer")) map.removeSource("target-buffer");
  }

  function circleGeoJSON(lon, lat, radiusM, steps = 96) {
    // approx equirectangular small-circle in degrees
    const dlat = radiusM / 111320;
    const dlon = radiusM / (111320 * Math.cos((lat * Math.PI) / 180));
    const ring = [];
    for (let i = 0; i <= steps; i++) {
      const a = (i / steps) * 2 * Math.PI;
      ring.push([lon + Math.cos(a) * dlon, lat + Math.sin(a) * dlat]);
    }
    return {
      type: "FeatureCollection",
      features: [{ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [ring] } }],
    };
  }

  function setSpatialTarget(resolved, radiusM, query) {
    clearSpatialTarget();
    if (!resolved || !resolved.found || !resolved.centroid_lonlat) return;
    const [lon, lat] = resolved.centroid_lonlat;
    const el = document.createElement("div");
    el.className = "target-marker";
    el.title = resolved.display_name || query || "target";
    el.style.cssText =
      "width:22px;height:22px;border-radius:50%;background:#ff3d3d;" +
      "border:3px solid #fff;box-shadow:0 0 0 2px #ff3d3d,0 1px 4px rgba(0,0,0,0.4);" +
      "cursor:help;";
    targetMarker = new maplibregl.Marker({ element: el })
      .setLngLat([lon, lat])
      .setPopup(new maplibregl.Popup({ offset: 14 }).setHTML(
        `<div style="font-weight:600;margin-bottom:2px">${escapeHtml(query || "target")}</div>` +
        `<div style="font-size:12px;color:#444">${escapeHtml(resolved.display_name || "")}</div>` +
        (radiusM ? `<div style="font-size:12px;color:#444">filter radius: ${Number(radiusM).toFixed(0)} m</div>` : "")
      ))
      .addTo(map);

    if (radiusM && radiusM > 0) {
      map.addSource("target-buffer", { type: "geojson", data: circleGeoJSON(lon, lat, radiusM) });
      map.addLayer({
        id: "target-buffer-fill", type: "fill", source: "target-buffer",
        paint: { "fill-color": "#ff3d3d", "fill-opacity": 0.06 },
      });
      map.addLayer({
        id: "target-buffer-line", type: "line", source: "target-buffer",
        paint: { "line-color": "#ff3d3d", "line-width": 1.5, "line-opacity": 0.7, "line-dasharray": [2, 2] },
      });
    }
  }

  function setRouteLayer(geojson, note) {
    if (map.getLayer("route-line")) map.removeLayer("route-line");
    if (map.getSource("route")) map.removeSource("route");
    if (!geojson || !geojson.features || !geojson.features.length) return;
    map.addSource("route", { type: "geojson", data: geojson });
    map.addLayer({
      id: "route-line", type: "line", source: "route",
      paint: { "line-color": "#dc2626", "line-width": 5, "line-opacity": 0.92 },
    });
    const f = geojson.features[0];
    if (f && f.geometry && f.geometry.coordinates) {
      const coords = (f.geometry.type === "MultiLineString")
        ? [].concat(...f.geometry.coordinates)
        : f.geometry.coordinates;
      let x0 = 180, y0 = 90, x1 = -180, y1 = -90;
      coords.forEach(([x, y]) => {
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      });
      map.fitBounds([[x0, y0], [x1, y1]], { padding: 48, maxZoom: 14 });
    }
    if (note) appendLog("System", "Weight: " + note, "sys");
  }

  // -------------------------------------------------------------------
  // Hover popup with feature contributions
  // -------------------------------------------------------------------
  let hoverPopup = null;
  let hoverFetchId = 0;

  map.on("mousemove", "index-lines", (e) => {
    if (routeMode.active) return;
    if (!e.features || !e.features.length || !lastIndex) return;
    map.getCanvas().style.cursor = "pointer";
    const f = e.features[0];
    const v = f.properties && f.properties.v;
    const ll = e.lngLat;
    const myId = ++hoverFetchId;
    fetch(apiPath(`/api/edge-info?lon=${ll.lng}&lat=${ll.lat}&index_col=${encodeURIComponent(lastIndex.col)}`))
      .then((r) => r.ok ? r.json() : null)
      .then((info) => {
        if (myId !== hoverFetchId || !info) return;
        const name = info.name || `${info.highway || "edge"} #${info.edge_id}`;
        let html =
          `<div style="font-weight:600;margin-bottom:2px">${escapeHtml(name)}</div>` +
          `<div class="popup-row"><span class="k">${escapeHtml(lastIndex.col)}</span><span class="v">${fmtNum(info.index_value ?? v)}</span></div>`;
        if (info.length) {
          html += `<div class="popup-row"><span class="k">length</span><span class="v">${Number(info.length).toFixed(0)} m</span></div>`;
        }
        if (info.top_features && info.top_features.length) {
          html += `<div class="popup-mini">Top contributions:<ul>`;
          for (const t of info.top_features.slice(0, 3)) {
            html += `<li>${escapeHtml(t.feature)} · w=${fmtNum(t.weight)} · x=${fmtNum(t.raw_value)}</li>`;
          }
          html += `</ul></div>`;
        }
        if (hoverPopup) hoverPopup.remove();
        hoverPopup = new maplibregl.Popup({ closeButton: false, offset: 8, maxWidth: "300px" })
          .setLngLat(ll).setHTML(html).addTo(map);
      })
      .catch(() => {});
  });
  map.on("mouseleave", "index-lines", () => {
    map.getCanvas().style.cursor = "";
    hoverFetchId++;
    if (hoverPopup) { hoverPopup.remove(); hoverPopup = null; }
  });
  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  // -------------------------------------------------------------------
  // Click-to-route
  // -------------------------------------------------------------------
  const routeMode = { active: false, from: null, fromMarker: null, toMarker: null };
  function setRouteModeActive(on) {
    routeMode.active = !!on;
    routeModeBtn.setAttribute("aria-pressed", routeMode.active ? "true" : "false");
    routeBanner.classList.toggle("visible", routeMode.active);
    map.getCanvas().style.cursor = routeMode.active ? "crosshair" : "";
    if (!routeMode.active) {
      if (routeMode.fromMarker) { routeMode.fromMarker.remove(); routeMode.fromMarker = null; }
      if (routeMode.toMarker) { routeMode.toMarker.remove(); routeMode.toMarker = null; }
      routeMode.from = null;
    }
  }
  routeModeBtn.addEventListener("click", () => setRouteModeActive(!routeMode.active));
  routeCancelBtn.addEventListener("click", () => setRouteModeActive(false));

  map.on("click", async (e) => {
    if (!routeMode.active) return;
    const { lng, lat } = e.lngLat;
    if (!routeMode.from) {
      routeMode.from = [lng, lat];
      routeMode.fromMarker = new maplibregl.Marker({ color: "#16a34a" }).setLngLat([lng, lat]).addTo(map);
      appendLog("Route", `Start: ${lng.toFixed(5)}, ${lat.toFixed(5)}`, "sys");
      return;
    }
    const [flon, flat] = routeMode.from;
    routeMode.toMarker = new maplibregl.Marker({ color: "#dc2626" }).setLngLat([lng, lat]).addTo(map);
    const weightCol = routeWeightSelect.value || "length";
    try {
      const res = await fetch(apiPath("/api/route"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          from_lon: flon, from_lat: flat, to_lon: lng, to_lat: lat,
          weight_col: weightCol,
          weight_mode: weightCol === "length" ? "length" : "length_over_index",
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const d = await res.json();
      setRouteLayer(d.geojson, d.weight_note);
      appendLog("System", `Route ~${Math.round(d.length_m)} m （Weight: ${d.weight_note}）`, "sys");
    } catch (err) {
      appendLog("Error", String(err.message || err), "err");
    } finally {
      setRouteModeActive(false);
    }
  });

  // -------------------------------------------------------------------
  // Indices drawer
  // -------------------------------------------------------------------
  async function refreshIndices() {
    try {
      const r = await fetch(apiPath("/api/indices"));
      if (!r.ok) throw new Error(await r.text());
      const d = await r.json();
      renderIndicesList(d.indices || []);
      // Populate route weight select with index columns
      const opts = ['<option value="length">length(shortest)</option>'];
      (d.indices || []).forEach((it) => {
        opts.push(`<option value="${escapeHtml(it.index_col)}">${escapeHtml(it.index_name || it.index_col)}(comfort)</option>`);
      });
      routeWeightSelect.innerHTML = opts.join("");
    } catch (e) {
      indicesList.innerHTML = `<div style="color:#dc2626;font-size:12px;padding:8px;">Failed to load: ${escapeHtml(e.message || e)}</div>`;
    }
  }

  function renderIndicesList(items) {
    if (!items.length) {
      indicesList.innerHTML = `<div style="color:var(--muted);font-size:12px;text-align:center;padding:16px;">No saved indices yet.<br/>Create one in the chat below.</div>`;
      return;
    }
    indicesList.innerHTML = items.map((it) => {
      const t = it.timestamp ? new Date(it.timestamp).toLocaleString() : "";
      const stats = it.statistics || {};
      const morans = (it.morans_i != null) ? ` · I=${Number(it.morans_i).toFixed(2)}` : "";
      return (
        `<div class="index-item index-card" data-col="${escapeHtml(it.index_col)}">` +
          `<div class="name">${escapeHtml(it.index_name || it.index_col)}</div>` +
          `<div class="query">${escapeHtml(it.original_query || "")}</div>` +
          `<div class="meta">${escapeHtml(it.operator || "weighted_sum")} · ${escapeHtml(it.normalization || "robust")} · n=${stats.count != null ? stats.count.toLocaleString() : "?"}${morans} · ${escapeHtml(t)}</div>` +
        `</div>`
      );
    }).join("");
    Array.from(indicesList.querySelectorAll(".index-item")).forEach((el) => {
      el.addEventListener("click", () => {
        const col = el.dataset.col;
        Array.from(indicesList.querySelectorAll(".index-item")).forEach((x) => x.classList.remove("active"));
        el.classList.add("active");
        loadExistingIndex(col);
      });
    });
  }

  async function loadExistingIndex(col) {
    try {
      const r = await fetch(apiPath(`/api/index/${encodeURIComponent(col)}`));
      if (!r.ok) throw new Error(await r.text());
      const d = await r.json();
      setIndexLayer(d.index_col, d.value_summary, d.geojson, d.index_name);
      setSpatialTarget(d.spatial_target_resolved, d.spatial_filter_radius_m, d.spatial_target);
      if (d.reply) appendLog("System (saved)", d.reply, "sys");
      if (d.morans_i && d.morans_i.I != null) {
        appendLog("Spatial", `Moran's I = ${Number(d.morans_i.I).toFixed(4)} (n=${d.morans_i.n})`, "sys");
      }
      if (d.spatial_target_resolved && d.spatial_target_resolved.found) {
        const r2 = d.spatial_filter_radius_m ? ` · radius ${Number(d.spatial_filter_radius_m).toFixed(0)} m` : "";
        appendLog("Target", `${d.spatial_target} → ${d.spatial_target_resolved.display_name || ""}${r2}`, "sys");
      }
    } catch (e) {
      appendLog("Error", "Failed to load index: " + (e.message || e), "err");
    }
  }

  // Result card for analysis-only skills (correlate, multiscale_profile…)
  // that return numbers + text instead of a map layer.
  const SKILL_NAMES = {
    correlate: "Correlation",
    multiscale_profile: "Multiscale profile",
    cluster_lisa: "LISA clusters",
    composite_index: "Composite index",
    poi_review_search: "Review search",
    answer_directly: "Direct answer",
  };
  function renderAnalysisResult(d, opts) {
    const o = opts || {};
    const title = SKILL_NAMES[d.skill_name] || d.skill_name || "Analysis";
    if (d.reply && !o.skipReply) appendLog(title, d.reply, "sys");
    const ev = d.narrative_evidence || {};
    const lines = [];
    const stats = d.statistics || {};

    // correlate
    if (ev.pearson != null) {
      lines.push(`${ev.feature_a} × ${ev.feature_b}`);
      lines.push(`Pearson r = ${Number(ev.pearson).toFixed(3)} · Spearman ρ = ${Number(ev.spearman).toFixed(3)}` +
        (stats.n != null ? ` · n = ${Number(stats.n).toLocaleString()}` : ""));
    }

    // multiscale_profile
    if (ev.scale_profiles && typeof ev.scale_profiles === "object") {
      for (const [scale, p] of Object.entries(ev.scale_profiles)) {
        const lw = p.length_weighted || {};
        const st = p.index_stats || {};
        const km = lw.total_length_m != null ? (lw.total_length_m / 1000).toFixed(1) + " km" : "";
        const mean = st.mean != null ? `mean=${Number(st.mean).toFixed(3)}` : "";
        lines.push(`${scale} (${p.radius_m}m) top-k: ${mean} ${km}`.trim());
      }
    }
    if (ev.correlation_matrix && typeof ev.correlation_matrix === "object") {
      const cols = Object.keys(ev.correlation_matrix);
      const target = d.index_col;
      if (target && ev.correlation_matrix[target]) {
        for (const [other, v] of Object.entries(ev.correlation_matrix[target])) {
          if (other !== target && v != null) lines.push(`corr(${target}, ${other}) = ${Number(v).toFixed(3)}`);
        }
      } else if (cols.length === 0) { /* noop */ }
    }

    // cluster_lisa
    if (ev.lisa_summary && typeof ev.lisa_summary === "object") {
      const parts = Object.entries(ev.lisa_summary).map(([k, v]) => `${k}=${v}`);
      if (parts.length) lines.push("LISA: " + parts.join(" · "));
    }
    const mi = (ev.morans_i && ev.morans_i.I != null) ? ev.morans_i
             : (d.morans_i && d.morans_i.I != null) ? d.morans_i : null;
    if (mi) lines.push(`Moran's I = ${Number(mi.I).toFixed(4)} (n=${mi.n})`);

    // poi_review_search
    if (ev.review_snippets && ev.review_snippets.length) {
      for (const s of ev.review_snippets.slice(0, 5)) {
        const name = s.poi_name || s.poi_id || "POI";
        const text = (s.text || "").slice(0, 120);
        lines.push(`「${name}」: ${text}${text.length >= 120 ? "…" : ""}`);
      }
    }

    if (lines.length) appendLog("Evidence", lines.join("\n"), "sys");
  }

  refreshIndicesBtn.addEventListener("click", refreshIndices);
  drawerToggle.addEventListener("click", () => {
    drawer.classList.toggle("collapsed");
    localStorage.setItem(LS.DRAWER, drawer.classList.contains("collapsed") ? "1" : "0");
    updateDrawerToggle();
  });

  // -------------------------------------------------------------------
  // Selects (basemap / colormap / stretch)
  // -------------------------------------------------------------------
  basemapSelect.addEventListener("change", () => {
    localStorage.setItem(LS.BASE, basemapSelect.value);
    applyBasemap(basemapSelect.value);
    updateHashIndex(lastIndex ? lastIndex.col : null);
  });
  colormapSelect.addEventListener("change", () => {
    localStorage.setItem(LS.COLOR, colormapSelect.value);
    restyleIndex();
    updateHashIndex(lastIndex ? lastIndex.col : null);
  });
  stretchSelect.addEventListener("change", () => {
    localStorage.setItem(LS.STRETCH, stretchSelect.value);
    restyleIndex();
    updateHashIndex(lastIndex ? lastIndex.col : null);
  });
  function restyleIndex() {
    if (!lastIndex || !map.getLayer("index-lines")) return;
    const [lo, hi] = stretchRange(lastIndex.summary);
    lastIndex.vmin = lo; lastIndex.vmax = hi;
    const p = paintForIndex(lo, hi, getMode());
    map.setPaintProperty("index-lines", "line-color", p["line-color"]);
    map.setPaintProperty("index-lines", "line-width", p["line-width"]);
    map.setPaintProperty("index-lines", "line-opacity", p["line-opacity"]);
    showColorbar(lastIndex.col, lastIndex.summary, lastIndex.indexName);
  }

  // -------------------------------------------------------------------
  // Export current index
  // -------------------------------------------------------------------
  exportBtn.addEventListener("click", async () => {
    if (!lastIndex) {
      appendLog("System", "Select or create an index before exporting.", "sys");
      return;
    }
    const r = await fetch(apiPath(`/api/edges-geojson?color_by=${encodeURIComponent(lastIndex.col)}&simplify=4`));
    if (!r.ok) {
      appendLog("Error", "Export failed: " + await r.text(), "err");
      return;
    }
    const d = await r.json();
    const blob = new Blob([JSON.stringify(d.geojson)], { type: "application/geo+json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${lastIndex.col}.geojson`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // -------------------------------------------------------------------
  // Data files transparency panel
  // -------------------------------------------------------------------
  const datasetsBox = document.getElementById("datasetsBox");
  const rescanBtn = document.getElementById("rescanBtn");
  const addCityBtn = document.getElementById("addCityBtn");
  const addCityModal = document.getElementById("addCityModal");
  const syntaxCityBtn = document.getElementById("syntaxCityBtn");
  const syntaxModal = document.getElementById("syntaxModal");
  const apiSettingsBtn = document.getElementById("apiSettingsBtn");
  const apiModal = document.getElementById("apiModal");
  const DS_BADGES = {
    network_active: ["net", "Active network"],
    network_candidate: ["cand", "Candidate"],
    integrated: ["ok", "Integrated"],
    pending_integration: ["warn", "Pending"],
    partial_integration: ["warn", "Partial"],
    converted: ["conv", "Converted"],
    not_integrated: ["no", "Not integrated"],
    text_only: ["txt", "Text indexed"],
  };

  const citySelect = document.getElementById("citySelect");

  function dsFileRow(f) {
    const [cls, label] = DS_BADGES[f.role] || ["no", f.role];
    const metaParts = [];
    if (f.method) metaParts.push(f.method);
    if (f.size_mb != null) metaParts.push(`${f.size_mb}MB`);
    const colCount = f.columns?.length || f.columns_in_gpkg || f.registered_columns?.length;
    if (colCount) metaParts.push(`${colCount} cols`);
    if (f.note) metaParts.push(f.note);
    let html = `<div class="feat-row ds-row" data-file="${escapeHtml(f.name)}">` +
      `<span class="fname">${escapeHtml(f.name)}</span>`;
    if (f.missing) html += `<span class="ds-badge miss">Missing</span>`;
    html += `<span class="ds-badge ${cls}">${label}</span>`;
    if (metaParts.length) {
      html += `<span class="row-meta">${escapeHtml(metaParts.join(" · "))}</span>`;
    }
    if (f.text_columns && f.text_columns.length) {
      html += `<span class="ds-badge txt">Text ${f.text_columns.length}</span>`;
    }
    if (f.text_index) {
      const ti = f.text_index;
      html += `<span class="ds-badge ok">Reviews ${ti.n_chunks || 0}</span>`;
    }
    if (f.columns && f.columns.length) {
      html += `<details class="ds-cols" style="flex:1 1 100%;"><summary>${f.columns.length} feature columns</summary>` +
        `<div class="cols">${f.columns.map(escapeHtml).join("<br/>")}</div></details>`;
    }
    if (f.text_index?.text_columns?.length) {
      html += `<details class="ds-cols" style="flex:1 1 100%;"><summary>Text columns</summary>` +
        `<div class="cols">${f.text_index.text_columns.map(escapeHtml).join("<br/>")}</div></details>`;
    }
    if (f.role !== "network_active" && f.role !== "network_candidate" && !f.missing) {
      html += `<div class="feat-actions">` +
        `<button type="button" class="btn-sm ds-integrate" data-file="${escapeHtml(f.name)}">Integrate</button>` +
        `<button type="button" class="btn-sm ds-reintegrate" data-file="${escapeHtml(f.name)}">Reload</button>` +
        `<button type="button" class="btn-sm ds-delete" data-file="${escapeHtml(f.name)}">Delete</button>` +
        `<button type="button" class="btn-sm ds-addctx" data-file="${escapeHtml(f.name)}">+ Chat</button>` +
        `</div>`;
    }
    return html + `</div>`;
  }

  function bindDatasetActions(root) {
    if (!root) return;
    root.querySelectorAll(".ds-integrate").forEach((btn) => {
      btn.addEventListener("click", () => runFileIntegrate(btn.dataset.file, false));
    });
    root.querySelectorAll(".ds-reintegrate").forEach((btn) => {
      btn.addEventListener("click", () => runFileIntegrate(btn.dataset.file, true));
    });
    root.querySelectorAll(".ds-delete").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (!confirm("Delete " + btn.dataset.file + "?")) return;
        const r = await fetch(apiPath("/api/files/" + encodeURIComponent(btn.dataset.file)), { method: "DELETE" });
        if (!r.ok) { appendLog("Error", await r.text(), "err"); return; }
        refreshDatasets();
      });
    });
    root.querySelectorAll(".ds-addctx").forEach((btn) => {
      btn.addEventListener("click", () => {
        window.StreetRAG?.addContext?.({ type: "file", name: btn.dataset.file });
      });
    });
  }

  function formatIntegrateProgress(ev) {
    const d = ev.detail || {};
    if (d.message) return d.message;
    const parts = [d.phase || "integrate"];
    if (d.current != null && d.total != null) parts.push(`${d.current}/${d.total}`);
    if (d.pct != null) parts.push(`${Math.round(d.pct)}%`);
    return parts.join(" · ");
  }

  function setIntegrateButtonsDisabled(disabled) {
    document.querySelectorAll(".ds-integrate, .ds-reintegrate").forEach((btn) => {
      btn.disabled = disabled;
      btn.style.opacity = disabled ? "0.5" : "";
    });
  }

  async function runFileIntegrate(filename, reintegrate) {
    const url = apiPath("/api/files/" + encodeURIComponent(filename) + (reintegrate ? "/reintegrate-stream" : "/integrate-stream"));
    const action = reintegrate ? "Reintegrating" : "Integrating";
    appendLog("System", `${action} ${filename}… (this may take several minutes for large POI datasets)`, "sys");
    setIntegrateButtonsDisabled(true);
    window.StreetRAG?.showProgress?.(2, `${action} ${filename}…`);
    try {
      const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      if (!res.ok) throw new Error(await res.text());
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const p = parseSseBuffer(buf);
        buf = p.rest;
        for (const ev of p.events) {
          if (ev.type === "progress" && ev.detail) {
            const msg = formatIntegrateProgress(ev);
            const pct = ev.detail.pct ?? 50;
            window.StreetRAG?.showProgress?.(pct, msg);
          } else if (ev.type === "error") throw new Error(ev.message);
          else if (ev.type === "result") {
            const cols = ev.data?.columns_added || [];
            appendLog("System", `Integration done: ${cols.length} columns`, "sys");
          }
        }
      }
      window.StreetRAG?.hideProgress?.();
      refreshDatasets();
      window.refreshFeatures?.();
    } catch (e) {
      appendLog("Error", String(e.message || e), "err");
      window.StreetRAG?.hideProgress?.();
    } finally {
      setIntegrateButtonsDisabled(false);
    }
  }

  function renderDatasetsHtml(d) {
    const net = d.network || {};
    const fc = d.feature_counts || {};
    let html = "";
    if (d.data_dir) {
      html += `<div class="ds-note" style="margin-top:0;border:0;padding:0;">📁 ${escapeHtml(d.data_dir)}</div>`;
    }
    html += `<div class="ds-net">${escapeHtml(net.file || "No network set")}` +
      (net.n_edges ? ` · ${Number(net.n_edges).toLocaleString()} edges` : "") +
      (net.syntax_radii && net.syntax_radii.length ? ` · syntax radii ${net.syntax_radii.join("/")}m` : "") +
      `</div>`;
    html += `<div class="ds-counts">Features ${fc.total ?? "?"}: syntax ${fc.space_syntax ?? 0} · integrated ${fc.integrated ?? 0} · Composite index ${fc.composite_index ?? 0}` +
      (fc.review_chunks ? ` · reviews ${fc.review_chunks}` : "") + `</div>`;
    if (net.syntax_radii && net.syntax_radii.length) {
      const synNote = (fc.space_syntax || 0) > 0
        ? `Syntax computed (${fc.space_syntax} cols)`
        : `Syntax radii ${net.syntax_radii.join("/")}m set, not computed — click Syntax`;
      html += `<div class="ds-note" style="margin-top:4px;">${escapeHtml(synNote)}</div>`;
    }
    if (!d.ready) {
      html += `<div class="ds-note">City not ready. Use + City to download a network.</div>`;
      return html;
    }
    if ((d.networks || []).length) {
      html += `<div class="feat-group"><div class="ds-group">Street network</div>`;
      for (const f of d.networks) html += dsFileRow(f);
      html += `</div>`;
    }
    const groups = [
      [`Integrated sources`, (d.sources || []).filter((f) => f.role === "integrated" || f.role === "partial_integration")],
      [`Pending integration`, (d.sources || []).filter((f) => f.role === "pending_integration")],
      ["Text indexed only", (d.sources || []).filter((f) => f.role === "text_only")],
      ["Converted", (d.sources || []).filter((f) => f.role === "converted")],
      ["Not integrated", (d.sources || []).filter((f) => f.role === "not_integrated")],
    ];
    for (const [title, items] of groups) {
      if (!items.length) continue;
      html += `<div class="feat-group"><div class="ds-group">${title} (${items.length})</div>`;
      for (const f of items) html += dsFileRow(f);
      html += `</div>`;
    }
    if (d.multi_city_note) {
      html += `<div class="ds-note">${escapeHtml(d.multi_city_note)}</div>`;
    }
    return html;
  }

  function populateCitySelect(d) {
    if (!citySelect) return;
    const details = d.city_details || (d.cities || []).map((n) => ({ name: n }));
    citySelect.innerHTML = details.map((c) => {
      const name = typeof c === "string" ? c : c.name;
      let meta = "";
      if (typeof c === "object") {
        const syn = c.n_syntax ? ` · syntax${c.n_syntax}` : "";
        meta = c.network ? ` (${c.n_features || 0} feat${syn})` : "";
      }
      return `<option value="${escapeHtml(name)}"${name === d.city ? " selected" : ""}>🏙 ${escapeHtml(name)}${escapeHtml(meta)}</option>`;
    }).join("");
  }

  async function fetchDatasets() {
    const r = await fetch(apiPath("/api/datasets"));
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }

  async function refreshDatasets() {
    if (!datasetsBox) return;
    try {
      const d = await fetchDatasets();
      populateCitySelect(d);
      datasetsBox.innerHTML = renderDatasetsHtml(d);
      bindDatasetActions(datasetsBox);
    } catch (e) {
      datasetsBox.innerHTML = `<span style="color:#dc2626">Failed to load: ${escapeHtml(e.message || String(e))}</span>`;
    }
  }

  function openModal(el) {
    if (!el) return;
    el.classList.add("open");
    el.setAttribute("aria-hidden", "false");
  }
  function closeModal(el) {
    if (!el) return;
    el.classList.remove("open");
    el.setAttribute("aria-hidden", "true");
  }

  const CITY_STEP_LABELS = {
    download: "Download network",
    pois: "Download POIs",
    scan: "Scan directory",
    integrate: "Integrate data",
    activate: "Activate city",
    bootstrap: "Bootstrap",
    syntax: "Space Syntax",
  };

  async function loadSyntaxConfig() {
    const r = await fetch(apiPath("/api/syntax/config"));
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }

  async function openSyntaxModal() {
    const statusEl = document.getElementById("syntaxModalStatus");
    const radiiEl = document.getElementById("syntaxRadii");
    const progEl = document.getElementById("syntaxProgress");
    if (progEl) { progEl.style.display = "none"; progEl.textContent = ""; }
    openModal(syntaxModal);
    if (statusEl) statusEl.textContent = "Loading…";
    try {
      const cfg = await loadSyntaxConfig();
      if (radiiEl) radiiEl.value = (cfg.radii || cfg.default_radii || [500, 1500, 4500]).join(",");
      const title = document.getElementById("syntaxModalTitle");
      if (title) title.textContent = `Space Syntax · ${cfg.city || "—"}`;
      if (statusEl) {
        statusEl.textContent = cfg.has_syntax
          ? `Computed ${cfg.syntax_columns} syntax columns · radii ${(cfg.radii || []).join("/")} m`
          : `Syntax not computed · radii configured ${(cfg.radii || []).join("/")} m`;
      }
    } catch (e) {
      if (statusEl) statusEl.textContent = "Failed to load: " + (e.message || e);
    }
  }

  async function submitRunSyntax() {
    const radiiEl = document.getElementById("syntaxRadii");
    const progEl = document.getElementById("syntaxProgress");
    const runBtn = document.getElementById("syntaxRunBtn");
    const radii = (radiiEl && radiiEl.value || "").trim();
    if (runBtn) runBtn.disabled = true;
    if (progEl) { progEl.style.display = "block"; progEl.textContent = "Starting syntax…"; }
    appendLog("System", `Running space syntax: ${radii || "default radii"}`, "sys");
    clearActivity();
    try {
      const r = await fetch(apiPath("/api/syntax/run-stream"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ radii: radii || null }),
      });
      if (!r.ok) throw new Error(await r.text());
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parsed = parseSseBuffer(buf);
        buf = parsed.rest;
        for (const ev of parsed.events) {
          if (ev.type === "progress") {
            const msg = (ev.detail && ev.detail.message) || ev.step;
            appendActivity(msg);
            if (progEl) progEl.textContent = msg;
          } else if (ev.type === "error") {
            throw new Error(ev.message || "Syntax failed");
          } else if (ev.type === "result") {
            const n = ev.data && ev.data.syntax_columns;
            appendLog("System", `Syntax complete${n != null ? `, ${n} columns` : ""}`, "sys");
          }
        }
      }
      closeModal(syntaxModal);
      refreshDatasets();
      await loadBaseStreets();
      await refreshIndices();
    } catch (e) {
      appendLog("Error", "Syntax failed: " + (e.message || e), "err");
      if (progEl) progEl.textContent = String(e.message || e);
    } finally {
      if (runBtn) runBtn.disabled = false;
    }
  }

  async function submitAddCity() {
    const queryEl = document.getElementById("addCityQuery");
    const slugEl = document.getElementById("addCitySlug");
    const netEl = document.getElementById("addCityNetwork");
    const poisEl = document.getElementById("addCityPois");
    const syntaxEl = document.getElementById("addCitySyntax");
    const radiiEl = document.getElementById("addCityRadii");
    const progEl = document.getElementById("addCityProgress");
    const submitBtn = document.getElementById("addCitySubmit");
    const q = (queryEl && queryEl.value || "").trim();
    if (!q) {
      if (progEl) { progEl.style.display = "block"; progEl.textContent = "Enter OSM place name"; }
      return;
    }
    if (submitBtn) submitBtn.disabled = true;
    if (progEl) { progEl.style.display = "block"; progEl.textContent = "Starting download…"; }
    appendLog("System", `Adding city: ${q}`, "sys");
    clearActivity();
    try {
      const r = await fetch(apiPath("/api/cities/add-stream"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          osm_query: q,
          city_slug: (slugEl && slugEl.value || "").trim() || null,
          network_type: (netEl && netEl.value) || "drive",
          with_pois: !!(poisEl && poisEl.checked),
          run_syntax: !!(syntaxEl && syntaxEl.checked),
          syntax_radii: (radiiEl && radiiEl.value || "").trim() || "500,1500,4500",
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      let finalCity = null;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parsed = parseSseBuffer(buf);
        buf = parsed.rest;
        for (const ev of parsed.events) {
          if (ev.type === "progress") {
            const label = CITY_STEP_LABELS[ev.step] || ev.step;
            const msg = (ev.detail && ev.detail.message) || label;
            appendActivity(`${label}: ${msg}`);
            if (progEl) progEl.textContent = `${label} — ${msg}`;
          } else if (ev.type === "error") {
            throw new Error(ev.message || "Add failed");
          } else if (ev.type === "result") {
            finalCity = ev.data && ev.data.city;
          }
        }
      }
      closeModal(addCityModal);
      appendLog("System", `City ${finalCity || q} added and activated`, "sys");
      location.reload();
    } catch (e) {
      appendLog("Error", "Failed to add city: " + (e.message || e), "err");
      if (progEl) progEl.textContent = String(e.message || e);
    } finally {
      if (submitBtn) submitBtn.disabled = false;
    }
  }

  if (addCityBtn) addCityBtn.addEventListener("click", () => {
    const progEl = document.getElementById("addCityProgress");
    if (progEl) { progEl.style.display = "none"; progEl.textContent = ""; }
    openModal(addCityModal);
  });
  ["addCityModalClose", "addCityCancel"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("click", () => closeModal(addCityModal));
  });
  if (addCityModal) {
    addCityModal.addEventListener("click", (e) => {
      if (e.target === addCityModal) closeModal(addCityModal);
    });
  }
  const addCitySubmit = document.getElementById("addCitySubmit");
  if (addCitySubmit) addCitySubmit.addEventListener("click", submitAddCity);

  if (syntaxCityBtn) syntaxCityBtn.addEventListener("click", openSyntaxModal);
  ["syntaxModalClose", "syntaxModalCancel"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("click", () => closeModal(syntaxModal));
  });
  if (syntaxModal) {
    syntaxModal.addEventListener("click", (e) => {
      if (e.target === syntaxModal) closeModal(syntaxModal);
    });
  }
  const syntaxRunBtn = document.getElementById("syntaxRunBtn");
  if (syntaxRunBtn) syntaxRunBtn.addEventListener("click", submitRunSyntax);

  async function refreshApiButtonStatus() {
    if (!apiSettingsBtn) return;
    try {
      const r = await fetch(apiPath("/api/settings"));
      if (!r.ok) return;
      const d = await r.json();
      apiSettingsBtn.textContent = d.configured ? "API ✓" : "API";
      apiSettingsBtn.title = d.configured
        ? `API configured (${d.source}: ${d.masked_key})`
        : "OpenAI API not set — click to configure";
      apiSettingsBtn.style.borderColor = d.configured ? "#15803d" : "#b91c1c";
      apiSettingsBtn.style.color = d.configured ? "#15803d" : "#b91c1c";
    } catch (_) {}
  }

  async function openApiModal() {
    const statusEl = document.getElementById("apiModalStatus");
    const keyEl = document.getElementById("apiKeyInput");
    const llmEl = document.getElementById("llmModelInput");
    const embEl = document.getElementById("embedModelInput");
    const progEl = document.getElementById("apiSaveProgress");
    if (progEl) { progEl.style.display = "none"; progEl.textContent = ""; }
    if (keyEl) keyEl.value = "";
    openModal(apiModal);
    if (statusEl) statusEl.textContent = "Loading…";
    try {
      const d = await (await fetch(apiPath("/api/settings"))).json();
      if (llmEl) llmEl.value = d.llm_model || "gpt-4o";
      if (embEl) embEl.value = d.embedding_model || "text-embedding-3-small";
      if (statusEl) {
        if (d.configured) {
          statusEl.textContent = `Configured · source ${d.source}${d.masked_key ? " · " + d.masked_key : ""}`
            + (d.note_env_overrides_local ? " (OPENAI_API_KEY env overrides local file)" : "");
        } else {
          statusEl.textContent = "API key not set. Save to enable chat and embedding retrieval.";
        }
      }
    } catch (e) {
      if (statusEl) statusEl.textContent = "Failed to load: " + (e.message || e);
    }
  }

  async function saveApiSettings(clearKey) {
    const keyEl = document.getElementById("apiKeyInput");
    const llmEl = document.getElementById("llmModelInput");
    const embEl = document.getElementById("embedModelInput");
    const progEl = document.getElementById("apiSaveProgress");
    const saveBtn = document.getElementById("apiSaveBtn");
    if (saveBtn) saveBtn.disabled = true;
    if (progEl) { progEl.style.display = "block"; progEl.textContent = "Saving…"; }
    try {
      const body = {
        llm_model: (llmEl && llmEl.value || "").trim(),
        embedding_model: (embEl && embEl.value || "").trim(),
        clear_api_key: !!clearKey,
      };
      if (!clearKey && keyEl && keyEl.value.trim()) {
        body.openai_api_key = keyEl.value.trim();
      } else if (!clearKey) {
        body.openai_api_key = "";
      }
      const r = await fetch(apiPath("/api/settings"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(await r.text());
      const d = await r.json();
      appendLog("System", clearKey ? "Cleared local API key" : "API settings saved", "sys");
      if (progEl) {
        progEl.textContent = d.configured
          ? `Configured (${d.source}${d.masked_key ? ": " + d.masked_key : ""})`
          : "API key not configured";
      }
      await refreshApiButtonStatus();
      if (!clearKey) closeModal(apiModal);
    } catch (e) {
      appendLog("Error", "Failed to save API settings: " + (e.message || e), "err");
      if (progEl) progEl.textContent = String(e.message || e);
    } finally {
      if (saveBtn) saveBtn.disabled = false;
    }
  }

  if (apiSettingsBtn) apiSettingsBtn.addEventListener("click", openApiModal);
  ["apiModalClose", "apiModalCancel"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("click", () => closeModal(apiModal));
  });
  if (apiModal) {
    apiModal.addEventListener("click", (e) => {
      if (e.target === apiModal) closeModal(apiModal);
    });
  }
  const apiSaveBtn = document.getElementById("apiSaveBtn");
  if (apiSaveBtn) apiSaveBtn.addEventListener("click", () => saveApiSettings(false));
  const apiClearKeyBtn = document.getElementById("apiClearKeyBtn");
  if (apiClearKeyBtn) {
    apiClearKeyBtn.addEventListener("click", () => {
      if (confirm("Clear locally saved API key?")) saveApiSettings(true);
    });
  }
  refreshApiButtonStatus();

  if (citySelect) {
    citySelect.addEventListener("change", async () => {
      const name = citySelect.value;
      try {
        const r = await fetch(apiPath("/api/cities/activate"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name }),
        });
        if (!r.ok) throw new Error(await r.text());
        location.reload();
      } catch (e) {
        appendLog("Error", "Failed to switch city: " + (e.message || e), "err");
      }
    });
  }

  if (rescanBtn) {
    rescanBtn.addEventListener("click", async () => {
      rescanBtn.disabled = true;
      datasetsBox.textContent = "Scanning…";
      try {
        const r = await fetch(apiPath("/api/scan"), { method: "POST" });
        if (!r.ok) throw new Error(await r.text());
        appendLog("System", "Data directory rescanned", "sys");
      } catch (e) {
        appendLog("Error", "Scan failed: " + (e.message || e), "err");
      } finally {
        rescanBtn.disabled = false;
        refreshDatasets();
      }
    });
  }

  refreshDatasets();

  // -------------------------------------------------------------------
  // Upload & integrate
  // -------------------------------------------------------------------
  const uploadInput = document.getElementById("uploadInput");
  const uploadBtn = document.getElementById("uploadBtn");
  const integratorSelect = document.getElementById("integratorSelect");

  if (uploadBtn && uploadInput) {
    uploadBtn.addEventListener("click", async () => {
      const file = uploadInput.files && uploadInput.files[0];
      if (!file) {
        appendLog("System", "Select a file first", "err");
        return;
      }
      uploadBtn.disabled = true;
      try {
        const fd = new FormData();
        fd.append("file", file);
        const up = await fetch(apiPath("/api/upload"), { method: "POST", body: fd });
        if (!up.ok) throw new Error(await up.text());
        const upData = await up.json();
        if (upData.suggested_method && integratorSelect) {
          integratorSelect.value = upData.suggested_method;
        }
        appendLog("System", `Uploaded ${upData.filename} (${upData.geometry_type})`, "sys");
        if (upData.text_detected_message) {
          appendLog("System", upData.text_detected_message, "sys");
        }
        const method = (integratorSelect && integratorSelect.value) || upData.suggested_method || "snap_nearest";
        const ig = await fetch(apiPath("/api/integrate"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename: upData.filename, method_type: method, layer: upData.layers && upData.layers[0] }),
        });
        if (!ig.ok) throw new Error(await ig.text());
        const igData = await ig.json();
        let msg = `Integration done: ${(igData.columns_added || []).join(", ") || "(no numeric columns)"}`;
        if (igData.n_review_records || igData.n_chunks) {
          msg += ` · review index ${igData.n_chunks || igData.n_review_records} chunks`;
        }
        appendLog("System", msg, "sys");
        refreshDatasets();
      } catch (err) {
        appendLog("Error", String(err.message || err), "err");
      } finally {
        uploadBtn.disabled = false;
      }
    });
  }

  // -------------------------------------------------------------------
  // Chat
  // -------------------------------------------------------------------
  function pushChatHistory(role, content) {
    const c = (content || "").trim();
    if (!c) return;
    window.__streetHistory = window.__streetHistory || [];
    window.__streetHistory.push({ role, content: c });
    if (window.__streetHistory.length > 24) {
      window.__streetHistory = window.__streetHistory.slice(-24);
    }
  }

  function parseSseBuffer(buf) {
    const events = [];
    let rest = buf;
    let idx;
    while ((idx = rest.indexOf("\n\n")) >= 0) {
      const chunk = rest.slice(0, idx);
      rest = rest.slice(idx + 2);
      if (chunk.startsWith("data: ")) {
        try { events.push(JSON.parse(chunk.slice(6))); } catch (_) {}
      }
    }
    return { events, rest };
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = (input.value || "").trim();
    if (!text) return;
    window.__lastUserQuery = text;
    input.value = "";
    goBtn.disabled = true;
    clearActivity();
    appendLog("You", text, "you");
    pushChatHistory("user", text);
    const t0 = Date.now();
    let sawProposals = false;

    try {
      const res = await fetch(apiPath("/api/chat-stream"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          run_route_on_coords: true,
          context: window.StreetRAG?.getContext?.() || [],
          history: window.StreetRAG?.getHistory?.() || [],
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      if (!res.body) throw new Error("Empty response body");

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buffer = "";
      let final = null;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) {
          if (buffer.trim()) {
            const tail = buffer.trim();
            if (tail.startsWith("data: ")) {
              try {
                const ev = JSON.parse(tail.slice(6));
                if (ev.type === "result" && ev.data) final = ev.data;
                if (ev.type === "error") appendLog("System", "Error: " + (ev.message || ""), "err");
              } catch (_) {}
            }
          }
          break;
        }
        buffer += dec.decode(value, { stream: true });
        const p = parseSseBuffer(buffer);
        buffer = p.rest;
        for (const ev of p.events) {
          if (ev.type === "text_delta") {
            appendStreamText(ev.delta || "");
          } else if (ev.type === "thinking_delta") {
            appendThinking(ev.delta || "");
          } else if (ev.type === "tool_use") {
            if (ev.name !== "propose_index_options") {
              appendToolCard(ev.name, ev.arguments, "running");
            }
          } else if (ev.type === "tool_result") {
            if (ev.name !== "propose_index_options") {
              appendToolCard(ev.name, ev.result, "done");
            }
          } else if (ev.type === "visualize" && ev.column) {
            visualizeColumn(ev.column, ev.index_name);
          } else if (ev.type === "table") {
            window.StreetRAG?.showTable?.(ev);
          } else if (ev.type === "proposals") {
            sawProposals = true;
            if (streamBubble) {
              streamBubble.remove();
              streamBubble = null;
            }
            appendProposals(ev.proposals || []);
          } else if (ev.type === "progress") {
            appendActivity("· " + formatProgress(ev));
            if (ev.detail && ev.detail.pct != null) {
              window.StreetRAG?.showProgress?.(ev.detail.pct, ev.detail.message || ev.step);
            }
          } else if (ev.type === "error") {
            appendActivity("Error: " + (ev.message || JSON.stringify(ev)));
            appendLog("System", "Error: " + (ev.message || ""), "err");
            goBtn.disabled = false;
            return;
          } else if (ev.type === "result" && ev.data) {
            final = ev.data;
          }
        }
      }
      const ms = Date.now() - t0;
      appendActivity("— elapsed " + (ms / 1000).toFixed(1) + "s —");

      if (final) {
        if (final.mode === "route" && final.ok) {
          appendLog("System", final.reply || "", "sys");
          setRouteLayer(final.geojson, final.weight_note);
        } else if (final.index_col && final.render_map) {
          appendLog("System", final.reply || "", "sys");
          await visualizeColumn(final.index_col, final.index_name);
        } else if (final.geojson) {
          appendLog("System", final.reply || "", "sys");
          if (final.value_summary) {
            setIndexLayer(final.index_col, final.value_summary, final.geojson, final.index_name);
            setSpatialTarget(final.spatial_target_resolved, final.spatial_filter_radius_m, final.spatial_target);
            if (final.spatial_target_resolved && final.spatial_target_resolved.found) {
              const r2 = final.spatial_filter_radius_m ? ` · radius ${Number(final.spatial_filter_radius_m).toFixed(0)} m` : "";
              appendLog("Target", `${final.spatial_target} → ${final.spatial_target_resolved.display_name || ""}${r2}`, "sys");
            } else {
              clearSpatialTarget();
            }
            if (final.skill_name && final.skill_name !== "composite_index") {
              renderAnalysisResult(final, { skipReply: true });
            }
            await refreshIndices();
          }
        } else if (final.mode === "analysis") {
          renderAnalysisResult(final);
        } else if (final.reply && !sawProposals) {
          appendLog("System", final.reply, "sys");
        } else if (!sawProposals) {
          appendLog("System", JSON.stringify(final), "sys");
        }
      }
    } catch (err) {
      appendLog("Error", String(err.message || err), "err");
    } finally {
      const assistantReply = ((final && final.reply) || (streamBubble && streamBubble.textContent) || "").trim();
      if (assistantReply) pushChatHistory("assistant", assistantReply);
      goBtn.disabled = false;
      streamBubble = null;
      thinkDetails = null;
    }
  });

  window.StreetRAG = {
    getContext: () => (window.__streetContext || []),
    setContext: (ctx) => { window.__streetContext = ctx; },
    addContext: (item) => {
      window.__streetContext = window.__streetContext || [];
      if (!window.__streetContext.some((c) => c.type === item.type && c.name === item.name)) {
        window.__streetContext.push(item);
        window.dispatchEvent(new CustomEvent("street-context", { detail: window.__streetContext }));
      }
    },
    getHistory: () => (window.__streetHistory || []),
    getLastUserQuery: () => window.__lastUserQuery || "",
    visualizeColumn,
    refreshDatasets,
    refreshIndices,
    showTable: (ev) => {},
    showProgress: (pct, msg) => {},
    hideProgress: () => {},
  };
})();
