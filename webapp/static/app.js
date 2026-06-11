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
    load_settings: "加载配置",
    load_registry: "读取 feature registry",
    load_gpkg: "加载街道 GeoPackage",
    retrieve_features: "嵌入检索 · 特征召回",
    plan: "结构化规划（features/weights/operator）",
    agent: "Agent 选择 skill 并填参数",
    geocode: "解析空间目标",
    compute_index: "计算复合指数",
    multiscale_topk: "多尺度 top-k 统计",
    spatial_diag: "空间自相关 (Moran's I)",
    llm: "大模型生成解读",
    save_gpkg: "写入 GPKG",
    update_registry: "更新 registry / feature_list",
    complete: "完成",
    route: "路径规划",
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
    if (d.n_edges != null) parts.push(`边 ${d.n_edges} 条`);
    if (d.n_features) parts.push(`feat=${d.n_features}`);
    if (d.crs) parts.push(d.crs);
    if (d.index_col) parts.push("列 " + d.index_col);
    if (d.morans_i && d.morans_i.I != null) parts.push(`I=${Number(d.morans_i.I).toFixed(3)}`);
    if (d.query) parts.push(`q="${d.query}"`);
    if (d.display_name) parts.push(`→ ${d.display_name}`);
    if (d.filter_radius_m) parts.push(`半径 ${Number(d.filter_radius_m).toFixed(0)}m`);
    if (d.kept != null && d.total != null) parts.push(`保留 ${d.kept}/${d.total}`);
    if (d.message) parts.push(d.message);
    return parts.join(" · ");
  }
  function clearActivity() { if (activity) activity.textContent = ""; }
  function appendActivity(line) {
    if (!activity) return;
    activity.textContent = activity.textContent ? activity.textContent + "\n" + line : line;
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
      appendLog("错误", "无法加载路网: " + e, "err");
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
    if (map.getLayer("index-lines")) map.removeLayer("index-lines");
    if (map.getSource("index")) map.removeSource("index");
    if (!geojson || !geojson.features || !geojson.features.length) {
      hideColorbar();
      return;
    }
    const [lo, hi] = stretchRange(summary);
    lastIndex = { col, summary, vmin: lo, vmax: hi, indexName };
    map.addSource("index", { type: "geojson", data: geojson });
    map.addLayer({
      id: "index-lines", type: "line", source: "index",
      paint: paintForIndex(lo, hi, getMode()),
    });
    showColorbar(col, summary, indexName);
    updateHashIndex(col);
    const b = computeBounds(geojson);
    if (b) map.fitBounds(b, { padding: 32, maxZoom: 12 });
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
    if (note) appendLog("系统", "权重：" + note, "sys");
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
          html += `<div class="popup-mini">主要贡献：<ul>`;
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
      appendLog("路径", `起点：${lng.toFixed(5)}, ${lat.toFixed(5)}`, "sys");
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
      appendLog("系统", `路径 ~${Math.round(d.length_m)} m （权重：${d.weight_note}）`, "sys");
    } catch (err) {
      appendLog("错误", String(err.message || err), "err");
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
      const opts = ['<option value="length">length（最短）</option>'];
      (d.indices || []).forEach((it) => {
        opts.push(`<option value="${escapeHtml(it.index_col)}">${escapeHtml(it.index_name || it.index_col)}（最舒适）</option>`);
      });
      routeWeightSelect.innerHTML = opts.join("");
    } catch (e) {
      indicesList.innerHTML = `<div style="color:#dc2626;font-size:12px;padding:8px;">无法加载: ${escapeHtml(e.message || e)}</div>`;
    }
  }

  function renderIndicesList(items) {
    if (!items.length) {
      indicesList.innerHTML = `<div style="color:var(--muted);font-size:12px;text-align:center;padding:16px;">还没有保存的指数。<br/>用下方对话框创建一个吧。</div>`;
      return;
    }
    indicesList.innerHTML = items.map((it) => {
      const t = it.timestamp ? new Date(it.timestamp).toLocaleString() : "";
      const stats = it.statistics || {};
      const morans = (it.morans_i != null) ? ` · I=${Number(it.morans_i).toFixed(2)}` : "";
      return (
        `<div class="index-item" data-col="${escapeHtml(it.index_col)}">` +
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
      if (d.reply) appendLog("系统 (已存)", d.reply, "sys");
      if (d.morans_i && d.morans_i.I != null) {
        appendLog("空间诊断", `Moran's I = ${Number(d.morans_i.I).toFixed(4)} (n=${d.morans_i.n})`, "sys");
      }
      if (d.spatial_target_resolved && d.spatial_target_resolved.found) {
        const r2 = d.spatial_filter_radius_m ? ` · 半径 ${Number(d.spatial_filter_radius_m).toFixed(0)} m` : "";
        appendLog("空间目标", `${d.spatial_target} → ${d.spatial_target_resolved.display_name || ""}${r2}`, "sys");
      }
    } catch (e) {
      appendLog("错误", "加载指数失败: " + (e.message || e), "err");
    }
  }

  // Result card for analysis-only skills (correlate, multiscale_profile…)
  // that return numbers + text instead of a map layer.
  const SKILL_NAMES_ZH = {
    correlate: "相关性分析",
    multiscale_profile: "多尺度剖面",
    cluster_lisa: "空间聚类 (LISA)",
    composite_index: "复合指数",
  };
  function renderAnalysisResult(d, opts) {
    const o = opts || {};
    const title = SKILL_NAMES_ZH[d.skill_name] || d.skill_name || "分析";
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

    if (lines.length) appendLog("证据", lines.join("\n"), "sys");
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
      appendLog("系统", "请先选择或创建一个指数再导出。", "sys");
      return;
    }
    const r = await fetch(apiPath(`/api/edges-geojson?color_by=${encodeURIComponent(lastIndex.col)}&simplify=4`));
    if (!r.ok) {
      appendLog("错误", "导出失败: " + await r.text(), "err");
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
  const DS_BADGES = {
    network_active: ["net", "当前网络"],
    network_candidate: ["cand", "备选网络"],
    integrated: ["ok", "已整合"],
    converted: ["conv", "已转换"],
    not_integrated: ["no", "未整合"],
  };

  const citySelect = document.getElementById("citySelect");

  function dsFileRow(f) {
    const [cls, label] = DS_BADGES[f.role] || ["no", f.role];
    const size = f.size_mb != null ? ` <span style="color:var(--muted)">${f.size_mb}MB</span>` : "";
    const missing = f.missing ? `<span class="ds-badge miss">文件缺失</span>` : "";
    let html = `<div class="ds-file"><span class="fname">${escapeHtml(f.name)}</span>${size}` +
      `<span class="ds-badge ${cls}">${label}${f.role === "integrated" && f.columns ? " " + f.columns.length + "列" : ""}</span>${missing}`;
    if (f.method) html += `<span style="color:var(--muted)">${escapeHtml(f.method)}</span>`;
    if (f.note) html += `<span style="color:var(--muted)">${escapeHtml(f.note)}</span>`;
    if (f.columns && f.columns.length) {
      html += `<details class="ds-cols"><summary>查看 ${f.columns.length} 个特征列</summary>` +
        `<div class="cols">${f.columns.map(escapeHtml).join("<br/>")}</div></details>`;
    }
    return html + `</div>`;
  }

  async function refreshDatasets() {
    if (!datasetsBox) return;
    try {
      const r = await fetch(apiPath("/api/datasets"));
      if (!r.ok) throw new Error(await r.text());
      const d = await r.json();
      const net = d.network || {};
      const fc = d.feature_counts || {};

      if (citySelect) {
        citySelect.innerHTML = (d.cities || []).map((c) =>
          `<option value="${escapeHtml(c)}"${c === d.city ? " selected" : ""}>🏙 ${escapeHtml(c)}</option>`
        ).join("");
      }

      let html = "";
      html += `<div class="ds-net">${escapeHtml(net.file || "未设置网络")}` +
        (net.n_edges ? ` · ${Number(net.n_edges).toLocaleString()} 边` : "") +
        (net.syntax_radii && net.syntax_radii.length ? ` · 句法半径 ${net.syntax_radii.join("/")}m` : "") +
        `</div>`;
      html += `<div class="ds-counts">特征共 ${fc.total ?? "?"}：句法 ${fc.space_syntax ?? 0} · 外部整合 ${fc.integrated ?? 0} · 复合指数 ${fc.composite_index ?? 0}</div>`;

      if ((d.networks || []).length) {
        html += `<div class="ds-group">街道网络</div>`;
        for (const f of d.networks) html += dsFileRow(f);
      }
      const groups = [
        ["已整合数据源", (d.sources || []).filter((f) => f.role === "integrated")],
        ["已转换", (d.sources || []).filter((f) => f.role === "converted")],
        ["未整合", (d.sources || []).filter((f) => f.role === "not_integrated")],
      ];
      for (const [title, items] of groups) {
        if (!items.length) continue;
        html += `<div class="ds-group">${title} (${items.length})</div>`;
        for (const f of items) html += dsFileRow(f);
      }
      if (d.multi_city_note) {
        html += `<div class="ds-note">${escapeHtml(d.multi_city_note)}</div>`;
      }
      datasetsBox.innerHTML = html;
    } catch (e) {
      datasetsBox.innerHTML = `<span style="color:#dc2626">无法加载: ${escapeHtml(e.message || String(e))}</span>`;
    }
  }

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
        location.reload();  // base layer, indices, datasets all change with the city
      } catch (e) {
        appendLog("错误", "切换城市失败: " + (e.message || e), "err");
      }
    });
  }

  if (rescanBtn) {
    rescanBtn.addEventListener("click", async () => {
      rescanBtn.disabled = true;
      datasetsBox.textContent = "扫描中…";
      try {
        const r = await fetch(apiPath("/api/scan"), { method: "POST" });
        if (!r.ok) throw new Error(await r.text());
        appendLog("系统", "数据目录重新扫描完成", "sys");
      } catch (e) {
        appendLog("错误", "扫描失败: " + (e.message || e), "err");
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
        appendLog("系统", "请先选择文件", "err");
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
        appendLog("系统", `已上传 ${upData.filename} (${upData.geometry_type})`, "sys");
        const method = (integratorSelect && integratorSelect.value) || upData.suggested_method || "snap_nearest";
        const ig = await fetch(apiPath("/api/integrate"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename: upData.filename, method_type: method, layer: upData.layers && upData.layers[0] }),
        });
        if (!ig.ok) throw new Error(await ig.text());
        const igData = await ig.json();
        appendLog("系统", `集成完成: ${(igData.columns_added || []).join(", ")}`, "sys");
        refreshDatasets();
      } catch (err) {
        appendLog("错误", String(err.message || err), "err");
      } finally {
        uploadBtn.disabled = false;
      }
    });
  }

  // -------------------------------------------------------------------
  // Chat
  // -------------------------------------------------------------------
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
    input.value = "";
    goBtn.disabled = true;
    clearActivity();
    appendLog("你", text, "you");
    const t0 = Date.now();

    try {
      const res = await fetch(apiPath("/api/chat-stream"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, run_route_on_coords: true }),
      });
      if (!res.ok) throw new Error(await res.text());
      if (!res.body) throw new Error("无响应体");

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
                if (ev.type === "error") appendLog("系统", "错误: " + (ev.message || ""), "err");
              } catch (_) {}
            }
          }
          break;
        }
        buffer += dec.decode(value, { stream: true });
        const p = parseSseBuffer(buffer);
        buffer = p.rest;
        for (const ev of p.events) {
          if (ev.type === "progress") {
            appendActivity("· " + formatProgress(ev));
          } else if (ev.type === "error") {
            appendActivity("错误: " + (ev.message || JSON.stringify(ev)));
            appendLog("系统", "错误: " + (ev.message || ""), "err");
            goBtn.disabled = false;
            return;
          } else if (ev.type === "result" && ev.data) {
            final = ev.data;
          }
        }
      }
      const ms = Date.now() - t0;
      appendActivity("— 用时 " + (ms / 1000).toFixed(1) + "s —");

      if (final) {
        if (final.mode === "route" && final.ok) {
          appendLog("系统", final.reply || "", "sys");
          setRouteLayer(final.geojson, final.weight_note);
        } else if (final.geojson) {
          appendLog("系统", final.reply || "", "sys");
          if (final.value_summary) {
            setIndexLayer(final.index_col, final.value_summary, final.geojson, final.index_name);
            setSpatialTarget(final.spatial_target_resolved, final.spatial_filter_radius_m, final.spatial_target);
            if (final.spatial_target_resolved && final.spatial_target_resolved.found) {
              const r2 = final.spatial_filter_radius_m ? ` · 半径 ${Number(final.spatial_filter_radius_m).toFixed(0)} m` : "";
              appendLog("空间目标", `${final.spatial_target} → ${final.spatial_target_resolved.display_name || ""}${r2}`, "sys");
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
        } else {
          appendLog("系统", final.reply || JSON.stringify(final), "sys");
        }
      }
    } catch (err) {
      appendLog("错误", String(err.message || err), "err");
    } finally {
      goBtn.disabled = false;
    }
  });
})();
