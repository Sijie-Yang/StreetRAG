import { apiPath, state, addContext, escapeHtml } from "./state.js";

let listEl, searchEl;

export async function refreshFeatures() {
  if (!listEl) return;
  listEl.innerHTML = '<div class="hint">Loading…</div>';
  try {
    const r = await fetch(apiPath("/api/features"));
    const d = await r.json();
    state.features = d.features || [];
    render(state.features);
  } catch (e) {
    listEl.innerHTML = `<div class="hint err">${escapeHtml(e.message)}</div>`;
  }
}

function render(features) {
  if (!listEl) return;
  const q = (searchEl?.value || "").toLowerCase();
  const filtered = q
    ? features.filter((f) => f.name.toLowerCase().includes(q) || (f.description || "").toLowerCase().includes(q))
    : features;
  const groups = {};
  filtered.forEach((f) => {
    const g = f.source || "other";
    (groups[g] ||= []).push(f);
  });
  let html = "";
  for (const [g, items] of Object.entries(groups).sort()) {
    html += `<div class="feat-group"><div class="ds-group">${escapeHtml(g)} (${items.length})</div>`;
    for (const f of items) {
      const badge = f.in_storage ?? f.in_gpkg ? "ok" : "warn";
      html += `<div class="feat-row" data-col="${escapeHtml(f.name)}">` +
        `<span class="fname">${escapeHtml(f.name)}</span>` +
        `<span class="ds-badge ${badge}">${(f.in_storage ?? f.in_gpkg) ? "stored" : "pending"}</span>` +
        `<div class="feat-actions">` +
        `<button type="button" class="btn-sm feat-viz">Map</button>` +
        `<button type="button" class="btn-sm feat-chat">+ Chat</button>` +
        `<button type="button" class="btn-sm feat-del">Del</button>` +
        `</div></div>`;
    }
    html += `</div>`;
  }
  listEl.innerHTML = html || '<div class="hint">No features</div>';
  listEl.querySelectorAll(".feat-viz").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      const row = e.target.closest(".feat-row");
      const col = row?.dataset.col;
      if (!col) return;
      const stored = row.querySelector(".ds-badge")?.textContent?.includes("stored");
      if (!stored) {
        alert("This feature is not integrated yet. Click Integrate on the data file first.");
        return;
      }
      btn.disabled = true;
      try {
        await window.StreetRAG?.visualizeColumn?.(col, col);
      } finally {
        btn.disabled = false;
      }
    });
  });
  listEl.querySelectorAll(".feat-chat").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      addContext({ type: "feature", name: e.target.closest(".feat-row").dataset.col });
    });
  });
  listEl.querySelectorAll(".feat-del").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      const col = e.target.closest(".feat-row").dataset.col;
      if (!confirm("Delete feature " + col + "?")) return;
      await fetch(apiPath("/api/features/" + encodeURIComponent(col)), { method: "DELETE" });
      refreshFeatures();
    });
  });
}

export function initFeaturesPanel() {
  listEl = document.getElementById("featuresList");
  searchEl = document.getElementById("featuresSearch");
  searchEl?.addEventListener("input", () => render(state.features));
  window.refreshFeatures = refreshFeatures;
  refreshFeatures();
}
