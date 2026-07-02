import { apiPath, addContext, escapeHtml } from "./state.js";

export function enhanceIndicesList() {
  const list = document.getElementById("indicesList");
  if (!list) return;
  const obs = new MutationObserver(() => bindIndexActions(list));
  obs.observe(list, { childList: true, subtree: true });
  bindIndexActions(list);
}

function bindIndexActions(list) {
  list.querySelectorAll(".index-item").forEach((card) => {
    if (card.dataset.bound) return;
    card.dataset.bound = "1";
    const col = card.dataset.col;
    if (!col) return;
    const actions = document.createElement("div");
    actions.className = "index-actions";
    actions.innerHTML =
      `<button type="button" class="btn-sm idx-chat">+ Chat</button>` +
      `<button type="button" class="btn-sm idx-del">Delete</button>`;
    card.appendChild(actions);
    actions.querySelector(".idx-chat").addEventListener("click", (e) => {
      e.stopPropagation();
      addContext({ type: "index", name: col });
    });
    actions.querySelector(".idx-del").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm("Delete index " + col + "?")) return;
      await fetch(apiPath("/api/indices/" + encodeURIComponent(col)), { method: "DELETE" });
      document.getElementById("refreshIndicesBtn")?.click();
    });
  });
}

// Patch refreshIndices rendering via hook after load
export function initIndicesPanel() {
  enhanceIndicesList();
}
