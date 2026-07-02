import { state, on, removeContext, escapeHtml } from "./state.js";

let container;

export function initContextChips() {
  container = document.getElementById("contextChips");
  on("context", render);
  render(state.context);
}

function render(items) {
  if (!container) return;
  if (!items.length) {
    container.innerHTML = "";
    container.style.display = "none";
    return;
  }
  container.style.display = "flex";
  container.innerHTML = items.map((c) =>
    `<span class="ctx-chip" data-type="${escapeHtml(c.type)}" data-name="${escapeHtml(c.name)}">` +
    `${escapeHtml(c.type)}: ${escapeHtml(c.name)} <button type="button" class="ctx-rm">×</button></span>`
  ).join("");
  container.querySelectorAll(".ctx-rm").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const chip = e.target.closest(".ctx-chip");
      removeContext(chip.dataset.type, chip.dataset.name);
    });
  });
}
