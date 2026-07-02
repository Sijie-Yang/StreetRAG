import { on } from "./state.js";

let bar, label, backdrop;

export function initProgressBar() {
  bar = document.getElementById("globalProgress");
  label = document.getElementById("globalProgressLabel");
  backdrop = document.getElementById("globalProgressBackdrop");
  on("progress", ({ pct, message }) => show(pct, message));
}

export function show(pct, message) {
  if (backdrop) backdrop.style.display = "block";
  if (bar) {
    bar.style.display = "block";
    bar.querySelector(".fill").style.width = `${Math.min(100, Math.max(0, pct || 0))}%`;
  }
  if (label) {
    label.style.display = "block";
    label.textContent = message || "Working…";
  }
}

export function hide() {
  if (backdrop) backdrop.style.display = "none";
  if (bar) bar.style.display = "none";
  if (label) label.style.display = "none";
}

export function emitProgress(pct, message) {
  import("./state.js").then(({ emit }) => emit("progress", { pct, message }));
}
