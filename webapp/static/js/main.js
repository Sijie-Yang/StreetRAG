import { initContextChips } from "./context-chips.js";
import { initProgressBar, show, hide } from "./progress-bar.js";
import { initTablePanel, showTable } from "./table-panel.js";
import { initFeaturesPanel } from "./features-panel.js";
import { initIndicesPanel } from "./indices-panel.js";
import { state } from "./state.js";

function wireStreetRAG() {
  const SR = window.StreetRAG || {};
  SR.showTable = showTable;
  SR.showProgress = show;
  SR.hideProgress = hide;
  SR.getContext = () => state.context;
  SR.addContext = (item) => import("./state.js").then((m) => m.addContext(item));
  window.StreetRAG = SR;
}

function initDrawerTabs() {
  document.querySelectorAll(".drawer-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".drawer-tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".drawer-pane").forEach((p) => p.classList.add("hidden"));
      tab.classList.add("active");
      document.getElementById("pane-" + tab.dataset.pane)?.classList.remove("hidden");
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  wireStreetRAG();
  initProgressBar();
  initContextChips();
  initTablePanel();
  initDrawerTabs();
  initFeaturesPanel();
  initIndicesPanel();
});
