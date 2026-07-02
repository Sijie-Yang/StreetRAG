import { escapeHtml } from "./state.js";

let panel, titleEl, tableEl;

export function initTablePanel() {
  panel = document.getElementById("tablePanel");
  titleEl = document.getElementById("tablePanelTitle");
  tableEl = document.getElementById("tablePanelTable");
  document.getElementById("tablePanelClose")?.addEventListener("click", () => {
    panel?.classList.add("hidden");
  });
  document.getElementById("tablePanelExport")?.addEventListener("click", exportCsv);
}

export function showTable({ title, columns, rows }) {
  if (!panel) return;
  panel.classList.remove("hidden");
  if (titleEl) titleEl.textContent = title || "Table";
  const cols = columns || [];
  const hdr = `<thead><tr>${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>`;
  const body = (rows || []).map((r) =>
    `<tr>${r.map((v) => `<td>${escapeHtml(v)}</td>`).join("")}</tr>`
  ).join("");
  tableEl.innerHTML = `<table>${hdr}<tbody>${body}</tbody></table>`;
  tableEl.dataset.cols = JSON.stringify(cols);
  tableEl.dataset.rows = JSON.stringify(rows || []);
}

function exportCsv() {
  try {
    const cols = JSON.parse(tableEl.dataset.cols || "[]");
    const rows = JSON.parse(tableEl.dataset.rows || "[]");
    const lines = [cols.join(",")].concat(rows.map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")));
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "streetrag-table.csv";
    a.click();
  } catch (_) {}
}
