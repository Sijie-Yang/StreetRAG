/** Lightweight pub/sub state. */
const listeners = new Map();

export function on(event, fn) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(fn);
  return () => listeners.get(event)?.delete(fn);
}

export function emit(event, data) {
  listeners.get(event)?.forEach((fn) => {
    try { fn(data); } catch (_) {}
  });
}

export const state = {
  context: [], // {type, name}
  chatHistory: [],
  activeColumn: null,
  features: [],
  indices: [],
  datasets: null,
};

export function addContext(item) {
  if (state.context.some((c) => c.type === item.type && c.name === item.name)) return;
  state.context.push(item);
  window.__streetContext = state.context;
  emit("context", state.context);
}

export function removeContext(type, name) {
  state.context = state.context.filter((c) => !(c.type === type && c.name === name));
  window.__streetContext = state.context;
  emit("context", state.context);
}

export function apiPath(path) {
  try {
    if (typeof location !== "undefined" && location.protocol === "file:") {
      return (window.__STREETRAG_API__ || "http://127.0.0.1:8765") + path;
    }
  } catch (_) {}
  return path;
}

export function escapeHtml(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
