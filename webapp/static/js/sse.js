export function parseSseBuffer(buffer, onEvent) {
  const parts = buffer.split("\n\n");
  const rest = parts.pop() || "";
  for (const part of parts) {
    const line = part.split("\n").find((l) => l.startsWith("data: "));
    if (!line) continue;
    try {
      onEvent(JSON.parse(line.slice(6)));
    } catch (_) {}
  }
  return rest;
}

export async function consumeSse(url, { method = "GET", body, onEvent }) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(await res.text());
  if (!res.body) throw new Error("Empty response body");
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    buf = parseSseBuffer(buf, onEvent);
  }
  if (buf.trim()) parseSseBuffer(buf + "\n\n", onEvent);
}
