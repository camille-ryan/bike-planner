// Chat pane — talks to POST /chat over SSE, streams text into the log,
// and forwards tool calls to the map (draws routes, pulses anchors).
//
// The backend is stateless: we send the full message history on every
// turn. `history` here is the source of truth.
//
// Wrapped in an IIFE so `const`s at the top don't collide with the
// same-named symbols in app.js (both scripts share the global scope).
(function () {
"use strict";

const CHAT_API_BASE = "/api";
const CHAT_PATH = "/chat";

const logEl    = document.getElementById("chat-log");
const formEl   = document.getElementById("chat-form");
const inputEl  = document.getElementById("chat-input");
const sendBtn  = document.getElementById("chat-send");
const resetBtn = document.getElementById("chat-reset");

console.log("[chat.js] loaded, elements:", {
  log: !!logEl, form: !!formEl, input: !!inputEl, send: !!sendBtn, reset: !!resetBtn,
});
if (!formEl || !inputEl || !sendBtn) {
  console.error("[chat.js] chat pane elements missing — chat disabled");
  throw new Error("chat pane elements missing");
}

let history = [];
let inflight = null;

// ---------- rendering ----------

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  div.textContent = text;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
  return div;
}

function addToolCall(name, input, output) {
  const div = document.createElement("div");
  div.className = "chat-msg tool";
  const summary = summarizeTool(name, input, output);
  div.innerHTML = `<span class="tool-name">🔧 ${escape(name)}</span> ${escape(summary)}`;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
}

function escape(s) {
  return String(s ?? "").replace(/[&<>]/g, c =>
    ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));
}

function summarizeTool(name, input, output) {
  if (output && output.error) return `— error: ${output.error}`;
  switch (name) {
    case "search_anchors": {
      const n = output?.results?.length || 0;
      return `“${input.query}” → ${n} anchor${n === 1 ? "" : "s"}`;
    }
    case "route":
      return `${input.from_ref || input.from_lonlat} → ${input.to_ref || input.to_lonlat}  ⇒  ${output?.total_km ?? "?"} km`;
    case "stations_near":
      return `${output?.stations?.length || 0} stations within ${input.radius_km || 15} km of ${input.lon.toFixed(2)},${input.lat.toFixed(2)}`;
    case "split_into_stages":
      return `${output?.n_days ?? "?"} stages @ ~${input.target_km_per_day} km/day`;
    default:
      return "";
  }
}

// ---------- map integration ----------
// We piggyback on window.map (the maplibre instance created by app.js).
// Layer ids are namespaced with `chat-` so app.js overlays are unaffected.

function ensureChatRouteLayer() {
  const map = window.map;
  if (!map || map.getSource("chat-route")) return;
  map.addSource("chat-route", {
    type: "geojson",
    data: { type: "FeatureCollection", features: [] },
  });
  map.addLayer({
    id: "chat-route-line",
    type: "line",
    source: "chat-route",
    paint: {
      "line-color": "#2a7fbf",
      "line-width": 4,
      "line-opacity": 0.85,
    },
  });
}

// Accumulate route segments drawn this turn so multi-leg plans
// (Graz → Prague, Prague → Copenhagen) show the whole route instead of
// only the last segment. `clearChatRouteLayer()` is called at the top
// of every new user turn to prevent yesterday's plan from lingering.
let chatRouteFeatures = [];

function clearChatRouteLayer() {
  chatRouteFeatures = [];
  const map = window.map;
  if (map && map.getSource("chat-route")) {
    map.getSource("chat-route").setData({ type: "FeatureCollection", features: [] });
  }
}

function drawRouteOnMap(polyline) {
  const map = window.map;
  if (!map || !polyline || polyline.length < 2) return;
  ensureChatRouteLayer();
  chatRouteFeatures.push({
    type: "Feature",
    geometry: { type: "LineString", coordinates: polyline },
    properties: {},
  });
  map.getSource("chat-route").setData({
    type: "FeatureCollection",
    features: chatRouteFeatures,
  });
  // Fit to bounds of every route currently drawn (min/max across all
  // segments), so a multi-leg plan zooms out to cover them all.
  let minLon =  Infinity, minLat =  Infinity;
  let maxLon = -Infinity, maxLat = -Infinity;
  for (const f of chatRouteFeatures) {
    for (const [lon, lat] of f.geometry.coordinates) {
      if (lon < minLon) minLon = lon;
      if (lon > maxLon) maxLon = lon;
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
    }
  }
  map.fitBounds([[minLon, minLat], [maxLon, maxLat]],
    { padding: 60, duration: 500 });
}

function ensureStagePinsLayer() {
  const map = window.map;
  if (!map || map.getSource("chat-stages")) return;
  map.addSource("chat-stages", {
    type: "geojson",
    data: { type: "FeatureCollection", features: [] },
  });
  map.addLayer({
    id: "chat-stages-circles",
    type: "circle",
    source: "chat-stages",
    paint: {
      "circle-radius": 7,
      "circle-color": "#f5a623",
      "circle-stroke-width": 2,
      "circle-stroke-color": "#fff",
    },
  });
  map.addLayer({
    id: "chat-stages-labels",
    type: "symbol",
    source: "chat-stages",
    layout: {
      "text-field": ["get", "label"],
      "text-size": 12,
      "text-offset": [0, 1.3],
      "text-anchor": "top",
    },
    paint: {
      "text-color": "#333",
      "text-halo-color": "#fff",
      "text-halo-width": 2,
    },
  });
}

// Accumulate stage pins across multiple split_into_stages calls in one
// turn. Cleared on each new user turn by clearChatRouteLayer's twin.
let chatStageFeatures = [];

function clearChatStagesLayer() {
  chatStageFeatures = [];
  const map = window.map;
  if (map && map.getSource("chat-stages")) {
    map.getSource("chat-stages").setData({ type: "FeatureCollection", features: [] });
  }
}

function drawStagesOnMap(stages) {
  const map = window.map;
  if (!map || !stages?.length) return;
  ensureStagePinsLayer();
  for (const s of stages) {
    chatStageFeatures.push({
      type: "Feature",
      geometry: { type: "Point", coordinates: s.to_lonlat },
      properties: { label: `Day ${s.day}: ${s.to_name || "?"} (${s.km} km)` },
    });
  }
  map.getSource("chat-stages").setData({
    type: "FeatureCollection",
    features: chatStageFeatures,
  });
}

function handleToolResult(name, input, output) {
  addToolCall(name, input, output);
  if (name === "route" && output?.polyline?.length) {
    drawRouteOnMap(output.polyline);
  }
  if (name === "split_into_stages" && output?.stages) {
    drawStagesOnMap(output.stages);
  }
}

// ---------- SSE parser ----------
// Fetch-with-body streaming; parse `event:` + `data:` lines manually.

async function streamChat(userText) {
  const userDiv = addMessage("user", userText);
  history.push({ role: "user", content: userText });

  // Reset the map layers we own so a fresh plan starts on a clean map.
  clearChatRouteLayer();
  clearChatStagesLayer();

  sendBtn.disabled = true;
  sendBtn.textContent = "Thinking…";

  // Live status line above the assistant text — replaces the blank
  // "waiting" gap that made completion hard to spot. Cleared on `done`.
  const statusDiv = document.createElement("div");
  statusDiv.className = "chat-msg status";
  statusDiv.textContent = "⏳ Planning…";
  logEl.appendChild(statusDiv);
  logEl.scrollTop = logEl.scrollHeight;

  const t0 = performance.now();
  let toolCount = 0;
  const tickStatus = () => {
    const sec = ((performance.now() - t0) / 1000).toFixed(0);
    statusDiv.textContent = toolCount
      ? `⏳ Planning… ${toolCount} tool call${toolCount === 1 ? "" : "s"}, ${sec}s`
      : `⏳ Planning… ${sec}s`;
  };
  const statusTimer = setInterval(tickStatus, 500);

  // Don't create the assistant bubble yet — tool calls are about to
  // stream in and we want the final text to land BELOW them, not above.
  // We'll create it lazily on the first text delta.
  let asstDiv = null;
  let asstText = "";

  const controller = new AbortController();
  inflight = controller;

  let resp;
  try {
    resp = await fetch(`${CHAT_API_BASE}${CHAT_PATH}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
      signal: controller.signal,
    });
  } catch (e) {
    asstDiv.remove();
    addMessage("error", "⚠️  Couldn't reach the planner. Is the API up?");
    console.error("[chat.js] fetch failed:", e);
    sendBtn.disabled = false; sendBtn.textContent = "Send";
    return;
  }
  if (!resp.ok) {
    asstDiv.remove();
    const body = await resp.text();
    console.error("[chat.js] HTTP", resp.status, body);
    let friendly;
    if (resp.status === 502 || resp.status === 503) {
      friendly = "The API is still starting up — try again in a moment.";
    } else if (resp.status === 500) {
      friendly = "The planner errored on the server. Check api logs.";
    } else {
      friendly = `Request failed (HTTP ${resp.status}).`;
    }
    addMessage("error", `⚠️  ${friendly}`);
    sendBtn.disabled = false; sendBtn.textContent = "Send";
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const ev = parseSSE(raw);
      if (!ev) continue;
      if (ev.event === "text") {
        asstText += ev.data.delta || "";
        // Lazy-create the assistant bubble on first delta so it lands
        // below any tool calls that have already streamed in.
        if (!asstDiv) asstDiv = addMessage("assistant", "");
        // Render as markdown so tables/lists format properly. `marked`
        // is loaded from the CDN via a <script> tag in index.html.
        if (typeof marked !== "undefined") {
          asstDiv.innerHTML = marked.parse(asstText);
        } else {
          asstDiv.textContent = asstText;
        }
        logEl.scrollTop = logEl.scrollHeight;
      } else if (ev.event === "tool_call") {
        handleToolResult(ev.data.name, ev.data.input, ev.data.output);
        toolCount += 1;
        tickStatus();
      } else if (ev.event === "error") {
        // Server-side friendly error mid-stream. Attach it to the
        // in-flight assistant bubble so the user sees the failure in
        // context, not as a stray red line at the bottom.
        if (!asstText) asstDiv.remove();
        addMessage("error", `⚠️  ${ev.data.message || "Unknown error."}`);
      } else if (ev.event === "done") {
        // stop_reason is on ev.data.stop_reason if we ever want to show it
      }
    }
  }
  // Replace the live status with a compact "done" footer so
  // completion is unmistakable, then stop the ticker.
  clearInterval(statusTimer);
  const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
  statusDiv.className = "chat-msg done";
  statusDiv.textContent =
    `✓ Done in ${elapsed}s${toolCount ? ` · ${toolCount} tool call${toolCount === 1 ? "" : "s"}` : ""}`;

  // Save assistant text into history so the next turn has full context.
  // (The backend actually needs the block-structured `content` — we
  //  approximate it here as a plain string; Claude tolerates that on the
  //  next turn's input.)
  if (asstText.trim()) {
    history.push({ role: "assistant", content: asstText });
  }
  // if no text arrived, asstDiv is still null — nothing to remove.
  sendBtn.disabled = false; sendBtn.textContent = "Send";
  inflight = null;
}

function parseSSE(raw) {
  let event = "message";
  let dataLines = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return { event, data: dataLines.join("\n") };
  }
}

// ---------- form wiring ----------

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const txt = inputEl.value.trim();
  if (!txt || sendBtn.disabled) return;
  inputEl.value = "";
  streamChat(txt);
});

// Enter to send, Shift+Enter for newline
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});

resetBtn.addEventListener("click", () => {
  if (inflight) inflight.abort();
  history = [];
  logEl.innerHTML = "";
  const map = window.map;
  if (map) {
    if (map.getSource("chat-route")) map.getSource("chat-route").setData({ type: "FeatureCollection", features: [] });
    if (map.getSource("chat-stages")) map.getSource("chat-stages").setData({ type: "FeatureCollection", features: [] });
  }
});

})();  // end IIFE
