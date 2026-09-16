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
const saveBtn  = document.getElementById("chat-save");
const loadSel  = document.getElementById("chat-load");

// Abort the SSE if no bytes arrive for this many ms. Route planning
// can take ~2 min end-to-end; the API's own per-request timeout is
// 120s, so 90s of silence with no delta / tool_call / done is a real
// stall (Anthropic slow, WSL sleep, network drop). We surface it as a
// clean error instead of hanging the UI forever.
const CHAT_STREAM_IDLE_MS = 90_000;

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

// Accumulate route polylines DURING planning so the user watches the
// map build up as Claude iterates. On ✓ Done we collapse to just the
// longest polyline via `finalizeChatMap()` — accumulated sub-legs
// overlap the full route and would otherwise read as loops at every
// city stop.
let chatRoutePolylines = [];

// Latest `split_into_stages` result, used by the GPX-download button
// to slice the polyline into per-day tracks.
let chatLatestStages = null;

function clearChatRouteLayer() {
  chatRoutePolylines = [];
  chatLatestStages = null;
  const map = window.map;
  if (map && map.getSource("chat-route")) {
    map.getSource("chat-route").setData({ type: "FeatureCollection", features: [] });
  }
}

function _routesToFeatures(polylines) {
  return polylines.map(p => ({
    type: "Feature",
    geometry: { type: "LineString", coordinates: p },
    properties: {},
  }));
}

function drawRouteOnMap(polyline) {
  const map = window.map;
  if (!map || !polyline || polyline.length < 2) return;
  chatRoutePolylines.push(polyline);
  ensureChatRouteLayer();
  map.getSource("chat-route").setData({
    type: "FeatureCollection",
    features: _routesToFeatures(chatRoutePolylines),
  });
  // Fit to bounds of the drawn polyline.
  let minLon =  Infinity, minLat =  Infinity;
  let maxLon = -Infinity, maxLat = -Infinity;
  for (const [lon, lat] of polyline) {
    if (lon < minLon) minLon = lon;
    if (lon > maxLon) maxLon = lon;
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
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
  // Numbered overnight pins — three layers stacked:
  //   1. outer circle (dark purple fill, white ring) — reads as "stop"
  //      and is visually distinct from POI markers (amber viewpoints,
  //      teal water) and rail stations (blue) already on the map
  //   2. day number rendered white in the center of the circle
  //   3. city + km label rendered below, matches other on-map labels
  map.addLayer({
    id: "chat-stages-circles",
    type: "circle",
    source: "chat-stages",
    paint: {
      "circle-radius": 14,
      "circle-color": "#6d4aff",
      "circle-stroke-width": 2.5,
      "circle-stroke-color": "#fff",
      "circle-opacity": 0.95,
    },
  });
  // MapLibre's default text-font stack is
  // `["Open Sans Regular","Arial Unicode MS Regular"]`, which the
  // OpenFreeMap positron tile server DOESN'T ship — a request for
  // that font's glyph range returns 404. That 404 doesn't just kill
  // the symbol layer's text; empirically it also tanks the sibling
  // circle layer's WebGL draw call. Setting an explicit font that
  // IS in the OpenFreeMap glyph pack (Noto Sans Bold + Regular) both
  // makes the text render AND unblocks the circles.
  map.addLayer({
    id: "chat-stages-daynum",
    type: "symbol",
    source: "chat-stages",
    layout: {
      "text-field": ["to-string", ["get", "day"]],
      "text-size": 14,
      "text-font": ["Noto Sans Bold"],
      "text-anchor": "center",
      "text-allow-overlap": true,
      "text-ignore-placement": true,
    },
    paint: {
      "text-color": "#fff",
    },
  });
  map.addLayer({
    id: "chat-stages-labels",
    type: "symbol",
    source: "chat-stages",
    layout: {
      "text-field": ["get", "label"],
      "text-size": 12,
      "text-font": ["Noto Sans Regular"],
      "text-offset": [0, 1.6],
      "text-anchor": "top",
    },
    paint: {
      "text-color": "#222",
      "text-halo-color": "#fff",
      "text-halo-width": 2,
    },
  });
}

// Stage pins are stored per (from_ref, to_ref) leg key. When Claude
// re-splits the same leg at a different km/day, we REPLACE that leg's
// stages rather than accumulate — otherwise the map ends up with two
// pin sets from the same leg overlapping.
let chatStagesByLeg = new Map();

function clearChatStagesLayer() {
  chatStagesByLeg.clear();
  const map = window.map;
  if (map && map.getSource("chat-stages")) {
    map.getSource("chat-stages").setData({ type: "FeatureCollection", features: [] });
  }
}

function _stagesToFeatures() {
  const feats = [];
  for (const stages of chatStagesByLeg.values()) {
    for (const s of stages) {
      feats.push({
        type: "Feature",
        geometry: { type: "Point", coordinates: s.to_lonlat },
        properties: {
          // day number rendered inside the circle (chat-stages-daynum)
          day: s.day,
          // city + km label rendered below (chat-stages-labels)
          label: `${s.to_name || "?"} · ${s.km} km`,
        },
      });
    }
  }
  return feats;
}

function drawStagesOnMap(stages, legKey) {
  const map = window.map;
  if (!map || !stages?.length) return;
  ensureStagePinsLayer();
  chatStagesByLeg.set(legKey, stages);
  map.getSource("chat-stages").setData({
    type: "FeatureCollection",
    features: _stagesToFeatures(),
  });
  // Cache the freshest stages snapshot for GPX generation. Same
  // per-leg dedup so the GPX matches what's on the map.
  const flat = [];
  for (const legStages of chatStagesByLeg.values()) flat.push(...legStages);
  chatLatestStages = flat;
}

// Collapse the accumulated overlays down to just the "final" plan:
//  - keep only the longest polyline (removes sub-leg overlap loops)
//  - per-leg stage dedup is already live-applied by drawStagesOnMap
// Called from the SSE `done` path so the user sees planning progress
// during the run, then a clean map at the end. Returns the longest
// polyline so the GPX button has something to work with.
function finalizeChatMap() {
  if (!chatRoutePolylines.length) return null;
  let longest = chatRoutePolylines[0];
  for (const p of chatRoutePolylines) {
    if (p.length > longest.length) longest = p;
  }
  const map = window.map;
  if (map && map.getSource("chat-route")) {
    map.getSource("chat-route").setData({
      type: "FeatureCollection",
      features: _routesToFeatures([longest]),
    });
  }
  return longest;
}

// ---------- GPX export ----------

function _hav_m(lon1, lat1, lon2, lat2) {
  const R = 6371000;
  const p1 = lat1 * Math.PI / 180, p2 = lat2 * Math.PI / 180;
  const dp = p2 - p1, dl = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dp/2)**2 + Math.cos(p1)*Math.cos(p2)*Math.sin(dl/2)**2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

function _nearestVertexIdx(polyline, lon, lat) {
  let bestI = 0, bestD = Infinity;
  for (let i = 0; i < polyline.length; i++) {
    const d = _hav_m(lon, lat, polyline[i][0], polyline[i][1]);
    if (d < bestD) { bestD = d; bestI = i; }
  }
  return bestI;
}

function _xmlEscape(s) {
  return String(s ?? "").replace(/[<>&'"]/g, c =>
    ({"<":"&lt;", ">":"&gt;", "&":"&amp;", "'":"&apos;", '"':"&quot;"}[c]));
}

function buildGpx(polyline, stages, tripName = "Bike tour") {
  // Slice the polyline into per-stage tracks. If no stages, emit ONE
  // track covering the whole polyline.
  const now = new Date().toISOString().replace(/\.\d+Z$/, "Z");
  let xml =
    '<?xml version="1.0" encoding="UTF-8"?>\n' +
    '<gpx version="1.1" creator="bike-routing-planner" ' +
    'xmlns="http://www.topografix.com/GPX/1/1">\n' +
    `  <metadata>\n` +
    `    <name>${_xmlEscape(tripName)}</name>\n` +
    `    <time>${now}</time>\n` +
    `  </metadata>\n`;

  const tracks = [];
  if (stages && stages.length) {
    // Find the polyline index closest to each stage's from/to lonlat.
    let prevIdx = 0;
    for (const s of stages) {
      const toIdx = _nearestVertexIdx(polyline, s.to_lonlat[0], s.to_lonlat[1]);
      // Slice from prevIdx (inclusive) to toIdx (inclusive)
      const lo = Math.min(prevIdx, toIdx);
      const hi = Math.max(prevIdx, toIdx);
      const seg = polyline.slice(lo, hi + 1);
      if (seg.length >= 2) {
        tracks.push({
          name: `Day ${s.day}: ${s.from_name ?? "?"} → ${s.to_name ?? "?"} (${s.km} km)`,
          seg,
        });
      }
      prevIdx = toIdx;
    }
  } else {
    tracks.push({ name: tripName, seg: polyline });
  }

  for (const t of tracks) {
    xml += `  <trk>\n    <name>${_xmlEscape(t.name)}</name>\n    <trkseg>\n`;
    for (const [lon, lat] of t.seg) {
      xml += `      <trkpt lat="${lat.toFixed(6)}" lon="${lon.toFixed(6)}"/>\n`;
    }
    xml += `    </trkseg>\n  </trk>\n`;
  }
  xml += "</gpx>\n";
  return xml;
}

function downloadGpx(polyline) {
  if (!polyline) {
    addMessage("error", "⚠️  No route to export yet.");
    return;
  }
  // chatLatestStages is already dedup'd per-leg by drawStagesOnMap and
  // ordered as Claude called split_into_stages. But leg-local day
  // numbers overlap (each leg has its own day 1, day 2, …), so we
  // can't dedup by day here. Order the stages by where their `to`
  // vertex falls along the polyline — that gives the tour order
  // regardless of what order Claude computed the legs in, and also
  // renumbers them 1..N for the GPX display.
  const raw = chatLatestStages ?? [];
  const withIdx = raw.map(s => ({
    stage: s,
    polyIdx: _nearestVertexIdx(polyline, s.to_lonlat[0], s.to_lonlat[1]),
  }));
  withIdx.sort((a, b) => a.polyIdx - b.polyIdx);
  // Drop duplicates whose `to` snaps to the same polyline vertex.
  const stages = [];
  let lastIdx = -1;
  for (let i = 0; i < withIdx.length; i++) {
    if (withIdx[i].polyIdx === lastIdx) continue;
    lastIdx = withIdx[i].polyIdx;
    stages.push({ ...withIdx[i].stage, day: stages.length + 1 });
  }

  const firstName = stages[0]?.from_name ?? "start";
  const lastName  = stages.at(-1)?.to_name ?? "end";
  const tripName  = `${firstName} → ${lastName}`;
  const gpx = buildGpx(polyline, stages, tripName);

  const blob = new Blob([gpx], { type: "application/gpx+xml" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url;
  const stamp = new Date().toISOString().slice(0, 10);
  a.download = `${tripName.replace(/[^\w -]+/g, "_")} ${stamp}.gpx`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function handleToolResult(name, input, output) {
  addToolCall(name, input, output);
  if (name === "route" && output?.polyline?.length) {
    drawRouteOnMap(output.polyline);
    // Also populate the sidebar's route state so the paired-SPT viz
    // ("Show route data" toggle) works after a chat-driven plan.
    // Before this, only sidebar-form routes populated state, and any
    // demo that opened the chat then flipped the debug toggle just
    // saw "plan a route first" — the interview regression.
    // We synthesize the same GeoJSON Feature shape trunk_router.route
    // returns for the sidebar, minimum fields the viz reads.
    try {
      const s = window.sidebarState;
      if (s && Array.isArray(output.chain_city_idx) && output.chain_city_idx.length) {
        s.routesByProfile = s.routesByProfile || {};
        s.routesByProfile["views"] = {
          type: "Feature",
          geometry: { type: "LineString", coordinates: output.polyline },
          properties: {
            chain_city_idx: output.chain_city_idx,
            chain_names:    output.chain_names || [],
            gross_length_m: (output.total_km || 0) * 1000,
            bridges: [],  // chat.route doesn't propagate the full bridges array
          },
        };
      }
    } catch (e) {
      console.warn("[chat.js] sidebarState.routesByProfile sync failed:", e);
    }
  }
  if (name === "split_into_stages" && output?.stages) {
    // Key stage pins by the (from_ref, to_ref) leg so a re-split of
    // the same leg REPLACES its pin set. Without this, Claude's
    // per-leg + full-tour + retry splits all pile pins on the map.
    const legKey = `${input?.from_ref ?? "?"}→${input?.to_ref ?? "?"}`;
    drawStagesOnMap(output.stages, legKey);
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

  // Cleanup + user-visible error helper. Any exit path (offline, HTTP
  // error, stream idle, mid-stream exception) routes through this so
  // the UI always resets and the user sees what happened.
  let idleTimer = null;
  const finish = (errMsg) => {
    if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
    clearInterval(statusTimer);
    if (asstDiv && !asstText) asstDiv.remove();
    statusDiv.remove();
    if (errMsg) addMessage("error", `⚠️  ${errMsg}`);
    sendBtn.disabled = false; sendBtn.textContent = "Send";
    inflight = null;
  };

  // Fail fast when the browser knows it's offline. `navigator.onLine`
  // is false-negative-prone (returns true if any interface is up) but
  // reliably catches full network loss.
  if (typeof navigator !== "undefined" && navigator.onLine === false) {
    finish("You appear to be offline. Reconnect and try again.");
    history.pop();  // don't leave the failed user turn in history
    return;
  }

  let resp;
  try {
    resp = await fetch(`${CHAT_API_BASE}${CHAT_PATH}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
      signal: controller.signal,
    });
  } catch (e) {
    const msg = (e && e.name === "AbortError")
      ? "Request cancelled."
      : "Couldn't reach the planner. Is the API up? Check your connection.";
    console.error("[chat.js] fetch failed:", e);
    finish(msg);
    history.pop();
    return;
  }
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    console.error("[chat.js] HTTP", resp.status, body);
    let friendly;
    if (resp.status === 502 || resp.status === 503) {
      friendly = "The API is still starting up — try again in a moment.";
    } else if (resp.status === 500) {
      friendly = "The planner errored on the server. Check api logs.";
    } else {
      friendly = `Request failed (HTTP ${resp.status}).`;
    }
    finish(friendly);
    history.pop();
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let didStreamIdleAbort = false;

  // Arm the stream-idle watchdog: if no bytes arrive within
  // CHAT_STREAM_IDLE_MS we abort the fetch and surface a clean error.
  // Each successful read resets it.
  const armIdleTimer = () => {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      didStreamIdleAbort = true;
      try { controller.abort(); } catch { /* ignore */ }
    }, CHAT_STREAM_IDLE_MS);
  };
  armIdleTimer();

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      armIdleTimer();
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
          if (asstDiv && !asstText) { asstDiv.remove(); asstDiv = null; }
          addMessage("error", `⚠️  ${ev.data.message || "Unknown error."}`);
        } else if (ev.event === "done") {
          // stop_reason is on ev.data.stop_reason if we ever want to show it
        }
      }
    }
  } catch (e) {
    if (didStreamIdleAbort) {
      finish(`No response from the planner for ${Math.round(CHAT_STREAM_IDLE_MS/1000)}s — aborted. Try again.`);
    } else if (e && e.name === "AbortError") {
      finish("Request cancelled.");
    } else {
      console.error("[chat.js] stream read failed:", e);
      finish("Connection dropped mid-response. Try again.");
    }
    history.pop();
    return;
  }
  if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
  // Replace the live status with a compact "done" footer so
  // completion is unmistakable, then stop the ticker.
  clearInterval(statusTimer);
  const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
  statusDiv.className = "chat-msg done";
  statusDiv.textContent =
    `✓ Done in ${elapsed}s${toolCount ? ` · ${toolCount} tool call${toolCount === 1 ? "" : "s"}` : ""}`;
  // Collapse the accumulated overlays to the final plan (drops sub-leg
  // polylines that were shown live so the user could watch planning
  // progress). Returns the polyline the GPX button will use.
  const chatFinalPolyline = finalizeChatMap();
  // If a route was drawn, offer a GPX download. Multi-track file: one
  // <trk> per day when stages exist, one <trk> for the whole route
  // when no split was done.
  if (chatFinalPolyline) {
    const btn = document.createElement("button");
    btn.textContent = "⬇ Download GPX";
    btn.className = "secondary";
    btn.style.cssText = "margin-left:0.6rem; font-size:0.75rem; padding:0.1rem 0.5rem;";
    btn.addEventListener("click", () => downloadGpx(chatFinalPolyline));
    statusDiv.appendChild(btn);
  }

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

// ---------- session save / load (localStorage) ----------

const SS_INDEX_KEY = "chat:index";        // JSON array of session ids
const SS_SESSION_PREFIX = "chat:session:"; // per-session payload

function _loadIndex() {
  try { return JSON.parse(localStorage.getItem(SS_INDEX_KEY) || "[]"); }
  catch { return []; }
}

function _saveIndex(idx) {
  localStorage.setItem(SS_INDEX_KEY, JSON.stringify(idx));
}

function _titleFromHistory(hist) {
  const firstUser = hist.find(m => m.role === "user");
  if (!firstUser) return "(empty)";
  const t = typeof firstUser.content === "string"
    ? firstUser.content
    : JSON.stringify(firstUser.content);
  return t.length > 48 ? t.slice(0, 45) + "…" : t;
}

function saveCurrentSession() {
  if (!history.length) {
    addMessage("error", "⚠️  Nothing to save yet.");
    return;
  }
  const id = new Date().toISOString().replace(/[:.]/g, "-");
  const payload = {
    id,
    savedAt: new Date().toISOString(),
    title: _titleFromHistory(history),
    history,
    // Serialize map state so a reload can restore visuals without
    // needing to re-run tools.
    polylines: chatRoutePolylines,
    stagesByLeg: Array.from(chatStagesByLeg.entries()),
  };
  try {
    localStorage.setItem(SS_SESSION_PREFIX + id, JSON.stringify(payload));
    const idx = _loadIndex();
    idx.unshift({ id, savedAt: payload.savedAt, title: payload.title });
    // Keep the most-recent 30 sessions so we don't burn through the
    // ~5 MB localStorage budget.
    while (idx.length > 30) {
      const drop = idx.pop();
      localStorage.removeItem(SS_SESSION_PREFIX + drop.id);
    }
    _saveIndex(idx);
    refreshLoadMenu();
    addMessage("status", `💾 Saved as "${payload.title}"`);
  } catch (e) {
    console.error("[chat.js] save failed:", e);
    addMessage("error", "⚠️  Save failed (localStorage full?)");
  }
}

function loadSession(id) {
  const raw = localStorage.getItem(SS_SESSION_PREFIX + id);
  if (!raw) { addMessage("error", "⚠️  Session not found."); return; }
  let payload;
  try { payload = JSON.parse(raw); }
  catch { addMessage("error", "⚠️  Session corrupted."); return; }

  if (inflight) inflight.abort();
  history = payload.history || [];
  logEl.innerHTML = "";
  clearChatRouteLayer();
  clearChatStagesLayer();

  // Replay message log so the user sees what was in the conversation.
  for (const m of history) {
    if (m.role === "user") {
      addMessage("user", typeof m.content === "string" ? m.content
                                                        : JSON.stringify(m.content));
    } else if (m.role === "assistant") {
      const txt = typeof m.content === "string" ? m.content
                                                : JSON.stringify(m.content);
      const div = addMessage("assistant", "");
      if (typeof marked !== "undefined") div.innerHTML = marked.parse(txt);
      else div.textContent = txt;
    }
  }

  // Restore map state.
  if (Array.isArray(payload.polylines)) {
    for (const p of payload.polylines) drawRouteOnMap(p);
    finalizeChatMap();
  }
  if (Array.isArray(payload.stagesByLeg)) {
    for (const [key, stages] of payload.stagesByLeg) {
      drawStagesOnMap(stages, key);
    }
  }
  addMessage("status", `📂 Loaded "${payload.title}"`);
}

function refreshLoadMenu() {
  if (!loadSel) return;
  const idx = _loadIndex();
  loadSel.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = idx.length ? `Load (${idx.length})…` : "Load…";
  loadSel.appendChild(placeholder);
  for (const entry of idx) {
    const opt = document.createElement("option");
    opt.value = entry.id;
    const stamp = entry.savedAt.slice(0, 16).replace("T", " ");
    opt.textContent = `${stamp} · ${entry.title}`;
    loadSel.appendChild(opt);
  }
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

if (saveBtn) {
  saveBtn.addEventListener("click", (e) => { e.preventDefault(); saveCurrentSession(); });
}
if (loadSel) {
  loadSel.addEventListener("change", (e) => {
    const id = loadSel.value;
    if (id) { loadSession(id); loadSel.value = ""; }
  });
  refreshLoadMenu();
}

})();  // end IIFE
