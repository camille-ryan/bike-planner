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

// Abort the SSE if no bytes arrive for this many ms. Between LLM
// round text-deltas, `: tick` and `round_start` heartbeats, and
// `tool_start` events, we normally emit bytes every few seconds.
// A 5-minute silence is a real stall — Anthropic dropped us, WSL
// sleep, network drop, or a tool that hangs (see #9 for split
// stalling on the whole-trip pass; that's fixed but the ceiling is
// still cheap insurance).
const CHAT_STREAM_IDLE_MS = 300_000;

console.log("[chat.js] loaded, elements:", {
  log: !!logEl, form: !!formEl, input: !!inputEl, send: !!sendBtn, reset: !!resetBtn,
});
if (!formEl || !inputEl || !sendBtn) {
  console.error("[chat.js] chat pane elements missing — chat disabled");
  throw new Error("chat pane elements missing");
}

let history = [];
let inflight = null;

// ---------- Session spend tracking ----------
// Every LLM round the backend emits a `usage` SSE event with the
// four token counts. Multiply by the current model's per-M-token
// prices and accumulate a running $ total for the browser session.
// Rates from Anthropic's public pricing for Sonnet 5 as of 2026:
//   input:       $3    / 1M tokens
//   output:      $15   / 1M tokens
//   cache_write: $3.75 / 1M tokens (1.25× input)
//   cache_read:  $0.30 / 1M tokens (0.1×  input)
// If you switch models (or Anthropic revises pricing), update here.
const SPEND_RATES = {
  input:       3.00e-6,
  output:     15.00e-6,
  cache_write: 3.75e-6,
  cache_read:  0.30e-6,
};
const sessionSpend = { usd: 0, in_tok: 0, out_tok: 0, cw_tok: 0, cr_tok: 0 };
const spendUsdEl    = document.getElementById("chat-spend-usd");
const spendDetailEl = document.getElementById("chat-spend-detail");

function _fmtUsd(v) {
  return v < 0.01 ? `<$0.01` : `$${v.toFixed(2)}`;
}

function _fmtTokShort(n) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000)     return `${(n / 1_000).toFixed(0)}k`;
  return String(n);
}

function _renderSpend() {
  if (spendUsdEl) spendUsdEl.textContent = _fmtUsd(sessionSpend.usd);
  if (spendDetailEl) {
    const total = sessionSpend.in_tok + sessionSpend.out_tok
                + sessionSpend.cw_tok + sessionSpend.cr_tok;
    spendDetailEl.textContent = total > 0
      ? ` · ${_fmtTokShort(total)} tokens`
      : "";
  }
}

function handleUsage(data) {
  const inp = data.input       || 0;
  const out = data.output      || 0;
  const cw  = data.cache_write || 0;
  const cr  = data.cache_read  || 0;
  const cost =
      inp * SPEND_RATES.input
    + out * SPEND_RATES.output
    + cw  * SPEND_RATES.cache_write
    + cr  * SPEND_RATES.cache_read;
  sessionSpend.usd    += cost;
  sessionSpend.in_tok += inp;
  sessionSpend.out_tok+= out;
  sessionSpend.cw_tok += cw;
  sessionSpend.cr_tok += cr;
  _renderSpend();
}

function resetSpend() {
  sessionSpend.usd = 0;
  sessionSpend.in_tok = sessionSpend.out_tok = 0;
  sessionSpend.cw_tok = sessionSpend.cr_tok = 0;
  _renderSpend();
}
_renderSpend();

// Detect the user's unit preference from their first message so map
// pins + GPX names match. Cached across the session; recomputed only
// when the log is cleared (Reset button).
let chatUnitMode = "km";  // "km" | "mi"

function detectUnitMode(text) {
  if (!text) return "km";
  const t = String(text).toLowerCase();
  // Look for a distance-y "mile" or "mi" reference; ignore "km".
  if (/\bmiles?\b|\bmi\b|\bmi\/day\b/.test(t)) return "mi";
  return "km";
}

function fmtDistance(km) {
  // Consistent 1-decimal formatting in the current unit.
  if (km == null || !isFinite(km)) return "?";
  if (chatUnitMode === "mi") {
    const mi = km * 0.6213711922;
    return `${mi.toFixed(1)} mi`;
  }
  return `${km.toFixed(1)} km`;
}

// ---------- rendering ----------

// Standard chat-log scroll behavior: auto-scroll to bottom ONLY when
// the user hasn't scrolled up to read something mid-stream. The flag
// tracks the user's intent — updated by the scroll listener below —
// and every append site replaces its unconditional scroll with a
// call to `scrollToBottomIfPinned()` which respects the flag.
let userScrolledUp = false;
logEl.addEventListener("scroll", () => {
  const dist = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight;
  userScrolledUp = dist > 100;
});

function scrollToBottomIfPinned() {
  if (!userScrolledUp) logEl.scrollTop = logEl.scrollHeight;
}

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  div.textContent = text;
  logEl.appendChild(div);
  scrollToBottomIfPinned();
  return div;
}

function addToolCall(name, input, output) {
  const div = document.createElement("div");
  div.className = "chat-msg tool";
  const summary = summarizeTool(name, input, output);
  div.innerHTML = `<span class="tool-name">🔧 ${escape(name)}</span> ${escape(summary)}`;
  logEl.appendChild(div);
  scrollToBottomIfPinned();
}

// Live "🔧 <name> (running…)" indicator that turns into the real
// tool_call bubble when the result comes back. Keyed by tool_use_id
// so `tool_start` and its matching `tool_call` line up even when
// multiple tools run in one round.
const pendingToolMsgs = new Map();

function addToolPending(id, name, input, container) {
  const div = document.createElement("div");
  div.className = "chat-msg tool pending";
  const argHint = summarizeTool(name, input, null);
  div.innerHTML =
    `<span class="tool-name">🔧 ${escape(name)}</span> ` +
    `${escape(argHint)} <span class="running">(running…)</span>`;
  (container || logEl).appendChild(div);
  scrollToBottomIfPinned();
  pendingToolMsgs.set(id, div);
}

function resolveToolPending(id, name, input, output) {
  const div = pendingToolMsgs.get(id);
  if (!div) return false;
  pendingToolMsgs.delete(id);
  const summary = summarizeTool(name, input, output);
  div.className = "chat-msg tool";
  div.innerHTML =
    `<span class="tool-name">🔧 ${escape(name)}</span> ${escape(summary)}`;
  return true;
}

// ---------- multi-agent bubble tracking ----------
// Each agent (supervisor / seg[i] / merge) gets its own bubble container
// with a header line, an inline tool tray, and its own streaming text
// area. `agent_start` creates it; text / tool_start / tool_call events
// tagged with `agent_id` route to it; `agent_end` swaps the header
// to a completed state.
const agentBubbles = new Map();  // agent_id -> {container, header, textDiv, tools, textBuf}

function _agentBubbleTitle(role, data) {
  if (role === "supervisor") return "▸ Planning corridor…";
  if (role === "merge")      return "▸ Merging plan…";
  if (role === "ask")        return "▸ Need a bit more info";
  if (role === "extract")    return "▸ Reading the plan…";
  if (role === "segment") {
    const i    = data.segment_i;
    const from = data.from_name ?? "?";
    const to   = data.to_name   ?? "?";
    return `▸ Segment ${i}: ${from} → ${to}`;
  }
  if (role === "lodging") {
    return `▸ Lodging near ${data.overnight_name ?? "?"}…`;
  }
  if (role === "transit") {
    return `▸ Train: ${data.from_name ?? "?"} → ${data.to_name ?? "?"}…`;
  }
  return `▸ Agent ${data.agent_id || ""}`;
}

function _agentBubbleDone(role, data) {
  const s = data.status === "failed" ? "✗" : "✓";
  const wall = data.wall_ms ? ` · ${(data.wall_ms / 1000).toFixed(1)}s` : "";
  if (role === "supervisor") return `${s} Corridor planned${wall}`;
  if (role === "merge")      return `${s} Plan complete${wall}`;
  if (role === "ask")        return "▸ Awaiting your reply";
  if (role === "extract")    return `${s} Read the plan${wall}`;
  if (role === "segment") {
    const i    = data.segment_i;
    const from = agentBubbles.get(`seg[${i}]`)?.headerData?.from_name ?? "?";
    const to   = agentBubbles.get(`seg[${i}]`)?.headerData?.to_name   ?? "?";
    return `${s} Segment ${i}: ${from} → ${to}${wall}`;
  }
  if (role === "lodging") {
    const hd = agentBubbles.get(data.agent_id)?.headerData || {};
    return `${s} Lodging near ${hd.overnight_name ?? "?"}${wall}`;
  }
  if (role === "transit") {
    const hd = agentBubbles.get(data.agent_id)?.headerData || {};
    return `${s} Train ${hd.from_name ?? "?"} → ${hd.to_name ?? "?"}${wall}`;
  }
  return `${s} Agent ${data.agent_id || ""}${wall}`;
}

function handleAgentStart(data) {
  const agentId = data.agent_id;
  if (!agentId || agentBubbles.has(agentId)) return;
  // Supervisor firing means a FRESH plan is starting. If old segment
  // state is on the map from a previous plan, clear it now so new
  // `segment_committed` events populate a clean slate. Enrichment /
  // extract / ask flows don't fire the supervisor and leave the
  // map intact.
  if (data.role === "supervisor"
      && (chatRoutePolylines.size > 0 || chatStagesByLeg.size > 0)) {
    clearChatRouteLayer();
    clearChatStagesLayer();
  }
  const container = document.createElement("div");
  container.className = `chat-agent-bubble role-${data.role}`;
  const header = document.createElement("div");
  header.className = "chat-agent-header";
  header.textContent = _agentBubbleTitle(data.role, data);
  const tools = document.createElement("div");
  tools.className = "chat-agent-tools";
  const textDiv = document.createElement("div");
  textDiv.className = "chat-agent-text";
  container.appendChild(header);
  container.appendChild(tools);
  container.appendChild(textDiv);
  logEl.appendChild(container);
  scrollToBottomIfPinned();
  agentBubbles.set(agentId, {
    container, header, tools, textDiv,
    textBuf: "",
    headerData: {
      segment_i:      data.segment_i,
      from_name:      data.from_name,
      to_name:        data.to_name,
      role:           data.role,
      overnight_i:    data.overnight_i,
      overnight_ref:  data.overnight_ref,
      overnight_name: data.overnight_name,
      pair_i:         data.pair_i,
    },
  });
}

function handleAgentEnd(data) {
  const agentId = data.agent_id;
  const bubble = agentBubbles.get(agentId);
  if (!bubble) return;
  const role = bubble.headerData?.role || data.role;
  bubble.header.textContent = _agentBubbleDone(role, data);
  if (data.status === "failed") {
    bubble.container.classList.add("failed");
    if (data.error) {
      const err = document.createElement("div");
      err.className = "chat-agent-error";
      err.textContent = `⚠️  ${data.error}`;
      bubble.container.appendChild(err);
    }
  } else {
    bubble.container.classList.add("done");
  }
}

function clearAgentBubbles() {
  agentBubbles.clear();
}

// Coordinator-committed segment: this is the sole source of truth
// for what shows on the map after multi-agent planning. Each segment
// commits ONCE with its final polyline + globally-numbered overnights.
function handleSegmentCommit(data) {
  const segKey = `seg[${data.segment_i}]`;
  if (Array.isArray(data.polyline) && data.polyline.length > 1) {
    drawRouteOnMap(data.polyline, segKey);
  }
  const overnights = Array.isArray(data.overnights) ? data.overnights : [];
  if (overnights.length) {
    // Build a stages-shaped array that `drawStagesOnMap` understands.
    // Skip the first overnight (start-of-segment = end-of-previous)
    // to avoid double-pinning the corridor hubs.
    const stages = [];
    for (let i = 1; i < overnights.length; i++) {
      const prev = overnights[i - 1];
      const cur  = overnights[i];
      if (!Array.isArray(cur.lonlat) || cur.lonlat.length !== 2) continue;
      // km per stage: prefer the segment agent's `km_from_prev` if it
      // was submitted, otherwise fall back to a haversine along the
      // polyline (rough, but avoids "undefined km" in labels).
      let km = (typeof cur.km_from_prev === "number" && isFinite(cur.km_from_prev))
        ? cur.km_from_prev
        : null;
      if (km == null && Array.isArray(data.polyline) && data.polyline.length > 1
          && Array.isArray(prev.lonlat) && prev.lonlat.length === 2) {
        km = _hav_m(prev.lonlat[0], prev.lonlat[1], cur.lonlat[0], cur.lonlat[1]) / 1000;
      }
      stages.push({
        day:         cur.day,
        from_ref:    prev.ref,
        from_name:   prev.name,
        to_ref:      cur.ref,
        to_name:     cur.name,
        to_lonlat:   cur.lonlat,
        km:          km != null ? Math.round(km * 10) / 10 : null,
      });
    }
    if (stages.length) drawStagesOnMap(stages, segKey);
  }
}

// ---------- Enrichment renderers (lodging + transit) ----------

function renderLodgingCards(bubble, input) {
  const hotels = Array.isArray(input?.hotels) ? input.hotels : [];
  const summary = input?.summary || "";
  const overnightName = bubble.headerData?.overnight_name || "";
  const wrap = document.createElement("div");
  wrap.className = "lodging-cards";
  if (summary) {
    const s = document.createElement("div");
    s.className = "lodging-summary";
    s.textContent = summary;
    wrap.appendChild(s);
  }
  if (!hotels.length) {
    const empty = document.createElement("div");
    empty.className = "lodging-empty";
    empty.textContent = "No nearby lodging in the OSM dataset for this overnight.";
    wrap.appendChild(empty);
  }
  for (const h of hotels) {
    const card = document.createElement("div");
    card.className = "lodging-card";
    // Every card gets a clickable title. OSM website is preferred;
    // fall back to a Google search for "<name> <town> hotel" so the
    // user can always click through.
    const href = h.website
      ? h.website
      : `https://www.google.com/search?q=${encodeURIComponent(
            (h.name || "hotel") + " " + overnightName)}`;
    const title = `<a href="${escape(href)}" target="_blank" rel="noopener">${escape(h.name || "(unnamed)")}</a>`;
    const meta = [];
    if (h.subtype)  meta.push(escape(h.subtype));
    if (h.stars)    meta.push(`${escape(String(h.stars))}★`);
    if (h.distance_m != null) {
      const dm = Math.round(Number(h.distance_m));
      // Flag far-from-town-center options prominently — bike-tourists
      // don't want a 10 km detour to their hotel after 60 km riding.
      let distLabel;
      if (dm >= 3000) {
        distLabel = `⚠ ${(dm / 1000).toFixed(1)} km from town`;
      } else if (dm >= 1500) {
        distLabel = `${(dm / 1000).toFixed(1)} km from town`;
      } else {
        distLabel = `${dm} m from town`;
      }
      meta.push(distLabel);
    }
    if (!h.website) meta.push("no OSM website — link is a Google search");
    card.innerHTML =
      `<div class="lodging-card-title">${title}</div>` +
      (meta.length ? `<div class="lodging-card-meta">${meta.join(" · ")}</div>` : "");
    wrap.appendChild(card);
  }
  bubble.textDiv.appendChild(wrap);
  scrollToBottomIfPinned();
}

function renderTransitLinks(bubble, input) {
  const wrap = document.createElement("div");
  wrap.className = "transit-block";
  if (typeof input?.direct_rail === "boolean") {
    const badge = document.createElement("div");
    badge.className = `transit-badge ${input.direct_rail ? "direct" : "transfer"}`;
    badge.textContent = input.direct_rail
      ? "✓ Direct-train available"
      : "⚠ Transfer required — no direct train found";
    wrap.appendChild(badge);
  }
  const summary = input?.summary || "";
  if (summary) {
    const s = document.createElement("div");
    s.className = "transit-summary";
    s.textContent = summary;
    wrap.appendChild(s);
  }
  const urls = Array.isArray(input?.booking_urls) ? input.booking_urls : [];
  if (urls.length) {
    const ul = document.createElement("ul");
    ul.className = "transit-links";
    for (const u of urls) {
      const li = document.createElement("li");
      li.innerHTML =
        `<a href="${escape(u.url)}" target="_blank" rel="noopener">Book on ${escape(u.operator || "operator")}</a>`;
      ul.appendChild(li);
    }
    wrap.appendChild(ul);
  }
  bubble.textDiv.appendChild(wrap);
  scrollToBottomIfPinned();
}

function escape(s) {
  return String(s ?? "").replace(/[&<>]/g, c =>
    ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));
}

function summarizeTool(name, input, output) {
  if (output && output.error) return `— error: ${output.error}`;
  // Post-result branches (output present) show what came back.
  // Pre-result branches (output === null, from the "running…" row)
  // show the arg shape without "→ undefined" placeholders.
  const done = output != null;
  switch (name) {
    case "search_anchors":
      return done
        ? `“${input.query}” → ${output?.results?.length || 0} anchor${(output?.results?.length || 0) === 1 ? "" : "s"}`
        : `“${input.query}”`;
    case "route": {
      const via = Array.isArray(input.via_refs) && input.via_refs.length
        ? ` via ${input.via_refs.length}` : "";
      const path = `${input.from_ref || input.from_lonlat} → ${input.to_ref || input.to_lonlat}${via}`;
      return done ? `${path}  ⇒  ${output?.total_km ?? "?"} km` : path;
    }
    case "stations_near":
      return done
        ? `${output?.stations?.length || 0} stations within ${input.radius_km || 15} km of ${input.lon.toFixed(2)},${input.lat.toFixed(2)}`
        : `${input.radius_km || 15} km around ${input.lon.toFixed(2)},${input.lat.toFixed(2)}`;
    case "stations_along_route": {
      const via = Array.isArray(input.via_refs) && input.via_refs.length
        ? ` via ${input.via_refs.length}` : "";
      const path = `${input.from_ref || "?"} → ${input.to_ref || "?"}${via}`;
      return done
        ? `${path}  ⇒  ${output?.anchors?.length ?? "?"} rail-served anchors`
        : path;
    }
    case "direct_rail_service":
      return done
        ? `${input.from_ref} ↔ ${input.to_ref}  ⇒  ${output?.direct ? "direct" : "no direct"}`
        : `${input.from_ref} ↔ ${input.to_ref}`;
    case "direct_rail_service_batch": {
      const n = Array.isArray(input.pairs) ? input.pairs.length : 0;
      const nd = done
        ? (output?.results?.filter(r => r && r.direct).length ?? "?")
        : null;
      return done ? `${n} pair${n === 1 ? "" : "s"}  ⇒  ${nd} direct` : `${n} pair${n === 1 ? "" : "s"}`;
    }
    case "split_into_stages": {
      const via = Array.isArray(input.via_refs) && input.via_refs.length
        ? ` via ${input.via_refs.length}` : "";
      const path = `${input.from_ref || "?"} → ${input.to_ref || "?"}${via} @ ~${input.target_km_per_day || 80} km/day`;
      return done ? `${path}  ⇒  ${output?.n_days ?? "?"} stages` : path;
    }
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
// map build up as Claude iterates. Keyed by the (from_ref, to_ref,
// via_refs) triple of the route call, so a revised call for the
// same corridor REPLACES its polyline instead of overlaying — the
// old array-append behavior painted loops at every revision, exactly
// what the user was seeing during planning. On ✓ Done we collapse
// to just the longest polyline via `finalizeChatMap()`, since even
// with dedup, sub-legs still overlap the full corridor.
let chatRoutePolylines = new Map();

// Latest `split_into_stages` result, used by the GPX-download button
// to slice the polyline into per-day tracks.
let chatLatestStages = null;

function clearChatRouteLayer() {
  chatRoutePolylines = new Map();
  chatLatestStages = null;
  const map = window.map;
  if (map && map.getSource("chat-route")) {
    map.getSource("chat-route").setData({ type: "FeatureCollection", features: [] });
  }
}

function _routesToFeatures(polylines) {
  const arr = Array.isArray(polylines) ? polylines : Array.from(polylines);
  return arr.map(p => ({
    type: "Feature",
    geometry: { type: "LineString", coordinates: p },
    properties: {},
  }));
}

// A tool-call's (from_ref, to_ref) identifies the corridor endpoints.
// via_refs is intentionally NOT part of the key: a revised call for
// the same (from, to) — say the model first tries direct Graz→CPH
// then re-routes via the rail hubs — should REPLACE the previous
// polyline, not paint on top of it. Different (from, to) pairs
// (say per-segment sub-agent polylines) still coexist naturally
// because they have distinct keys.
function _routeKey(input) {
  const from = input?.from_ref ?? input?.from_lonlat ?? "?";
  const to   = input?.to_ref   ?? input?.to_lonlat   ?? "?";
  return `${from}→${to}`;
}

function drawRouteOnMap(polyline, key) {
  const map = window.map;
  if (!map || !polyline || polyline.length < 2) return;
  chatRoutePolylines.set(key ?? String(chatRoutePolylines.size), polyline);
  ensureChatRouteLayer();
  map.getSource("chat-route").setData({
    type: "FeatureCollection",
    features: _routesToFeatures(chatRoutePolylines.values()),
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
          // city + distance label rendered below (chat-stages-labels)
          label: `${s.to_name || "?"} · ${fmtDistance(s.km)}`,
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

// Called from the SSE `done` path. Historically this collapsed the
// accumulated overlays to just the longest polyline because single-
// agent runs produced overlapping sub-leg polylines. With multi-
// agent, each SEGMENT is a distinct piece of the corridor (Graz→
// Wien, Wien→Brno, …) — we want all of them to stay visible. So
// we now leave the polylines as-drawn and just concatenate them
// for GPX export purposes. Returns the concatenation (in insertion
// order, which is segment order since agent_start fires per-
// segment) so the GPX button has something to save.
function finalizeChatMap() {
  if (chatRoutePolylines.size === 0) return null;
  const all = Array.from(chatRoutePolylines.values());
  // Concat in insertion order — dedup consecutive duplicate points
  // at the seams (segment K's end coord = segment K+1's start).
  const concat = [];
  for (const seg of all) {
    for (const pt of seg) {
      const last = concat[concat.length - 1];
      if (last && last[0] === pt[0] && last[1] === pt[1]) continue;
      concat.push(pt);
    }
  }
  return concat;
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
          name: `Day ${s.day}: ${s.from_name ?? "?"} → ${s.to_name ?? "?"} (${fmtDistance(s.km)})`,
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

function handleToolResult(name, input, output, agentId) {
  // Chat-bubble rendering is now done by the tool_call caller so it
  // can update the "(running…)" pending row in place. This function
  // just handles the side effects (map draw, sidebar sync).
  //
  // Multi-agent: SEGMENT agents call `route` / `split_into_stages`
  // many times mid-planning (verify, revise, re-verify). Drawing
  // every intermediate polyline creates "ghost loops" as the agent
  // iterates. Filter out those calls here — the coordinator emits
  // one `segment_committed` event per segment when its plan is
  // finalized, and THAT is where the frontend renders the segment's
  // final polyline + pins. Similarly filter supervisor / merge /
  // enrichment agents — their tool calls (search_anchors, rail_path,
  // direct_rail_service, search_lodging) don't need to touch the map.
  if (agentId && agentId !== "main") return;
  if (name === "route" && output?.polyline?.length) {
    drawRouteOnMap(output.polyline, _routeKey(input));
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
    // Key stage pins by the full (from, to, via_refs) so a re-split
    // of the same corridor REPLACES its pin set. A key that ignores
    // via_refs would let split(Wien→Praha) and split(Wien→Praha via
    // Brno) both survive and paint their day pins on top of each
    // other.
    drawStagesOnMap(output.stages, _routeKey(input));
  }
}

// ---------- SSE parser ----------
// Fetch-with-body streaming; parse `event:` + `data:` lines manually.

// Enrichment triggers — mirror of _looks_like_enrichment_request in
// api/app/chat.py. Used to (a) NOT clear the map layers on a booking
// follow-up (user wants to keep seeing the route while lodging cards
// fill in) and (b) leave the plan visible.
const ENRICHMENT_TRIGGERS = [
  "book lodging", "book train", "book hotel", "find lodging",
  "find hotel", "book trains", "book the trains", "lodging plan",
  "hotels please", "book everything", "yes please book",
];

function _isEnrichmentFollowup(userText) {
  if (!userText) return false;
  const t = String(userText).toLowerCase();
  if (!ENRICHMENT_TRIGGERS.some(trig => t.includes(trig))) return false;
  // Needs a prior assistant turn to enrich.
  return history.some(m => m.role === "assistant");
}

async function streamChat(userText) {
  const userDiv = addMessage("user", userText);
  history.push({ role: "user", content: userText });

  // Detect unit preference from the first turn's user message. If they
  // say "50 miles/day" we render pins / GPX in miles; else km. This
  // stays sticky across follow-up turns in the same session (Reset
  // clears history and re-detects on the next first message).
  if (history.filter(m => m.role === "user").length === 1) {
    chatUnitMode = detectUnitMode(userText);
  }

  // NEVER preemptively wipe the map. The map should reflect the
  // MOST RECENT state — the segment_committed events overwrite
  // polylines / pins by segment_i as they arrive, so an iteration
  // ("fix the Wien-Praha stretch") replaces just that segment's
  // slot and leaves the rest of the plan visible. Users can hit
  // Reset to clear the whole session state.

  sendBtn.disabled = true;
  sendBtn.textContent = "Thinking…";

  // Live status line above the assistant text — replaces the blank
  // "waiting" gap that made completion hard to spot. Cleared on `done`.
  const statusDiv = document.createElement("div");
  statusDiv.className = "chat-msg status";
  statusDiv.textContent = "⏳ Planning…";
  logEl.appendChild(statusDiv);
  scrollToBottomIfPinned();

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
  // The assistant text stream may span many LLM rounds, each of which
  // can produce a short rationale before its tool calls plus a chunk
  // of narrative. Each round gets its own bubble (reset by
  // `round_start`) so text and tool rows interleave naturally in the
  // log. `asstText` accumulates across rounds for the history entry
  // saved on `done`; `asstDiv` / `asstBubbleText` are for the current
  // bubble only.
  let asstDiv = null;
  let asstText = "";
  let asstBubbleText = "";

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

  clearAgentBubbles();

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
        // Multi-agent path: text / tool_start / tool_call events carry
        // an `agent_id` and route to that agent's dedicated bubble.
        // Single-agent path uses `agent_id="main"` and the same code
        // renders it as one supervisor-less flat log (main's bubble is
        // never created via agent_start, so we hit the addMessage
        // fallback and behave like before).
        const agentId  = ev.data && ev.data.agent_id;
        const isMulti  = agentId && agentId !== "main";
        const bubble   = isMulti ? agentBubbles.get(agentId) : null;

        if (ev.event === "agent_start") {
          handleAgentStart(ev.data);
        } else if (ev.event === "agent_end") {
          handleAgentEnd(ev.data);
        } else if (ev.event === "round_start") {
          // Start a new bubble for this round's text so a "rationale
          // sentence + narrative" chunk lands right above THIS round's
          // tool rows, not glommed onto whatever the previous round
          // wrote. In multi-agent mode, each agent has its own bubble
          // that already accumulates within-agent text — the per-round
          // reset only applies to the single-agent flat log.
          if (bubble) {
            bubble.textBuf = "";
            bubble.textDiv.innerHTML = "";
          } else {
            asstDiv = null;
            asstBubbleText = "";
          }
        } else if (ev.event === "text") {
          const delta = ev.data.delta || "";
          asstText += delta;
          if (bubble) {
            bubble.textBuf += delta;
            if (typeof marked !== "undefined") {
              bubble.textDiv.innerHTML = marked.parse(bubble.textBuf);
            } else {
              bubble.textDiv.textContent = bubble.textBuf;
            }
            scrollToBottomIfPinned();
          } else {
            asstBubbleText += delta;
            // Lazy-create the bubble on first delta of the current
            // round so it lands below any tool calls that already
            // streamed in.
            if (!asstDiv) asstDiv = addMessage("assistant", "");
            if (typeof marked !== "undefined") {
              asstDiv.innerHTML = marked.parse(asstBubbleText);
            } else {
              asstDiv.textContent = asstBubbleText;
            }
            scrollToBottomIfPinned();
          }
        } else if (ev.event === "tool_start") {
          const container = bubble ? bubble.tools : null;
          addToolPending(ev.data.id, ev.data.name, ev.data.input, container);
          tickStatus();
        } else if (ev.event === "tool_call") {
          // Enrichment terminal tools carry their user-facing payload
          // in `input` (the sub-agent's submit_lodging / submit_transit
          // args). Render them as cards inside the bubble's text area
          // instead of a stock 🔧 tool row.
          if (bubble && ev.data.name === "submit_lodging") {
            renderLodgingCards(bubble, ev.data.input);
            resolveToolPending(
              ev.data.id, ev.data.name, ev.data.input, ev.data.output);
            toolCount += 1;
            tickStatus();
            continue;
          }
          if (bubble && ev.data.name === "submit_transit") {
            renderTransitLinks(bubble, ev.data.input);
            resolveToolPending(
              ev.data.id, ev.data.name, ev.data.input, ev.data.output);
            toolCount += 1;
            tickStatus();
            continue;
          }
          // If we already showed a "(running…)" pending row for this
          // id, mutate it in place; otherwise append a fresh bubble
          // in the right container.
          if (!resolveToolPending(
                ev.data.id, ev.data.name, ev.data.input, ev.data.output)) {
            const container = bubble ? bubble.tools : null;
            const div = document.createElement("div");
            div.className = "chat-msg tool";
            const summary = summarizeTool(ev.data.name, ev.data.input, ev.data.output);
            div.innerHTML =
              `<span class="tool-name">🔧 ${escape(ev.data.name)}</span> ${escape(summary)}`;
            (container || logEl).appendChild(div);
          }
          handleToolResult(ev.data.name, ev.data.input, ev.data.output, agentId);
          toolCount += 1;
          tickStatus();
        } else if (ev.event === "usage") {
          // Per-round token count from an LLM call; feeds the
          // session spend badge in the chat header.
          handleUsage(ev.data);
        } else if (ev.event === "segment_committed") {
          // Coordinator's per-segment commit: renumbered overnights +
          // final polyline. This is the SINGLE place segment map
          // state lands. Intermediate route/split calls from segment
          // agents are filtered out in `handleToolResult`.
          handleSegmentCommit(ev.data);
        } else if (ev.event === "error") {
          // Server-side friendly error mid-stream. Attach it to the
          // in-flight assistant bubble so the user sees the failure in
          // context, not as a stray red line at the bottom.
          if (asstDiv && !asstText) { asstDiv.remove(); asstDiv = null; }
          addMessage("error", `⚠️  ${ev.data.message || "Unknown error."}`);
        } else if (ev.event === "done") {
          // stop_reason is on ev.data.stop_reason if we ever want to show it.
          // In multi-agent mode, per-agent `done` events precede the
          // final coordinator `done`; we treat them uniformly.
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
    polylines: Array.from(chatRoutePolylines.entries()),
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

  // Restore map state. Handle both the new [key, polyline][] and the
  // legacy polyline[] shape so older saved sessions still load.
  if (Array.isArray(payload.polylines)) {
    for (const entry of payload.polylines) {
      if (Array.isArray(entry) && entry.length === 2
          && typeof entry[0] === "string") {
        // New shape: [key, polyline]
        drawRouteOnMap(entry[1], entry[0]);
      } else {
        // Legacy shape: raw polyline
        drawRouteOnMap(entry);
      }
    }
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
  resetSpend();
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
