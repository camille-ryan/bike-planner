// Bike-routing front-end. Single-file JS, vanilla (no framework).
// Talks to the FastAPI service via the /api/ prefix proxied by nginx.

const API = "/api";

// --- map setup ---------------------------------------------------------

const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: {
      osm: {
        type: "raster",
        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        maxzoom: 19,
      },
    },
    layers: [{ id: "osm", type: "raster", source: "osm" }],
  },
  center: [13.5, 51],
  zoom: 5,
  hash: true,
});
map.addControl(new maplibregl.NavigationControl(), "top-right");
map.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: "metric" }), "bottom-left");

// --- state -------------------------------------------------------------

const state = {
  // 1st click = start, 2nd = end, every additional click inserts a via-point
  // at whichever segment midpoint is closest to the click. Required for the
  // long-corridor case where BRouter can't plan ~1300km point-to-point.
  waypoints: [],      // [[lon, lat], ...]
  routes: [],         // GeoJSON Features from /route
  activeIdx: 0,
  legs: [],           // from /stages
  poiMarkers: { viewpoint: [], lodging: [], food: [], bike_service: [], water: [] },
  waypointMarkers: [],
};

// --- map sources / layers (initialized once map loads) -----------------

map.on("load", () => {
  map.addSource("route-active", { type: "geojson", data: emptyFC() });
  map.addSource("route-alts",   { type: "geojson", data: emptyFC() });
  map.addSource("legs",         { type: "geojson", data: emptyFC() });

  map.addLayer({
    id: "route-alts-line",
    type: "line",
    source: "route-alts",
    paint: { "line-color": "#888", "line-width": 3, "line-opacity": 0.45, "line-dasharray": [1, 1.5] },
  });
  map.addLayer({
    id: "route-active-line",
    type: "line",
    source: "route-active",
    paint: { "line-color": "#2c5", "line-width": 5, "line-opacity": 0.95 },
  });
  map.addLayer({
    id: "legs-points",
    type: "circle",
    source: "legs",
    paint: {
      "circle-radius": 8,
      "circle-color": "#fff",
      "circle-stroke-color": "#c52",
      "circle-stroke-width": 3,
    },
  });
  map.addLayer({
    id: "legs-labels",
    type: "symbol",
    source: "legs",
    layout: {
      "text-field": ["get", "n"],
      "text-size": 11,
      "text-font": ["Noto Sans Regular"],
      "text-allow-overlap": true,
    },
    paint: { "text-color": "#000" },
  });
});

// --- helpers -----------------------------------------------------------

function emptyFC() { return { type: "FeatureCollection", features: [] }; }

function fmtKm(m) { return (m / 1000).toFixed(1) + " km"; }
function fmtH(s)  { const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60); return `${h}h${String(m).padStart(2, "0")}`; }

async function api(path, params) {
  const u = new URL(API + path, location.origin);
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== undefined && v !== null) u.searchParams.set(k, String(v));
  }
  const r = await fetch(u);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

// --- click to place waypoints -----------------------------------------

map.on("click", (e) => {
  insertWaypoint([e.lngLat.lng, e.lngLat.lat]);
  refreshWaypointMarkers();
  const ready = state.waypoints.length >= 2;
  document.getElementById("route-btn").disabled = !ready;
  document.getElementById("stages-btn").disabled = !ready;
});

// Insert by closest-segment-midpoint: a click between Prague and Berlin lands
// in the right slot regardless of click order. First two clicks just append
// (start, end); subsequent clicks pick the segment they're nearest to.
function insertWaypoint(lonlat) {
  if (state.waypoints.length < 2) {
    state.waypoints.push(lonlat);
    return;
  }
  let bestIdx = state.waypoints.length;  // default: append at end
  let bestDist = Infinity;
  for (let i = 0; i < state.waypoints.length - 1; i++) {
    const a = state.waypoints[i], b = state.waypoints[i + 1];
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const d = haversine(lonlat, mid);
    if (d < bestDist) { bestDist = d; bestIdx = i + 1; }
  }
  state.waypoints.splice(bestIdx, 0, lonlat);
}

function refreshWaypointMarkers() {
  state.waypointMarkers.forEach(m => m.remove());
  state.waypointMarkers = [];
  state.waypoints.forEach((p, i) => {
    const isStart = i === 0;
    const isEnd = i === state.waypoints.length - 1;
    const color = isStart ? "#2c5" : (isEnd ? "#c52" : "#888");
    const label = isStart ? "S" : (isEnd ? "E" : String(i));
    const el = document.createElement("div");
    el.style.cssText = `width:22px;height:22px;border-radius:50%;background:${color};color:white;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:13px;border:2px solid white;box-shadow:0 1px 3px rgba(0,0,0,0.3);cursor:grab;`;
    el.textContent = label;
    const m = new maplibregl.Marker({ element: el, draggable: false }).setLngLat(p).addTo(map);
    state.waypointMarkers.push(m);
  });
}

document.getElementById("clear-btn").addEventListener("click", clearAll);

function clearAll() {
  state.waypoints = [];
  state.routes = [];
  state.legs = [];
  state.activeIdx = 0;
  document.getElementById("route-btn").disabled = true;
  document.getElementById("stages-btn").disabled = true;
  state.waypointMarkers.forEach(m => m.remove());
  state.waypointMarkers = [];
  if (map.getSource("route-active")) map.getSource("route-active").setData(emptyFC());
  if (map.getSource("route-alts"))   map.getSource("route-alts").setData(emptyFC());
  if (map.getSource("legs"))         map.getSource("legs").setData(emptyFC());
  document.getElementById("results").innerHTML = "";
  document.getElementById("elevation").innerHTML = "";
}

// --- route ------------------------------------------------------------

document.getElementById("route-btn").addEventListener("click", async () => {
  if (state.waypoints.length < 2) return;
  const profile = document.getElementById("profile").value;
  const alternatives = +document.getElementById("alternatives").value;
  const rerank = document.getElementById("rerank").checked;
  const lonlats = state.waypoints.map(p => p.join(",")).join("|");
  setBusy(`Routing ${state.waypoints.length} waypoints…`);
  try {
    const r = await api("/route", { lonlats, profile, alternatives, rerank });
    state.routes = r.routes;
    state.activeIdx = 0;
    renderRoutes();
  } catch (e) {
    setError(e.message);
  }
});

function renderRoutes() {
  const active = state.routes[state.activeIdx];
  const others = state.routes.filter((_, i) => i !== state.activeIdx);
  map.getSource("route-active").setData({ type: "FeatureCollection", features: active ? [active] : [] });
  map.getSource("route-alts").setData({ type: "FeatureCollection", features: others });

  if (active) {
    const bbox = lineBbox(active.geometry.coordinates);
    map.fitBounds(bbox, { padding: 60, maxZoom: 13 });
    drawElevation(active.geometry.coordinates);
  }

  const html = state.routes.map((r, i) => {
    const p = r.properties;
    const km = fmtKm(+p["track-length"] || 0);
    const t  = fmtH(+p["total-time"] || 0);
    const climb = `${Math.round(+(p["filtered ascend"] || p["filtered-ascend"]) || 0)} m`;
    const sc = p.scoring;
    const cls = i === state.activeIdx ? "active" : "";
    const name = sc ? `Alt ${p.alternativeidx} · score ${sc.composite_score}` : `Alt ${p.alternativeidx}`;
    let extras = "";
    if (sc) extras = `<div class="stat"><span>curvy descent</span><span>${sc.curvy_descent_penalty}</span></div>
                     <div class="stat"><span>viewpoints near</span><span>${sc.viewpoints_near_route}</span></div>`;
    return `<div class="route-card ${cls}" data-idx="${i}">
      <div class="name">${name}</div>
      <div class="stat"><span>distance</span><span>${km}</span></div>
      <div class="stat"><span>climb</span><span>${climb}</span></div>
      <div class="stat"><span>time</span><span>${t}</span></div>
      ${extras}
    </div>`;
  }).join("");
  document.getElementById("results").innerHTML = html;
  document.querySelectorAll(".route-card").forEach(el => {
    el.addEventListener("click", () => {
      state.activeIdx = +el.dataset.idx;
      renderRoutes();
    });
  });
}

function lineBbox(coords) {
  let n = 90, s = -90, e = -180, w = 180;
  for (const c of coords) {
    if (c[0] < w) w = c[0];
    if (c[0] > e) e = c[0];
    if (c[1] < n) n = c[1];
    if (c[1] > s) s = c[1];
  }
  return [[w, n], [e, s]];
}

// --- elevation profile ------------------------------------------------

function drawElevation(coords) {
  const el = document.getElementById("elevation");
  el.innerHTML = "";
  if (!coords.length || coords[0].length < 3) return;
  const w = el.clientWidth, h = el.clientHeight - 4;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.style.cssText = "width:100%;height:100%;";
  // Cumulative distances + elevations
  let dist = 0, prev = coords[0];
  const pts = [{ d: 0, e: prev[2] }];
  for (let i = 1; i < coords.length; i++) {
    const c = coords[i];
    dist += haversine(prev, c);
    pts.push({ d: dist, e: c[2] });
    prev = c;
  }
  const eMin = Math.min(...pts.map(p => p.e));
  const eMax = Math.max(...pts.map(p => p.e));
  const span = Math.max(1, eMax - eMin);
  const dMax = pts[pts.length - 1].d || 1;
  const xy = pts.map(p => [
    (p.d / dMax) * w,
    h - ((p.e - eMin) / span) * (h - 12) - 6
  ]);
  const path = "M " + xy.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(" L ");
  const poly = document.createElementNS("http://www.w3.org/2000/svg", "path");
  poly.setAttribute("d", path + ` L ${w} ${h} L 0 ${h} Z`);
  poly.setAttribute("fill", "rgba(44, 197, 85, 0.25)");
  poly.setAttribute("stroke", "#2c5");
  poly.setAttribute("stroke-width", "1");
  svg.appendChild(poly);
  // Labels
  const lbl = document.createElement("div");
  lbl.style.cssText = "position:absolute;top:4px;left:8px;font-size:11px;color:#555;background:rgba(255,255,255,0.85);padding:1px 4px;border-radius:2px;";
  lbl.textContent = `${Math.round(eMin)}–${Math.round(eMax)} m · ${(dMax / 1000).toFixed(1)} km · climb ${pts.reduce((s, p, i) => i ? s + Math.max(0, p.e - pts[i-1].e) : 0, 0).toFixed(0)} m`;
  el.appendChild(svg);
  el.appendChild(lbl);
}

function haversine(p1, p2) {
  const R = 6371000;
  const toRad = d => d * Math.PI / 180;
  const dlat = toRad(p2[1] - p1[1]);
  const dlon = toRad(p2[0] - p1[0]);
  const a = Math.sin(dlat / 2) ** 2 + Math.cos(toRad(p1[1])) * Math.cos(toRad(p2[1])) * Math.sin(dlon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

// --- stages ------------------------------------------------------------

document.getElementById("stages-btn").addEventListener("click", async () => {
  if (state.waypoints.length < 2) return;
  // Stages currently splits a single from→to route. With multiple waypoints
  // we treat the first and last as the corridor endpoints; if you want
  // stages tied to your via-points, wait for that feature in a later version.
  const profile = document.getElementById("profile").value;
  setBusy("Planning stages…");
  try {
    const r = await api("/stages", {
      from: state.waypoints[0].join(","),
      to:   state.waypoints[state.waypoints.length - 1].join(","),
      profile,
      target_km: 100,
      lodging_radius_m: 3000,
    });
    state.legs = r.legs;
    renderLegs(r);
  } catch (e) {
    setError(e.message);
  }
});

function renderLegs(stagesResp) {
  const features = stagesResp.legs.map((l, i) => ({
    type: "Feature",
    properties: { n: i + 1 },
    geometry: { type: "Point", coordinates: l.end },
  }));
  map.getSource("legs").setData({ type: "FeatureCollection", features });
  const html = `
    <h2>Stages (${stagesResp.total_legs} legs · ${(stagesResp.total_length_m / 1000).toFixed(0)} km total)</h2>
    ${stagesResp.legs.map((l, i) => `
      <div class="leg-card">
        <div class="leg-num">Leg ${i + 1} · ${(l.length_m / 1000).toFixed(1)} km · ${l.ascend_m} m climb</div>
        <div class="lodging">${
          l.lodging.length === 0
            ? "<em>no lodging within 3 km</em>"
            : l.lodging.slice(0, 5).map(p =>
                `<span class="lodging-item">${p.subtype}: ${p.name || "(unnamed)"} · ${p.distance_m} m</span>`
              ).join("")
        }</div>
      </div>
    `).join("")}
  `;
  document.getElementById("results").innerHTML = html;
}

// --- POI overlay -------------------------------------------------------

const POI_COLORS = {
  viewpoint: "#1e88e5",
  lodging:   "#7e57c2",
  food:      "#ef6c00",
  bike_service: "#00897b",
  water:     "#0097a7",
};

function refreshPois() {
  const cats = [...document.querySelectorAll('#layers input[type=checkbox]')].filter(c => c.checked).map(c => c.dataset.cat);
  // remove existing markers
  for (const cat of Object.keys(state.poiMarkers)) {
    state.poiMarkers[cat].forEach(m => m.remove());
    state.poiMarkers[cat] = [];
  }
  if (cats.length === 0 || map.getZoom() < 9) return;  // too zoomed out
  const b = map.getBounds();
  const bbox = `${b.getWest().toFixed(4)},${b.getSouth().toFixed(4)},${b.getEast().toFixed(4)},${b.getNorth().toFixed(4)}`;
  api("/pois", { bbox, category: cats.join(","), limit: 500 }).then(r => {
    for (const p of r.items) {
      const el = document.createElement("div");
      el.style.cssText = `width:10px;height:10px;border-radius:50%;background:${POI_COLORS[p.category] || "#666"};border:1.5px solid white;box-shadow:0 0 2px rgba(0,0,0,0.4);cursor:pointer;`;
      const popup = new maplibregl.Popup({ offset: 8 }).setHTML(
        `<strong>${p.name || "(unnamed)"}</strong><br><small>${p.category}/${p.subtype}</small>`
      );
      const m = new maplibregl.Marker({ element: el }).setLngLat([p.lon, p.lat]).setPopup(popup).addTo(map);
      state.poiMarkers[p.category]?.push(m);
    }
  }).catch(e => console.warn("pois fetch failed", e));
}

let poiTimer = null;
map.on("moveend", () => {
  clearTimeout(poiTimer);
  poiTimer = setTimeout(refreshPois, 250);
});
document.querySelectorAll('#layers input[type=checkbox]').forEach(c => {
  c.addEventListener("change", refreshPois);
});

// --- status helpers ---------------------------------------------------

function setBusy(msg) {
  document.getElementById("results").innerHTML = `<div class="route-card"><em>${msg}</em></div>`;
}

function setError(msg) {
  document.getElementById("results").innerHTML = `<div class="route-card" style="border-color:#c52;"><strong>Error</strong><div class="stat">${msg}</div></div>`;
}
