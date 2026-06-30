// Bike-routing front-end. Single-file JS, vanilla (no framework).
// Talks to the FastAPI service via the /api/ prefix proxied by nginx.

const API = "/api";

// V3 multi-profile routing: every Route click fires one request per
// profile in parallel; the map renders all 5 result lines colored by
// profile. Color set matches the bake-time / city-routes palette so
// the UI stays visually consistent across the codebase.
const PROFILES = ["direct", "vineyard_lover", "forest_lover", "views", "water"];
const PROFILE_COLORS = {
  direct:         "#888888",
  vineyard_lover: "#7a3380",
  forest_lover:   "#1f7a3a",
  views:          "#d4623a",
  water:          "#2a6dc4",
};
// Fixed profile used for non-routing UI bits (SPT cell visualization,
// anchor availability check). `direct` is the broadest, most-populated
// profile; using a single profile here avoids re-introducing a dropdown.
const UI_PROFILE = "direct";

// --- map setup ---------------------------------------------------------

const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: {
      basemap: {
        // CartoDB Dark Matter — free, no API key, designed for data
        // overlays. Four subdomains in the tile list let MapLibre
        // parallelise tile requests across them.
        type: "raster",
        tiles: [
          "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
          "https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
          "https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
          "https://d.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        ],
        tileSize: 256,
        attribution:
          '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors · ' +
          '© <a href="https://carto.com/attributions">CARTO</a>',
        maxzoom: 19,
      },
    },
    layers: [{ id: "basemap", type: "raster", source: "basemap" }],
  },
  center: [13.5, 51],
  zoom: 5,
  hash: true,
});
map.addControl(new maplibregl.NavigationControl(), "top-right");
map.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: "metric" }), "bottom-left");

// --- state -------------------------------------------------------------

const state = {
  // Each waypoint: { role: 'start'|'mid'|'end', coord: [lon, lat]|null,
  //                  input: HTMLInputElement, row: HTMLElement }
  // The chainless SPT engine routes pairwise; midpoints chain N legs.
  waypoints: [],
  pickArmed: null,    // waypoint expecting next map click, or null
  // routesByProfile: { profile: GeoJSON Feature } — one merged
  // multi-leg route per profile. Cleared on Clear; populated by
  // routeNow with the 5 parallel /trunk/route results.
  routesByProfile: {},
  routeReqId: 0,      // increments per routeNow; stale responses are discarded
  poiMarkers: { viewpoint: [], lodging: [], food: [], bike_service: [], water: [] },
  waypointMarkers: [],
  cities: null,
  anchorMarkers: [],
  shownCellIdx: null,
};

// --- map sources / layers (initialized once map loads) -----------------

map.on("load", () => {
  // tolerance: 0 disables MapLibre's Douglas-Peucker simplification of
  // the route geometry at low zoom — the default 0.375 px tolerance
  // drops vertices aggressively at zoom <10, which makes a long
  // multi-leg route look like a series of disjoint segments. The route
  // is at most ~10k points; the memory cost of keeping every vertex at
  // every zoom is negligible.
  map.addSource("routes-multi", {
    type: "geojson", data: emptyFC(), tolerance: 0,
  });
  map.addLayer({
    id: "routes-multi-line",
    type: "line",
    source: "routes-multi",
    paint: {
      "line-color": [
        "match", ["get", "profile"],
        "direct",         PROFILE_COLORS.direct,
        "vineyard_lover", PROFILE_COLORS.vineyard_lover,
        "forest_lover",   PROFILE_COLORS.forest_lover,
        "views",          PROFILE_COLORS.views,
        "water",          PROFILE_COLORS.water,
        "#aaa",
      ],
      "line-width": 4,
      "line-opacity": 0.85,
    },
  });

  map.addSource("cell-gradient", { type: "geojson", data: emptyFC() });
  // SPT visualization: each non-seed vertex's edge to its parent_local,
  // colored by cost-from-anchor. Lives below the route line so an
  // active route stays readable on top.
  map.addLayer({
    id: "cell-gradient-lines",
    type: "line",
    source: "cell-gradient",
    layout: { "line-cap": "butt", "line-join": "miter" },
    paint: {
      "line-width": [
        "interpolate", ["linear"], ["zoom"],
        6,  1,
        10, 1.5,
        13, 2.5,
        16, 4,
      ],
      "line-color": [
        "interpolate", ["linear"], ["get", "cost"],
        0,       "#10b981",
        50000,   "#facc15",
        100000,  "#dc2626",
      ],
      "line-opacity": 0.7,
    },
  }, "routes-multi-line");
});

// --- helpers -----------------------------------------------------------

function emptyFC() { return { type: "FeatureCollection", features: [] }; }
function fmtKm(m) { return (m / 1000).toFixed(1) + " km"; }

async function api(path, params) {
  const u = new URL(API + path, location.origin);
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== undefined && v !== null) u.searchParams.set(k, String(v));
  }
  const r = await fetch(u);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

// --- waypoint inputs --------------------------------------------------
//
// Inputs are the source of truth. Map click only fires when a 📍 button
// has been "armed" — otherwise clicks just hit the basemap and do
// nothing for routing (anchor markers still handle their own clicks).

function parseLonLat(s) {
  const m = (s || "").trim().match(/^(-?\d+(?:\.\d+)?)[ ,;\t]+(-?\d+(?:\.\d+)?)$/);
  if (!m) return null;
  const a = +m[1], b = +m[2];
  if (!isFinite(a) || !isFinite(b)) return null;
  return [a, b];
}

function bindWaypoint(row, role) {
  const wp = {
    role,
    coord: null,
    input: row.querySelector(".wp-input"),
    pickBtn: row.querySelector(".wp-pick"),
    row,
  };
  wp.input.addEventListener("input", () => {
    wp.coord = parseLonLat(wp.input.value);
    refreshWaypointMarkers();
    updateRouteButton();
  });
  wp.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !document.getElementById("route-btn").disabled) {
      routeNow();
    }
  });
  wp.pickBtn.addEventListener("click", () => armPick(wp));
  return wp;
}

function armPick(wp) {
  // Toggle off if same waypoint is clicked again.
  for (const w of state.waypoints) w.pickBtn.classList.remove("armed");
  if (state.pickArmed === wp) {
    state.pickArmed = null;
    map.getCanvas().style.cursor = "";
    return;
  }
  state.pickArmed = wp;
  wp.pickBtn.classList.add("armed");
  map.getCanvas().style.cursor = "crosshair";
}

function disarmPick() {
  if (!state.pickArmed) return;
  state.pickArmed.pickBtn.classList.remove("armed");
  state.pickArmed = null;
  map.getCanvas().style.cursor = "";
}

map.on("click", (e) => {
  if (!state.pickArmed) return;
  const wp = state.pickArmed;
  const lon = e.lngLat.lng, lat = e.lngLat.lat;
  wp.coord = [lon, lat];
  wp.input.value = `${lon.toFixed(5)},${lat.toFixed(5)}`;
  disarmPick();
  refreshWaypointMarkers();
  updateRouteButton();
});

function addMidpoint(coord = null) {
  const endIdx = state.waypoints.findIndex(w => w.role === "end");
  if (endIdx < 0) return null;
  const row = document.createElement("div");
  row.className = "waypoint-row";
  row.dataset.role = "mid";
  row.innerHTML = `
    <span class="wp-label mid">·</span>
    <input type="text" class="wp-input" placeholder="lon,lat" />
    <button class="wp-pick" title="Pick from map">📍</button>
    <button class="wp-remove" title="Remove midpoint">✕</button>
  `;
  document.getElementById("midpoints").appendChild(row);
  const wp = bindWaypoint(row, "mid");
  state.waypoints.splice(endIdx, 0, wp);
  row.querySelector(".wp-remove").addEventListener("click", () => removeWp(wp));
  if (coord) {
    wp.coord = coord;
    wp.input.value = `${coord[0].toFixed(5)},${coord[1].toFixed(5)}`;
  }
  renumberMidpoints();
  refreshWaypointMarkers();
  updateRouteButton();
  return wp;
}

function removeWp(wp) {
  const i = state.waypoints.indexOf(wp);
  if (i < 0) return;
  if (state.pickArmed === wp) disarmPick();
  state.waypoints.splice(i, 1);
  wp.row.remove();
  renumberMidpoints();
  refreshWaypointMarkers();
  updateRouteButton();
}

function renumberMidpoints() {
  state.waypoints
    .filter(w => w.role === "mid")
    .forEach((w, i) => {
      w.row.querySelector(".wp-label").textContent = String(i + 1);
    });
}

function updateRouteButton() {
  const allValid = state.waypoints.length >= 2 &&
                   state.waypoints.every(w => w.coord);
  document.getElementById("route-btn").disabled = !allValid;
}

function refreshWaypointMarkers() {
  state.waypointMarkers.forEach(m => m.remove());
  state.waypointMarkers = [];
  let midNum = 0;
  state.waypoints.forEach((w) => {
    if (!w.coord) return;
    let color, label;
    if (w.role === "start")    { color = "#2c5"; label = "S"; }
    else if (w.role === "end") { color = "#c52"; label = "E"; }
    else                       { color = "#888"; label = String(++midNum); }
    const el = document.createElement("div");
    el.style.cssText = `width:22px;height:22px;border-radius:50%;background:${color};color:white;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:13px;border:2px solid white;box-shadow:0 1px 3px rgba(0,0,0,0.3);`;
    el.textContent = label;
    const m = new maplibregl.Marker({ element: el }).setLngLat(w.coord).addTo(map);
    state.waypointMarkers.push(m);
  });
}

document.getElementById("clear-btn").addEventListener("click", clearAll);
document.getElementById("add-midpoint").addEventListener("click", () => addMidpoint());

function clearAll() {
  // Clear inputs but keep start/end rows; remove midpoints entirely.
  for (const w of [...state.waypoints]) {
    if (w.role === "mid") {
      w.row.remove();
    } else {
      w.input.value = "";
      w.coord = null;
    }
  }
  state.waypoints = state.waypoints.filter(w => w.role !== "mid");
  disarmPick();
  state.routesByProfile = {};
  state.waypointMarkers.forEach(m => m.remove());
  state.waypointMarkers = [];
  if (map.getSource("routes-multi")) map.getSource("routes-multi").setData(emptyFC());
  document.getElementById("results").innerHTML = "";
  document.getElementById("elevation").innerHTML = "";
  updateRouteButton();
}

// --- route ------------------------------------------------------------

document.getElementById("route-btn").addEventListener("click", routeNow);

async function routeNow() {
  const valid = state.waypoints.filter(w => w.coord);
  if (valid.length < 2) return;
  const legCount = valid.length - 1;
  // Bump the request id so any in-flight earlier routeNow becomes
  // stale and its late-arriving responses are dropped instead of
  // overwriting this one.
  const myReqId = ++state.routeReqId;
  setBusy(`Routing ${legCount} leg${legCount > 1 ? "s" : ""} × ${PROFILES.length} profiles…`);

  // Fire one multi-leg routing pipeline per profile in parallel.
  // Per-profile failures (e.g. a degenerate route under one profile)
  // are caught locally so the rest still render.
  const results = await Promise.all(PROFILES.map(async (profile) => {
    try {
      const legs = await Promise.all(
        Array.from({ length: legCount }, (_, i) =>
          api("/trunk/route", {
            from: valid[i].coord.join(","),
            to:   valid[i + 1].coord.join(","),
            profile,
          })
        )
      );
      const route = mergeLegs(legs.map(r => r.route));
      route.properties = route.properties || {};
      route.properties.profile = profile;
      return { profile, route };
    } catch (e) {
      return { profile, error: e.message };
    }
  }));
  if (myReqId !== state.routeReqId) return;   // superseded

  state.routesByProfile = {};
  for (const r of results) {
    if (r.route) state.routesByProfile[r.profile] = r.route;
  }
  renderRoutes(results);
}

function mergeLegs(legs) {
  // Concatenate coordinates; drop the first point of each leg after the
  // first to avoid a duplicated vertex at the join. Sum gross_length_m
  // and vertex_count; concat chain_names with dedup.
  const coords = [];
  let totalLen = 0;
  let totalNodes = 0;
  const cities = [];
  const bridges = [];
  for (const leg of legs) {
    const lc = leg.geometry.coordinates;
    if (coords.length > 0 && lc.length > 0) coords.push(...lc.slice(1));
    else coords.push(...lc);
    const lp = leg.properties || {};
    totalLen += +lp.gross_length_m || 0;
    totalNodes += +lp.vertex_count || 0;
    const lcities = lp.chain_names || [];
    for (const c of lcities) {
      if (cities[cities.length - 1] !== c) cities.push(c);
    }
    for (const b of (lp.bridges || [])) bridges.push(b);
  }
  return {
    type: "Feature",
    geometry: { type: "LineString", coordinates: coords },
    properties: {
      creator: "trunk-router",
      chain_names: cities,
      gross_length_m: totalLen,
      vertex_count: totalNodes,
      leg_count: legs.length,
      bridges,
    },
  };
}

function renderRoutes(results) {
  // results: [{profile, route?, error?}, …] in PROFILES order.
  const features = PROFILES
    .map(p => state.routesByProfile[p])
    .filter(Boolean);
  map.getSource("routes-multi").setData({
    type: "FeatureCollection",
    features,
  });

  // Fit to the union bbox of every successful profile's geometry.
  if (features.length > 0) {
    let n = -90, s = 90, e = -180, w = 180;
    for (const f of features) {
      for (const c of f.geometry.coordinates) {
        if (c[0] < w) w = c[0]; if (c[0] > e) e = c[0];
        if (c[1] < s) s = c[1]; if (c[1] > n) n = c[1];
      }
    }
    map.fitBounds([[w, s], [e, n]], { padding: 60, maxZoom: 13 });
  }

  // One result card per profile. Border color = profile color so the
  // sidebar reads like a legend of the map lines.
  const html = (results || []).map((r) => {
    const color = PROFILE_COLORS[r.profile] || "#888";
    if (r.error) {
      return `<div class="route-card" style="border-left:4px solid ${color}">` +
        `<div class="name" style="color:${color}">${r.profile}</div>` +
        `<div class="stat" style="color:#a52"><span>error</span><span>${r.error}</span></div>` +
        `</div>`;
    }
    const p = r.route.properties || {};
    const km = fmtKm(+p.gross_length_m || 0);
    const cities = (p.chain_names || []).join(" → ");
    const nBridges = (p.bridges || []).length;
    return `<div class="route-card" style="border-left:4px solid ${color}">
      <div class="name" style="color:${color}">${r.profile}</div>
      <div class="stat"><span>distance</span><span>${km}</span></div>
      <div class="stat"><span>nodes</span><span>${(+p.vertex_count || 0).toLocaleString()}</span></div>
      ${nBridges ? `<div class="stat"><span>bridges</span><span>${nBridges}</span></div>` : ""}
      ${cities ? `<div class="stat" style="grid-template-columns: 1fr;"><span><em>${cities}</em></span></div>` : ""}
    </div>`;
  }).join("");
  document.getElementById("results").innerHTML = html;
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
  for (const cat of Object.keys(state.poiMarkers)) {
    state.poiMarkers[cat].forEach(m => m.remove());
    state.poiMarkers[cat] = [];
  }
  if (cats.length === 0 || map.getZoom() < 9) return;
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

// --- Anchors + SPT overlay ---------------------------------------------
//
// Anchor list comes from /live/cities (Postgres-backed). Click an
// anchor to overlay /spt/cell/<idx> — the per-anchor SPT from the
// chainless preprocess, colored by cost-from-anchor — and load that
// anchor's amenities (POIs grouped by category) into the sidebar.

async function loadCities() {
  // Filter to anchors that have an npz on disk under UI_PROFILE.
  // spts-multi writes all 5 profile npzs in lockstep, so a vertex
  // having one means it has all five — a single-profile check is
  // sufficient for "is this anchor routable yet".
  const profile = UI_PROFILE;
  try {
    const r = await api("/live/cities", { profile });
    state.cities = r.cities;
    document.getElementById("show-anchors").disabled = false;
    if (document.getElementById("show-anchors").checked) renderAnchors();
  } catch (e) {
    state.cities = null;
    document.getElementById("show-anchors").checked = false;
    document.getElementById("show-anchors").disabled = true;
    clearAnchorMarkers();
  }
}

function clearAnchorMarkers() {
  state.anchorMarkers.forEach(m => m.remove());
  state.anchorMarkers = [];
}

function renderAnchors() {
  clearAnchorMarkers();
  if (!state.cities) return;
  for (const c of state.cities) {
    const el = document.createElement("div");
    const isCity = c.place === "city";
    const size = isCity ? 12 : 8;
    el.style.cssText = `width:${size}px;height:${size}px;border-radius:50%;background:#1d4ed8;border:1.5px solid white;box-shadow:0 0 2px rgba(0,0,0,0.4);cursor:pointer;`;
    el.title = c.name;
    el.addEventListener("click", (ev) => {
      ev.stopPropagation();
      toggleSpt(c.city_idx, c.name);
    });
    const m = new maplibregl.Marker({ element: el }).setLngLat([c.lon, c.lat]).addTo(map);
    state.anchorMarkers.push(m);
  }
}

async function toggleSpt(idx, name) {
  if (state.shownCellIdx === idx) {
    map.getSource("cell-gradient").setData(emptyFC());
    state.shownCellIdx = null;
    document.getElementById("results").innerHTML = "";
    return;
  }
  setBusy(`Loading SPT for ${name}…`);
  try {
    const profile = UI_PROFILE;
    // Default cost filter — sharp unsubsampled view of the ~15 km
    // bike-cost vicinity. At 30 km Graz produces 78 MB / 528 K edges
    // and MapLibre is laggy; 15 km is roughly a quarter of that and
    // still shows the full local road network. Read from the input
    // box so users can widen/narrow.
    const maxCostInput = document.getElementById("max-cost-km");
    const max_cost_km = Math.max(1, +maxCostInput?.value || 15);
    const max_cost = max_cost_km * 1000;
    // Fetch the SPT and the amenities in parallel — independent reads.
    const [spt, amenities] = await Promise.all([
      api(`/spt/cell/${idx}`, { profile, max_cost }),
      api(`/spt/cell/${idx}/amenities`).catch(e => {
        console.warn("amenities fetch failed", e);
        return null;
      }),
    ]);
    map.getSource("cell-gradient").setData(spt);
    state.shownCellIdx = idx;
    // Stretch the color ramp across the *shown* cost range, not the
    // full-SPT range. Otherwise filtering to a small max_cost (say
    // 15 km of a 100 km SPT) leaves every shown edge in the bottom
    // 15% of the ramp — visually all green.
    const lo = spt.shown_cost_min ?? spt.cost_min ?? 0;
    const hi = spt.shown_cost_max ?? spt.cost_max ?? 100000;
    const mid = lo + (hi - lo) / 2;
    map.setPaintProperty("cell-gradient-lines", "line-color", [
      "interpolate", ["linear"], ["get", "cost"],
      lo,  "#10b981",
      mid, "#facc15",
      hi,  "#dc2626",
    ]);
    showSptInfo(name, spt, amenities);
  } catch (e) {
    setError(e.message);
  }
}

function showSptInfo(name, spt, amenities) {
  const lo = spt.cost_min, hi = spt.cost_max;
  const km = (n) => (n / 1000).toFixed(1) + " km-equiv";
  const filtered = spt.filtered_max_cost
    ? `, filtered to <${(spt.filtered_max_cost/1000).toFixed(0)} km`
    : "";
  const renderMode = spt.subsampled
    ? `subsampled (grid ~${(spt.grid_deg * 111).toFixed(2)} km)`
    : "all kept edges";
  let amenityHtml = "";
  if (amenities && amenities.by_category) {
    const cats = Object.entries(amenities.by_category);
    if (cats.length > 0) {
      amenityHtml = `
        <div class="route-card">
          <div class="name">${name}: amenities (${amenities.footprint})</div>
          ${cats.map(([cat, info]) => `
            <div class="stat"><span>${cat}</span><span>${info.count}</span></div>
            <div class="hint" style="margin-top:-2px;font-size:10px;">${info.samples.slice(0, 4).map(s => s.name).join(" · ")}${info.count > 4 ? " …" : ""}</div>
          `).join("")}
        </div>`;
    } else {
      amenityHtml = `<div class="route-card"><div class="name">${name}: no POIs in footprint</div></div>`;
    }
  }
  const showLo = spt.shown_cost_min ?? lo;
  const showHi = spt.shown_cost_max ?? hi;
  document.getElementById("results").innerHTML = `
    <div class="route-card">
      <div class="name">${name}: SPT${filtered}</div>
      <div class="stat"><span>reachable nodes</span><span>${(spt.total_visited || 0).toLocaleString()}</span></div>
      <div class="stat"><span>shown</span><span>${spt.features.length.toLocaleString()} edges (${renderMode})</span></div>
      <div class="stat"><span>shown cost range</span><span>${km(showLo)} → ${km(showHi)}</span></div>
      <div class="stat"><span>full SPT cost range</span><span>${lo == null ? "—" : km(lo) + " → " + km(hi)}</span></div>
    </div>
    ${amenityHtml}
  `;
}

document.getElementById("show-anchors").addEventListener("change", (e) => {
  if (e.target.checked) renderAnchors();
  else {
    clearAnchorMarkers();
    map.getSource("cell-gradient").setData(emptyFC());
    state.shownCellIdx = null;
  }
});

// --- Biome overlay (Resolve 2017 ecoregions) ---------------------------
//
// Static GeoJSON at /data/ecoregions_europe.geojson — clipped to Europe
// (-15..40 lon, 33..72 lat), simplified to ~500 m tolerance, 1.3 MB.
// Loaded lazily on first toggle, then just shown/hidden afterwards.

let biomeLoaded = false;

async function ensureBiomeLayers() {
  if (biomeLoaded) return;
  setBusy("Loading ecoregions…");
  try {
    const r = await fetch("/data/ecoregions_europe.geojson");
    if (!r.ok) throw new Error(`ecoregions: ${r.status}`);
    const fc = await r.json();
    map.addSource("ecoregions", { type: "geojson", data: fc });
    // Fill colored by the dataset's own COLOR_BIO field (one color per
    // biome, set in the source shapefile). Place beneath the gradient
    // line layer so SPT viz still pops.
    map.addLayer({
      id: "ecoregions-fill",
      type: "fill",
      source: "ecoregions",
      paint: {
        "fill-color": ["get", "COLOR_BIO"],
        "fill-opacity": 0.30,
      },
    }, "cell-gradient-lines");
    map.addLayer({
      id: "ecoregions-outline",
      type: "line",
      source: "ecoregions",
      paint: {
        "line-color": ["get", "COLOR_BIO"],
        "line-width": 0.6,
        "line-opacity": 0.8,
      },
    }, "cell-gradient-lines");
    map.on("click", "ecoregions-fill", (e) => {
      const f = e.features?.[0];
      if (!f) return;
      const p = f.properties;
      new maplibregl.Popup({ offset: 8 })
        .setLngLat(e.lngLat)
        .setHTML(
          `<strong>${p.ECO_NAME}</strong>` +
          `<br><small>${p.BIOME_NAME}</small>`
        )
        .addTo(map);
    });
    map.on("mouseenter", "ecoregions-fill", () => map.getCanvas().style.cursor = "crosshair");
    map.on("mouseleave", "ecoregions-fill", () => map.getCanvas().style.cursor = "");
    biomeLoaded = true;
    document.getElementById("results").innerHTML = "";
  } catch (e) {
    setError(`biome load failed: ${e.message}`);
    throw e;
  }
}

document.getElementById("show-biome").addEventListener("change", async (e) => {
  if (e.target.checked) {
    try {
      await ensureBiomeLayers();
    } catch {
      e.target.checked = false;
      return;
    }
    map.setLayoutProperty("ecoregions-fill", "visibility", "visible");
    map.setLayoutProperty("ecoregions-outline", "visibility", "visible");
  } else if (biomeLoaded) {
    map.setLayoutProperty("ecoregions-fill", "visibility", "none");
    map.setLayoutProperty("ecoregions-outline", "visibility", "none");
  }
});

// --- Passenger rail overlays -------------------------------------------
// Static GeoJSON at /data/rail_lines.geojson, /data/rail_stations.geojson,
// produced by `export-rails`. Lines come from OSM track geometry,
// spatially filtered to lines that pass within 200 m of any GTFS-served
// station. Stations come from the national GTFS feed and represent the
// definitive "served" list (with route count per stop).
let railLinesLoaded = false;
let railStationsLoaded = false;

async function ensureRailLinesLayer() {
  if (railLinesLoaded) return;
  setBusy("Loading rail lines…");
  try {
    const r = await fetch("/data/rail_lines.geojson");
    if (!r.ok) throw new Error(`rail_lines: ${r.status}`);
    const fc = await r.json();
    map.addSource("rail-lines", { type: "geojson", data: fc, tolerance: 0 });
    map.addLayer({
      id: "rail-lines-layer",
      type: "line",
      source: "rail-lines",
      paint: {
        // Dark slate with subtle "ties" effect via dasharray; visible
        // against both green/forest backgrounds and the white basemap.
        "line-color": "#2a2a3a",
        "line-width": [
          "interpolate", ["linear"], ["zoom"],
          7,  0.8,
          10, 1.6,
          14, 2.4,
        ],
        "line-opacity": 0.85,
      },
    });
    railLinesLoaded = true;
    document.getElementById("results").innerHTML = "";
  } catch (e) {
    setError(`rail_lines load failed: ${e.message}`);
    throw e;
  }
}

async function ensureRailStationsLayer() {
  if (railStationsLoaded) return;
  setBusy("Loading rail stations…");
  try {
    const r = await fetch("/data/rail_stations.geojson");
    if (!r.ok) throw new Error(`rail_stations: ${r.status}`);
    const fc = await r.json();
    map.addSource("rail-stations", { type: "geojson", data: fc });
    map.addLayer({
      id: "rail-stations-circle",
      type: "circle",
      source: "rail-stations",
      paint: {
        // Radius scales gently with n_routes (busier = bigger dot).
        "circle-radius": [
          "interpolate", ["linear"], ["get", "n_routes"],
          1,  3,
          10, 5,
          50, 8,
        ],
        "circle-color": "#c43a3a",
        "circle-stroke-color": "#ffffff",
        "circle-stroke-width": 1.2,
        "circle-opacity": 0.9,
      },
    });
    map.addLayer({
      id: "rail-stations-label",
      type: "symbol",
      source: "rail-stations",
      // Only label busier stops so the map stays readable.
      filter: [">=", ["get", "n_routes"], 4],
      layout: {
        "text-field": ["get", "name"],
        "text-size": 11,
        "text-offset": [0, 0.9],
        "text-anchor": "top",
        "text-allow-overlap": false,
      },
      paint: {
        "text-color": "#2a2a3a",
        "text-halo-color": "#ffffff",
        "text-halo-width": 1.4,
      },
    });
    railStationsLoaded = true;
    document.getElementById("results").innerHTML = "";
  } catch (e) {
    setError(`rail_stations load failed: ${e.message}`);
    throw e;
  }
}

function _bindRailToggle(toggleId, ensureFn, layerIds) {
  document.getElementById(toggleId).addEventListener("change", async (e) => {
    if (e.target.checked) {
      try {
        await ensureFn();
      } catch {
        e.target.checked = false;
        return;
      }
      for (const lid of layerIds) {
        map.setLayoutProperty(lid, "visibility", "visible");
      }
    } else {
      for (const lid of layerIds) {
        if (map.getLayer(lid)) {
          map.setLayoutProperty(lid, "visibility", "none");
        }
      }
    }
  });
}

_bindRailToggle("show-rail-lines",    ensureRailLinesLayer,    ["rail-lines-layer"]);
_bindRailToggle("show-rail-stations", ensureRailStationsLayer, ["rail-stations-circle", "rail-stations-label"]);


// --- Terrain hillshade overlay ---------------------------------------
// Public DEM tiles from AWS Open Data, terrarium-encoded. MapLibre
// renders shaded relief from the raster-dem source via the hillshade
// layer type — no compositing on our side, no auth.

let hillshadeLoaded = false;

function ensureHillshade() {
  if (hillshadeLoaded) return;
  map.addSource("terrain-dem", {
    type: "raster-dem",
    tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"],
    encoding: "terrarium",
    tileSize: 256,
    maxzoom: 14,
    attribution: "DEM: <a href='https://registry.opendata.aws/terrain-tiles/'>AWS Open Data</a>",
  });
  // Put hillshade above basemap but beneath any vector overlays.
  // beforeId is left undefined so it lands on top; we'll move it
  // beneath the routes if/when they exist via map.moveLayer.
  map.addLayer({
    id: "hillshade",
    type: "hillshade",
    source: "terrain-dem",
    paint: {
      "hillshade-exaggeration": 0.5,
      "hillshade-shadow-color": "#000",
      "hillshade-highlight-color": "#fff",
      "hillshade-accent-color": "#666",
    },
  });
  // Keep the multi-profile route lines on top of the shading.
  if (map.getLayer("routes-multi-line")) map.moveLayer("routes-multi-line");
  hillshadeLoaded = true;
}

// --- Coverage gap overlay -------------------------------------------
// Static GeoJSON at /data/web_overlays/coverage_gap.geojson produced
// by `dump_coverage_gap.py`. Each feature is a road edge whose both
// endpoints sit >15 km euclidean from every anchor — i.e., outside
// the "comfortable" zone where paired-SPT routing is accurate.
// Filtered to length_m ≥ 100 m so urban connector noise doesn't
// drown the corridor patterns.

let coverageGapLoaded = false;

async function ensureCoverageGapLayer() {
  if (coverageGapLoaded) return;
  setBusy("Loading coverage gap…");
  try {
    const r = await fetch("/data/coverage_gap.geojson?ts=" + Date.now());
    if (!r.ok) throw new Error(`coverage_gap: ${r.status}`);
    const fc = await r.json();
    map.addSource("coverage-gap", { type: "geojson", data: fc, tolerance: 0 });
    map.addLayer({
      id: "coverage-gap-line",
      type: "line",
      source: "coverage-gap",
      paint: {
        "line-color": "#dc2626",
        "line-width": [
          "interpolate", ["linear"], ["zoom"],
          7,  0.6,
          10, 1.2,
          14, 2.0,
        ],
        "line-opacity": 0.65,
      },
    });
    coverageGapLoaded = true;
    document.getElementById("results").innerHTML = "";
  } catch (e) {
    setError(`coverage_gap load failed: ${e.message}`);
    throw e;
  }
}

document.getElementById("show-coverage-gap").addEventListener("change", async (e) => {
  if (e.target.checked) {
    try { await ensureCoverageGapLayer(); }
    catch { e.target.checked = false; return; }
    map.setLayoutProperty("coverage-gap-line", "visibility", "visible");
  } else if (coverageGapLoaded) {
    map.setLayoutProperty("coverage-gap-line", "visibility", "none");
  }
});

// --- Promoted corridor villages overlay ------------------------------
// Static GeoJSON at /data/promoted_villages_corridor.geojson produced
// by promote_villages_corridor.py. Yellow dots = OSM place=village
// candidates within 200 m of a primary/trunk/motorway road, 5 km
// spaced. Visual sanity-check before committing to an INSERT + full
// preprocess rebuild.

let promotedCorridorLoaded = false;

async function ensurePromotedCorridorLayer() {
  if (promotedCorridorLoaded) return;
  setBusy("Loading corridor village candidates…");
  try {
    const r = await fetch("/data/promoted_villages_corridor.geojson?ts=" + Date.now());
    if (!r.ok) throw new Error(`promoted_villages_corridor: ${r.status}`);
    const fc = await r.json();
    map.addSource("promoted-corridor", { type: "geojson", data: fc });
    map.addLayer({
      id: "promoted-corridor-circles",
      type: "circle",
      source: "promoted-corridor",
      paint: {
        "circle-color": "#facc15",
        "circle-radius": [
          "interpolate", ["linear"], ["zoom"],
          5,  3,
          10, 5,
          14, 8,
        ],
        "circle-stroke-color": "#000",
        "circle-stroke-width": 1.2,
        "circle-opacity": 0.9,
      },
    });
    map.on("click", "promoted-corridor-circles", (e) => {
      const p = e.features[0].properties;
      document.getElementById("results").innerHTML =
        `<div class="route-card"><strong>${p.name}</strong>` +
        `<div class="stat"><span>population</span><span>${p.population ?? "—"}</span></div>` +
        `<div class="stat"><span>osm_id</span><span>${p.osm_id}</span></div></div>`;
    });
    map.on("mouseenter", "promoted-corridor-circles", () => map.getCanvas().style.cursor = "pointer");
    map.on("mouseleave", "promoted-corridor-circles", () => map.getCanvas().style.cursor = "");
    promotedCorridorLoaded = true;
    document.getElementById("results").innerHTML = "";
  } catch (e) {
    setError(`promoted_villages_corridor load failed: ${e.message}`);
    throw e;
  }
}

document.getElementById("show-promoted-corridor").addEventListener("change", async (e) => {
  if (e.target.checked) {
    try { await ensurePromotedCorridorLayer(); }
    catch { e.target.checked = false; return; }
    map.setLayoutProperty("promoted-corridor-circles", "visibility", "visible");
  } else if (promotedCorridorLoaded) {
    map.setLayoutProperty("promoted-corridor-circles", "visibility", "none");
  }
});

document.getElementById("show-hillshade").addEventListener("change", (e) => {
  if (e.target.checked) {
    ensureHillshade();
    map.setLayoutProperty("hillshade", "visibility", "visible");
  } else if (hillshadeLoaded) {
    map.setLayoutProperty("hillshade", "visibility", "none");
  }
});

// --- Way-graph experiment overlays ------------------------------------
// /data/way_city_graph.geojson + way_city_anchors.geojson produced by
// build_way_graph.py. Chain backbone derived from highway topology
// (trunk + primary + motorway) instead of overlapping SPTs — Voronoi
// adjacency on the road subgraph yields "no other anchor sits between
// these two along the corridor". Pure visualization layer for now.

let wayGraphEdgesLoaded = false;
let wayGraphNodesLoaded = false;
let wayGraphPolygonsLoaded = false;

// SPT done-status for the polygon-bounded direct-profile build. Populated
// by ensureWayGraphNodes (so anchor colors reflect done-ness) and refreshed
// every 30 s while the build is running.
let sptStatus = { done: 0, total: 0, doneSet: new Set(), refByIdx: new Map() };
let sptStatusTimer = null;

const API_BASE = (location.hostname === "localhost" || location.hostname === "127.0.0.1")
  ? "http://localhost:8001"
  : `${location.protocol}//${location.hostname}:8001`;

async function fetchSptStatus() {
  try {
    const r = await fetch(`${API_BASE}/way-graph/spt-status?profile=direct_polygon&_=${Date.now()}`);
    if (!r.ok) throw new Error(`spt-status: ${r.status}`);
    const d = await r.json();
    sptStatus.done = d.done;
    sptStatus.total = d.total;
    sptStatus.doneSet = new Set(d.done_indices);
    sptStatus.refByIdx = new Map(d.cities.map(c => [c.city_idx, c.ref]));
    // Inverse map ref -> city_idx, so anchor features can be tagged.
    sptStatus.idxByRef = new Map(d.cities.map(c => [c.ref, c.city_idx]));
    const badge = document.getElementById("spt-status-badge");
    if (badge) {
      const pct = d.total ? (100 * d.done / d.total).toFixed(1) : "0.0";
      badge.textContent = `${d.done.toLocaleString()} / ${d.total.toLocaleString()} (${pct}%)`;
    }
    return d;
  } catch (e) {
    const badge = document.getElementById("spt-status-badge");
    if (badge) badge.textContent = "(status unavailable)";
    console.warn("spt-status fetch failed:", e);
    return null;
  }
}

function startSptStatusPolling() {
  if (sptStatusTimer) return;
  fetchSptStatus().then(refreshAnchorDoneColoring);
  sptStatusTimer = setInterval(() => {
    fetchSptStatus().then(refreshAnchorDoneColoring);
  }, 30000);
}

function refreshAnchorDoneColoring() {
  if (!wayGraphNodesLoaded || !map.getSource("way-graph-nodes")) return;
  // Re-stamp each anchor feature with spt_done = doneSet.has(city_idx).
  // We don't have city_idx in the geojson, so map via ref.
  const src = map.getSource("way-graph-nodes");
  const data = src._data;
  if (!data || !data.features) return;
  for (const f of data.features) {
    const ref = f.properties.ref;
    const idx = sptStatus.idxByRef ? sptStatus.idxByRef.get(ref) : undefined;
    f.properties.city_idx = idx ?? -1;
    f.properties.spt_done = idx !== undefined && sptStatus.doneSet.has(idx);
  }
  src.setData(data);
}

async function ensureWayGraphEdges() {
  if (wayGraphEdgesLoaded) return;
  setBusy("Loading way-graph edges…");
  try {
    const r = await fetch("/data/way_city_graph.geojson?ts=" + Date.now());
    if (!r.ok) throw new Error(`way_city_graph: ${r.status}`);
    const fc = await r.json();
    map.addSource("way-graph-edges", { type: "geojson", data: fc });
    map.addLayer({
      id: "way-graph-edges-line",
      type: "line",
      source: "way-graph-edges",
      paint: {
        // Gradient by edge cost: green (short) → yellow → red (long).
        "line-color": [
          "interpolate", ["linear"], ["get", "cost_km"],
          0,   "#22c55e",
          10,  "#84cc16",
          25,  "#facc15",
          50,  "#f97316",
          100, "#dc2626",
        ],
        "line-width": [
          "interpolate", ["linear"], ["zoom"],
          6,  1.0,
          10, 2.2,
          14, 3.0,
        ],
        "line-opacity": 0.85,
      },
    });
    map.on("click", "way-graph-edges-line", (e) => {
      const p = e.features[0].properties;
      document.getElementById("results").innerHTML =
        `<div class="route-card"><strong>${p.a_name} ↔ ${p.b_name}</strong>` +
        `<div class="stat"><span>cost</span><span>${p.cost_km} km</span></div>` +
        `<div class="stat"><span>a</span><span>${p.a}</span></div>` +
        `<div class="stat"><span>b</span><span>${p.b}</span></div></div>`;
    });
    map.on("mouseenter", "way-graph-edges-line", () => map.getCanvas().style.cursor = "pointer");
    map.on("mouseleave", "way-graph-edges-line", () => map.getCanvas().style.cursor = "");
    wayGraphEdgesLoaded = true;
    document.getElementById("results").innerHTML = "";
  } catch (e) {
    setError(`way_city_graph load failed: ${e.message}`);
    throw e;
  }
}

async function ensureWayGraphNodes() {
  if (wayGraphNodesLoaded) return;
  setBusy("Loading way-graph anchors…");
  try {
    const r = await fetch("/data/way_city_anchors.geojson?ts=" + Date.now());
    if (!r.ok) throw new Error(`way_city_anchors: ${r.status}`);
    const fc = await r.json();
    map.addSource("way-graph-nodes", { type: "geojson", data: fc });
    map.addLayer({
      id: "way-graph-nodes-circles",
      type: "circle",
      source: "way-graph-nodes",
      paint: {
        // SPT done? bright green. Else fall back to db=blue / village=orange.
        // Faded if not in chain graph at all.
        "circle-color": [
          "case",
          ["get", "spt_done"], "#16a34a",
          ["==", ["get", "kind"], "db"], "#2563eb",
          "#f97316",
        ],
        "circle-radius": [
          "interpolate", ["linear"], ["zoom"],
          5,  2,
          10, 4,
          14, 7,
        ],
        "circle-stroke-color": "#000",
        "circle-stroke-width": 1.0,
        "circle-opacity": ["case", ["get", "in_graph"], 0.9, 0.3],
      },
    });
    // Kick off the SPT-status polling so anchors light up as the build
    // progresses. Safe to call repeatedly — guarded by sptStatusTimer.
    startSptStatusPolling();
    map.on("click", "way-graph-nodes-circles", async (e) => {
      const p = e.features[0].properties;
      const ref = p.ref;
      const cityIdx = (sptStatus.idxByRef && sptStatus.idxByRef.get(ref)) ?? -1;
      // Highlight this anchor's SPT polygon (if polygons layer enabled).
      await highlightPolygonForRef(ref);
      // Load the SPT visualization if this anchor's SPT has been built.
      if (p.spt_done) {
        await loadSptForCityIdx(cityIdx, p.name);
      } else {
        // Clear any previously-loaded SPT and show the info card.
        if (map.getSource("way-graph-spt")) {
          map.getSource("way-graph-spt").setData({ type: "FeatureCollection", features: [] });
        }
        document.getElementById("results").innerHTML =
          `<div class="route-card"><strong>${p.name}</strong>` +
          `<div class="stat"><span>ref</span><span>${ref}</span></div>` +
          `<div class="stat"><span>kind</span><span>${p.kind} / ${p.place}</span></div>` +
          `<div class="stat"><span>population</span><span>${p.population ?? "—"}</span></div>` +
          `<div class="stat"><span>in chain graph</span><span>${p.in_graph}</span></div>` +
          `<div class="stat"><span>SPT</span><span>pending</span></div></div>`;
      }
    });
    map.on("mouseenter", "way-graph-nodes-circles", () => map.getCanvas().style.cursor = "pointer");
    map.on("mouseleave", "way-graph-nodes-circles", () => map.getCanvas().style.cursor = "");
    wayGraphNodesLoaded = true;
    document.getElementById("results").innerHTML = "";
  } catch (e) {
    setError(`way_city_anchors load failed: ${e.message}`);
    throw e;
  }
}

async function ensureWayGraphPolygons() {
  if (wayGraphPolygonsLoaded) return;
  setBusy("Loading SPT polygons…");
  try {
    const r = await fetch("/data/way_city_spt_polygons.geojson?ts=" + Date.now());
    if (!r.ok) throw new Error(`spt_polygons: ${r.status}`);
    const fc = await r.json();
    map.addSource("way-graph-polygons", { type: "geojson", data: fc });
    map.addLayer({
      id: "way-graph-polygons-line",
      type: "line",
      source: "way-graph-polygons",
      paint: {
        "line-color": "#1e40af",
        "line-width": 0.8,
        "line-opacity": 0.5,
      },
    });
    // Highlighted polygon (one at a time, set via setData on click).
    map.addSource("way-graph-polygon-highlight", {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] },
    });
    map.addLayer({
      id: "way-graph-polygon-highlight-line",
      type: "line",
      source: "way-graph-polygon-highlight",
      paint: { "line-color": "#16a34a", "line-width": 2.5 },
    });
    wayGraphPolygonsLoaded = true;
    document.getElementById("results").innerHTML = "";
  } catch (e) {
    setError(`spt polygons load failed: ${e.message}`);
    throw e;
  }
}

async function highlightPolygonForRef(ref) {
  if (!wayGraphPolygonsLoaded) return;
  const src = map.getSource("way-graph-polygons");
  if (!src || !src._data) return;
  const match = src._data.features.find(f => f.properties.ref === ref);
  const highlight = map.getSource("way-graph-polygon-highlight");
  if (highlight) {
    highlight.setData({
      type: "FeatureCollection",
      features: match ? [match] : [],
    });
  }
}

document.getElementById("show-way-graph-polygons").addEventListener("change", async (e) => {
  if (e.target.checked) {
    try { await ensureWayGraphPolygons(); }
    catch { e.target.checked = false; return; }
    for (const lid of ["way-graph-polygons-line", "way-graph-polygon-highlight-line"]) {
      map.setLayoutProperty(lid, "visibility", "visible");
    }
  } else if (wayGraphPolygonsLoaded) {
    for (const lid of ["way-graph-polygons-line", "way-graph-polygon-highlight-line"]) {
      map.setLayoutProperty(lid, "visibility", "none");
    }
  }
});

// --- SPT-on-click: fetch + render the polygon-bounded SPT for an anchor.
// Source/layer are added lazily on first click; subsequent clicks just
// setData() to swap which anchor's SPT is shown.
// No client-side cost cap — the API ships every edge up to ~50k features
// (HARD_FEATURE_CAP in api/app/main.py) and adaptive-strides if larger.

async function ensureWayGraphSptLayer() {
  if (map.getSource("way-graph-spt")) return;
  map.addSource("way-graph-spt", {
    type: "geojson",
    data: { type: "FeatureCollection", features: [] },
  });
  map.addLayer({
    id: "way-graph-spt-line",
    type: "line",
    source: "way-graph-spt",
    paint: {
      // Green near anchor → yellow → red far. Stops cover the typical
      // direct-profile range (~5-100 km bike-equivalent in dense areas).
      "line-color": [
        "interpolate", ["linear"], ["get", "cost"],
        0,      "#22c55e",
        10000,  "#84cc16",
        30000,  "#facc15",
        60000,  "#f97316",
        100000, "#dc2626",
      ],
      "line-width": [
        "interpolate", ["linear"], ["zoom"],
        8,  0.8,
        12, 1.6,
        16, 2.4,
      ],
      "line-opacity": 0.9,
    },
  });
}

// Remember which anchor's SPT is currently shown so the detail slider
// can reload the same one when the cap changes.
let currentSptCityIdx = null;
let currentSptLabel = null;

async function loadSptForCityIdx(city_idx, label) {
  if (city_idx === undefined || city_idx === null || city_idx < 0) {
    return;
  }
  await ensureWayGraphSptLayer();
  currentSptCityIdx = city_idx;
  currentSptLabel = label;
  const maxFeatures = +document.getElementById("spt-max-features").value || 300000;
  const url = `${API_BASE}/way-graph/spt/${city_idx}?profile=direct_polygon&max_features=${maxFeatures}`;
  setBusy(`Loading SPT for ${label} (cap ${maxFeatures.toLocaleString()})…`);
  try {
    const r = await fetch(url);
    if (r.status === 404) {
      setError(`SPT not built yet for ${label}`);
      map.getSource("way-graph-spt").setData({ type: "FeatureCollection", features: [] });
      return;
    }
    if (!r.ok) throw new Error(`spt: ${r.status}`);
    const fc = await r.json();
    map.getSource("way-graph-spt").setData(fc);
    document.getElementById("results").innerHTML =
      `<div class="route-card"><strong>${label}</strong> SPT` +
      `<div class="stat"><span>edges shown</span><span>${fc.kept_count.toLocaleString()}</span></div>` +
      `<div class="stat"><span>total visited</span><span>${fc.total_visited.toLocaleString()}</span></div>` +
      `<div class="stat"><span>cost range</span><span>${(fc.cost_min/1000).toFixed(1)} – ${(fc.cost_max/1000).toFixed(1)} km</span></div></div>`;
  } catch (e) {
    setError(`SPT load failed: ${e.message}`);
  }
}

// Reload the currently-displayed SPT whenever the detail slider changes.
document.getElementById("spt-max-features").addEventListener("change", () => {
  if (currentSptCityIdx !== null) {
    loadSptForCityIdx(currentSptCityIdx, currentSptLabel);
  }
});

// Kick the badge update immediately on page load so the status shows up
// before any layers are toggled on.
fetchSptStatus();

document.getElementById("show-way-graph-edges").addEventListener("change", async (e) => {
  if (e.target.checked) {
    try { await ensureWayGraphEdges(); }
    catch { e.target.checked = false; return; }
    map.setLayoutProperty("way-graph-edges-line", "visibility", "visible");
  } else if (wayGraphEdgesLoaded) {
    map.setLayoutProperty("way-graph-edges-line", "visibility", "none");
  }
});

document.getElementById("show-way-graph-nodes").addEventListener("change", async (e) => {
  if (e.target.checked) {
    try { await ensureWayGraphNodes(); }
    catch { e.target.checked = false; return; }
    map.setLayoutProperty("way-graph-nodes-circles", "visibility", "visible");
  } else if (wayGraphNodesLoaded) {
    map.setLayoutProperty("way-graph-nodes-circles", "visibility", "none");
  }
});

// --- status helpers ---------------------------------------------------

function setBusy(msg) {
  document.getElementById("results").innerHTML = `<div class="route-card"><em>${msg}</em></div>`;
}

function setError(msg) {
  document.getElementById("results").innerHTML = `<div class="route-card" style="border-color:#c52;"><strong>Error</strong><div class="stat">${msg}</div></div>`;
}

// --- Scenicness raster overlays ---------------------------------------
// One image overlay per signal raster, produced by
// `scenicness-bake --export-rasters`. The bake writes
// data/scenicness/<column>.png plus a manifest.json giving the bbox
// and per-signal colormap info. We fetch the manifest, render one
// toggle per signal, and lazily add MapLibre image sources/layers on
// first activation.

const SCENICNESS_MANIFEST_URL = "/data/scenicness/manifest.json";
const scenicnessLoaded = new Set();     // columns whose layer is in the style
let scenicnessManifest = null;

async function initScenicnessOverlays() {
  const root = document.getElementById("scenicness-toggles");
  if (!root) return;
  try {
    const r = await fetch(SCENICNESS_MANIFEST_URL + "?ts=" + Date.now());
    if (!r.ok) throw new Error(`manifest: ${r.status}`);
    scenicnessManifest = await r.json();
  } catch (e) {
    root.innerHTML =
      `<p class="hint" style="color:#a52;">No scenicness manifest yet ` +
      `(<code>${e.message}</code>). Run <code>scenicness-bake ` +
      `--export-rasters</code> to generate.</p>`;
    return;
  }
  const sigs = scenicnessManifest.signals || {};
  const items = Object.keys(sigs).map(col => {
    const s = sigs[col];
    return `<label class="checkbox" title="${s.description || ""}">` +
      `<input type="checkbox" data-scenicness="${col}" />` +
      `${s.name || col}</label>`;
  });
  root.innerHTML = items.length
    ? items.join("\n")
    : `<p class="hint">No signals in manifest.</p>`;
  for (const cb of root.querySelectorAll("input[data-scenicness]")) {
    cb.addEventListener("change", e => toggleScenicness(
      e.target.dataset.scenicness, e.target.checked,
    ));
  }
}

function toggleScenicness(column, on) {
  if (!scenicnessManifest) return;
  const sig = scenicnessManifest.signals[column];
  if (!sig) return;
  const layerId = `scenic-${column}`;
  const sourceId = `scenic-${column}-src`;

  if (on && !scenicnessLoaded.has(column)) {
    if (sig.tiles) {
      // Scalable path: a Web-Mercator XYZ raster tile pyramid. MapLibre
      // lazy-loads only the {z}/{x}/{y}.png tiles in view; missing tiles
      // (transparent areas we skipped) just 404 → rendered as empty.
      map.addSource(sourceId, {
        type: "raster",
        tiles: [`/data/scenicness/${sig.tiles}`],
        tileSize: 256,
        minzoom: sig.minzoom ?? scenicnessManifest.minzoom ?? 0,
        maxzoom: sig.maxzoom ?? scenicnessManifest.maxzoom ?? 12,
        bounds: scenicnessManifest.bbox,
      });
    } else {
      // Legacy path: one image overlay covering the whole bbox. Only
      // viable for small extents (a continental single PNG is ~34 GB).
      const [minLon, minLat, maxLon, maxLat] = scenicnessManifest.bbox;
      map.addSource(sourceId, {
        type: "image",
        url: `/data/scenicness/${sig.png}?ts=${Date.now()}`,
        // MapLibre image source corners, clockwise from top-left.
        coordinates: [
          [minLon, maxLat],
          [maxLon, maxLat],
          [maxLon, minLat],
          [minLon, minLat],
        ],
      });
    }
    // Insert just below the route/anchor layers so overlays don't
    // hide them. Pick the first layer in the active style whose id
    // starts with one of the "things we want on top" prefixes; if
    // none found, addLayer with no anchor (= on top).
    const topAnchor = map.getStyle().layers
      .map(l => l.id)
      .find(id => id.startsWith("graz-wien-") || id === "anchors"
                || id === "route-line");
    map.addLayer({
      id: layerId,
      type: "raster",
      source: sourceId,
      paint: { "raster-opacity": 0.55 },
    }, topAnchor);
    scenicnessLoaded.add(column);
  } else if (scenicnessLoaded.has(column)) {
    map.setLayoutProperty(
      layerId, "visibility", on ? "visible" : "none",
    );
  }
}

// --- bootstrap ---------------------------------------------------------

map.on("load", () => {
  // Bind the start/end input rows from the DOM into state.waypoints.
  // Inputs start empty; user enters coords (or clicks 📍 to pick from
  // the map) and presses Enter or clicks Route. No default route runs
  // on load.
  const startRow = document.querySelector('.waypoint-row[data-role="start"]');
  const endRow   = document.querySelector('.waypoint-row[data-role="end"]');
  state.waypoints = [
    bindWaypoint(startRow, "start"),
    bindWaypoint(endRow,   "end"),
  ];
  refreshWaypointMarkers();
  updateRouteButton();
  loadCities();
  initScenicnessOverlays();
});
