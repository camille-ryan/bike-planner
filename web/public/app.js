// Bike-routing front-end. Single-file JS, vanilla (no framework).
// Talks to the FastAPI service via the /api/ prefix proxied by nginx.

const API = "/api";

// Default route on app load.
const DEFAULT_START = [15.4395, 47.0707];   // Graz
const DEFAULT_END   = [12.5683, 55.6761];   // København

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
  routes: [],         // merged GeoJSON Feature(s) from leg concat
  activeIdx: 0,
  routeReqId: 0,      // increments per routeNow; stale responses are discarded
  poiMarkers: { viewpoint: [], lodging: [], food: [], bike_service: [], water: [] },
  waypointMarkers: [],
  cities: null,
  anchorMarkers: [],
  shownCellIdx: null,
};

// --- map sources / layers (initialized once map loads) -----------------

map.on("load", () => {
  map.addSource("route-active", { type: "geojson", data: emptyFC() });
  map.addLayer({
    id: "route-active-line",
    type: "line",
    source: "route-active",
    paint: { "line-color": "#2c5", "line-width": 5, "line-opacity": 0.95 },
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
  }, "route-active-line");
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
  state.routes = [];
  state.activeIdx = 0;
  state.waypointMarkers.forEach(m => m.remove());
  state.waypointMarkers = [];
  if (map.getSource("route-active")) map.getSource("route-active").setData(emptyFC());
  document.getElementById("results").innerHTML = "";
  document.getElementById("elevation").innerHTML = "";
  updateRouteButton();
}

// --- route ------------------------------------------------------------

document.getElementById("route-btn").addEventListener("click", routeNow);

async function routeNow() {
  const valid = state.waypoints.filter(w => w.coord);
  if (valid.length < 2) return;
  const profile = document.getElementById("profile").value;
  const legCount = valid.length - 1;
  // Bump the request id so any in-flight earlier routeNow (e.g. the
  // ~10 s cold first-page-load default route) becomes stale and its
  // late-arriving response is dropped instead of overwriting this one.
  const myReqId = ++state.routeReqId;
  setBusy(`Routing ${legCount} leg${legCount > 1 ? "s" : ""}…`);
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
    if (myReqId !== state.routeReqId) return;   // superseded
    state.routes = [mergeLegs(legs.map(r => r.route))];
    state.activeIdx = 0;
    renderRoutes();
  } catch (e) {
    if (myReqId !== state.routeReqId) return;   // superseded
    setError(e.message);
  }
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

function renderRoutes() {
  const active = state.routes[state.activeIdx];
  map.getSource("route-active").setData({
    type: "FeatureCollection",
    features: active ? [active] : [],
  });

  if (active) {
    const bbox = lineBbox(active.geometry.coordinates);
    map.fitBounds(bbox, { padding: 60, maxZoom: 13 });
  }

  const html = state.routes.map((r) => {
    const p = r.properties || {};
    const km = fmtKm(+p.gross_length_m || 0);
    const cities = (p.chain_names || []).join(" → ");
    const nBridges = (p.bridges || []).length;
    return `<div class="route-card active">
      <div class="name">trunk route</div>
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
  // Filter to anchors that have an npz on disk for the active profile.
  // While the chainless preprocess is mid-flight, this prevents the
  // user from clicking anchors that aren't yet routable.
  const profile = document.getElementById("profile").value;
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
    const profile = document.getElementById("profile").value;
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

// --- Land cover overlay (OSM landuse, AT/CZ/DE/DK) ---------------------
//
// Static GeoJSON at /data/landcover_corridor.geojson, ~26 MB / 75 K
// polygons rolled up to 5 classes from OSM landuse + natural tags.
// Coverage: only the four current corridor countries (Austria, Czech
// Republic, Germany, Denmark). Adding new tour countries means
// re-running ingest/build_landuse_overlay.py for those countries
// and concatenating into the same file.

const LANDCOVER_COLORS = {
  forest:       "#1e6b3a",
  agricultural: "#d4b366",
  urban:        "#7a7a7a",
  water:        "#3d6fa3",
  wetland:      "#4a8a8a",
};

let landcoverLoaded = false;

async function ensureLandcoverLayers() {
  if (landcoverLoaded) return;
  setBusy("Loading landcover (~26 MB)…");
  try {
    const r = await fetch("/data/landcover_corridor.geojson");
    if (!r.ok) throw new Error(`landcover: ${r.status}`);
    const fc = await r.json();
    map.addSource("landcover", { type: "geojson", data: fc });
    map.addLayer({
      id: "landcover-fill",
      type: "fill",
      source: "landcover",
      paint: {
        "fill-color": [
          "match", ["get", "class"],
          "forest",       LANDCOVER_COLORS.forest,
          "agricultural", LANDCOVER_COLORS.agricultural,
          "urban",        LANDCOVER_COLORS.urban,
          "water",        LANDCOVER_COLORS.water,
          "wetland",      LANDCOVER_COLORS.wetland,
          "#666",
        ],
        "fill-opacity": 0.45,
      },
    }, "cell-gradient-lines");
    landcoverLoaded = true;
    document.getElementById("results").innerHTML = "";
  } catch (e) {
    setError(`landcover load failed: ${e.message}`);
    throw e;
  }
}

document.getElementById("show-landcover").addEventListener("change", async (e) => {
  if (e.target.checked) {
    try {
      await ensureLandcoverLayers();
    } catch {
      e.target.checked = false;
      return;
    }
    map.setLayoutProperty("landcover-fill", "visibility", "visible");
  } else if (landcoverLoaded) {
    map.setLayoutProperty("landcover-fill", "visibility", "none");
  }
});

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
  // Keep route lines on top of the shading.
  for (const id of ["graz-wien-no-canopy", "graz-wien-with-canopy"]) {
    if (map.getLayer(id)) map.moveLayer(id);
  }
  hillshadeLoaded = true;
}

document.getElementById("show-hillshade").addEventListener("change", (e) => {
  if (e.target.checked) {
    ensureHillshade();
    map.setLayoutProperty("hillshade", "visibility", "visible");
  } else if (hillshadeLoaded) {
    map.setLayoutProperty("hillshade", "visibility", "none");
  }
});

// --- Elevation profile chart ------------------------------------------
// Lightweight SVG line chart drawn into the #elevation div under the
// map. Used by the Graz→Wien comparison overlay to plot both variants'
// elevation profiles side-by-side. Colors come from COMPARE_STYLES so
// the chart and the map line colors stay in sync.

function renderElevationProfile(features) {
  const root = document.getElementById("elevation");
  root.innerHTML = "";
  const series = features
    .map(f => ({
      name: f.properties.name,
      variant: f.properties.variant,
      points: (f.properties.profile || []).filter(p => p[1] != null),
      color: (COMPARE_STYLES[f.properties.variant]
        || { color: "#888" }).color,
    }))
    .filter(s => s.points.length > 1);
  if (!series.length) return;

  const w = root.clientWidth || 800;
  const h = root.clientHeight || 140;
  const ml = 36, mr = 8, mt = 6, mb = 18;
  const innerW = w - ml - mr;
  const innerH = h - mt - mb;

  // x: cumulative distance km. Use the max across both series.
  const maxKm = Math.max(...series.map(s => s.points[s.points.length - 1][0]));
  const allElevs = series.flatMap(s => s.points.map(p => p[1]));
  const minE = Math.min(...allElevs);
  const maxE = Math.max(...allElevs);
  const elevRange = Math.max(maxE - minE, 1);

  const sx = (km) => ml + (km / maxKm) * innerW;
  const sy = (e)  => mt + innerH - ((e - minE) / elevRange) * innerH;

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("width", w);
  svg.setAttribute("height", h);
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.style.font = "10px system-ui, sans-serif";

  // y-axis grid + labels (every 200 m)
  const yStep = elevRange > 800 ? 200 : 100;
  const yStart = Math.ceil(minE / yStep) * yStep;
  for (let e = yStart; e <= maxE; e += yStep) {
    const y = sy(e);
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", ml); line.setAttribute("x2", w - mr);
    line.setAttribute("y1", y);  line.setAttribute("y2", y);
    line.setAttribute("stroke", "#eee"); line.setAttribute("stroke-width", "1");
    svg.appendChild(line);
    const txt = document.createElementNS(svgNS, "text");
    txt.setAttribute("x", ml - 4); txt.setAttribute("y", y + 3);
    txt.setAttribute("text-anchor", "end"); txt.setAttribute("fill", "#666");
    txt.textContent = `${e}m`;
    svg.appendChild(txt);
  }
  // x-axis labels (every 50 km)
  for (let km = 0; km <= maxKm; km += 50) {
    const x = sx(km);
    const txt = document.createElementNS(svgNS, "text");
    txt.setAttribute("x", x); txt.setAttribute("y", h - 4);
    txt.setAttribute("text-anchor", "middle"); txt.setAttribute("fill", "#666");
    txt.textContent = `${km}km`;
    svg.appendChild(txt);
  }
  // Series lines
  for (const s of series) {
    const path = document.createElementNS(svgNS, "path");
    let d = "";
    for (let i = 0; i < s.points.length; i++) {
      const [km, e] = s.points[i];
      d += (i === 0 ? "M" : "L") + sx(km).toFixed(1) + " " + sy(e).toFixed(1);
    }
    path.setAttribute("d", d);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", s.color);
    path.setAttribute("stroke-width", "1.5");
    path.setAttribute("stroke-opacity", "0.85");
    svg.appendChild(path);
  }
  // Legend
  const legend = document.createElementNS(svgNS, "g");
  legend.setAttribute("transform", `translate(${ml + 8}, ${mt + 4})`);
  for (let i = 0; i < series.length; i++) {
    const s = series[i];
    const y = i * 12;
    const sw = document.createElementNS(svgNS, "line");
    sw.setAttribute("x1", 0); sw.setAttribute("x2", 14);
    sw.setAttribute("y1", y); sw.setAttribute("y2", y);
    sw.setAttribute("stroke", s.color); sw.setAttribute("stroke-width", "2");
    legend.appendChild(sw);
    const t = document.createElementNS(svgNS, "text");
    t.setAttribute("x", 18); t.setAttribute("y", y + 3); t.setAttribute("fill", "#333");
    t.textContent = s.name;
    legend.appendChild(t);
  }
  svg.appendChild(legend);

  root.appendChild(svg);
}

// --- Graz → Wien canopy-on/off comparison overlay ---------------------
// Static GeoJSON at /data/graz_wien_compare.geojson. Two features
// keyed by `variant`:
//   - "no_canopy"   gray  baseline V2 (elev + curv on, no canopy term)
//   - "with_canopy" green V2 + 0.9× multiplier where canopy_frac > 0
// Built by pgrouting/export_route_compare.py; rerun before/after a
// canopy recompute to refresh either variant in place.

const COMPARE_STYLES = {
  no_canopy:   { color: "#8a8d92", label: "no canopy term" },
  with_canopy: { color: "#2ca02c", label: "with canopy bonus" },
};

let grazWienCompareLoaded = false;

async function ensureGrazWienCompareLayers() {
  if (grazWienCompareLoaded) return;
  setBusy("Loading Graz→Wien comparison…");
  try {
    const r = await fetch("/data/graz_wien_compare.geojson?ts=" + Date.now());
    if (!r.ok) throw new Error(`compare: ${r.status}`);
    const fc = await r.json();
    map.addSource("graz-wien-compare", { type: "geojson", data: fc });
    // no_canopy underneath, with_canopy on top.
    map.addLayer({
      id: "graz-wien-no-canopy",
      type: "line",
      source: "graz-wien-compare",
      filter: ["==", ["get", "variant"], "no_canopy"],
      paint: {
        "line-color": COMPARE_STYLES.no_canopy.color,
        "line-width": 4,
        "line-opacity": 0.85,
      },
    });
    map.addLayer({
      id: "graz-wien-with-canopy",
      type: "line",
      source: "graz-wien-compare",
      filter: ["==", ["get", "variant"], "with_canopy"],
      paint: {
        "line-color": COMPARE_STYLES.with_canopy.color,
        "line-width": 4,
        "line-opacity": 0.85,
      },
    });
    for (const id of ["graz-wien-no-canopy", "graz-wien-with-canopy"]) {
      map.on("click", id, (e) => {
        const p = e.features[0].properties;
        document.getElementById("results").innerHTML =
          `<div class="route-card"><strong>${p.name}</strong>` +
          `<div class="stat">Edges: ${p.edges}</div>` +
          `<div class="stat">Length: ${p.length_km} km</div>` +
          `<div class="stat">Climb: ${p.climb_m} m</div>` +
          `<div class="stat">Under canopy: ${p.canopy_km ?? 0} km</div></div>`;
      });
      map.on("mouseenter", id, () => map.getCanvas().style.cursor = "crosshair");
      map.on("mouseleave", id, () => map.getCanvas().style.cursor = "");
    }
    grazWienCompareLoaded = true;
    map.fitBounds([[15.0, 46.9], [16.8, 48.4]], { padding: 60, duration: 500 });
    renderElevationProfile(fc.features);

    const byVariant = Object.fromEntries(
      fc.features.map(f => [f.properties.variant, f.properties])
    );
    const lines = ["no_canopy", "with_canopy"]
      .filter(v => byVariant[v])
      .map(v => {
        const p = byVariant[v];
        return `<div class="stat" style="color:${COMPARE_STYLES[v].color}">` +
          `▬ ${p.name}: ${p.edges} edges, ${p.length_km} km, ` +
          `${p.climb_m} m climb, ${p.canopy_km ?? 0} km under canopy</div>`;
      }).join("");
    const missing = ["no_canopy", "with_canopy"].filter(v => !byVariant[v]);
    const note = missing.length
      ? `<div class="stat"><em>Missing variant(s): ${missing.join(", ")}. ` +
        `Re-run export_route_compare.py to generate them.</em></div>`
      : `<div class="stat"><em>V2 cost held constant (elev + curv always on); ` +
        `the only difference is the per-edge canopy multiplier ` +
        `(1 - 0.1 × canopy_frac).</em></div>`;
    document.getElementById("results").innerHTML =
      `<div class="route-card"><strong>Graz→Wien canopy comparison</strong>` +
      lines + note + `</div>`;
  } catch (e) {
    setError(`compare load failed: ${e.message}`);
    throw e;
  }
}

document.getElementById("show-graz-wien-compare").addEventListener("change", async (e) => {
  if (e.target.checked) {
    try {
      await ensureGrazWienCompareLayers();
    } catch {
      e.target.checked = false;
      return;
    }
    map.setLayoutProperty("graz-wien-no-canopy", "visibility", "visible");
    map.setLayoutProperty("graz-wien-with-canopy", "visibility", "visible");
  } else if (grazWienCompareLoaded) {
    map.setLayoutProperty("graz-wien-no-canopy", "visibility", "none");
    map.setLayoutProperty("graz-wien-with-canopy", "visibility", "none");
    document.getElementById("elevation").innerHTML = "";
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
  const startRow = document.querySelector('.waypoint-row[data-role="start"]');
  const endRow   = document.querySelector('.waypoint-row[data-role="end"]');
  state.waypoints = [
    bindWaypoint(startRow, "start"),
    bindWaypoint(endRow,   "end"),
  ];
  // Seed defaults.
  state.waypoints[0].coord = DEFAULT_START;
  state.waypoints[0].input.value = DEFAULT_START.join(",");
  state.waypoints[1].coord = DEFAULT_END;
  state.waypoints[1].input.value = DEFAULT_END.join(",");
  refreshWaypointMarkers();
  updateRouteButton();
  loadCities();
  initScenicnessOverlays();
  routeNow();
});
