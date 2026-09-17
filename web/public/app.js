// Bike-routing front-end. AI-native: the chat pane on the right is the
// only route-entry surface. This file owns the map + debug/inspection
// overlays; chat.js talks to the same map instance via window.map and
// synthesizes route state into window.sidebarState so the debug viz
// keeps working after a chat-driven plan.

const API = "/api";

// --- map setup ---------------------------------------------------------

const map = new maplibregl.Map({
  container: "map",
  // OpenFreeMap "positron" — free, no API key, hosted vector tiles
  // based on planet OSM. See https://openfreemap.org.
  style: "https://tiles.openfreemap.org/styles/positron",
  center: [13.5, 51],
  zoom: 5,
  hash: true,
});
map.addControl(new maplibregl.NavigationControl(), "top-right");
map.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: "metric" }), "bottom-left");
// Expose the map so chat.js (a separate script) can add its own
// overlay sources/layers. Top-level `const` in a plain <script> is
// NOT attached to window automatically.
window.map = map;
// Chat.js writes route state here so the paired-SPT debug viz
// (`Show route data`) keeps working after a chat-driven route.
Object.defineProperty(window, "sidebarState", { get: () => state });

// --- state -------------------------------------------------------------

const state = {
  // routesByProfile: { profile: GeoJSON Feature } — populated by
  // chat.js after every `route` tool call. Read by loadRouteSpts /
  // loadRouteTrunks in the debug overlays below.
  routesByProfile: {},
  poiMarkers: { viewpoint: [], lodging: [], food: [], bike_service: [], water: [] },
};

// --- helpers -----------------------------------------------------------

function emptyFC() { return { type: "FeatureCollection", features: [] }; }
function fmtKm(m) { return (m / 1000).toFixed(1) + " km"; }

// The old sidebar's `#results` panel was removed in the AI-native
// refactor, but a dozen overlay-load and click-to-inspect handlers
// still wrote to it. Null-op when the element is gone so those
// handlers don't throw mid-load — a mid-load throw was masking
// successful layer additions as "overlay X load failed" and making
// the toggle look like it did nothing.
function setResults(html) {
  const el = document.getElementById("results");
  if (el) el.innerHTML = html;
}

async function api(path, params) {
  const u = new URL(API + path, location.origin);
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== undefined && v !== null) u.searchParams.set(k, String(v));
  }
  const r = await fetch(u);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
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
    });
    map.addLayer({
      id: "ecoregions-outline",
      type: "line",
      source: "ecoregions",
      paint: {
        "line-color": ["get", "COLOR_BIO"],
        "line-width": 0.6,
        "line-opacity": 0.8,
      },
    });
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
    setResults("");
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
    setResults("");
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
    setResults("");
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
  // Keep the chat route on top of the shading so plans stay visible
  // when the terrain layer is on.
  if (map.getLayer("chat-route-line")) map.moveLayer("chat-route-line");
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
    const r = await fetch(`${API_BASE}/way-graph/spt-status?profile=views_polygon&_=${Date.now()}`);
    if (!r.ok) throw new Error(`spt-status: ${r.status}`);
    const d = await r.json();
    sptStatus.done = d.done;
    sptStatus.total = d.total;
    sptStatus.doneSet = new Set(d.done_indices);
    sptStatus.refByIdx = new Map(d.cities.map(c => [c.city_idx, c.ref]));
    // Inverse map ref -> city_idx, so anchor features can be tagged.
    sptStatus.idxByRef = new Map(d.cities.map(c => [c.ref, c.city_idx]));
    // city_idx -> [lon, lat] — direct coord lookup that avoids the
    // 82 MB cities.json fetch. The spt-status payload is compact
    // (~200 KB for 2k anchors).
    sptStatus.coordByIdx = new Map(
      d.cities.map(c => [c.city_idx, [c.lon, c.lat]]));
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
  // Build a shallow-cloned FeatureCollection so MapLibre re-processes
  // it — passing back the same reference is a no-op in some versions,
  // which leaves anchors visually stuck at their pre-poll color even
  // though the underlying spt_done value is correct.
  const cloned = {
    type: "FeatureCollection",
    features: data.features.map(f => {
      const ref = f.properties.ref;
      const idx = sptStatus.idxByRef ? sptStatus.idxByRef.get(ref) : undefined;
      return {
        type: "Feature",
        geometry: f.geometry,
        properties: {
          ...f.properties,
          city_idx: idx ?? -1,
          spt_done: idx !== undefined && sptStatus.doneSet.has(idx),
        },
      };
    }),
  };
  src.setData(cloned);
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
      setResults(
        `<div class="route-card"><strong>${p.a_name} ↔ ${p.b_name}</strong>` +
        `<div class="stat"><span>cost</span><span>${p.cost_km} km</span></div>` +
        `<div class="stat"><span>a</span><span>${p.a}</span></div>` +
        `<div class="stat"><span>b</span><span>${p.b}</span></div></div>`
      );
    });
    map.on("mouseenter", "way-graph-edges-line", () => map.getCanvas().style.cursor = "pointer");
    map.on("mouseleave", "way-graph-edges-line", () => map.getCanvas().style.cursor = "");
    wayGraphEdgesLoaded = true;
    setResults("");
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
      // Anchor-as-endpoint pick-mode used to live here for the old
      // sidebar route form. Removed with the AI-native refactor:
      // routing is chat-only now, so anchor clicks always fall through
      // to the debug/inspection paths (paired-trunk compare, SPT viz).
      const cityIdx = (sptStatus.idxByRef && sptStatus.idxByRef.get(ref)) ?? -1;
      // Paired-trunk compare mode: first click = A, second click = B,
      // then fetch (A, B) from both v1 and v2 and overlay.
      if (trunkVizState.active && cityIdx >= 0) {
        if (trunkVizState.aIdx === null) {
          trunkVizState.aIdx = cityIdx;
          trunkVizState.aName = p.name;
          document.getElementById("trunk-viz-status").textContent =
            `A=${p.name}, click 2nd anchor for B`;
        } else {
          loadTrunkBlobViz(trunkVizState.aIdx, cityIdx, trunkVizState.aName, p.name);
          trunkVizState.aIdx = null;
        }
        return;
      }
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
        setResults(
          `<div class="route-card"><strong>${p.name}</strong>` +
          `<div class="stat"><span>ref</span><span>${ref}</span></div>` +
          `<div class="stat"><span>kind</span><span>${p.kind} / ${p.place}</span></div>` +
          `<div class="stat"><span>population</span><span>${p.population ?? "—"}</span></div>` +
          `<div class="stat"><span>in chain graph</span><span>${p.in_graph}</span></div>` +
          `<div class="stat"><span>SPT</span><span>pending</span></div></div>`
        );
      }
    });
    map.on("mouseenter", "way-graph-nodes-circles", () => map.getCanvas().style.cursor = "pointer");
    map.on("mouseleave", "way-graph-nodes-circles", () => map.getCanvas().style.cursor = "");
    wayGraphNodesLoaded = true;
    setResults("");
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
    setResults("");
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
  const url = `${API_BASE}/way-graph/spt/${city_idx}?profile=views_polygon&max_features=${maxFeatures}`;
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
    setResults(
      `<div class="route-card"><strong>${label}</strong> SPT` +
      `<div class="stat"><span>edges shown</span><span>${fc.kept_count.toLocaleString()}</span></div>` +
      `<div class="stat"><span>total visited</span><span>${fc.total_visited.toLocaleString()}</span></div>` +
      `<div class="stat"><span>cost range</span><span>${(fc.cost_min/1000).toFixed(1)} – ${(fc.cost_max/1000).toFixed(1)} km</span></div></div>`
    );
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

// --- Paired-trunk viz --------------------------------------------------
// Click anchor A, then anchor B. Fetches /trunk/blob/{A}/{B}?db=... and
// renders as edges colored red (near A) → green (far from A) by haversine
// distance. Toggle the DB dropdown to compare v1 vs v2.

const trunkVizState = {
  active: false,
  aIdx: null,
  aName: null,
  lastAB: null,   // remember for the DB-switch reload
};

function ensureTrunkVizLayer() {
  if (map.getSource("trunk-viz")) return;
  map.addSource("trunk-viz", {
    type: "geojson",
    data: { type: "FeatureCollection", features: [] },
  });
  map.addLayer({
    id: "trunk-viz-line",
    type: "line",
    source: "trunk-viz",
    filter: ["==", ["geometry-type"], "LineString"],
    paint: {
      // Red near A → green far. Ramp matches polygon-SPT viz semantics.
      "line-color": [
        "interpolate", ["linear"], ["get", "dist_from_a"],
        0,     "#dc2626",   // red at A
        5000,  "#f97316",
        15000, "#facc15",
        30000, "#84cc16",
        60000, "#22c55e",   // green far
      ],
      "line-width": [
        "interpolate", ["linear"], ["zoom"],
        8,  1.0,
        12, 1.8,
        16, 2.6,
      ],
      "line-opacity": 0.9,
    },
  });
  map.addLayer({
    id: "trunk-viz-frontier",
    type: "circle",
    source: "trunk-viz",
    filter: ["all", ["==", ["geometry-type"], "Point"], ["get", "is_frontier"]],
    paint: {
      "circle-color": "#facc15",
      "circle-radius": 4,
      "circle-stroke-color": "#000",
      "circle-stroke-width": 1,
    },
  });
}

function haversineM(lon1, lat1, lon2, lat2) {
  const R = 6371000;
  const toRad = d => d * Math.PI / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const s = Math.sin(dLat / 2) ** 2
          + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
}

async function loadTrunkBlobViz(aIdx, bIdx, aName, bName) {
  ensureTrunkVizLayer();
  const db = document.getElementById("trunk-viz-db").value;
  const statusEl = document.getElementById("trunk-viz-status");
  statusEl.textContent = `loading (${aName || aIdx} → ${bName || bIdx}) from ${db}…`;
  try {
    const r = await fetch(`${API_BASE}/trunk/blob/${aIdx}/${bIdx}?db=${db}&profile=views`);
    if (r.status === 404) {
      statusEl.textContent = `no trunk (${aIdx}, ${bIdx}) in ${db}`;
      map.getSource("trunk-viz").setData({ type: "FeatureCollection", features: [] });
      return;
    }
    if (!r.ok) throw new Error(`trunk blob: ${r.status}`);
    const fc = await r.json();
    // Find A's coord from the anchors source. If unavailable, use the
    // first vertex we can find in the feature collection with matching
    // properties.vid = A's snap_vertex.
    let anchor = null;
    if (map.getSource("way-graph-nodes")) {
      const src = map.getSource("way-graph-nodes")._data;
      if (src && src.features) {
        const a = src.features.find(f =>
          sptStatus.idxByRef && sptStatus.idxByRef.get(f.properties.ref) === aIdx);
        if (a) anchor = a.geometry.coordinates;
      }
    }
    // Fallback: use first vertex coord (approximation).
    if (!anchor && fc.features.length > 0) {
      const first = fc.features.find(f => f.geometry.type === "Point");
      if (first) anchor = first.geometry.coordinates;
    }
    // Attach dist_from_a to every feature.
    for (const f of fc.features) {
      let refLon, refLat;
      if (f.geometry.type === "Point") {
        [refLon, refLat] = f.geometry.coordinates;
      } else {
        // LineString: use midpoint for coloring.
        const c = f.geometry.coordinates;
        refLon = (c[0][0] + c[1][0]) / 2;
        refLat = (c[0][1] + c[1][1]) / 2;
      }
      f.properties.dist_from_a = anchor
        ? haversineM(anchor[0], anchor[1], refLon, refLat)
        : 0;
    }
    map.getSource("trunk-viz").setData(fc);
    statusEl.textContent =
      `${aName || aIdx} → ${bName || bIdx} · ${fc.n_vertices} verts, ` +
      `${fc.n_frontier} frontier · ${db}`;
    trunkVizState.lastAB = { aIdx, bIdx, aName, bName };
  } catch (e) {
    statusEl.textContent = `trunk viz failed: ${e.message}`;
  }
}

document.getElementById("trunk-viz-mode").addEventListener("change", (e) => {
  trunkVizState.active = e.target.checked;
  trunkVizState.aIdx = null;
  const statusEl = document.getElementById("trunk-viz-status");
  if (trunkVizState.active) {
    statusEl.textContent = "click 1st anchor for A";
  } else {
    statusEl.textContent = "off";
    if (map.getSource("trunk-viz")) {
      map.getSource("trunk-viz").setData({ type: "FeatureCollection", features: [] });
    }
  }
});

// Re-render the last-shown trunk when the user swaps DBs.
document.getElementById("trunk-viz-db").addEventListener("change", () => {
  if (trunkVizState.lastAB) {
    const { aIdx, bIdx, aName, bName } = trunkVizState.lastAB;
    loadTrunkBlobViz(aIdx, bIdx, aName, bName);
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

// --- Per-anchor polygon overlay for the planned route -------------------
//
// For each chain anchor in the last route, outline its polygon and drop
// a clickable anchor point. Clicking a point highlights that anchor's
// polygon so you can spot which anchor owns a given corridor. Purpose:
// troubleshooting — inspect polygon extents along the route without the
// visual clutter of every SPT edge.

let routeSptsLayerReady = false;
let routeSptsAllPolygons = null;          // cached way_city_spt_polygons.geojson
let routeSptsPolyByIdx   = null;          // Map<city_idx, GeoJSON polygon>

function ensureRouteSptsLayer() {
  if (routeSptsLayerReady) return;

  // Polygon outlines (all route anchors, thin white line).
  map.addSource("route-spt-polygons", {
    type: "geojson",
    data: { type: "FeatureCollection", features: [] },
  });
  map.addLayer({
    id: "route-spt-polygons-line",
    type: "line",
    source: "route-spt-polygons",
    paint: {
      "line-color": "#ffffff",
      "line-width": 1.2,
      "line-opacity": 0.6,
    },
  });

  // Highlighted polygon (fill + thick outline; set on anchor click).
  map.addSource("route-spt-polygon-highlight", {
    type: "geojson",
    data: { type: "FeatureCollection", features: [] },
  });
  map.addLayer({
    id: "route-spt-polygon-highlight-fill",
    type: "fill",
    source: "route-spt-polygon-highlight",
    paint: {
      "fill-color": "#facc15",
      "fill-opacity": 0.18,
    },
  });
  map.addLayer({
    id: "route-spt-polygon-highlight-line",
    type: "line",
    source: "route-spt-polygon-highlight",
    paint: {
      "line-color": "#facc15",
      "line-width": 3,
      "line-opacity": 0.95,
    },
  });

  // Anchor points along the route, clickable.
  map.addSource("route-spt-anchors", {
    type: "geojson",
    data: { type: "FeatureCollection", features: [] },
  });
  map.addLayer({
    id: "route-spt-anchors-circle",
    type: "circle",
    source: "route-spt-anchors",
    paint: {
      "circle-radius": [
        "interpolate", ["linear"], ["zoom"], 6, 3, 10, 5, 14, 7,
      ],
      "circle-color": "#38bdf8",
      "circle-stroke-color": "#0c4a6e",
      "circle-stroke-width": 1.5,
    },
  });
  map.addLayer({
    id: "route-spt-anchors-label",
    type: "symbol",
    source: "route-spt-anchors",
    layout: {
      "text-field": ["get", "name"],
      "text-size": 11,
      "text-anchor": "left",
      "text-offset": [0.6, 0],
      "text-optional": true,
    },
    paint: {
      "text-color": "#0f172a",
      "text-halo-color": "#ffffff",
      "text-halo-width": 1.5,
    },
  });

  // Click an anchor → highlight that polygon.
  map.on("click", "route-spt-anchors-circle", (e) => {
    const f = e.features?.[0];
    if (!f) return;
    const ci = f.properties?.city_idx;
    const poly = routeSptsPolyByIdx?.get(ci);
    const badge = document.getElementById("route-spts-status");
    if (poly) {
      map.getSource("route-spt-polygon-highlight").setData({
        type: "FeatureCollection", features: [poly],
      });
      if (badge) badge.textContent =
        `highlighted: ${f.properties?.name || ci}`;
    } else {
      map.getSource("route-spt-polygon-highlight").setData(
        { type: "FeatureCollection", features: [] });
      if (badge) badge.textContent =
        `no polygon for ${f.properties?.name || ci}`;
    }
  });
  map.on("mouseenter", "route-spt-anchors-circle", () => {
    map.getCanvas().style.cursor = "pointer";
  });
  map.on("mouseleave", "route-spt-anchors-circle", () => {
    map.getCanvas().style.cursor = "";
  });

  routeSptsLayerReady = true;
}

async function fetchAllPolygonsOnce() {
  if (routeSptsAllPolygons) return routeSptsAllPolygons;
  const r = await fetch("/data/way_city_spt_polygons.geojson?ts=" + Date.now());
  if (!r.ok) throw new Error(`spt_polygons: ${r.status}`);
  routeSptsAllPolygons = await r.json();
  return routeSptsAllPolygons;
}

let routeSptsAnchorsGeoJSON = null;
async function fetchAnchorsGeojsonOnce() {
  if (routeSptsAnchorsGeoJSON) return routeSptsAnchorsGeoJSON;
  const r = await fetch("/data/way_city_anchors.geojson?ts=" + Date.now());
  if (!r.ok) throw new Error(`way_city_anchors: ${r.status}`);
  routeSptsAnchorsGeoJSON = await r.json();
  return routeSptsAnchorsGeoJSON;
}

async function loadRouteSpts() {
  ensureRouteSptsLayer();
  const route = state.routesByProfile["views"]
             || Object.values(state.routesByProfile)[0];
  const chainIdx = route?.properties?.chain_city_idx;
  const chainNames = route?.properties?.chain_names || [];
  const badge = document.getElementById("route-spts-status");
  if (!chainIdx || chainIdx.length === 0) {
    badge.textContent = "(plan a route first)";
    map.getSource("route-spt-polygons").setData({ type: "FeatureCollection", features: [] });
    map.getSource("route-spt-anchors").setData({ type: "FeatureCollection", features: [] });
    map.getSource("route-spt-polygon-highlight").setData(
      { type: "FeatureCollection", features: [] });
    return;
  }
  badge.textContent = "loading polygons + anchors…";

  // Build ref ↔ city_idx map from sptStatus so we can join the polygon
  // geojson (keyed by ref) to chain city indices.
  const refForIdx = new Map();
  if (sptStatus.refByIdx) {
    for (const ci of chainIdx) {
      const ref = sptStatus.refByIdx.get(ci);
      if (ref) refForIdx.set(ref, ci);
    }
  }

  // Polygon outlines for every anchor in the chain.
  routeSptsPolyByIdx = new Map();
  try {
    const allPolys = await fetchAllPolygonsOnce();
    const polyFeats = [];
    for (const f of allPolys.features || []) {
      const ci = refForIdx.get(f.properties?.ref);
      if (ci === undefined) continue;
      const feat = {
        ...f,
        properties: { ...(f.properties || {}), city_idx: ci },
      };
      polyFeats.push(feat);
      routeSptsPolyByIdx.set(ci, feat);
    }
    map.getSource("route-spt-polygons").setData({
      type: "FeatureCollection", features: polyFeats,
    });
  } catch (e) {
    console.warn("route-spt polygon outline load failed:", e.message);
  }

  // Anchor points — coord lookup via sptStatus.coordByIdx (built from
  // the compact spt-status endpoint, ~200 KB for 2k anchors). If the
  // page just loaded and sptStatus isn't populated yet, re-fetch it.
  if (!sptStatus.coordByIdx || sptStatus.coordByIdx.size === 0) {
    await fetchSptStatus();
  }
  const anchorFeats = [];
  for (let i = 0; i < chainIdx.length; i++) {
    const ci = chainIdx[i];
    const c = sptStatus.coordByIdx?.get(ci);
    if (!c) continue;
    anchorFeats.push({
      type: "Feature",
      geometry: { type: "Point", coordinates: c },
      properties: {
        city_idx: ci,
        name: chainNames[i] || `city ${ci}`,
        leg: i,
      },
    });
  }
  map.getSource("route-spt-anchors").setData({
    type: "FeatureCollection", features: anchorFeats,
  });
  map.getSource("route-spt-polygon-highlight").setData(
    { type: "FeatureCollection", features: [] });

  badge.textContent =
    `${anchorFeats.length} anchors · ${routeSptsPolyByIdx.size} polygons · click an anchor to highlight`;
}

document.getElementById("show-route-spts").addEventListener("change", async (e) => {
  ensureRouteSptsLayer();
  if (e.target.checked) {
    for (const id of ["route-spt-polygons-line", "route-spt-anchors-circle",
                      "route-spt-anchors-label",
                      "route-spt-polygon-highlight-fill",
                      "route-spt-polygon-highlight-line"]) {
      map.setLayoutProperty(id, "visibility", "visible");
    }
    try { await loadRouteSpts(); }
    catch (err) { setError(`route SPTs: ${err.message}`); }
  } else {
    for (const id of ["route-spt-polygons-line", "route-spt-anchors-circle",
                      "route-spt-anchors-label",
                      "route-spt-polygon-highlight-fill",
                      "route-spt-polygon-highlight-line"]) {
      map.setLayoutProperty(id, "visibility", "none");
    }
  }
});

// --- Paired trunks along route ---------------------------------------
// Same idea as `show-route-spts`, but pulls the actual (A, B) paired
// trunks the router walks — the routing data itself. Useful for spotting
// where a walk terminates unexpectedly (chain-handoff bridge).
//
// Coloring is PER-TRUNK normalized: each vertex's ramp position is its
// distance from A divided by the trunk's max distance from A. Every
// trunk uses the full red→green range regardless of its physical size.
let routeTrunksLayerReady = false;
function ensureRouteTrunksLayer() {
  if (routeTrunksLayerReady) return;
  if (!map || typeof map.addSource !== "function") return;
  if (!map.getSource("route-trunks")) {
    map.addSource("route-trunks", {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] },
      lineMetrics: true,   // required for line-gradient
    });
  }
  if (!map.getLayer("route-trunks-line")) {
    map.addLayer({
      id: "route-trunks-line",
      type: "line",
      source: "route-trunks",
      paint: {
        // Color by trunk_progress ∈ [0, 1] over a red → orange →
        // yellow → green palette. First paired-SPT on the chain is
        // red, the last is green; middle trunks get orange/yellow.
        // Handoffs between adjacent trunks show up as visible color
        // transitions along the corridor. `to-number` coerces a
        // missing property to 0, so features loaded before this
        // change still render (as red).
        "line-color": [
          "interpolate", ["linear"],
          ["to-number", ["get", "trunk_progress"]],
          0.0,  "#dc2626",  // red
          0.33, "#f97316",  // orange
          0.66, "#facc15",  // yellow
          1.0,  "#16a34a",  // green
        ],
        "line-width": [
          "interpolate", ["linear"], ["zoom"],
          8,  0.9,
          12, 1.6,
          16, 2.4,
        ],
        "line-opacity": 0.85,
      },
    });
  }
  routeTrunksLayerReady = true;
}

const ROUTE_TRUNKS_CONCURRENCY = 6;

// Cache city coord lookup so we don't refetch on every DB change.
let _cityCoordsPromise = null;
function fetchCityCoords() {
  if (_cityCoordsPromise) return _cityCoordsPromise;
  _cityCoordsPromise = (async () => {
    const r = await fetch("/data/spt/views/cities.json?ts=" + Date.now());
    if (!r.ok) throw new Error(`cities.json: ${r.status}`);
    const d = await r.json();
    const m = new Map();
    for (const c of d) m.set(c.city_idx, [c.lon, c.lat]);
    return m;
  })().catch(err => {
    _cityCoordsPromise = null;
    throw err;
  });
  return _cityCoordsPromise;
}

async function loadRouteTrunks() {
  ensureRouteTrunksLayer();
  const route = state.routesByProfile["views"]
             || Object.values(state.routesByProfile)[0];
  const chainIdx = route?.properties?.chain_city_idx;
  const chainNames = route?.properties?.chain_names || [];
  const badge = document.getElementById("route-trunks-status");
  const src = map.getSource("route-trunks");
  if (!chainIdx || chainIdx.length < 2) {
    if (badge) badge.textContent = "(plan a route first)";
    if (src) src.setData({ type: "FeatureCollection", features: [] });
    return;
  }
  const db = document.getElementById("trunk-viz-db").value;
  // The router emits {leg: i, skipped: true} for every chain-Dijkstra
  // leg it jumped past via task-#46 skip-lookahead. Filter those out
  // so the viz shows only trunks the router actually walked.
  const bridges = route?.properties?.bridges || [];
  const skippedLegs = new Set(
    bridges.filter(b => b.skipped && typeof b.leg === "number")
           .map(b => b.leg),
  );
  const pairs = [];
  for (let i = 0; i < chainIdx.length - 1; i++) {
    if (skippedLegs.has(i)) continue;   // router jumped past this leg
    pairs.push({ aIdx: chainIdx[i], bIdx: chainIdx[i + 1],
                 aName: chainNames[i], bName: chainNames[i + 1] });
  }
  if (badge) badge.textContent =
    `loading 0 / ${pairs.length}… (${db}, ${skippedLegs.size} skipped)`;

  // Each pair gets a `trunk_progress` in [0, 1] = its position in
  // the walked chain, normalised. The layer paint interpolates the
  // color across a red → orange → yellow → green palette using this
  // value, so the FIRST paired-SPT is red, the LAST is green, and
  // middle trunks smoothly get orange/yellow. Handoffs between
  // trunks show up as color transitions along the corridor at a
  // glance — much clearer than the previous per-branch gradient
  // (which lost trunk grouping) or a 2-color alternation (which
  // gave a jarring red/green/red pattern on adjacent trunks).
  const _lastIdx = Math.max(pairs.length - 1, 1);
  for (let i = 0; i < pairs.length; i++) {
    pairs[i].trunk_idx = i;
    pairs[i].trunk_progress = i / _lastIdx;
  }

  const all = [];
  let done = 0;
  let totalChains = 0;
  const queue = [...pairs];
  async function worker() {
    while (queue.length) {
      const { aIdx, bIdx, aName, bName, trunk_idx, trunk_progress }
        = queue.shift();
      try {
        // Chain-mode endpoint: one LineString per succ leaf-to-root
        // walk. Feature count = ~n_leaves per trunk (tens to
        // hundreds), not ~2 × n_vertices.
        const url = `${API_BASE}/trunk/chains/${aIdx}/${bIdx}?db=${db}&profile=views`;
        const r = await fetch(url);
        if (r.ok) {
          const fc = await r.json();
          totalChains += (fc.n_chains || 0);
          for (const f of fc.features || []) {
            f.properties = f.properties || {};
            f.properties.trunk_a = aIdx;
            f.properties.trunk_b = bIdx;
            f.properties.trunk_idx = trunk_idx;
            f.properties.trunk_progress = trunk_progress;
            f.properties.trunk_a_name = aName || String(aIdx);
            f.properties.trunk_b_name = bName || String(bIdx);
            all.push(f);
          }
        }
      } catch (e) {
        // best-effort
      }
      done++;
      if (badge) badge.textContent =
        `loading ${done} / ${pairs.length}… (${db}, ${totalChains.toLocaleString()} chains)`;
      if (done % 4 === 0 || done === pairs.length) {
        if (src) src.setData({ type: "FeatureCollection", features: all });
      }
    }
  }
  await Promise.all(Array.from({ length: ROUTE_TRUNKS_CONCURRENCY }, worker));
  if (src) src.setData({ type: "FeatureCollection", features: all });
  if (badge) badge.textContent =
    `${pairs.length} walked trunks · ${totalChains.toLocaleString()} chains `
    + `· ${skippedLegs.size} skipped · ${db}`;
}

document.getElementById("show-route-trunks").addEventListener("change", async (e) => {
  try {
    ensureRouteTrunksLayer();
    if (e.target.checked) {
      map.setLayoutProperty("route-trunks-line", "visibility", "visible");
      await loadRouteTrunks();
    } else {
      map.setLayoutProperty("route-trunks-line", "visibility", "none");
    }
  } catch (err) {
    console.error("route-trunks toggle failed:", err);
    setError(`route trunks: ${err.message}`);
  }
});

// Reload trunks when the DB dropdown changes so the user can compare
// v2c vs v2d along a whole route.
document.getElementById("trunk-viz-db").addEventListener("change", () => {
  const cb = document.getElementById("show-route-trunks");
  if (cb && cb.checked) {
    loadRouteTrunks().catch(err => {
      console.error("route-trunks db-swap reload failed:", err);
      setError(`route trunks: ${err.message}`);
    });
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
//
// Both write to the small #status-hint slot at the bottom of the
// overlay sidebar (replaces the removed #results panel). Null-safe so
// script order / missing element doesn't crash callers.

function setBusy(msg) {
  const el = document.getElementById("status-hint");
  if (el) el.textContent = msg;
  else console.info("[busy]", msg);
}

function setError(msg) {
  console.error("[err]", msg);
  const el = document.getElementById("status-hint");
  if (el) el.innerHTML = `<span style="color:#c33">${msg}</span>`;
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
  initScenicnessOverlays();
});

// --- Consolidated debug toggles ----------------------------------------
//
// Two coarse toggles that fan out to the legacy per-layer checkboxes
// (kept hidden in index.html). Simpler mental model:
//   * "Show city graph"  — chain edges + anchors, click anchor → its polygon.
//   * "Show route data"  — polygons + walked paired trunks for the current route.

async function _fireHiddenToggle(id, on) {
  const cb = document.getElementById(id);
  if (!cb) return;
  if (cb.checked !== on) {
    cb.checked = on;
    cb.dispatchEvent(new Event("change"));
  }
}

document.getElementById("show-city-graph").addEventListener("change", async (e) => {
  const on = e.target.checked;
  const badge = document.getElementById("city-graph-status");
  badge.textContent = on ? "loading…" : "off";
  await _fireHiddenToggle("show-way-graph-edges", on);
  await _fireHiddenToggle("show-way-graph-nodes", on);
  // Load polygons in the background so click-to-highlight works, but keep
  // the "all polygons" line layer hidden — the user opts in to just the
  // clicked anchor's polygon.
  if (on) {
    try {
      await ensureWayGraphPolygons();
      map.setLayoutProperty("way-graph-polygons-line", "visibility", "none");
      map.setLayoutProperty("way-graph-polygon-highlight-line",
                            "visibility", "visible");
      badge.textContent = "on · click an anchor to see its polygon";
    } catch (err) {
      badge.textContent = `err: ${err.message}`;
    }
  } else if (wayGraphPolygonsLoaded) {
    map.setLayoutProperty("way-graph-polygon-highlight-line", "visibility", "none");
    const src = map.getSource("way-graph-polygon-highlight");
    if (src) src.setData({ type: "FeatureCollection", features: [] });
  }
});

document.getElementById("show-route-data").addEventListener("change", async (e) => {
  const on = e.target.checked;
  const badge = document.getElementById("route-data-status");
  badge.textContent = on ? "loading…" : "off";
  await _fireHiddenToggle("show-route-spts",   on);
  await _fireHiddenToggle("show-route-trunks", on);
  badge.textContent = on ? "on" : "off";
});
