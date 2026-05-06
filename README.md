# Bike Routing — Graz → Copenhagen

A self-supported bike-tour planner for the Graz → Copenhagen corridor. OpenStreetMap data + BRouter routing + scenic POI overlays (vistas, lodging, food, protected areas, EuroVelo / national bike networks). Dockerized.

## Architecture

| Service     | Stack                                       | Status        |
|-------------|---------------------------------------------|---------------|
| ingest      | Python + osmium + SpatiaLite                | ✅ Phase 1    |
| brouter     | OpenJDK + abrensch/brouter 1.7.9            | ✅ Phase 2    |
| api         | Python + FastAPI + httpx + SpatiaLite + numpy/scipy | ✅ Phase 3 |
| web         | nginx + MapLibre GL JS                      | ✅ Phase 4    |
| preprocess  | Python + pyosmium + scipy + shapely         | ✅ Phase 5    |

The runtime layers stack:

- **Phase 1-4** are the original BRouter-backed system. BRouter is a battle-tested cycle-routing engine; this gets you correct routes for any (start, end) at the cost of multi-minute wall time on continental distances.
- **Phase 5 (long-distance optimization)** layers two things on top: (a) auto-waypoint insertion + per-leg routing cache, both transparent to BRouter; (b) a parallel SPT-based router built on a Python road graph, no BRouter at query time, sub-second on any distance once the per-profile preprocess has run.

The corridor data comes from:
- **Geofabrik** country PBFs (Austria, Czech Republic, Germany, Denmark) for POI extraction.
- **BRouter segment tiles** (`.rd5`) covering 5°E–20°E × 45°N–60°N for routing graph + SRTM elevation.

## Prerequisites

- Linux host with Docker + Docker Compose (works on WSL2). About 10 GB free disk for the full corridor.
- Your user must be in the `docker` group:
  ```sh
  sudo usermod -aG docker $USER
  # Log out + back in (or open a new shell), then verify:
  docker ps
  ```
  *(Without this, Docker commands fail with "permission denied while trying to connect to the docker API".)*

---

## Phase 1 — data ingest pipeline ✅

Downloads OSM PBFs, BRouter `.rd5` tiles, and extracts POIs into a SpatiaLite database.

### Smoke test (~700 MB, a few minutes)

```sh
cd /path/to/bike
docker compose run --rm ingest --test
```

Downloads Austria PBF (~700 MB) + the Graz BRouter tile `E15_N45.rd5` (~250 MB), extracts POIs, builds `data/pois/pois.sqlite`. Idempotent — reruns skip cached files.

Expected output ends with a summary like:
```
[summary] /data/pois/pois.sqlite
  total POIs: ~80000
    bike_service     ~1000
    food            ~30000
    lodging         ~10000
    viewpoint        ~3500
    water           ~30000
```

### Full corridor (~7 GB, longer download)

```sh
docker compose run --rm ingest
```

### Verify the SpatiaLite DB by hand

```sh
docker run --rm -v "$(pwd)/data:/data" debian:bookworm-slim bash -c '
  apt-get update -qq && apt-get install -qq -y sqlite3 libsqlite3-mod-spatialite > /dev/null
  sqlite3 /data/pois/pois.sqlite "SELECT load_extension(\"mod_spatialite\"); SELECT category, COUNT(*) FROM pois GROUP BY category;"
'
```

---

## Phase 2 — BRouter routing engine ✅

Brings up a BRouter HTTP routing server using the `.rd5` segment tiles fetched in Phase 1, plus a custom routing profile `lht.brf` tuned for the user's bike, fitness, and preferences.

### Custom profile: `lht.brf` (Long Haul Trucker)

Located at `brouter/profiles/lht.brf` and bind-mounted into the container at `/opt/brouter/customprofiles/`. Differs from the standard `trekking.brf` baseline:

- **Heavy uphill cost** (`uphillcost=250`, `uphillcutoff=0.5%`) — strong climb avoidance. BRouter's DSL only supports linear cost-per-percent above a cutoff, so this is a practical approximation of the user's preferred quadratic-on-grade model.
- **Zero downhill cost** — straight downhills are *preferred*, not just neutral. The "curvature × |grade|" penalty for curvy descents is computed in the Phase 3 Python re-rank pass (BRouter's DSL doesn't expose intra-way curvature cleanly).
- **40mm-tire surface filter** — `tracktype=grade1-3` OK, `grade4` heavily penalized, `grade5` effectively excluded. Sand/mud/very_bad smoothness rejected.
- **Scenic biases on by default** — rivers, forest, low-traffic, low-noise.
- **EuroVelo / national / regional cycle networks treated as ideal** (cost=1).

### Start the routing server

You need the BRouter `.rd5` tiles already on disk. The smoke-test ingest fetches one tile (covers Graz). Either run the full ingest, or just the smoke test if you only want to test routes near Graz.

```sh
docker compose up -d brouter
docker compose logs -f brouter
```

The server listens on port `17777` on all interfaces, so it's reachable over Tailscale at `http://desktop-nk6flc3.tail9115a7.ts.net:17777`.

### Smoke-test the routing API

From the desktop or your laptop:

```sh
# Graz (15.43, 47.07) → Vienna (16.37, 48.21), GeoJSON output
curl 'http://desktop-nk6flc3.tail9115a7.ts.net:17777/brouter?lonlats=15.43,47.07|16.37,48.21&profile=lht&alternativeidx=0&format=geojson' \
  | python3 -m json.tool | head -40
```

Useful URL params:
| param            | example                     | notes                                                     |
|------------------|-----------------------------|-----------------------------------------------------------|
| `lonlats`        | `15.43,47.07\|16.37,48.21`  | one or more `lon,lat` pairs, joined by `\|`                |
| `profile`        | `lht`                       | matches a `.brf` filename in customprofiles or profiles2  |
| `alternativeidx` | `0`–`3`                     | get up to four ranked alternative routes                  |
| `format`         | `geojson` / `gpx` / `kml`   | response format                                            |

### Long routes need via-points

BRouter's search is roughly exponential in straight-line distance. Point-to-point routes longer than ~600 km tend to exceed the 1800 s `maxRunningTime` cap (and even with more time, hold the JVM in heavy memory). The corridor target is ~1300 km Graz → Copenhagen — to plan that end-to-end, pass intermediate via-points:

```sh
# Single shot: Graz → Prague → Berlin → Copenhagen
curl -fsS 'http://localhost:17777/brouter?lonlats=15.43,47.07|14.42,50.08|13.40,52.52|12.57,55.68&profile=lht&format=geojson'
```

Two or three via-points across the corridor lets BRouter plan each segment in tens of seconds instead of timing out. The web UI takes any number of clicks: 1st = start, 2nd = end, every subsequent click inserts an intermediate via-point between them. Each leg is solved by BRouter and the results are stitched into one continuous polyline.

### Compare profiles

The Dockerfile bundles BRouter's standard profiles too — `trekking`, `trekking-noferries`, `gravel`, `fastbike-lowtraffic`. Swap `profile=lht` for any of those to see how routing changes.

```sh
for p in lht trekking gravel fastbike-lowtraffic; do
  echo "=== $p ==="
  curl -s "http://localhost:17777/brouter?lonlats=15.43,47.07|16.37,48.21&profile=$p&format=geojson" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);
p=d['features'][0]['properties'];
print(f\"  distance: {p['track-length']} m, duration: {p['total-time']} s, climb: {p['filtered ascend']} m\")"
done
```

### Iterating on the profile

Edits to `brouter/profiles/lht.brf` take effect on the **next request** — BRouter reloads profiles on demand and the file is bind-mounted, no rebuild needed.

If something is wrong with the syntax, BRouter responds with a parse-error message in the body.

---

## Project layout

```
bike/
├── docker-compose.yml
├── .gitignore
├── README.md
├── ingest/
│   ├── Dockerfile
│   ├── config.py             # corridor bbox, country list, POI tag filters
│   ├── download_util.py      # resumable HTTP fetcher
│   ├── download_osm.py       # Geofabrik PBFs
│   ├── download_brouter.py   # BRouter .rd5 segment tiles
│   ├── extract_pois.py       # osmium tags-filter wrapper
│   ├── build_db.py           # PBF → GeoJSON-Seq → SpatiaLite
│   └── main.py               # orchestrator (--test or full)
├── brouter/
│   ├── Dockerfile            # OpenJDK + BRouter 1.7.9
│   └── profiles/
│       └── lht.brf           # custom Long Haul Trucker profile
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py           # FastAPI endpoints
│       ├── settings.py       # paths, BRouter URL, default profile
│       ├── geo.py            # haversine + bearing math
│       ├── brouter.py        # async HTTP client to BRouter
│       ├── pois.py           # SpatiaLite POI lookups
│       ├── scoring.py        # curvy-descent + scenic re-rank
│       └── stages.py         # ~100 km leg splitter + lodging clusters
└── data/                     # populated by ingest (gitignored)
    ├── osm/         *.osm.pbf
    ├── brouter/     *.rd5    # consumed by brouter container
    └── pois/        *.osm.pbf, pois.sqlite
```

---

## Phase 3 — FastAPI service ✅

A small Python service that wraps BRouter and the SpatiaLite POI DB into a clean HTTP API. Adds the curvature×grade penalty and scenic re-rank that BRouter's DSL can't express directly.

### Start the API (brings BRouter up too as a dependency)

```sh
docker compose up -d api
docker compose logs -f api
```

Listens on port `8000`. From the laptop: `http://desktop-nk6flc3.tail9115a7.ts.net:8000`.

### Endpoints

#### `GET /health`

```sh
curl http://localhost:8001/health
# {"status":"ok"}
```

#### `GET /route?from=lon,lat&to=lon,lat&profile=lht&alternatives=N&rerank=true`

Proxies to BRouter and (optionally) re-ranks alternatives by composite score:
`raw_cost + 5·curvy_descent_penalty − 50·viewpoints_near_route`.

```sh
# Single route Graz → Vienna
curl 'http://localhost:8001/route?from=15.43,47.07&to=16.37,48.21&profile=lht'

# Three alternatives, re-ranked for scenicness + curvy-descent avoidance
curl 'http://localhost:8001/route?from=15.43,47.07&to=16.37,48.21&profile=lht&alternatives=3&rerank=true' \
  | python3 -m json.tool
```

Each route in the response gets a `properties.scoring` block:
```json
{
  "track_length_km": 197.4,
  "ascend_m": 1240,
  "raw_cost": 4321,
  "curvy_descent_penalty": 87.3,
  "viewpoints_near_route": 5,
  "composite_score": 4506.5
}
```

#### `GET /pois?bbox=minlon,minlat,maxlon,maxlat&category=lodging,food&limit=500`

```sh
# All viewpoints near Graz
curl 'http://localhost:8001/pois?bbox=15.3,47.0,15.6,47.2&category=viewpoint&limit=200'
```

Categories: `viewpoint`, `lodging`, `food`, `bike_service`, `water`.

#### `GET /stages?from=lon,lat&to=lon,lat&target_km=100&lodging_radius_m=3000`

Splits a route into legs of about `target_km` each and returns nearby lodging POIs at every leg-end. Useful for daily-stage planning on a multi-day tour.

```sh
# Graz → Copenhagen, 100km daily stages
curl 'http://localhost:8001/stages?from=15.43,47.07&to=12.57,55.68&target_km=100' \
  | python3 -m json.tool
```

Returns:
```json
{
  "from": [15.43, 47.07],
  "to": [12.57, 55.68],
  "profile": "lht",
  "target_km": 100,
  "total_legs": 13,
  "total_length_m": 1287000,
  "legs": [
    {
      "start": [15.43, 47.07],
      "end":   [14.91, 47.55],
      "length_m": 100123,
      "ascend_m": 720,
      "lodging": [
        {"name": "Gasthaus ...", "category": "lodging", "subtype": "guest_house",
         "lon": 14.92, "lat": 47.54, "distance_m": 850, ...}
      ]
    }
  ]
}
```

### Scoring weights

The composite score weights live in [`api/app/scoring.py`](api/app/scoring.py) at the top of the file (`W_CURVY_DESCENT`, `W_VIEWPOINT_BONUS`, `VIEWPOINT_BUFFER_M`). They're first-pass guesses — calibrate against a few real Graz → Vienna runs to dial them in.

---

## Phase 4 — MapLibre web UI ✅

A static single-page app that drives the API. nginx serves the page on port `8080` and reverse-proxies `/api/*` to the FastAPI container — so the browser only ever talks to one origin (no CORS gymnastics).

### Bring the whole stack up

```sh
# Build everything (first time only — ~3 minutes)
docker compose build

# Bring up brouter + api + web
docker compose up -d

# Watch logs
docker compose logs -f
```

Then open **`http://desktop-nk6flc3.tail9115a7.ts.net:8080`** from your laptop.

### How to use the UI

1. **Click the map** to drop a green start pin.
2. **Click again** to drop a red end pin.
3. Pick a profile + number of alternatives + whether to re-rank.
4. **`Route`** → routes drawn on map; sidebar lists each alternative with distance, climb, time, and (if re-rank is on) the scoring breakdown. Click an alternative card to make it the active route.
5. **`Stages (~100 km)`** → splits the route into daily legs and drops a numbered marker at each leg-end with up to 5 lodging options nearby.
6. **POI overlays** — toggle viewpoints / lodging / food / bike services / drinking water. POIs auto-load while you pan and zoom (only at zoom ≥ 9; capped at 500 markers per fetch).
7. The strip at the bottom is an **elevation profile** of the active route (BRouter's per-coordinate SRTM elevation).
8. **`Clear`** wipes pins, route, legs, and POI selections.

### Going up / down independently

```sh
docker compose up -d brouter         # routing only
docker compose up -d api             # API + brouter (compose pulls in deps)
docker compose up -d web             # whole stack
docker compose down                  # stop everything
docker compose down -v               # plus drop named volumes (no data here, but)
docker compose restart api           # reload API after editing code
```

The `api/app/` directory is **not** bind-mounted; editing scoring weights or endpoints requires `docker compose up -d --build api`. The BRouter `lht.brf` profile **is** bind-mounted — edits to `brouter/profiles/lht.brf` apply on the next routing request, no restart.

---

## Project layout

```
bike/
├── docker-compose.yml
├── .gitignore
├── README.md
├── ingest/                   # Phase 1 — data pipeline (one-shot)
│   └── ...
├── brouter/                  # Phase 2 — routing engine
│   ├── Dockerfile
│   └── profiles/
│       └── lht.brf
├── api/                      # Phase 3 — FastAPI service
│   └── ...
├── web/                      # Phase 4 — static SPA
│   ├── Dockerfile
│   ├── nginx.conf
│   └── public/
│       ├── index.html
│       ├── app.js
│       └── style.css
└── data/                     # populated by ingest (gitignored)
    ├── osm/         *.osm.pbf
    ├── brouter/     *.rd5
    └── pois/        *.osm.pbf, pois.sqlite
```

---

## Phase 5 — long-distance routing optimization ✅

Two layered optimizations on top of the BRouter stack, plus a parallel
SPT engine that bypasses BRouter entirely for long routes. The full
empirical numbers are in [`PERF_NOTES.md`](PERF_NOTES.md).

### 5a. Auto-waypoint insertion + per-leg cache

These two are transparent to existing BRouter callers — same `/route`
endpoint, just faster. Auto-waypoint kicks in on long 2-point routes
(>250 km) by inserting `place=city|town` anchors along the great-circle
corridor. Per-leg cache keys each `(profile, from, to)` segment to disk
in `data/cache/legs.sqlite`.

#### Check it's working

```sh
# A) Direct point-to-point goes through auto-waypointing automatically.
#    First call is cold; second hits the cache.
curl -s "http://localhost:8001/route?from=15.43,47.07&to=12.57,55.68" \
  -o /tmp/route1.json -w "first call: %{time_total}s\n"

curl -s "http://localhost:8001/route?from=15.43,47.07&to=12.57,55.68" \
  -o /tmp/route2.json -w "second call: %{time_total}s\n"
# Expect: first ~100s cold, second ~0.6s warm.

# B) See which anchors got inserted.
python3 -c "import json; d=json.load(open('/tmp/route2.json')); \
  print(d['routes'][0]['properties']['cache-hits']); \
  [print(f'  {a[\"name\"]}') for a in d.get('auto_waypoints', [])]"

# C) Inspect / clear the leg cache.
curl -s http://localhost:8001/cache/stats | python3 -m json.tool
curl -s -X POST http://localhost:8001/cache/clear        # nuke all
curl -s -X POST 'http://localhost:8001/cache/clear?profile=lht'
```

### 5b. SPT preprocess + graph-Voronoi router

A separate, optional engine. Run a per-profile preprocess that builds:
- A Python routable road graph from the same Geofabrik PBFs.
- Multi-source Dijkstra labeling every road node with its nearest
  anchor city (graph-Voronoi cell assignment).
- One per-city SPT covering each city's `cell ∪ adjacent cells`.
- Concave-hull polygons per cell for the click-to-show UI.

At query time the API plans a city sequence on the small city graph,
then walks per-cell gradient pointers across the SPTs — no BRouter, no
runtime search. Sub-second on any distance once preprocess has run.

#### Run the preprocess

Smoke (Austria, ~6 min, ~1.5 GB output):
```sh
docker compose --profile preprocess run --rm preprocess --profile lht --countries austria
```

Full corridor (~30-60 min, ~7-10 GB output, requires PBFs from Phase 1):
```sh
docker compose --profile preprocess run --rm preprocess \
  --profile lht --countries austria,czech-republic,germany,denmark
```

Output lands at `data/spt/lht/`:
```
graph_nodes.npz          dense (lon, lat, osm_id) for every road node
graph_edges.npz          (src, dst, cost, length_m) directed
spt_fwd.npz              global multi-source forward SPT (cost, parent, city_idx)
spt_rev.npz              global multi-source reverse SPT
global_assignment.npz    int32 city_idx per node (cell membership)
cities.json              anchor list with lon/lat/place/population
city_graph.json          (from_city, to_city, weight) adjacency
cells.geojson            FeatureCollection of cell polygons
spt/<idx>.npz            per-city SPT over (cell + neighbors)
```

#### Check it's working

```sh
# A) Verify the SPT engine is registered.
curl -s 'http://localhost:8001/cells/cities?profile=lht' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
                print(f'profile={d[\"profile\"]} cities={d[\"count\"]}')"

# B) Route an Austrian pair via SPT. Expect ~50ms warm.
curl -s 'http://localhost:8001/spt/route?from=15.43,47.07&to=14.29,48.30&profile=lht' \
  -o /tmp/spt.json -w "time=%{time_total}s\n"

python3 -c "
import json
d = json.load(open('/tmp/spt.json'))
p = d['route']['properties']
print(f'cities: {p[\"cities\"]}')
print(f'track-length: {p[\"track-length\"]/1000:.1f} km, {p[\"node-count\"]:,} nodes')
"

# C) Cell polygon for a specific anchor (Graz is city_idx 2 in Austria).
curl -s 'http://localhost:8001/cells/2?profile=lht' \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f\"city: {d['city']['name']}\")
print(f\"polygon: {d['polygon']['geometry']['type']}, \"
      f\"node_count={d['polygon']['properties']['node_count']:,}\")
print(f'top neighbors:')
for n in d['neighbors'][:5]:
    print(f\"  {n['name']:30s} weight={n['weight']:.0f}\")
"
```

#### Use it from the UI

In the controls panel:
- **Engine** select: switch to **SPT (precomputed)**. Click the map for start + end points (intermediate via-points are ignored — the city graph plans the corridor). Click **Route**.
- **Voronoi cells** panel: enable **Show city anchors**. Blue dots appear at every anchor; click any one to overlay its Voronoi cell polygon and see its neighbor list.

Switching the **Profile** select reloads anchors / cells for that profile (only `lht` exists today; v2 will add scenic / fast variants once the cost function ports more of `.brf`).

### Concurrent reads + the atomic-swap

The preprocess writes to `data/spt/<profile>.tmp/` and atomically
renames to `data/spt/<profile>/` on completion, so a live API mmap'd
to the previous output keeps serving from the (now-renamed) old
directory until clients trigger fresh loads. If you hit "Failed to
fetch" mid-preprocess anyway, just wait for it to finish and reload.

### Known v1 gaps

- **No elevation in the SPT cost function.** Cells reflect flat-distance topology. Routes don't penalize climbs the way BRouter's `lht.brf` does. Adding SRTM ingest is the v2 priority.
- **No scenic biases** (`estimated_*_class` BRouter terms are computed during BRouter's own preprocess; we don't replicate them).
- **One profile only.** v1 cost function is hardcoded in `preprocess/cost.py`. The `--profile` flag currently only renames the output dir.
- **Edge cases on disconnected components.** Some rural islands in the road graph aren't reachable from any anchor; those nodes have `cost=inf` and are excluded from cells.

---

## Quick reference: end-to-end test from a fresh clone

```sh
# 1) Make sure your user can talk to Docker.
sudo usermod -aG docker $USER
# (log out + back in OR open a new shell session)

# 2) Smoke-test ingest with a single country (~700 MB download).
docker compose run --rm ingest --test

# 3) Build + start the routing/api/web stack.
docker compose up -d --build

# 4) Verify each layer.
curl -fsS http://localhost:17777/brouter?lonlats=15.43,47.07'|'16.37,48.21'&'profile=lht'&'format=geojson | head -c 200
curl -fsS http://localhost:8001/health
curl -fsS 'http://localhost:8001/route?from=15.43,47.07&to=16.37,48.21&profile=lht'
open http://localhost:8080   # or visit it from your laptop via Tailscale

# 5) (Optional) Build the SPT engine for instant long-distance routing.
docker compose --profile preprocess run --rm preprocess --profile lht --countries austria
curl -fsS 'http://localhost:8001/spt/route?from=15.43,47.07&to=14.29,48.30&profile=lht' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
    print('SPT route:', d['route']['properties']['cities'])"
```

For the full corridor (Graz → Copenhagen end-to-end), rerun `docker compose run --rm ingest` (no `--test`) — that downloads all 4 country PBFs and 9 BRouter tiles, ~7 GB. Then optionally run the SPT preprocess against all four countries (`--countries austria,czech-republic,germany,denmark`) for sub-second long-distance routing.

## Remote access

This stack is designed to run on `desktop-nk6flc3.tail9115a7.ts.net` (over Tailscale) and be hit from a laptop. All Docker port bindings use `0.0.0.0` so they're reachable on any interface; just point the laptop at `http://desktop-nk6flc3.tail9115a7.ts.net:<port>` once the relevant phase is up:

| Port  | Service       |
|-------|---------------|
| 17777 | BRouter       |
| 8001  | FastAPI (API container exposes 8000 internally; mapped to 8001 because port 8000 is taken by another local service) |
| 8080  | Web UI        |

## Project layout (current)

```
bike/
├── docker-compose.yml
├── README.md
├── PERF_NOTES.md          # empirical timings + tradeoffs
├── ingest/                # Phase 1 — POI + BRouter tile pipeline
├── brouter/               # Phase 2 — routing engine container + lht.brf
├── api/                   # Phase 3 — FastAPI service
│   └── app/
│       ├── main.py
│       ├── brouter.py     # BRouter HTTP client + per-leg cache wrapper
│       ├── leg_cache.py   # SQLite-backed routing leg cache (Phase 5a)
│       ├── anchors.py     # auto-waypoint selector (Phase 5a)
│       ├── cells_api.py   # /cells/* endpoints (Phase 5b)
│       ├── spt_router.py  # SPT-based routing engine (Phase 5b)
│       ├── pois.py
│       ├── scoring.py     # curvy-descent + scenic re-rank (BRouter only)
│       └── stages.py
├── web/                   # Phase 4 — static SPA + nginx proxy
├── preprocess/            # Phase 5b — SPT preprocess pipeline
│   ├── extract_graph.py   # pyosmium streaming PBF -> graph
│   ├── cost.py            # per-edge cost function (port of lht.brf)
│   ├── spt.py             # global multi-source Dijkstra
│   ├── cells.py           # alpha-shape polygons + city graph
│   ├── per_city_spt.py    # per-city subgraph SPTs
│   ├── save.py
│   └── main.py
└── data/                  # populated by ingest + preprocess (gitignored)
    ├── osm/      *.osm.pbf
    ├── brouter/  *.rd5
    ├── pois/     *.osm.pbf, pois.sqlite
    ├── cache/    legs.sqlite
    └── spt/      <profile>/(graph_nodes.npz, spt/*.npz, cells.geojson, ...)
```
