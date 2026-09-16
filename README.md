# Bike Routing — an AI-native tour planner

> Ask a chat agent to plan a 4,000 km bike tour with rail-accessible
> overnights, then watch the map paint the plan while the agent
> compares routes, hunts for train stations, and enriches your daily
> stages with nearby destinations. Every planning decision is a tool
> call you can see.

![Placeholder for a demo screenshot — Phase 7 will replace this](NOTES/demo.png)

## What this is (portfolio version)

I built this as a demonstration of the skills required for a data-
scientist-to-AI-engineer transition. The routing engine is
DS-flavoured — a custom paired-SPT preprocess over OpenStreetMap
graphs of Austria, Czechia, Germany and Denmark — but the *product*
around it is an agent: a Claude Sonnet–driven planner that uses a
half-dozen tools to compose a bikeable multi-day itinerary.

**What's demonstrated**

- **Tool use as UX.** Custom tools (`route`, `stations_along_route`,
  `pois_near_anchor`, `split_into_stages`, …) designed to *steer* the
  model to good decisions in few round-trips instead of running away
  in a per-candidate loop. See `NOTES/tool-design.md` for the
  before/after story.
- **Streaming SSE + robust API integration** — the agent's plan
  arrives incrementally, tool call by tool call, with client-side
  offline detection, per-request timeout, and idle-watchdog abort.
- **Prompt engineering as system design** — the system prompt
  explicitly separates "compose a plan" from "reformat a plan
  already produced" so a "make it a table" turn doesn't re-run the
  route.
- **MCP server** (Phase 3) — the same tools also expose over
  Anthropic's Model Context Protocol so Claude Desktop can drive
  the planner without the web UI.
- **Multi-agent orchestration** (Phase 4) — a LangGraph fork with
  a planner → router → per-stage enricher (parallel fan-out) →
  composer → critic loop, benchmarked side-by-side with the native
  single-agent SDK on the same prompt. Written up in
  `NOTES/langchain-comparison.md`.
- **Evaluation harness + observability dashboard** (Phases 5-6) —
  gold prompt set, LLM-as-judge scoring, JSONL trace log, static
  Chart.js dashboard for cost/latency/quality by backend.

**What's underneath** — the routing engine itself is described in
`NOTES/routing-engine.md`: multi-source Dijkstra chain graphs,
proximity-based detour filtering, per-anchor SPT polygon bounds,
paired-trunk SQLite blobs. Sub-second query time for 1,500 km routes.

---

## Live demo prompts

Once you're at the running app (see quickstart below), try:

```
Route Graz to Vienna, split into 3-day stages, ~80 km/day. Enrich
each riding day with 2-3 viewpoints within 2 km of the route.
```

```
30 days Graz → Copenhagen, 2-3 days in major cities, ~50 mi/day,
rail-accessible overnights.
```

Watch the map fill in as the agent iterates. Toggle "Show route data"
in the left overlays panel to see the paired-SPT polygons the router
walked underneath.

---

## Original project name

A self-supported bike-tour planner for the Graz → Copenhagen corridor.
OpenStreetMap data + a Postgres/pgRouting-backed SPT preprocess + scenic
POI overlays. Dockerized.

## Architecture

| Service     | Stack                                       | Notes                                |
|-------------|---------------------------------------------|--------------------------------------|
| ingest      | Python + osmium + SpatiaLite                | One-shot PBF + POI download          |
| postgres    | PostGIS + pgRouting (16-3.5-3.7.3)          | Persistent road graph store          |
| pgrouting   | Python + psycopg + pyosmium                 | SPT preprocess (multi-source SQL B-F)|
| api         | Python + FastAPI + numpy/scipy + SpatiaLite | Serves SPT routes + POI lookups      |
| web         | nginx + MapLibre GL JS                      | Static SPA                           |

The routing pipeline is two-stage:
- **Preprocess** (pgrouting service): stream PBFs into Postgres, snap
  `place=city|town` anchors to graph vertices, run a multi-source
  Bellman-Ford wave relaxation until every reachable node is labeled
  with its nearest anchor + parent pointer toward it. Output is
  per-city SPTs on disk plus a small city-graph adjacency.
- **Query** (api service): plan a city sequence on the small city graph,
  walk per-cell gradient pointers across the SPTs to assemble the path.
  Sub-second on any distance once the preprocess has run.

The corridor data comes from **Geofabrik** country PBFs (Austria, Czech
Republic, Germany, Denmark).

## Prerequisites

- Linux host with Docker + Docker Compose (works on WSL2). About 10 GB
  free disk for the full corridor.
- Your user must be in the `docker` group:
  ```sh
  sudo usermod -aG docker $USER
  # Log out + back in (or open a new shell), then verify:
  docker ps
  ```

---

## Phase 1 — data ingest

Downloads OSM PBFs and extracts POIs + routing anchors into a
SpatiaLite database.

### Smoke test (Austria only, ~700 MB)

```sh
cd /path/to/bike
docker compose run --rm ingest --test
```

Downloads Austria PBF (~700 MB), extracts POIs and `place=city|town`
anchors, builds `data/pois/pois.sqlite`. Idempotent — reruns skip
cached files.

### Full corridor (~7 GB)

```sh
docker compose run --rm ingest
```

### Verify the SpatiaLite DB

```sh
docker run --rm -v "$(pwd)/data:/data" debian:bookworm-slim bash -c '
  apt-get update -qq && apt-get install -qq -y sqlite3 libsqlite3-mod-spatialite > /dev/null
  sqlite3 /data/pois/pois.sqlite "SELECT load_extension(\"mod_spatialite\"); SELECT category, COUNT(*) FROM pois GROUP BY category;"
'
```

---

## Phase 2 — SPT preprocess (pgrouting)

Brings up Postgres + the preprocess container, streams PBFs into
`ways` / `ways_vertices_pgr`, snaps anchors, runs the multi-source
Bellman-Ford wave relaxation, and materializes per-city SPTs.

### Bring up Postgres

```sh
docker compose up -d postgres
docker compose exec -u postgres postgres psql -U bike -d bike -c "
  CREATE EXTENSION IF NOT EXISTS postgis;
  CREATE EXTENSION IF NOT EXISTS pgrouting;"
docker compose exec -u postgres postgres psql -U bike -d bike \
  -f /docker-entrypoint-initdb.d/schema.sql 2>/dev/null \
  || docker compose run --rm pgrouting --help > /dev/null   # ensures image builds
```

(For now the schema is applied by hand — see `pgrouting/schema.sql`.)

### Run the preprocess

Smoke (Austria):
Subcommands (run independently):
```sh
docker compose --profile preprocess run --rm pgrouting ingest         --countries austria
docker compose --profile preprocess run --rm pgrouting snap           --countries austria
docker compose --profile preprocess run --rm pgrouting recompute-cost --profile views
```

The V4 polygon-SPT pipeline runs outside `main.py`:
```
export_cells.py                  edges → per-1°-cell .npz files
compute_anchor_spt_polygons.py   1.5-hop hull + 5 km discs per anchor
compute_spts_polygon.py          polygon-bounded multi-source Dijkstras
adapt_polygon_to_paired.py       cities.json + city_graph.json
build_polygon_paired_db.py       SQLite trunk DB consumed by the API
```

Output lands at `data/spt/<profile>/`:
```
cities.json         anchor list (city_idx, name, lon, lat, snap_vids, ...)
city_graph.json     (from_city, to_city, weight, geom) adjacency
<idx>.npz           per-anchor polygon SPT: node_global, parent, cost, coords_lonlat
paired_trunks.db    SQLite trunk DB consumed by /trunk/route
```

---

## Phase 3 — FastAPI service

A small Python service that serves bike routes and POI lookups.

```sh
docker compose up -d api
docker compose logs -f api
```

Listens on port `8000` inside the container, mapped to `8001` on the host.

### Endpoints

#### `GET /health`

```sh
curl http://localhost:8001/health
# {"status":"ok"}
```

#### `GET /trunk/route?from=lon,lat&to=lon,lat&profile=views`

Paired-trunk routing — hot-path ~25 ms. Snaps endpoints to the nearest
anchor pair, looks up the precomputed trunk path from the SQLite
trunk DB, and stitches in first/last-mile straight segments.

#### `GET /way-graph/spt/{city_idx}?profile=direct_polygon`

Per-anchor polygon-bounded SPT for visualization in the UI.

#### `GET /way-graph/spt-status?profile=direct_polygon`

Which anchor SPTs have been written so far — lets the UI light up
anchors as the overnight build progresses.

#### `GET /pois?bbox=minlon,minlat,maxlon,maxlat&category=lodging,food&limit=500`

```sh
curl 'http://localhost:8001/pois?bbox=15.3,47.0,15.6,47.2&category=viewpoint&limit=200'
```

Categories: `viewpoint`, `lodging`, `food`, `bike_service`, `water`.

---

## Phase 4 — MapLibre web UI

A static single-page app driven by the API. nginx serves it on port
`8080` and reverse-proxies `/api/*` to the FastAPI container.

```sh
docker compose up -d --build
```

Open `http://localhost:8080` (or via Tailscale,
`http://desktop-nk6flc3.tail9115a7.ts.net:8080`).


---

## Project layout

```
bike/
├── docker-compose.yml
├── README.md
├── PERF_NOTES.md
├── ingest/                # Phase 1 — OSM + POI download
│   ├── Dockerfile
│   ├── config.py
│   ├── download_osm.py
│   ├── download_util.py
│   ├── extract_pois.py    # osmium tags-filter wrapper
│   ├── build_db.py        # PBF → GeoJSON-Seq → SpatiaLite
│   └── main.py
├── pgrouting/             # Phase 2 — SQL-backed preprocess
│   ├── Dockerfile
│   ├── schema.sql         # ways, ways_vertices_pgr, anchors
│   ├── cost.py            # per-edge bike cost function (port of lht.brf semantics)
│   ├── ingest_pbf.py      # streaming PBF -> Postgres staging tables
│   ├── snap_anchors.py    # SpatiaLite anchors -> Postgres + KNN snap
│   ├── export_cells.py    # ways -> per-1°-cell .npz files
│   ├── compute_anchor_spt_polygons.py
│   ├── compute_spts_polygon.py    # polygon-bounded multi-source Dijkstra
│   ├── adapt_polygon_to_paired.py
│   ├── build_polygon_paired_db.py # SQLite trunk DB
│   ├── config.py
│   └── main.py            # ingest, snap, dem-*, landcover-*, recompute-cost, …
├── api/                   # Phase 3 — FastAPI service
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py        # /health, /trunk/route, /way-graph/*, /pois
│       ├── settings.py
│       ├── geo.py
│       ├── pois.py        # SpatiaLite POI lookups
│       └── trunk_router.py  # paired-trunk lookup
├── web/                   # Phase 4 — static SPA + nginx proxy
└── data/                  # populated by ingest + preprocess (gitignored)
    ├── osm/      *.osm.pbf
    ├── pois/     *.osm.pbf, pois.sqlite
    └── spt/      <profile>/(cities.json, city_graph.json, spt/*.npz)
```

## Port bindings

Port bindings use `0.0.0.0` so they're reachable on any interface.

| Port  | Service              |
|-------|----------------------|
| 8001  | FastAPI (8000 in container) |
| 8080  | Web UI               |
