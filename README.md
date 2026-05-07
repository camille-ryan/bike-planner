# Bike Routing — Graz → Copenhagen

A self-supported bike-tour planner for the Graz → Copenhagen corridor.
OpenStreetMap data + a Postgres/pgRouting-backed SPT preprocess + scenic
POI overlays (vistas, lodging, food, protected areas, EuroVelo / national
bike networks). Dockerized.

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
```sh
docker compose --profile preprocess run --rm pgrouting all --countries austria
```

Full corridor:
```sh
docker compose --profile preprocess run --rm pgrouting all \
    --countries austria,czech-republic,germany,denmark
```

Subcommands (run independently):
```sh
docker compose --profile preprocess run --rm pgrouting ingest --countries austria
docker compose --profile preprocess run --rm pgrouting snap   --countries austria
docker compose --profile preprocess run --rm pgrouting spts   --profile lht
```

Output lands at `data/spt/<profile>/`:
```
cities.json         anchor list (city_idx, name, lon, lat, snap_vertex_id, ...)
city_graph.json     (from_city, to_city, weight) adjacency
spt/<idx>.npz       per-city SPT: node_global, parent_local, cost
```

The pgrouting pipeline does **not** currently emit `graph_nodes.npz`,
`global_assignment.npz`, or `cells.geojson` — the API consumer side
is being rewritten to query Postgres directly instead of reading those
legacy artifacts.

---

## Phase 3 — FastAPI service

A small Python service that serves SPT routes and POI lookups.

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

#### `GET /spt/route?from=lon,lat&to=lon,lat&profile=lht`

SPT-based routing — sub-second on any distance once preprocess has run.
Snaps endpoints to the road graph, plans a city sequence on the city
graph, walks per-city SPT parent pointers to assemble the path.

> **Note:** the underlying loader is being rewritten to read from
> Postgres rather than the legacy npz artifacts; this endpoint is
> in-flight until the pgrouting preprocess settles.

#### `GET /cells/cities?profile=lht`

List the anchors with precomputed cells.

#### `GET /cells/{city_idx}?profile=lht`

Voronoi cell polygon for a given anchor, plus its neighbor list.

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

The UI's BRouter "Engine" path is dead code pending cleanup — only the
SPT engine reflects the current backend.

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
├── pgrouting/             # Phase 2 — SQL-backed SPT preprocess
│   ├── Dockerfile
│   ├── schema.sql         # ways, ways_vertices_pgr, anchors, visited, city_adjacency
│   ├── cost.py            # per-edge bike cost function (port of lht.brf semantics)
│   ├── ingest_pbf.py      # streaming PBF -> Postgres staging tables
│   ├── snap_anchors.py    # SpatiaLite anchors -> Postgres + KNN snap
│   ├── compute_spts.py    # multi-source Bellman-Ford waves + per-city dump
│   ├── config.py
│   └── main.py            # subcommands: ingest, snap, spts, all
├── api/                   # Phase 3 — FastAPI service
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py        # /health, /spt/route, /cells/*, /pois
│       ├── settings.py
│       ├── geo.py
│       ├── pois.py        # SpatiaLite POI lookups
│       ├── cells_api.py   # legacy npz-format reader (rewrite pending)
│       └── spt_router.py  # legacy npz-format reader (rewrite pending)
├── web/                   # Phase 4 — static SPA + nginx proxy
└── data/                  # populated by ingest + preprocess (gitignored)
    ├── osm/      *.osm.pbf
    ├── pois/     *.osm.pbf, pois.sqlite
    └── spt/      <profile>/(cities.json, city_graph.json, spt/*.npz)
```

## Remote access

This stack is designed to run on `desktop-nk6flc3.tail9115a7.ts.net`
(over Tailscale) and be hit from a laptop. Port bindings use `0.0.0.0`
so they're reachable on any interface.

| Port  | Service              |
|-------|----------------------|
| 5432  | Postgres             |
| 8001  | FastAPI (8000 in container) |
| 8080  | Web UI               |
