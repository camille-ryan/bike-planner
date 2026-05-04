# Bike Routing — Graz → Copenhagen

A self-supported bike-tour planner for the Graz → Copenhagen corridor. OpenStreetMap data + BRouter routing + scenic POI overlays (vistas, lodging, food, protected areas, EuroVelo / national bike networks). Dockerized.

## Architecture

| Service  | Stack                          | Status        |
|----------|--------------------------------|---------------|
| ingest   | Python + osmium + SpatiaLite   | ✅ Phase 1    |
| brouter  | OpenJDK + abrensch/brouter 1.7.9 | ✅ Phase 2  |
| api      | Python + FastAPI + httpx + SpatiaLite | ✅ Phase 3 |
| web      | MapLibre + nginx               | (Phase 4)     |

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

### Compare profiles

The Dockerfile bundles BRouter's standard profiles too — `trekking`, `trekking-noferries`, `gravel`, `fastbike-lowtraffic`. Swap `profile=lht` for any of those to see how routing changes.

```sh
for p in lht trekking gravel fastbike-lowtraffic; do
  echo "=== $p ==="
  curl -s "http://localhost:17777/brouter?lonlats=15.43,47.07|16.37,48.21&profile=$p&format=geojson" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);
p=d['features'][0]['properties'];
print(f\"  distance: {p['track-length']} m, duration: {p['total-time']} s, climb: {p['filtered-ascend']} m\")"
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
curl http://localhost:8000/health
# {"status":"ok"}
```

#### `GET /route?from=lon,lat&to=lon,lat&profile=lht&alternatives=N&rerank=true`

Proxies to BRouter and (optionally) re-ranks alternatives by composite score:
`raw_cost + 5·curvy_descent_penalty − 50·viewpoints_near_route`.

```sh
# Single route Graz → Vienna
curl 'http://localhost:8000/route?from=15.43,47.07&to=16.37,48.21&profile=lht'

# Three alternatives, re-ranked for scenicness + curvy-descent avoidance
curl 'http://localhost:8000/route?from=15.43,47.07&to=16.37,48.21&profile=lht&alternatives=3&rerank=true' \
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
curl 'http://localhost:8000/pois?bbox=15.3,47.0,15.6,47.2&category=viewpoint&limit=200'
```

Categories: `viewpoint`, `lodging`, `food`, `bike_service`, `water`.

#### `GET /stages?from=lon,lat&to=lon,lat&target_km=100&lodging_radius_m=3000`

Splits a route into legs of about `target_km` each and returns nearby lodging POIs at every leg-end. Useful for daily-stage planning on a multi-day tour.

```sh
# Graz → Copenhagen, 100km daily stages
curl 'http://localhost:8000/stages?from=15.43,47.07&to=12.57,55.68&target_km=100' \
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

## Roadmap

- **Phase 4:** MapLibre web UI for click-to-route, POI overlays, elevation profile, and stage markers.

## Remote access

This stack is designed to run on `desktop-nk6flc3.tail9115a7.ts.net` (over Tailscale) and be hit from a laptop. All Docker port bindings use `0.0.0.0` so they're reachable on any interface; just point the laptop at `http://desktop-nk6flc3.tail9115a7.ts.net:<port>` once the relevant phase is up:

| Port  | Service       |
|-------|---------------|
| 17777 | BRouter       |
| 8000  | FastAPI (Phase 3) |
| 8080  | Web UI (Phase 4)  |
