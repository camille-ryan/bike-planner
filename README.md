# Bike Routing — Graz → Copenhagen

A self-supported bike-tour planner for the Graz → Copenhagen corridor. OpenStreetMap data + BRouter routing + scenic POI overlays (vistas, lodging, food, protected areas, EuroVelo / national bike networks). Dockerized.

## Architecture

| Service  | Stack                  | Status        |
|----------|------------------------|---------------|
| ingest   | Python + osmium + SpatiaLite | ✅ Phase 1 |
| brouter  | nrenner/brouter (Java) | (Phase 2)     |
| api      | Python + FastAPI       | (Phase 3)     |
| web      | MapLibre + nginx       | (Phase 4)     |

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

## Phase 1 — data ingest pipeline ✅

Downloads OSM PBFs, BRouter `.rd5` tiles, and extracts POIs into a SpatiaLite database.

### Smoke test (~700 MB, a few minutes)

```sh
cd /path/to/bike
docker compose run --rm ingest --test
```

This downloads:
- Austria PBF (~700 MB)
- One BRouter tile covering Graz (`E15_N45.rd5`, ~250 MB)

Then runs `osmium tags-filter` to extract viewpoints / lodging / food POIs and loads them into a SpatiaLite database at `data/pois/pois.sqlite`.

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

Full Austria + Czech Republic + Germany + Denmark + 9 BRouter tiles.

### Verify the SpatiaLite DB by hand

```sh
docker run --rm -v "$(pwd)/data:/data" debian:bookworm-slim bash -c '
  apt-get update -qq && apt-get install -qq -y sqlite3 libsqlite3-mod-spatialite > /dev/null
  sqlite3 /data/pois/pois.sqlite "SELECT load_extension(\"mod_spatialite\"); SELECT category, COUNT(*) FROM pois GROUP BY category;"
'
```

### Re-running

`ingest` is idempotent: downloaded files are cached on disk, so reruns skip what's already there. The POI table is dropped and rebuilt every run (cheap once the PBFs are local).

## Project layout

```
bike/
├── docker-compose.yml
├── .gitignore
├── ingest/
│   ├── Dockerfile
│   ├── config.py             # corridor bbox, country list, POI tag filters
│   ├── download_util.py      # resumable HTTP fetcher
│   ├── download_osm.py       # Geofabrik PBFs
│   ├── download_brouter.py   # BRouter .rd5 segment tiles
│   ├── extract_pois.py       # osmium tags-filter wrapper
│   ├── build_db.py           # PBF → GeoJSON-Seq → SpatiaLite
│   └── main.py               # orchestrator (--test or full)
└── data/                     # populated by ingest (gitignored)
    ├── osm/         *.osm.pbf
    ├── brouter/     *.rd5
    └── pois/        *.osm.pbf, pois.sqlite
```

## Roadmap

- **Phase 2:** BRouter routing engine + custom `lht.brf` profile (quadratic uphill cost, downhill bonus modulated by curvature, 40mm-tire-friendly surface filter).
- **Phase 3:** FastAPI service exposing `/route`, `/pois?bbox=`, `/stages?route_id=` with a Python re-rank pass for scenic scoring.
- **Phase 4:** MapLibre web UI for click-to-route, POI overlays, elevation profile, and stage markers (~100 km legs).

## Remote access

This stack is designed to run on the desktop (`desktop-nk6flc3.tail9115a7.ts.net` over Tailscale) and be hit from a laptop. All Docker port bindings use `0.0.0.0` so they're reachable on any interface; just point the laptop at `http://desktop-nk6flc3.tail9115a7.ts.net:<port>` once the relevant phase is up.
