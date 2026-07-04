# Routing DB rebuild pipeline

Rebuild the paired-trunks routing DB from a fresh postgres ingest, in
12 stages. Wall clock: **~7-10 h** on the 4-country (AT/CZ/DE/DK) data.

## Orchestrator

Use `../run_full_rebuild.sh` at the repo root. Resumability is driven
by `pipeline_status.py` — only stages whose outputs are stale (or
missing) actually run.

```bash
# Full rebuild — resumes from whatever's already current
./run_full_rebuild.sh

# Force stages 8 + 9 (paired-db + prune) to rerun
FORCE_STAGES=8,9 ./run_full_rebuild.sh

# Dry-run — inspect what would run without running it
DRY_RUN=1 ./run_full_rebuild.sh

# Ignore all resumability and rerun everything
RESUME=0 ./run_full_rebuild.sh

# Alternate ntfy topic for completion / failure pings
NTFY_TOPIC=my-topic ./run_full_rebuild.sh
```

Env knobs: `PG_DB` (default `bike_v2_test`), `SPT_PROFILE` (default
`views`), `NTFY_TOPIC` (default `bike-rebuild`), `FORCE_STAGES` (comma
list), `DRY_RUN=1`, `RESUME=0`.

Logs land in `data/spt/logs/pipeline_<ts>/`:
- `pipeline.log` — aggregate orchestrator log
- `stage-<n>-<name>.log` — per-stage docker container output

## Stage summary

| # | Name | Script | Runtime | Resume check |
|---|------|--------|---------|--------------|
| 1 | build_paved | `chain/build_ways_paved.py` | ~3 h | `ways_paved` table + GIST index in postgres |
| 2 | anchors | `chain/select_anchors_bottom_up.py` | ~10 s | `data/way_city_anchors.geojson` |
| 3 | chain_land | `chain/connect_anchors_pairs.py` | ~30 min | `data/way_city_graph.json` |
| 4 | chain_ferry | `chain/augment_way_city_graph_with_ferries.py` | ~5 min | `data/way_city_graph.geojson` |
| 5 | anchor_polys | `chain/compute_anchor_spt_polygons.py` | ~30 s | `data/way_city_spt_polygons.geojson` |
| 6 | spt_polygon | `spt/compute_spts_polygon.py` | ~90 min | `data/spt/<profile>_polygon/*.npz` |
| 7 | adapt_paired | `paired/adapt_polygon_to_paired.py` | ~14 min | `data/spt/<profile>/city_graph.json` |
| 8 | build_paired | `paired/build_polygon_paired_db_v2.py` | ~4 h | `data/spt/<profile>/paired_trunks_v2c.db` |
| 9 | prune | `paired/prune_paired_trunks.py` | ~25 min | `data/spt/<profile>/paired_trunks_v2d.db` |
| 10 | symlink | — (native) | instant | `paired_trunks.db` → `paired_trunks_v2d.db` |
| 11 | api_restart | — (native) | ~5 min preload | `bike-api` container running |
| 12 | verify | curl Graz→Cph | ~10 s | worst non-skipped bridge < 100 m |

## Detailed stages

### 1. `build_paved` — denormalized paved subgraph

**Purpose**: Postgres CTAS-like build of `ways_paved` — a denormalized
paved-only subgraph with per-vertex coords inline + GIST index on
`src_pt`. Feeds stage 3's per-anchor Dijkstra (bbox subgraph loads
become O(edges-in-bbox) instead of O(all-edges)).

**Inputs**: postgres tables `ways`, `way_tags`, `ways_vertices_pgr`.
**Outputs**: postgres table `ways_paved` with GIST index `ways_paved_src_pt_idx`.
**Env vars**: `PGDATABASE`, `PYTHONUNBUFFERED`.
**Resume check**: `ways_paved` exists AND has ≥ 1 row AND
`ways_paved_src_pt_idx` exists.
**Known failure modes**: postgres OOM during INSERT (needs
`work_mem=512MB`, `maintenance_work_mem=2GB` — already set in the
script). Multi-hour if `ORDER BY ST_GeoHash` is present (was removed
in task #35 — verify the script's INSERT has no ORDER BY).

### 2. `anchors` — pick chain anchors

**Purpose**: greedy Poisson-disk on OSM settlements + ferry piers
across the 4 countries.
**Inputs**: postgres `anchors` table (populated by ingest), OSM village
geojsonseq files under `data/osm/*-villages.geojsonseq`, ferry piers
geojsonseq at `data/ferry_piers.geojsonseq`.
**Outputs**: `data/way_city_anchors.geojson`.
**Env vars**: `PGDATABASE`.
**Resume check**: output file exists.
**Known failure modes**: missing villages / ferry_piers files silently
skipped (a warning goes to stdout).

### 3. `chain_land` — land↔land chain graph

**Purpose**: per-anchor SSSP on the paved subgraph (via `ways_paved`
GIST bbox), producing directed chain edges A→B for each anchor's
sector-nearest chain neighbors. Task #35's revised scope: land only.
**Inputs**: `data/way_city_anchors.geojson`, postgres `ways_paved`.
**Outputs**: `data/way_city_graph.json` (from_city / to_city / weight
parallel arrays), `data/way_city_graph.geojson` (visualization).
**Env vars**: `PGDATABASE`.
**Resume check**: output file newer than the anchors file.
**Known failure modes**: `ways_paved` missing → immediate fail (rerun
stage 1 first).

### 4. `chain_ferry` — pier↔pier + pier↔land chain edges

**Purpose**: BFS over the ferry-only subgraph (`ways WHERE is_ferry`)
to add pier↔pier chain edges (multi-hop ferries collapsed to a single
chain edge with polyline geom) + KDTree pier↔land nearest-K edges.
Task #34's design.
**Inputs**: postgres `ways` (filtered by `is_ferry`), current
`data/way_city_graph.json`.
**Outputs**: `data/way_city_graph.json` (in-place edit — appends ferry
edges, dedups against existing pairs).
**Env vars**: `PGDATABASE`.
**Resume check**: `data/way_city_graph.geojson` newer than
`way_city_graph.json`.
**Known failure modes**: none commonly seen.

### 5. `anchor_polys` — per-anchor SPT bounding polygons

**Purpose**: per-anchor polygon = 5 km disc + convex hull of 1.5-hop
chain-neighbor geom polylines. Bounds each anchor's polygon-SPT in
stage 6.
**Inputs**: `data/way_city_graph.json`, `data/way_city_anchors.geojson`.
**Outputs**: `data/way_city_spt_polygons.geojson`.
**Env vars**: none.
**Resume check**: output file newer than the chain graph.
**Known failure modes**: none commonly seen.

### 6. `spt_polygon` — polygon-bounded per-anchor SPTs

**Purpose**: tiled multiprocessing Dijkstra from each anchor's seed,
bounded by the anchor's polygon. Writes one NPZ per anchor.
**Inputs**: `data/way_city_spt_polygons.geojson`, cell edge files under
`data/cells/<profile>/`.
**Outputs**: `data/spt/<profile>_polygon/<city_idx>.npz` per anchor
(one NPZ contains `node_global`, `parent`, `is_frontier`,
`coords_lonlat`, `cost`).
**Env vars**: `SPT_WORKERS` (default 4), `SPT_TILE_DEG` (1.0),
`SPT_BUFFER_DEG` (1.0), `SPT_PROFILE`.
**Resume check**: `<profile>_polygon/` npz count matches
`cities.json` anchor count AND newest NPZ mtime > polygons file
mtime. Orchestrator wipes the NPZ dir before running so partial state
never survives.
**Known failure modes**: memory pressure with SPT_WORKERS > 4 (each
worker holds ~1 GB); disk I/O bottleneck on WSL bind mount (~20-30 min
total wait for writes to complete after workers finish).

### 7. `adapt_paired` — adapter to paired-SPT format

**Purpose**: reads polygon NPZs + chain graph, writes `cities.json`
(anchor metadata) and `city_graph.json` (chain edges keyed by
city_idx) in the format the paired-db builder expects. Also snaps
each anchor's polygon-SPT seed to a canonical vid.
**Inputs**: `data/spt/<profile>_polygon/*.npz`,
`data/way_city_graph.json`.
**Outputs**: `data/spt/<profile>/cities.json`,
`data/spt/<profile>/city_graph.json`.
**Env vars**: `SPT_PROFILE`.
**Resume check**: output `city_graph.json` newer than the polygon
NPZ directory.
**Known failure modes**: none commonly seen.

### 8. `build_paired` — polygon → paired trunk sqlite DB

**Purpose**: for every chain edge (A, B), builds A's F-only-ancestors-
of-is_frontier-leaves + b_frontier slice, packs vid+succ+lat+lon per
kept vertex into a sqlite BLOB. Writes `trunk_blobs` and
`trunk_termini` tables.
**Inputs**: `data/spt/<profile>_polygon/*.npz`,
`data/spt/<profile>/cities.json`, `data/spt/<profile>/city_graph.json`.
**Outputs**: `data/spt/<profile>/paired_trunks_v2c.db` (~6 GB).
**Env vars**: `SPT_PROFILE`, `PAIRED_DB_NAME`.
**Resume check**: `paired_trunks_v2c.db` newer than
`city_graph.json`.
**Known failure modes**: multi-hour build (~4 h); no incremental
progress log — trust it.

### 9. `prune` — iterative entry-point pruning

**Purpose**: walk succ chains forward from every chain-neighbor entry
terminus; keep only reached vertices; repeat until termini fixed
point. Task #40 iterative variant.
**Inputs**: `paired_trunks_v2c.db`, `city_graph.json`, `cities.json`.
**Outputs**: `paired_trunks_v2d.db` (~2.5 GB, 41% of v2c).
**Env vars**: `SPT_PROFILE`, `PAIRED_DB_NAME` (source),
`OUT_DB_NAME` (dest), `BORDER_KM` (default 25), `MAX_PASSES` (default 10).
**Resume check**: `paired_trunks_v2d.db` newer than `v2c.db`.
**Known failure modes**: **Loads all 6 GB of blobs into RAM** —
requires the API to be stopped first (WSL VM has 11 GB total, API
preload uses ~5.7 GB, pruner ~6 GB). Orchestrator stops API before
this stage automatically. Converges in ~8 passes, ~25 min total.

### 10. `symlink` — swap deployed DB

**Purpose**: point `paired_trunks.db` → `paired_trunks_v2d.db`.
**Inputs**: `paired_trunks_v2d.db`.
**Outputs**: `data/spt/<profile>/paired_trunks.db` symlink.
**Native** — no docker.
**Resume check**: symlink exists, points at `paired_trunks_v2d.db`,
target file newer than v2c.

### 11. `api_restart` — reload API

**Purpose**: `docker compose up -d --no-deps --force-recreate api`.
Loads new trunks into RAM (~1-5 min preload depending on DB size).
**Resume check**: `bike-api` container running.
**Known failure modes**: OOM during preload if v2c was accidentally
symlinked instead of v2d.

### 12. `verify` — curl Graz→Cph

**Purpose**: sanity route Graz(15.4404,47.0707)→Cph(12.5683,55.6761).
Fail if any non-skipped bridge > 100 m.
**Resume check**: always re-runs (runtime check).
**Known failure modes**: preload not finished when verify starts —
orchestrator polls `/health` up to 10 min.

## Stale artifacts

Files under `data/spt/<profile>/` that don't come from the current
pipeline and can be deleted if disk space is needed:

- `paired_trunks_v2e.db`, `paired_trunks_v2f.db` — deleted 2026-07-03,
  came from failed gateway-seed experiments in `_build_pair`.
- `paired_trunks_v2b.db-shm`, `paired_trunks_v2b.db-wal` — sqlite
  sidecars from a partial build, no matching main DB.
- `paired_trunks_v1.db` — pre-v2 builder output. Kept for UI trunk-viz
  reference; router does not read it.
- `paired_trunks_v2_original.db` — provenance unknown; kept for
  reference.

## Adding or reordering stages

When you add / remove / reorder a stage:

1. Update the `stage <n>` calls in `run_full_rebuild.sh`.
2. Update the stage list in `pipeline_status.py::_stages()`.
3. Update the stage table above.
4. Keep the numbering monotonic — the orchestrator uses `n` as the log
   filename prefix and container name suffix.

## Memory model (WSL 11 GB VM)

The paired-db build and pruner are the biggest single consumers. The
API preload also holds ~5.7 GB. **Never run two heavy jobs
concurrently**:

- API up + pruner running → OOM (7.4 + 6 = ~13 GB)
- Paired-db build + pruner → OOM
- SPT compute at `SPT_WORKERS > 4` → OOM (each worker ~1 GB)

The orchestrator handles the API-stop-before-prune case automatically.
For manual runs of stage 6 or 9, verify with `free -h` before starting.
