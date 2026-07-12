#!/usr/bin/env bash
# Rebuild the routing DB end-to-end (14 stages, ~7-10 h wall clock).
#
# PIPELINE STAGES
#   1  build_paved         Materializes ways_paved from postgres.ways.
#   2  classify_piers      Sea vs river pier classification.
#   3  anchors             Bottom-up anchor selection (cities + villages
#                          + sea piers).
#   4  chain_land          Crow-flies chain graph via 60° sectors with
#                          30° overlap.
#   5  chain_ferry         Ferry pier chain edges appended.
#   6  chain_proximity     Multi-source Dijkstra reachability filter +
#                          proximity-based triangle removal. Replaces
#                          the old bidir_reach + dedup_chain stages.
#                          For each candidate (A, B): if the shortest
#                          road path from a 5 km disc around A to a
#                          5 km disc around B passes within 5 km of a
#                          third anchor C, drop (A, B) and enqueue
#                          (A, C) and (C, B). Emits real-road geom
#                          alongside cost.
#   7  anchor_polys        Anchor SPT polygons (5 km disc + neighbor hulls).
#   8  spt_polygon         Per-anchor bounded Dijkstra on cell npz files.
#   9  adapt_paired        Translate chain graph → per-directed-edge
#                          metadata using SPT costs.
#   10 build_paired        Materialize per-edge trunk blobs into v2c.db.
#                          INCREMENTAL — reuses existing v2c.db and
#                          adds only new (src, dst) pairs. Set
#                          BUILD_PAIRED_FRESH=1 to force a clean rebuild.
#   11 prune               v2c.db → v2d.db (iterative entry-point prune).
#   12 symlink             paired_trunks.db → paired_trunks_v2d.db.
#   13 api_restart         Restart bike-api container (preloads new DB).
#   14 verify              curl Graz→Copenhagen route; fail if any
#                          non-skipped bridge > 100 m.
#
# ENV KNOBS
#   PG_DB           postgres db   (default bike_v2_test)
#   SPT_PROFILE     cost profile  (default views)
#   NTFY_TOPIC      ntfy topic    (default SMJVoZsEr7s6TKGb — user's)
#   FORCE_STAGES    comma-list to run even if output looks current
#                   e.g. "9,10" — usually not needed unless
#                   is_current gets confused
#   SKIP_STAGES     comma-list to unconditionally skip. Useful for a
#                   partial rerun (e.g. SKIP_STAGES=1,2,3 to reuse
#                   ingest state).
#   DRY_RUN=1       print each stage's docker command without running
#   RESUME=1        (default) skip stages whose output pipeline_status
#                   reports as [x]. Set RESUME=0 to force a full run.
#
# USAGE
#   ./run_full_rebuild.sh                          # start-to-finish
#   FORCE_STAGES=9,10 ./run_full_rebuild.sh        # rerun adapt + build
#   BUILD_PAIRED_FRESH=1 ./run_full_rebuild.sh     # nuke v2c.db first
#   DRY_RUN=1 ./run_full_rebuild.sh                # inspect without running
#
# LOGS
#   data/spt/logs/pipeline_<ts>/pipeline.log       aggregate
#   data/spt/logs/pipeline_<ts>/stage-<n>-<name>.log per-stage
#
# NOTES
# - Stage 11 (pruner) loads ~6 GB of blobs into RAM; API's ~5.7 GB
#   preload would OOM together on the 11 GB WSL VM. Orchestrator stops
#   API before stage 11 and stage 13 restarts it.
# - Failures halt immediately: log FAIL, ntfy, exit 1. The failing
#   container is NOT removed (docker run --rm is dropped for the failing
#   stage, so `docker logs <name>` can be used for postmortem).
# - Resumability is driven by pipeline_status.py — see that script's
#   stage list to add/reorder.
set -euo pipefail

: "${PG_DB:=bike_v2_test}"
: "${SPT_PROFILE:=views}"
: "${NTFY_TOPIC:=SMJVoZsEr7s6TKGb}"
: "${FORCE_STAGES:=}"
: "${SKIP_STAGES:=}"          # comma-list of stages to always skip
: "${DRY_RUN:=0}"
: "${RESUME:=1}"

REPO_ROOT="/mnt/e/proj/bike"
PIPELINE_TS="$(date +%Y%m%d-%H%M%S)"
DATA="$REPO_ROOT/data"
LOG_DIR="$DATA/spt/logs/pipeline_${PIPELINE_TS}"
mkdir -p "$LOG_DIR"
AGG_LOG="$LOG_DIR/pipeline.log"
TOTAL=14

trap 'ntfy_send "bike-rebuild INTERRUPTED"' INT TERM

log()       { echo "=== [$(date -Iseconds)] $*" | tee -a "$AGG_LOG"; }
ntfy_send() { curl -fsS -m 5 -d "$*" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true; }

# The web container bind-mounts $REPO_ROOT/web/public — NOT $REPO_ROOT/data —
# so viz files (anchors, chain graph, SPT polygons) need to be copied
# from the live pipeline output into the web tree. This helper syncs
# them; called at pipeline end and any time a stage rewrites one of
# these files.
publish_web_data() {
  local WEB="$REPO_ROOT/web/public/data"
  local FILES=(
    way_city_anchors.geojson
    way_city_graph.geojson
    way_city_graph.json
    way_city_spt_polygons.geojson
  )
  for f in "${FILES[@]}"; do
    if [[ -f "$DATA/$f" ]]; then
      cp -u "$DATA/$f" "$WEB/$f" 2>/dev/null || true
    fi
  done
}

# is_current N — exit 0 iff stage N's output is current (skip).
# Precedence: SKIP_STAGES (always skip) > FORCE_STAGES (always run) >
# pipeline_status --check-current (inspects data).
# The docker-based check needs -v /app so pipeline_status.py is present
# and -e PGDATABASE so it hits the right DB.
is_current() {
  local n="$1"
  [[ ",${SKIP_STAGES}," == *",${n},"* ]] && return 0
  [[ "$RESUME" != "1" ]] && return 1
  [[ ",${FORCE_STAGES}," == *",${n},"* ]] && return 1
  docker compose --profile preprocess run --rm --no-deps \
    -v "$REPO_ROOT/pgrouting:/app" \
    -e PGDATABASE="$PG_DB" -e SPT_PROFILE="$SPT_PROFILE" \
    --entrypoint python3 pgrouting /app/pipeline_status.py \
    --stage "$n" --check-current >/dev/null 2>&1
}

# stage N NAME SCRIPT [-e VAR=VAL ...]
# SCRIPT is the path inside the container (e.g. /app/chain/build_ways_paved.py).
# Docker compose expects: run [OPTIONS] SERVICE [COMMAND] [ARGS], so we
# put the entrypoint + env vars as options, then `pgrouting` service, then
# script as the command argument.
stage() {
  local n="$1" name="$2" script="$3"; shift 3
  local cname="rebuild_s${n}_${name}"
  local logfile="$LOG_DIR/stage-${n}-${name}.log"

  if is_current "$n"; then
    log "STAGE $n/$TOTAL $name: SKIP (output current)"
    return 0
  fi

  log "STAGE $n/$TOTAL $name: START -> $cname"
  if [[ "$DRY_RUN" = "1" ]]; then
    echo "DRY_RUN: docker compose --profile preprocess run -d --rm --name $cname"
    echo "         -v $REPO_ROOT/pgrouting:/app"
    echo "         -e PGDATABASE=$PG_DB -e SPT_PROFILE=$SPT_PROFILE -e PYTHONUNBUFFERED=1"
    echo "         $* --entrypoint python3 pgrouting $script"
    return 0
  fi

  # Forward any BUILD_PAIRED_FRESH / BIDIR_* / INTERMED_* / other tool-
  # specific env vars into the container. Bare `-e VAR` (no `=VAL`)
  # forwards the current shell's value, so ./run_full_rebuild.sh
  # BUILD_PAIRED_FRESH=1 … actually reaches build_paired.
  docker compose --profile preprocess run -d --rm --name "$cname" \
    -v "$REPO_ROOT/pgrouting:/app" \
    -e PGDATABASE="$PG_DB" -e SPT_PROFILE="$SPT_PROFILE" \
    -e PYTHONUNBUFFERED=1 \
    -e BUILD_PAIRED_FRESH \
    -e BIDIR_COST_CAP_MULT -e BIDIR_BUFFER_FRAC \
    -e BIDIR_MAX_BBOX_EDGES -e BIDIR_CELL_CACHE_SIZE \
    -e INTERMED_ANCHOR_RADIUS_M -e INTERMED_COST_CAP_MULT \
    -e INTERMED_CELL_CACHE_SIZE -e INTERMED_MAX_BBOX_EDGES \
    -e INTERMED_BUFFER_FRAC -e INTERMED_MAX_PATH_LEN \
    -e INTERMED_MAX_ROUNDS \
    -e ADAPT_MAX_TRUNK_KM \
    -e CROW_SECTOR_WIDTH_DEG -e CROW_SECTOR_STRIDE_DEG \
    "$@" \
    --entrypoint python3 \
    pgrouting "$script" >/dev/null
  docker logs -f "$cname" >>"$logfile" 2>&1 &
  local logs_pid=$!
  local rc; rc=$(docker wait "$cname")
  wait "$logs_pid" 2>/dev/null || true
  if [[ "$rc" != "0" ]]; then
    log "STAGE $n $name: FAIL — exit=$rc (log: $logfile)"
    ntfy_send "bike-rebuild FAILED at stage $n ($name): exit=$rc"
    exit 1
  fi
  log "STAGE $n/$TOTAL $name: OK"
  # Sync any newly-produced viz files into web/public/data as soon as
  # the stage that writes them completes. Keeps the dashboard fresh
  # mid-run, and guarantees files are published even if a later stage
  # fails before the exit trap fires.
  publish_web_data
}

# EXIT trap: publish one last time no matter how the script exits.
# Prevents stale-web-data bug where a mid-pipeline failure (e.g. verify
# crash, ctrl-C) skipped the end-of-run publish and left the dashboard
# showing data from the run before last.
trap 'publish_web_data' EXIT

# ---- 14 stages -------------------------------------------------------

log "== rebuild START (ts=$PIPELINE_TS, profile=$SPT_PROFILE, db=$PG_DB) =="
# Bring web/public/data up to date on entry so anything the previous
# run produced is visible even if the current run doesn't touch that
# stage (e.g. SKIP_STAGES=1..7).
publish_web_data

stage 1 build_paved    /app/chain/build_ways_paved.py
stage 2 classify_piers /app/chain/classify_piers.py
stage 3 anchors        /app/chain/select_anchors_bottom_up.py
stage 4 chain_land     /app/chain/crow_flies_chain_graph.py
stage 5 chain_ferry    /app/chain/augment_way_city_graph_with_ferries.py
# Multi-source Dijkstra reachability + proximity-based triangle removal
# (replaces bidir_reach_filter.py + dedup_chain_triangles.py). For
# each candidate (A, B) it multi-source-Dijkstras a 5 km disc around A
# to a 5 km disc around B, traces the shortest path via predecessors,
# and checks whether that path passes within 5 km of a third anchor C.
# If it does, drops (A, B) and enqueues (A, C) + (C, B). Emits real-
# road geom alongside cost. Pier↔pier ferry edges preserved
# unconditionally.
stage 6 chain_proximity /app/chain/build_chain_graph_proximity.py

stage 7 anchor_polys /app/chain/compute_anchor_spt_polygons.py

if ! is_current 8; then
  log "wiping NPZs before stage 8 (SPT compute)"
  if [[ "$DRY_RUN" = "1" ]]; then
    echo "DRY_RUN: find $DATA/spt/${SPT_PROFILE}_polygon -name '*.npz' -delete"
  else
    find "$DATA/spt/${SPT_PROFILE}_polygon" -name '*.npz' -delete 2>/dev/null || true
  fi
fi
stage 8 spt_polygon /app/spt/compute_spts_polygon.py \
  -e SPT_WORKERS=4 -e SPT_TILE_DEG=1.0 -e SPT_BUFFER_DEG=1.0

stage 9 adapt_paired /app/paired/adapt_polygon_to_paired.py
stage 10 build_paired /app/paired/build_polygon_paired_db_v2.py \
  -e PAIRED_DB_NAME=paired_trunks_v2c.db

# Stage 11 (pruner) loads ~6 GB of blobs into RAM; the API preload holds
# ~5.7 GB. Together they OOM the 11 GB WSL VM. Stop API before, restart
# in stage 13 after the pruner has released its RAM.
if ! is_current 11; then
  log "stopping API before pruner"
  [[ "$DRY_RUN" = "1" ]] || docker compose stop api
fi
stage 11 prune /app/paired/prune_paired_trunks.py \
  -e PAIRED_DB_NAME=paired_trunks_v2c.db \
  -e OUT_DB_NAME=paired_trunks_v2d.db

# ---- Native stages (no docker container) -----------------------------

log "STAGE 12/$TOTAL symlink: paired_trunks.db -> paired_trunks_v2d.db"
if is_current 12; then
  log "STAGE 12 symlink: SKIP (already pointing at v2d)"
else
  if [[ "$DRY_RUN" = "1" ]]; then
    echo "DRY_RUN: ln -sfn paired_trunks_v2d.db $DATA/spt/$SPT_PROFILE/paired_trunks.db"
  else
    ln -sfn paired_trunks_v2d.db "$DATA/spt/$SPT_PROFILE/paired_trunks.db"
    log "STAGE 12 symlink: OK"
  fi
fi

log "STAGE 13/$TOTAL api_restart"
if [[ "$DRY_RUN" = "1" ]]; then
  echo "DRY_RUN: docker compose up -d --no-deps --force-recreate api"
else
  docker compose up -d --no-deps --force-recreate api >/dev/null
  log "STAGE 13 api_restart: OK (preload takes ~1-5 min)"
fi

log "STAGE 14/$TOTAL verify: waiting for API preload…"
if [[ "$DRY_RUN" = "1" ]]; then
  echo "DRY_RUN: curl http://localhost:8001/trunk/route Graz->Cph"
else
  # Poll /health until it returns 200 (up to 10 min).
  for i in $(seq 1 60); do
    if curl -fsS -m 3 "http://localhost:8001/health" >/dev/null 2>&1; then
      break
    fi
    sleep 10
  done
  # Test Graz→Cph route; fail if any non-skipped bridge is > 100 m.
  # Route JSON polylines can be several hundred KB — pass via stdin
  # rather than argv (`python3 -c "…$route_json…"` blows past
  # ARG_MAX with a real graph and dies "Argument list too long").
  route_file="$LOG_DIR/stage-14-verify-route.json"
  curl -fsS -G \
    -d from=15.4404,47.0707 -d to=12.5683,55.6761 -d profile="$SPT_PROFILE" \
    http://localhost:8001/trunk/route \
    2>>"$LOG_DIR/stage-14-verify.log" > "$route_file"
  worst=$(ROUTE_FILE="$route_file" python3 - <<'PYEOF' 2>>"$LOG_DIR/stage-14-verify.log"
import json, os
d = json.load(open(os.environ["ROUTE_FILE"]))
bs = [b for b in d.get('route',{}).get('properties',{}).get('bridges',[])
      if not b.get('skipped')]
worst = max((b.get('distance_m') or 0) for b in bs) if bs else 0
print(f'{worst:.1f}')
PYEOF
)
  if [[ -z "$worst" ]]; then
    log "STAGE 14 verify: FAIL — could not parse route response"
    ntfy_send "bike-rebuild verify FAILED — could not parse route response"
    exit 1
  fi
  worst_int="${worst%.*}"
  if (( worst_int > 100 )); then
    log "STAGE 14 verify: FAIL — worst non-skipped bridge = ${worst} m > 100 m limit"
    ntfy_send "bike-rebuild verify FAILED — worst bridge ${worst} m"
    exit 1
  fi
  log "STAGE 14 verify: OK (worst non-skipped bridge ${worst} m)"
fi

# publish_web_data is called by the EXIT trap.
log "== rebuild COMPLETE (ts=$PIPELINE_TS) =="
ntfy_send "bike-rebuild complete (profile=$SPT_PROFILE)"
