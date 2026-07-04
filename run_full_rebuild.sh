#!/usr/bin/env bash
# Rebuild the routing DB end-to-end (12 stages, ~7-10 h wall clock).
#
# ENV KNOBS
#   PG_DB           postgres db  (default bike_v2_test)
#   SPT_PROFILE     cost profile (default views)
#   NTFY_TOPIC      ntfy topic   (default bike-rebuild)
#   FORCE_STAGES    comma-list of stages to run even if their output
#                   is [x] current per pipeline_status.py (e.g. "8,9")
#   DRY_RUN=1       print each stage's docker command without running
#   RESUME=1        (default) skip stages whose output pipeline_status
#                   reports as [x]. Set RESUME=0 to force a full run.
#
# USAGE
#   ./run_full_rebuild.sh                       # resume from where it stopped
#   FORCE_STAGES=8,9 ./run_full_rebuild.sh      # rerun paired-db + prune
#   DRY_RUN=1 ./run_full_rebuild.sh             # inspect without running
#
# LOGS
#   data/spt/logs/pipeline_<ts>/pipeline.log       aggregate
#   data/spt/logs/pipeline_<ts>/stage-<n>-<name>.log per-stage
#
# NOTES
# - Stage 9 (pruner) needs the API stopped so ~7 GB RAM is free.
#   Orchestrator stops it automatically; stage 11 restarts.
# - Failures halt immediately: log FAIL, ntfy, exit 1. The failing
#   container is NOT removed (docker run --rm is dropped for the failing
#   stage, so `docker logs <name>` can be used for postmortem).
# - Resumability is driven by pipeline_status.py — see that script's
#   stage list to add/reorder.
set -euo pipefail

: "${PG_DB:=bike_v2_test}"
: "${SPT_PROFILE:=views}"
: "${NTFY_TOPIC:=bike-rebuild}"
: "${FORCE_STAGES:=}"
: "${DRY_RUN:=0}"
: "${RESUME:=1}"

REPO_ROOT="/mnt/e/proj/bike"
PIPELINE_TS="$(date +%Y%m%d-%H%M%S)"
DATA="$REPO_ROOT/data"
LOG_DIR="$DATA/spt/logs/pipeline_${PIPELINE_TS}"
mkdir -p "$LOG_DIR"
AGG_LOG="$LOG_DIR/pipeline.log"
TOTAL=12

trap 'ntfy_send "bike-rebuild INTERRUPTED"' INT TERM

log()       { echo "=== [$(date -Iseconds)] $*" | tee -a "$AGG_LOG"; }
ntfy_send() { curl -fsS -m 5 -d "$*" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true; }

# is_current N — exit 0 iff stage N's output is current (skip).
# Invoked inside the pgrouting container so psycopg + postgres:5432 work.
is_current() {
  local n="$1"
  [[ "$RESUME" != "1" ]] && return 1
  [[ ",${FORCE_STAGES}," == *",${n},"* ]] && return 1
  docker compose --profile preprocess run --rm --no-deps \
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

  docker compose --profile preprocess run -d --rm --name "$cname" \
    -v "$REPO_ROOT/pgrouting:/app" \
    -e PGDATABASE="$PG_DB" -e SPT_PROFILE="$SPT_PROFILE" \
    -e PYTHONUNBUFFERED=1 \
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
}

# ---- 12 stages -------------------------------------------------------

log "== rebuild START (ts=$PIPELINE_TS, profile=$SPT_PROFILE, db=$PG_DB) =="

stage 1 build_paved  /app/chain/build_ways_paved.py
stage 2 anchors      /app/chain/select_anchors_bottom_up.py
stage 3 chain_land   /app/chain/connect_anchors_pairs.py
stage 4 chain_ferry  /app/chain/augment_way_city_graph_with_ferries.py
stage 5 anchor_polys /app/chain/compute_anchor_spt_polygons.py

# Stage 6 needs the polygon-SPT NPZ dir clean before recompute.
# Only wipe when we're actually going to run stage 6.
if ! is_current 6; then
  log "wiping NPZs before stage 6"
  if [[ "$DRY_RUN" = "1" ]]; then
    echo "DRY_RUN: find $DATA/spt/${SPT_PROFILE}_polygon -name '*.npz' -delete"
  else
    find "$DATA/spt/${SPT_PROFILE}_polygon" -name '*.npz' -delete 2>/dev/null || true
  fi
fi
stage 6 spt_polygon /app/spt/compute_spts_polygon.py \
  -e SPT_WORKERS=4 -e SPT_TILE_DEG=1.0 -e SPT_BUFFER_DEG=1.0

stage 7 adapt_paired /app/paired/adapt_polygon_to_paired.py
stage 8 build_paired /app/paired/build_polygon_paired_db_v2.py \
  -e PAIRED_DB_NAME=paired_trunks_v2c.db

# Stage 9 (pruner) loads ~6 GB of blobs into RAM; the API preload holds
# ~5.7 GB. Together they OOM the 11 GB WSL VM. Stop API before, restart
# in stage 11 after the pruner has released its RAM.
if ! is_current 9; then
  log "stopping API before pruner"
  [[ "$DRY_RUN" = "1" ]] || docker compose stop api
fi
stage 9 prune /app/paired/prune_paired_trunks.py \
  -e PAIRED_DB_NAME=paired_trunks_v2c.db \
  -e OUT_DB_NAME=paired_trunks_v2d.db

# ---- Native stages (no docker container) -----------------------------

log "STAGE 10/$TOTAL symlink: paired_trunks.db -> paired_trunks_v2d.db"
if is_current 10; then
  log "STAGE 10 symlink: SKIP (already pointing at v2d)"
else
  if [[ "$DRY_RUN" = "1" ]]; then
    echo "DRY_RUN: ln -sfn paired_trunks_v2d.db $DATA/spt/$SPT_PROFILE/paired_trunks.db"
  else
    ln -sfn paired_trunks_v2d.db "$DATA/spt/$SPT_PROFILE/paired_trunks.db"
    log "STAGE 10 symlink: OK"
  fi
fi

log "STAGE 11/$TOTAL api_restart"
if [[ "$DRY_RUN" = "1" ]]; then
  echo "DRY_RUN: docker compose up -d --no-deps --force-recreate api"
else
  docker compose up -d --no-deps --force-recreate api >/dev/null
  log "STAGE 11 api_restart: OK (preload takes ~1-5 min)"
fi

log "STAGE 12/$TOTAL verify: waiting for API preload…"
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
  route_json="$(curl -fsS -G \
    -d from=15.4404,47.0707 -d to=12.5683,55.6761 -d profile="$SPT_PROFILE" \
    http://localhost:8001/trunk/route 2>>"$LOG_DIR/stage-12-verify.log")"
  echo "$route_json" >>"$LOG_DIR/stage-12-verify.log"
  worst=$(python3 -c "
import json, sys
d = json.loads('''$route_json''')
bs = [b for b in d.get('route',{}).get('properties',{}).get('bridges',[])
      if not b.get('skipped')]
worst = max((b.get('distance_m') or 0) for b in bs) if bs else 0
print(f'{worst:.1f}')
" 2>>"$LOG_DIR/stage-12-verify.log")
  if [[ -z "$worst" ]]; then
    log "STAGE 12 verify: FAIL — could not parse route response"
    ntfy_send "bike-rebuild verify FAILED — could not parse route response"
    exit 1
  fi
  worst_int="${worst%.*}"
  if (( worst_int > 100 )); then
    log "STAGE 12 verify: FAIL — worst non-skipped bridge = ${worst} m > 100 m limit"
    ntfy_send "bike-rebuild verify FAILED — worst bridge ${worst} m"
    exit 1
  fi
  log "STAGE 12 verify: OK (worst non-skipped bridge ${worst} m)"
fi

log "== rebuild COMPLETE (ts=$PIPELINE_TS) =="
ntfy_send "bike-rebuild complete (profile=$SPT_PROFILE)"
