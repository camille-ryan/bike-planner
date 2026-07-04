#!/usr/bin/env bash
# DEPRECATED — superseded by run_full_rebuild.sh.
# Kept for reference; scheduled for deletion once run_full_rebuild.sh
# has been executed successfully end-to-end at least once.
# Notable gaps vs the new orchestrator: no iterative-pruner step,
# no skip-lookahead-aware verify, no per-stage logs, no ntfy.
#
# Ferry-terminals-as-anchors full rebuild.
#
# Pipeline (see /home/ryan/.claude/plans/splendid-swinging-pizza.md):
#   1. select_anchors_bottom_up.py    — anchors incl. 411 protected ferry piers
#   2. build_way_graph.py             — chain graph using ferry+seed subgraph
#   3. compute_anchor_spt_polygons.py — single-ring 1.5-hop convex hulls
#   4. wipe SPT NPZs
#   5. compute_spts_polygon.py        — polygon-bounded SPTs + is_frontier
#   6. adapt_polygon_to_paired.py     — cities.json + city_graph.json
#   7. build_polygon_paired_db_v2.py  — V1-style paired trunks DB
#   8. rebuild + restart api
set -euo pipefail
LOG=/tmp/claude-1000/-mnt-e-proj-bike/d51847db-70fb-4673-8a38-03aa71184840/scratchpad/ferry_anchors_rebuild.log
: > "$LOG"

run() {
  local name="$1"; shift
  echo "=== $name ===" | tee -a "$LOG"
  docker compose --profile preprocess run --rm --name "$name" \
    -v /mnt/e/proj/bike/pgrouting:/app \
    -e PGDATABASE=bike_v2_test -e SPT_PROFILE=views \
    -e PYTHONUNBUFFERED=1 \
    "$@" \
    --entrypoint python3 pgrouting "/app/$1" 2>&1 | tee -a "$LOG"
}

echo "=== select_anchors_bottom_up ===" | tee -a "$LOG"
docker compose --profile preprocess run --rm --name fa_select \
  -v /mnt/e/proj/bike/pgrouting:/app \
  -e PGDATABASE=bike_v2_test -e PYTHONUNBUFFERED=1 \
  --entrypoint python3 pgrouting /app/select_anchors_bottom_up.py 2>&1 | tee -a "$LOG"

echo "=== augment_way_city_graph_with_ferries ===" | tee -a "$LOG"
# Fast path: reuse existing land chain graph + add ferry & pier↔land
# edges via partial-index query on is_ferry (~5 min). Avoids the
# 127M-row JOIN in build_way_graph.py which chokes on 4-country DB
# without a rebuilt ways_paved denormalization.
docker compose --profile preprocess run --rm --name fa_augment \
  -v /mnt/e/proj/bike/pgrouting:/app \
  -e PGDATABASE=bike_v2_test -e PYTHONUNBUFFERED=1 \
  --entrypoint python3 pgrouting /app/augment_way_city_graph_with_ferries.py 2>&1 | tee -a "$LOG"

echo "=== compute_anchor_spt_polygons ===" | tee -a "$LOG"
docker compose --profile preprocess run --rm --name fa_poly \
  -v /mnt/e/proj/bike/pgrouting:/app \
  -e PGDATABASE=bike_v2_test -e PYTHONUNBUFFERED=1 \
  --entrypoint python3 pgrouting /app/compute_anchor_spt_polygons.py 2>&1 | tee -a "$LOG"

echo "=== wiping SPT NPZs ===" | tee -a "$LOG"
rm -rf /mnt/e/proj/bike/data/spt/views_polygon/*.npz
echo "  wiped $(ls /mnt/e/proj/bike/data/spt/views_polygon/*.npz 2>/dev/null | wc -l) remaining" | tee -a "$LOG"

echo "=== compute_spts_polygon ===" | tee -a "$LOG"
docker compose --profile preprocess run --rm --name fa_spt \
  -v /mnt/e/proj/bike/pgrouting:/app \
  -e PGDATABASE=bike_v2_test -e SPT_PROFILE=views \
  -e SPT_WORKERS=4 -e SPT_TILE_DEG=1.0 -e SPT_BUFFER_DEG=1.0 \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint python3 pgrouting /app/compute_spts_polygon.py 2>&1 | tee -a "$LOG"

echo "=== adapt_polygon_to_paired ===" | tee -a "$LOG"
docker compose --profile preprocess run --rm --name fa_adapt \
  -v /mnt/e/proj/bike/pgrouting:/app \
  -e PGDATABASE=bike_v2_test -e SPT_PROFILE=views \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint python3 pgrouting /app/adapt_polygon_to_paired.py 2>&1 | tee -a "$LOG"

echo "=== build_polygon_paired_db_v2 ===" | tee -a "$LOG"
docker compose --profile preprocess run --rm --name fa_paired \
  -v /mnt/e/proj/bike/pgrouting:/app \
  -e PGDATABASE=bike_v2_test -e SPT_PROFILE=views \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint python3 pgrouting /app/build_polygon_paired_db_v2.py 2>&1 | tee -a "$LOG"

echo "=== rebuild + restart api ===" | tee -a "$LOG"
docker compose build api 2>&1 | tee -a "$LOG"
docker compose up -d --no-deps api 2>&1 | tee -a "$LOG"
echo "=== DONE ===" | tee -a "$LOG"
