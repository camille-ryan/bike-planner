#!/usr/bin/env bash
# DEPRECATED — superseded by run_full_rebuild.sh.
# Kept for reference; will be deleted once the new orchestrator has
# been executed successfully at least once. Same gaps as
# run_ferry_anchors_rebuild.sh.
#
# Rerun SPT + adapter + paired-db after adding ferry chain edges
# to polygon inputs. Assumes NPZs already wiped.
set -euo pipefail
LOG=/tmp/claude-1000/-mnt-e-proj-bike/d51847db-70fb-4673-8a38-03aa71184840/scratchpad/ferry_rebuild.log
: > "$LOG"

echo "=== compute_spts_polygon (with is_frontier, ferry polygons) ===" | tee -a "$LOG"
docker compose --profile preprocess run --rm --name ferry_spt \
  -v /mnt/e/proj/bike/pgrouting:/app \
  -e PGDATABASE=bike_v2_test -e SPT_PROFILE=views \
  -e SPT_WORKERS=4 -e SPT_TILE_DEG=1.0 -e SPT_BUFFER_DEG=1.0 \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint python3 pgrouting /app/compute_spts_polygon.py 2>&1 | tee -a "$LOG"

echo "=== adapt_polygon_to_paired ===" | tee -a "$LOG"
docker compose --profile preprocess run --rm --name ferry_adapt \
  -v /mnt/e/proj/bike/pgrouting:/app \
  -e PGDATABASE=bike_v2_test -e SPT_PROFILE=views \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint python3 pgrouting /app/adapt_polygon_to_paired.py 2>&1 | tee -a "$LOG"

echo "=== build_polygon_paired_db_v2 (V1-style with is_frontier) ===" | tee -a "$LOG"
docker compose --profile preprocess run --rm --name ferry_paired \
  -v /mnt/e/proj/bike/pgrouting:/app \
  -e PGDATABASE=bike_v2_test -e SPT_PROFILE=views \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint python3 pgrouting /app/build_polygon_paired_db_v2.py 2>&1 | tee -a "$LOG"

echo "=== restart api ===" | tee -a "$LOG"
docker compose up -d --no-deps api 2>&1 | tee -a "$LOG"
echo "=== DONE ===" | tee -a "$LOG"
