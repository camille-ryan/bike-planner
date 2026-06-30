#!/usr/bin/env bash
# Overnight pipeline (per user instruction 2026-05-21):
#   1. Build ways_bike denormalized table (bike network + cost + GIST)
#   2. Build per-anchor SPT polygons (chain-graph derived)
#   3. Compute polygon-bounded SPTs for the direct profile
#   --- STOP HERE if anything above fails ---
#   4. Download missing DEM tiles (4-country bbox)
#   5. Ingest DEM into ways_vertices_pgr.elev_m
#   --- STOP. User must inspect direct SPTs before scenicness bake +
#       scenic-profile SPT compute. ---

set -euo pipefail

cd /mnt/e/proj/bike
LOG=/mnt/e/proj/bike/data/logs/pipeline.log
PG_MOUNT="-v /mnt/e/proj/bike/pgrouting:/app"
PYRUN="docker compose --profile preprocess run --rm $PG_MOUNT --entrypoint python3 pgrouting"

echo "=== [$(date)] STEP 1/5 build ways_bike ===" | tee -a "$LOG"
$PYRUN /app/build_ways_bike.py 2>&1 | tee -a "$LOG"

echo "=== [$(date)] STEP 2/5 compute anchor SPT polygons ===" | tee -a "$LOG"
$PYRUN /app/compute_anchor_spt_polygons.py 2>&1 | tee -a "$LOG"

echo "=== [$(date)] STEP 3/5 compute polygon-bounded SPTs (direct profile) ===" | tee -a "$LOG"
$PYRUN /app/compute_spts_polygon.py 2>&1 | tee -a "$LOG"

echo "=== [$(date)] STEP 3/5 DONE — direct SPTs at data/spt/direct_polygon/ ===" | tee -a "$LOG"

echo "=== [$(date)] STEP 4/5 download missing DEM tiles (4 countries) ===" | tee -a "$LOG"
docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
  dem-download --countries austria,czech-republic,germany,denmark 2>&1 | tee -a "$LOG"

echo "=== [$(date)] STEP 5/5 ingest DEM into ways_vertices_pgr.elev_m ===" | tee -a "$LOG"
docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
  dem-ingest 2>&1 | tee -a "$LOG"

echo "=== [$(date)] ALL DONE. Direct SPTs ready for review. ===" | tee -a "$LOG"
echo "=== Scenicness bake + scenic SPTs gated on user inspection. ===" | tee -a "$LOG"
