#!/usr/bin/env bash
# Scenic feature prep (gated on user review of direct SPTs).
#   1. Download missing DEM tiles for the 4-country bbox
#   2. Ingest DEM into ways_vertices_pgr.elev_m
#   3. Bake scenicness signals over the full corridor
# STOPS before any scenic-profile SPT compute (per user instruction).

set -euo pipefail

cd /mnt/e/proj/bike
LOG=/mnt/e/proj/bike/data/logs/pipeline.log
PG_MOUNT="-v /mnt/e/proj/bike/pgrouting:/app"

echo "=== [$(date)] SCENIC 1/3 download DEM tiles for 4 countries ===" | tee -a "$LOG"
docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
  dem-download --countries austria,czech-republic,germany,denmark 2>&1 | tee -a "$LOG"

echo "=== [$(date)] SCENIC 2/3 ingest DEM into ways_vertices_pgr.elev_m ===" | tee -a "$LOG"
docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
  dem-ingest 2>&1 | tee -a "$LOG"

echo "=== [$(date)] SCENIC 3/3 bake scenicness signals ===" | tee -a "$LOG"
docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
  scenicness-bake 2>&1 | tee -a "$LOG"

echo "=== [$(date)] SCENIC PREP DONE — gating on user review of direct SPTs ===" | tee -a "$LOG"
