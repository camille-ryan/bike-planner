#!/usr/bin/env bash
# Orchestrate the rest of the 4-country way-graph build:
#   1. Incremental ingest of all 4 PBFs (adds drivable-only ways)
#   2. Bottom-up anchor selection at 10 km
#   3. Chain graph build via pair-Dijkstra
#   4. Publish to web/public/data
#
# Assumes way_tags + village geojsonseq files are already extracted.

set -euo pipefail

cd /mnt/e/proj/bike
LOG=/mnt/e/proj/bike/data/logs/pipeline.log

PG_MOUNT="-v /mnt/e/proj/bike/pgrouting:/app"

echo "=== [$(date)] 1/4 incremental ingest (4 countries, adds drivable-only) ===" | tee -a "$LOG"
docker compose --profile preprocess run --rm \
  $PG_MOUNT pgrouting \
  ingest --countries austria,czech-republic,germany,denmark \
  2>&1 | tee -a "$LOG"

echo "=== [$(date)] 2/4 bottom-up anchor selection (10 km, 4 countries) ===" | tee -a "$LOG"
docker compose --profile preprocess run --rm \
  $PG_MOUNT --entrypoint python3 pgrouting \
  /app/select_anchors_bottom_up.py \
  2>&1 | tee -a "$LOG"

echo "=== [$(date)] 3/4 chain graph build (pair-Dijkstra) ===" | tee -a "$LOG"
docker compose --profile preprocess run --rm \
  $PG_MOUNT --entrypoint python3 pgrouting \
  /app/connect_anchors_pairs.py \
  2>&1 | tee -a "$LOG"

echo "=== [$(date)] 4/4 publish to web/public/data ===" | tee -a "$LOG"
mkdir -p web/public/data
cp -v data/way_city_graph.geojson data/way_city_anchors.geojson \
      web/public/data/ 2>&1 | tee -a "$LOG"
if [ -f data/way_city_anchors_orphans.geojson ]; then
  cp -v data/way_city_anchors_orphans.geojson web/public/data/ 2>&1 | tee -a "$LOG"
fi

echo "=== [$(date)] all done ===" | tee -a "$LOG"
