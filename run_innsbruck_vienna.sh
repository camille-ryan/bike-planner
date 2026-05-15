#!/usr/bin/env bash
# Full Austria scenicness pipeline (no bbox):
#   1. landcover-ingest  — re-ingest landcover polygons for all Austria
#   2. waterway-ingest   — re-ingest waterway polygons for all Austria
#   3. scenicness-bake   — tiled bake over the ways extent (NaN-aware
#                          kernels, downsampled PNGs for web)
#   4. sync              — copy PNGs + manifest into web/public/data/

set -e
cd /mnt/e/proj/bike
mkdir -p logs

TS=$(date +%Y%m%d-%H%M%S)
echo "=== run_innsbruck_vienna.sh start: $(date) ==="

echo
echo "=== [1] landcover-ingest (full austria) ==="
docker compose run --rm -e PGDATABASE=bike_v2_test pgrouting \
  landcover-ingest --countries austria \
  2>&1 | tee logs/iv-landcover-$TS.log

echo
echo "=== [2] waterway-ingest (full austria) ==="
docker compose run --rm -e PGDATABASE=bike_v2_test pgrouting \
  waterway-ingest --countries austria \
  2>&1 | tee logs/iv-waterway-$TS.log

echo
echo "=== [3] scenicness-bake (tiled, full ways extent) ==="
docker compose run --rm -e PGDATABASE=bike_v2_test pgrouting \
  scenicness-bake --res-m 20 --tile-size-deg 1.0 \
  --export-rasters /data/web_overlays/scenicness \
  2>&1 | tee logs/iv-bake-$TS.log

echo
echo "=== [4] sync overlays to web/public ==="
rm -f web/public/data/scenicness/*.png
cp data/web_overlays/scenicness/*.png web/public/data/scenicness/
cp data/web_overlays/scenicness/manifest.json web/public/data/scenicness/

echo
echo "=== run_innsbruck_vienna.sh DONE: $(date) ==="
