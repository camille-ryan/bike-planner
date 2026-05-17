#!/usr/bin/env bash
# Direct-profile-only first pass for the city-pair pre-compute.
# Used to sanity-check the geometries before committing to the full
# 5-profile run. See run_city_routes.sh for the full sweep.

set -e
cd /mnt/e/proj/bike
mkdir -p logs

TS=$(date +%Y%m%d-%H%M%S)
OUT_CTR=/data/web_overlays/city_routes.geojson
echo "=== run_city_routes_direct.sh start: $(date) ==="

echo
echo "=== [$(date +%H:%M:%S)] recompute-cost --profile direct ==="
docker compose run --rm -e PGDATABASE=bike_v2_test pgrouting \
  recompute-cost --profile direct 2>&1 | tee logs/citypair-recompute-direct-$TS.log

echo
echo "=== [$(date +%H:%M:%S)] route-city-pairs --profile direct ==="
docker compose run --rm -e PGDATABASE=bike_v2_test pgrouting \
  route-city-pairs --profile direct --out "$OUT_CTR" \
  2>&1 | tee logs/citypair-route-direct-$TS.log

echo
echo "=== [$(date +%H:%M:%S)] sync to web ==="
cp /mnt/e/proj/bike/data/web_overlays/city_routes.geojson \
   /mnt/e/proj/bike/web/public/data/

echo
echo "=== run_city_routes_direct.sh DONE: $(date) ==="
