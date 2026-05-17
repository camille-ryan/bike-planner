#!/usr/bin/env bash
# Pre-compute all city-pair routes across all 5 cost profiles.
#
# For each profile:
#   1. recompute-cost --profile <profile>  (≈ 30 min on 22M edges)
#   2. route-city-pairs --profile <profile> --out city_routes.geojson
#      (15 pairs × pgr_bdAstar with 30km line-buffer corridor;
#       INN-WIE ≈ 10 min, short pairs ≈ 1-2 min, total ≈ 60 min)
#
# Output is APPENDED to city_routes.geojson — the CLI handler
# de-dupes per (a,b,profile) so re-runs replace rather than duplicate.
# After all profiles: sync to web/public/data/.
#
# Total wall ≈ 5h routing + 2.5h recompute = ~7-8 hours.
# Per-pair pgr_bdAstar memory peak ≈ 5 GB postgres backend (under 10 GB cap).

set -e
cd /mnt/e/proj/bike
mkdir -p logs

TS=$(date +%Y%m%d-%H%M%S)
OUT_HOST=/mnt/e/proj/bike/data/web_overlays/city_routes.geojson
OUT_CTR=/data/web_overlays/city_routes.geojson
rm -f "$OUT_HOST"   # start fresh
echo "=== run_city_routes.sh start: $(date) ==="

PROFILES="direct vineyard_lover forest_lover views water"

for prof in $PROFILES; do
  echo
  echo "=== [$(date +%H:%M:%S)] recompute-cost --profile $prof ==="
  docker compose run --rm -e PGDATABASE=bike_v2_test pgrouting \
    recompute-cost --profile "$prof" \
    2>&1 | tee logs/citypair-recompute-$prof-$TS.log

  echo
  echo "=== [$(date +%H:%M:%S)] route-city-pairs --profile $prof ==="
  docker compose run --rm -e PGDATABASE=bike_v2_test pgrouting \
    route-city-pairs --profile "$prof" --out "$OUT_CTR" \
    2>&1 | tee logs/citypair-route-$prof-$TS.log
done

echo
echo "=== [$(date +%H:%M:%S)] sync to web/public ==="
cp "$OUT_HOST" /mnt/e/proj/bike/web/public/data/

echo
echo "=== run_city_routes.sh DONE: $(date) ==="
