#!/usr/bin/env bash
# Per-country supplement-off ingest for CZ/DE/DK.
# Austria is already loaded and bikeable; this script only adds the
# previously-missing bikeable edges for the other three countries.
#
# Each country is its own self-contained stream + resolve + commit, so
# a WSL VM restart mid-run only loses the in-flight country (others
# stay committed). Each phase is idempotent — re-launching after a
# crash skips countries that already have bikeable rows.
#
# Order: DK (7.6M edges, smoke test) → CZ (13M) → DE (88M, biggest).
#
# Launch:
#   nohup ./run_per_country_ingest.sh > logs/per-country-$(date +%Y%m%d-%H%M%S).log 2>&1 &

set -uo pipefail

cd /mnt/e/proj/bike
mkdir -p logs

PG_DB=bike_v2_test
PG_MOUNT="-v /mnt/e/proj/bike/pgrouting:/app"
PG_RUN="docker compose --profile preprocess run --rm $PG_MOUNT \
  -e PGDATABASE=$PG_DB -e PYTHONUNBUFFERED=1 -e INGEST_SUPPLEMENT_MODE=0 pgrouting"
PSQL="docker exec -i bike-postgres psql -U bike -d $PG_DB"

# Per-country approximate bbox (lat lon) for the "is this country
# already loaded?" idempotency check. Bikeable count > 100k → skip.
#   country         lat_min lat_max lon_min lon_max
declare -A BBOX=(
  [denmark]="54.0 58.0 8.0 13.0"
  [czech-republic]="48.5 51.1 12.0 19.0"
  [germany]="47.2 55.1 5.8 15.2"
)

step() { echo; echo "=== [$(date)] $1 ==="; }

# Idempotency probe: count bikeable rows whose SOURCE vertex lies in
# the country's bbox. Returns the integer count on stdout.
bikeable_in_bbox() {
  local C=$1
  read lat_min lat_max lon_min lon_max <<<"${BBOX[$C]}"
  # -qtA: quiet, tuples-only, unaligned → emits just the integer.
  # Inline SETs via -c so we don't get SET echoes.
  $PSQL -qtA \
    -c "SET work_mem='128MB'" \
    -c "SET max_parallel_workers_per_gather=0" \
    -c "SELECT COUNT(*)
          FROM ways w JOIN ways_vertices_pgr v ON v.id = w.source
         WHERE w.bike_excluded = false
           AND v.lat BETWEEN $lat_min AND $lat_max
           AND v.lon BETWEEN $lon_min AND $lon_max;" \
    | tail -n1
}

ingest_country() {
  local C=$1
  step "INGEST $C"
  local before
  before=$(bikeable_in_bbox "$C")
  echo "[per-country] $C: bikeable BEFORE = $before"
  if [ "$before" -gt 100000 ]; then
    echo "[per-country] $C: already has $before bikeable rows in bbox — SKIP"
    return 0
  fi
  echo "[per-country] streaming + resolving $C with INGEST_SUPPLEMENT_MODE=0 ..."
  if ! $PG_RUN ingest --countries "$C"; then
    echo "!!! [per-country] $C: ingest FAILED — leaving for retry on next launch"
    return 1
  fi
  local after
  after=$(bikeable_in_bbox "$C")
  echo "[per-country] $C: bikeable AFTER  = $after  (delta = $((after - before)))"
  if [ "$after" -le "$before" ]; then
    echo "!!! [per-country] $C: ingest did not add bikeable rows"
    return 1
  fi
}

step "PRECHECK"
echo "[per-country] ways/vertices totals BEFORE all ingests:"
$PSQL -c "SELECT COUNT(*) AS ways FROM ways; SELECT COUNT(*) AS vertices FROM ways_vertices_pgr;"

# DK first (smallest = smoke test), then CZ, then DE.
for C in denmark czech-republic germany; do
  ingest_country "$C" || { echo "!!! aborting on $C failure"; exit 1; }
done

step "FINAL TOTALS"
$PSQL -c "SELECT COUNT(*) AS ways, COUNT(*) FILTER (WHERE bike_excluded=false) AS bikeable FROM ways;"

echo
echo "=== [$(date)] PER-COUNTRY INGEST DONE ==="
echo "Next: ./run_views_only_postingest.sh"
