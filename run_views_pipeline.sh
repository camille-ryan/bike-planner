#!/usr/bin/env bash
# Run the remaining steps of the views-only routing pipeline:
#   1. spts-multi --profiles views   (~2-4 h)
#   2. paired --profile views        (~2-3 h)
#   3. Verify Graz→Cph route
#
# Launch:
#   nohup ./run_views_pipeline.sh > logs/views-pipeline-$(date +%Y%m%d-%H%M%S).log 2>&1 &
set -uo pipefail
cd /mnt/e/proj/bike
mkdir -p logs

PG_DB=bike_v2_test
PG_MOUNT="-v /mnt/e/proj/bike/pgrouting:/app"
PG_RUN="docker compose --profile preprocess run --rm $PG_MOUNT -e PGDATABASE=$PG_DB -e PYTHONUNBUFFERED=1 pgrouting"
PSQL="docker exec -i bike-postgres psql -U bike -d $PG_DB"

step() { echo; echo "=== [$(date)] STEP $1: $2 ==="; }

# ── STEP 1: spts-multi ──────────────────────────────────────────────
step 1 "spts-multi --profiles views (build per-anchor SPTs)"
if ! $PG_RUN spts-multi --profiles views; then
  echo "!!! spts-multi failed; aborting"; exit 1
fi

# ── STEP 2: paired ──────────────────────────────────────────────────
step 2 "paired --profile views (build trunk SQLite DB)"
if ! $PG_RUN paired --profile views; then
  echo "!!! paired failed; aborting"; exit 1
fi

# ── STEP 3: verify ──────────────────────────────────────────────────
step 3 "verify Graz→Cph route on views profile"
docker compose restart api 2>&1 | tail -3
sleep 5
ls -la /mnt/e/proj/bike/data/spt/views/paired_trunks.db 2>&1

GRAZ_ID=$($PSQL -qtA -c "SELECT id FROM anchors WHERE name ILIKE 'graz%' AND name NOT ILIKE '%graz-%' ORDER BY length(name) ASC LIMIT 1;" | tail -1)
CPH_ID=$($PSQL -qtA -c "SELECT id FROM anchors WHERE name ILIKE '%copenhagen%' OR name ILIKE 'kobenhavn%' OR name ILIKE 'københavn%' ORDER BY length(name) ASC LIMIT 1;" | tail -1)
echo "[verify] GRAZ_ID=$GRAZ_ID  CPH_ID=$CPH_ID"

if [ -n "$GRAZ_ID" ] && [ -n "$CPH_ID" ]; then
  echo "--- /trunk/route a=$GRAZ_ID b=$CPH_ID profile=views ---"
  docker exec bike-api curl -s -o /tmp/route.json -w "HTTP %{http_code} (%{time_total}s)\n" \
    "http://localhost:8000/trunk/route?a=$GRAZ_ID&b=$CPH_ID&profile=views"
  docker exec bike-api python3 -c "
import json
with open('/tmp/route.json') as f:
    feat = json.load(f)
geom = feat.get('geometry', {})
coords = geom.get('coordinates') or []
print(f'  profile=views coords={len(coords):,}')
"
else
  echo "!!! couldn't resolve Graz/Copenhagen anchor ids"
  exit 1
fi

echo
echo "=== [$(date)] VIEWS PIPELINE DONE ==="
