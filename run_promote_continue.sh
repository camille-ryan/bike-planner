#!/usr/bin/env bash
# Continuation orchestrator: resumes the promote-v2test pipeline after
# resume_resolve.py has restored the failed step 4 transaction.
#
# Skips: data migration (already done), ingest (resume_resolve handled it).
# Runs: steps 5–13 (snap → dem → canopy → bake → recompute → ways_bike → spts → paired → verify).
#
# IMPORTANT: do not run diagnostic SELECTs on tmp_* or ways while this is
# active — they can take AccessShareLocks that deadlock long transactions.
# Instead, watch the log file (PYTHONUNBUFFERED=1 in all docker runs).
#
# Launch:
#   nohup ./run_promote_continue.sh > logs/continue-$(date +%Y%m%d-%H%M%S).log 2>&1 &

set -euo pipefail

cd /mnt/e/proj/bike
mkdir -p data/logs logs

LOG=/mnt/e/proj/bike/data/logs/pipeline.log
PG_DB=bike_v2_test
PG_MOUNT="-v /mnt/e/proj/bike/pgrouting:/app"
# PYTHONUNBUFFERED=1 ensures live stdout — no more silent ingest blackouts
PG_RUN="docker compose --profile preprocess run --rm $PG_MOUNT -e PGDATABASE=$PG_DB -e PYTHONUNBUFFERED=1 pgrouting"
PSQL="docker exec -i bike-postgres psql -U bike -d $PG_DB"
COUNTRIES="austria,czech-republic,germany,denmark"
PROFILES=(direct vineyard_lover forest_lover views water)

step() {
  echo
  echo "=== [$(date)] STEP $1: $2 ===" | tee -a "$LOG"
}

# ── STEP 5 ─────────────────────────────────────────────────────────
step "5/13" "snap anchors for 4 countries"
$PG_RUN snap --countries "$COUNTRIES" 2>&1 | tee -a "$LOG"

# ── STEP 6 ─────────────────────────────────────────────────────────
step "6/13" "dem-ingest (with partial NULL elev index from step 3)"
$PG_RUN dem-ingest 2>&1 | tee -a "$LOG"

# ── STEP 7 ─────────────────────────────────────────────────────────
step "7/13" "canopy-compute (771K forest polygons over 4 countries)"
$PG_RUN canopy-compute 2>&1 | tee -a "$LOG"

# ── STEP 8 ─────────────────────────────────────────────────────────
step "8/13" "scenicness-bake + incremental overlay publishing"
./watch_overlays.sh >> "$LOG" 2>&1 &
WATCHER_PID=$!
echo "[orchestrator] watcher pid=$WATCHER_PID" | tee -a "$LOG"

BAKE_MAX_ATTEMPTS=${BAKE_MAX_ATTEMPTS:-5}
attempt=1
bake_ok=0
while [ "$attempt" -le "$BAKE_MAX_ATTEMPTS" ]; do
  echo "--- [$(date)] bake attempt $attempt/$BAKE_MAX_ATTEMPTS ---" | tee -a "$LOG"
  set +e
  $PG_RUN scenicness-bake --export-rasters /data/scenicness 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -eq 0 ]; then bake_ok=1; break; fi
  echo "!!! bake exited rc=$rc; resuming in 30s" | tee -a "$LOG"
  attempt=$((attempt+1)); sleep 30
done
kill "$WATCHER_PID" 2>/dev/null || true
wait "$WATCHER_PID" 2>/dev/null || true
if [ "$bake_ok" -ne 1 ]; then
  echo "!!! bake failed $BAKE_MAX_ATTEMPTS times, aborting" | tee -a "$LOG"
  exit 1
fi

rsync -a --update /mnt/e/proj/bike/data/scenicness/xyz/ \
  /mnt/e/proj/bike/web/public/data/scenicness/xyz/ 2>&1 | tee -a "$LOG"
cp -v /mnt/e/proj/bike/data/scenicness/manifest.json \
      /mnt/e/proj/bike/web/public/data/scenicness/manifest.json 2>&1 | tee -a "$LOG"

# ── STEP 9 ─────────────────────────────────────────────────────────
step "9/13" "recompute-cost × ${PROFILES[*]} (sequential — postgres 4GB cap)"
for P in "${PROFILES[@]}"; do
  echo "--- [$(date)] recompute-cost --profile $P ---" | tee -a "$LOG"
  $PG_RUN recompute-cost --profile "$P" 2>&1 | tee -a "$LOG"
done

# ── STEP 10 ────────────────────────────────────────────────────────
step "10/13" "build ways_bike + propagate cost columns"
docker compose --profile preprocess run --rm $PG_MOUNT \
  -e PGDATABASE=$PG_DB -e PYTHONUNBUFFERED=1 --entrypoint python3 pgrouting \
  /app/build_ways_bike.py 2>&1 | tee -a "$LOG" || \
  echo "[orchestrator] build_ways_bike.py may have already run; continuing" | tee -a "$LOG"
$PSQL <<SQL 2>&1 | tee -a "$LOG"
  DO \$\$
  DECLARE p text;
  BEGIN
    FOREACH p IN ARRAY ARRAY['direct','vineyard_lover','forest_lover','views','water']
    LOOP
      EXECUTE format('ALTER TABLE ways_bike ADD COLUMN IF NOT EXISTS cost_%I real', p);
      EXECUTE format('ALTER TABLE ways_bike ADD COLUMN IF NOT EXISTS reverse_cost_%I real', p);
      EXECUTE format(
        'UPDATE ways_bike b SET cost_%I = w.cost_%I, reverse_cost_%I = w.reverse_cost_%I
           FROM ways w WHERE w.gid = b.gid', p, p, p, p);
    END LOOP;
  END\$\$;
SQL

# ── STEP 11 ────────────────────────────────────────────────────────
step "11/13" "spts-multi (5 profiles, force cache rebuild)"
$PG_RUN spts-multi --force-cache 2>&1 | tee -a "$LOG"

# ── STEP 12 ────────────────────────────────────────────────────────
step "12/13" "paired --profile X × 5 profiles (sequential)"
for P in "${PROFILES[@]}"; do
  echo "--- [$(date)] paired --profile $P ---" | tee -a "$LOG"
  $PG_RUN paired --profile "$P" 2>&1 | tee -a "$LOG"
done

# ── STEP 13 ────────────────────────────────────────────────────────
step "13/13" "restart api + smoke test Graz→Cph"
docker compose restart api 2>&1 | tee -a "$LOG"
sleep 5
ls -la /mnt/e/proj/bike/data/spt/{direct,vineyard_lover,forest_lover,views,water}/paired_trunks.db 2>&1 | tee -a "$LOG"

GRAZ_ID=$($PSQL -tA -c "SELECT id FROM anchors WHERE name ILIKE 'graz%' AND name NOT ILIKE '%graz-%' ORDER BY length(name) ASC LIMIT 1;")
CPH_ID=$($PSQL -tA -c "SELECT id FROM anchors WHERE name ILIKE '%copenhagen%' OR name ILIKE 'kobenhavn%' OR name ILIKE 'københavn%' ORDER BY length(name) ASC LIMIT 1;")
echo "[verify] GRAZ_ID=$GRAZ_ID  CPH_ID=$CPH_ID" | tee -a "$LOG"

if [ -n "$GRAZ_ID" ] && [ -n "$CPH_ID" ]; then
  for P in "${PROFILES[@]}"; do
    echo "--- [$(date)] /trunk/route a=$GRAZ_ID b=$CPH_ID profile=$P ---" | tee -a "$LOG"
    docker exec bike-api curl -s -o /tmp/route.json -w "HTTP %{http_code} (%{time_total}s)\n" \
      "http://localhost:8000/trunk/route?a=$GRAZ_ID&b=$CPH_ID&profile=$P" 2>&1 | tee -a "$LOG"
    docker exec bike-api python3 -c "
import json
with open('/tmp/route.json') as f:
    feat = json.load(f)
geom = feat.get('geometry', {})
coords = geom.get('coordinates') or []
print(f'  profile=$P coords={len(coords):,}')
" 2>&1 | tee -a "$LOG"
  done
else
  echo "!!! couldn't resolve Graz/Copenhagen anchor ids" | tee -a "$LOG"
fi

echo
echo "=== [$(date)] PIPELINE DONE ===" | tee -a "$LOG"
