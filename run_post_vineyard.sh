#!/usr/bin/env bash
# Tail orchestrator: runs after `recompute-cost --profile vineyard_lover`
# completes. Skips the standalone views/water recomputes and instead
# derives them from cost_vineyard_lover algebraically (~5 min each via
# SQL instead of ~10h each via Python tile-streaming).
#
# Sequence:
#   D'. derive cost_views, cost_water from cost_vineyard_lover (SQL)
#   D.  build ways_bike + propagate cost columns (4 profiles)
#   E.  spts-multi (4 profiles)
#   F.  paired × 4 profiles
#   G.  restart api + smoke test Graz→Cph
#
# Launch:
#   nohup ./run_post_vineyard.sh > logs/post-vl-$(date +%Y%m%d-%H%M%S).log 2>&1 &

set -euo pipefail

cd /mnt/e/proj/bike
mkdir -p data/logs logs

LOG=/mnt/e/proj/bike/data/logs/pipeline.log
PG_DB=bike_v2_test
PG_MOUNT="-v /mnt/e/proj/bike/pgrouting:/app"
PG_RUN="docker compose --profile preprocess run --rm $PG_MOUNT -e PGDATABASE=$PG_DB -e PYTHONUNBUFFERED=1 pgrouting"
PSQL="docker exec -i bike-postgres psql -U bike -d $PG_DB"
PROFILES=(direct vineyard_lover views water)
PROFILES_PGARRAY="ARRAY['direct','vineyard_lover','views','water']"

step() {
  echo
  echo "=== [$(date)] STEP $1: $2 ===" | tee -a "$LOG"
}

# ── STEP D' ────────────────────────────────────────────────────────
step "D'" "derive cost_views, cost_water from cost_vineyard_lover (SQL)"
docker compose --profile preprocess run --rm $PG_MOUNT \
  -e PGDATABASE=$PG_DB -e PYTHONUNBUFFERED=1 --entrypoint python3 pgrouting \
  /app/derive_multi_axis_costs.py vineyard_lover views,water 2>&1 | tee -a "$LOG"

# ── STEP D ─────────────────────────────────────────────────────────
step "D" "build ways_bike + propagate cost columns (4 profiles)"
docker compose --profile preprocess run --rm $PG_MOUNT \
  -e PGDATABASE=$PG_DB -e PYTHONUNBUFFERED=1 --entrypoint python3 pgrouting \
  /app/build_ways_bike.py 2>&1 | tee -a "$LOG" || \
  echo "[orchestrator] build_ways_bike.py may have already run; continuing" | tee -a "$LOG"
$PSQL <<SQL 2>&1 | tee -a "$LOG"
  DO \$\$
  DECLARE p text;
  BEGIN
    FOREACH p IN ARRAY ${PROFILES_PGARRAY}
    LOOP
      EXECUTE format('ALTER TABLE ways_bike ADD COLUMN IF NOT EXISTS cost_%I real', p);
      EXECUTE format('ALTER TABLE ways_bike ADD COLUMN IF NOT EXISTS reverse_cost_%I real', p);
      EXECUTE format(
        'UPDATE ways_bike b SET cost_%I = w.cost_%I, reverse_cost_%I = w.reverse_cost_%I
           FROM ways w WHERE w.gid = b.gid', p, p, p, p);
    END LOOP;
  END\$\$;
SQL

# ── STEP E ─────────────────────────────────────────────────────────
step "E" "spts-multi (4 profiles, force cache rebuild)"
$PG_RUN spts-multi --force-cache --profiles "direct,vineyard_lover,views,water" 2>&1 | tee -a "$LOG"

# ── STEP F ─────────────────────────────────────────────────────────
step "F" "paired --profile X × 4 profiles (sequential)"
for P in "${PROFILES[@]}"; do
  echo "--- [$(date)] paired --profile $P ---" | tee -a "$LOG"
  $PG_RUN paired --profile "$P" 2>&1 | tee -a "$LOG"
done

# ── STEP G ─────────────────────────────────────────────────────────
step "G" "restart api + smoke test Graz→Cph"
docker compose restart api 2>&1 | tee -a "$LOG"
sleep 5
ls -la /mnt/e/proj/bike/data/spt/{direct,vineyard_lover,views,water}/paired_trunks.db 2>&1 | tee -a "$LOG"

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
