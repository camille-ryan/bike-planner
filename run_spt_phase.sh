#!/usr/bin/env bash
# SPT phase orchestrator — runs after run_4country_full.sh finishes.
# Targets the user-set goal: all 5 routing profiles attached, all per-anchor
# SPTs computed, all paired-SPT trunk DBs built, Graz→Copenhagen routable.
#
# Steps:
#   9.  Recompute cost_direct on ways (orchestrator only handled 4 multi-axis)
#   10. Propagate cost_direct onto ways_bike
#   11. spts-multi (5 profiles: direct, vineyard_lover, forest_lover, views, water)
#   12. paired --profile <P> per profile → paired_trunks.db
#   13. Verify trunk DBs present, restart API to preload, hit /trunk/route smoke test
#
# Idempotent / resumable: every step can be re-run.
#
# Launch:
#   nohup ./run_spt_phase.sh > logs/spt-phase-$(date +%Y%m%d-%H%M%S).log 2>&1 &

set -euo pipefail

cd /mnt/e/proj/bike
mkdir -p data/logs logs

LOG=/mnt/e/proj/bike/data/logs/pipeline.log
PG_MOUNT="-v /mnt/e/proj/bike/pgrouting:/app"
PROFILES=(direct vineyard_lover forest_lover views water)

step() {
  echo
  echo "=== [$(date)] STEP $1: $2 ===" | tee -a "$LOG"
}

# ── STEP 9 ─────────────────────────────────────────────────────────
step "9/13" "recompute-cost --profile direct"
docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
  recompute-cost --profile direct 2>&1 | tee -a "$LOG"

# ── STEP 10 ────────────────────────────────────────────────────────
step "10/13" "propagate cost_direct onto ways_bike"
docker exec -i bike-postgres psql -U bike -d bike <<'SQL' 2>&1 | tee -a "$LOG"
  ALTER TABLE ways_bike ADD COLUMN IF NOT EXISTS cost_direct real;
  ALTER TABLE ways_bike ADD COLUMN IF NOT EXISTS reverse_cost_direct real;
  UPDATE ways_bike b SET cost_direct = w.cost_direct,
                         reverse_cost_direct = w.reverse_cost_direct
    FROM ways w WHERE w.gid = b.gid;
SQL

# ── STEP 11 ────────────────────────────────────────────────────────
step "11/13" "spts-multi (5 profiles, force cache rebuild)"
# --force-cache: the npz graph cache was built before this rebake, so the
# scenic cost columns it captured are stale. Force a fresh dump from postgres.
docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
  spts-multi --force-cache 2>&1 | tee -a "$LOG"

# ── STEP 12 ────────────────────────────────────────────────────────
step "12/13" "paired --profile <P> for ${PROFILES[*]}"
for P in "${PROFILES[@]}"; do
  echo "--- [$(date)] paired --profile $P ---" | tee -a "$LOG"
  docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
    paired --profile "$P" 2>&1 | tee -a "$LOG"
done

# ── STEP 13 ────────────────────────────────────────────────────────
step "13/13" "verify trunk DBs + API smoke test"

# 13a: trunk DBs present + sized?
ls -la /mnt/e/proj/bike/data/spt/{direct,vineyard_lover,forest_lover,views,water}/paired_trunks.db 2>&1 | tee -a "$LOG"

# 13b: restart API so trunk_router.preload(DEFAULT_PROFILE) picks up the new DB
echo "--- restarting bike-api so it preloads the fresh trunk DB ---" | tee -a "$LOG"
docker compose restart api 2>&1 | tee -a "$LOG"

# 13c: smoke test — query each profile (Graz vertex → Copenhagen vertex via /trunk/route).
# Indices come from anchors table; resolve at runtime.
echo "--- looking up Graz + Copenhagen anchor ids ---" | tee -a "$LOG"
GRAZ_ID=$(docker exec bike-postgres psql -U bike -d bike -tA -c \
  "SELECT id FROM anchors WHERE name ILIKE 'graz%' AND name NOT ILIKE '%graz-%' ORDER BY length(name) ASC LIMIT 1;")
CPH_ID=$(docker exec bike-postgres psql -U bike -d bike -tA -c \
  "SELECT id FROM anchors WHERE name ILIKE '%copenhagen%' OR name ILIKE 'kobenhavn%' OR name ILIKE 'københavn%' ORDER BY length(name) ASC LIMIT 1;")
echo "[verify] GRAZ_ID=$GRAZ_ID  CPH_ID=$CPH_ID" | tee -a "$LOG"

if [ -z "$GRAZ_ID" ] || [ -z "$CPH_ID" ]; then
  echo "!!! couldn't resolve Graz/Copenhagen anchor ids — investigate manually" | tee -a "$LOG"
else
  # Give API a moment to come back up + preload trunk DB.
  sleep 5
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
props = feat.get('properties', {})
print(f'  profile=$P coords={len(coords):,} props_keys={sorted(props.keys())}')
" 2>&1 | tee -a "$LOG"
  done
fi

echo
echo "=== [$(date)] SPT PHASE DONE ===" | tee -a "$LOG"
echo "    Verify in browser: open the web app, route from Graz to Copenhagen" | tee -a "$LOG"
echo "    on each of the 5 profiles, toggle each of the 16 scenicness overlays." | tee -a "$LOG"
