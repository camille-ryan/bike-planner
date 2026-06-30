#!/usr/bin/env bash
# Recovery-B-Waytags: stage 2 of the CZ recovery.
# Discovered: the 12.6M new CZ bikeable rows have empty highway/access/
# bicycle tags. ingest_pbf.py writes only 8 cols (no highway/surface/
# etc.); those come from backfill_way_tags_to_ways.py JOINing against
# the way_tags sidecar. But way_tags is missing CZ entries.
#
# Steps:
#   1. Extract way_tags from CZ PBF (~5-15 min) — uses a temp PBFS
#      list with only CZ; existing AT/DE/DK entries preserved via
#      ON CONFLICT DO UPDATE merge.
#   2. backfill_way_tags_to_ways.py — CTAS swap of `ways` to JOIN
#      newly-populated way_tags onto the 12.6M empty-highway CZ rows
#      (~30-45 min for the 131M-row CTAS).
#   3. recompute_cost_views_ctas — now that highway is populated, cost
#      will compute correctly (~30 min for 12.6M new bikeable rows).
#   4. build_ways_bike (drop + rebuild, ~15-25 min) — pulls cost_views
#      into the bike-only edge table.
#   5. delete graph_multi.npz so spts-multi rebuilds.
#   6. clean up tiny (broken) CZ npz files from prior crashes.
#   7. spts-multi --profiles views (rebuilds cache, ~15-25 min;
#      then ~12-15h for the remaining ~2,970 anchors).
#   8. paired --profile views (~2h).
#   9. verify Graz→Cph route.
#
# Launch:
#   nohup ./run_recovery_b_waytags.sh > logs/recovery-b-waytags-$(date +%Y%m%d-%H%M%S).log 2>&1 &
set -uo pipefail
cd /mnt/e/proj/bike
mkdir -p logs

PG_DB=bike_v2_test
PG_MOUNT="-v /mnt/e/proj/bike/pgrouting:/app"
PG_RUN_PY="docker compose --profile preprocess run --rm $PG_MOUNT \
  -e PGDATABASE=$PG_DB -e PYTHONUNBUFFERED=1 --entrypoint python3 pgrouting"
PG_RUN_MAIN="docker compose --profile preprocess run --rm $PG_MOUNT \
  -e PGDATABASE=$PG_DB -e PYTHONUNBUFFERED=1 pgrouting"
PSQL="docker exec -i bike-postgres psql -U bike -d $PG_DB"

step() { echo; echo "=== [$(date)] STEP $1: $2 ==="; }

# ── STEP 2: backfill highway/etc into ways ─────────────────────────
step 2 "backfill_way_tags_to_ways.py (CTAS swap of full ways table)"
if ! $PG_RUN_PY backfill_way_tags_to_ways.py; then
  echo "!!! backfill FAILED — aborting"; exit 1
fi

# Sanity: CZ Prague should now have bikeable rows with highway populated
HW_CZ=$($PSQL -qtA -c "
WITH cz_vid AS (
  SELECT id FROM ways_vertices_pgr
  WHERE the_geom && ST_MakeEnvelope(14.37, 50.03, 14.47, 50.13, 4326) LIMIT 200
)
SELECT COUNT(*) FROM cz_vid c JOIN ways w ON w.source = c.id
WHERE NOT w.bike_excluded AND w.highway <> '';" | tail -1)
echo "[recovery-b-waytags] CZ Prague 5km bikeable+highway: $HW_CZ"
if [ "$HW_CZ" -lt 50 ]; then
  echo "!!! CZ rows still missing highway after backfill ($HW_CZ) — aborting"
  exit 1
fi

# ── STEP 3: recompute-cost-views for the now-highway-populated rows ─
step 3 "recompute_cost_views_ctas (process 12.6M new CZ bikeable rows)"
if ! $PG_RUN_PY recompute_cost_views_ctas.py; then
  echo "!!! recompute_cost_views FAILED — aborting"; exit 1
fi

# ── STEP 4: rebuild ways_bike ──────────────────────────────────────
step 4 "build_ways_bike.py (drop + recreate with full coverage)"
if ! $PG_RUN_PY build_ways_bike.py; then
  echo "!!! build_ways_bike FAILED — aborting"; exit 1
fi

# Sanity: ways_bike should have CZ edges with cost_views
CV_CZ=$($PSQL -qtA -c "
SELECT COUNT(*) FROM ways_bike
WHERE src_pt && ST_MakeEnvelope(14.37, 50.03, 14.47, 50.13, 4326)
  AND cost_views IS NOT NULL;" | tail -1)
echo "[recovery-b-waytags] CZ Prague 5km ways_bike w/ cost_views: $CV_CZ"
if [ "$CV_CZ" -lt 50 ]; then
  echo "!!! ways_bike CZ coverage broken ($CV_CZ) — aborting"; exit 1
fi

# ── STEP 5: invalidate graph cache ─────────────────────────────────
step 5 "delete graph_multi.npz cache"
ls -la data/spt_cache/graph_multi.npz 2>&1 | tail -2
rm -f data/spt_cache/graph_multi.npz
echo "[recovery-b-waytags] cache deleted"

# ── STEP 6: clean tiny CZ npz files ────────────────────────────────
step 6 "delete tiny CZ npz files (< 10 KB)"
N_TINY=$(find data/spt/views/spt -name "*.npz" -size -10k 2>/dev/null | wc -l)
echo "[recovery-b-waytags] found $N_TINY tiny npz files to delete"
find data/spt/views/spt -name "*.npz" -size -10k -delete 2>&1 | tail -3
N_REMAINING=$(ls data/spt/views/spt/*.npz 2>/dev/null | wc -l)
echo "[recovery-b-waytags] $N_REMAINING healthy AT npz files remain"

# ── STEP 7: spts-multi ─────────────────────────────────────────────
step 7 "spts-multi --profiles views (rebuilds cache, processes remaining anchors)"
if ! $PG_RUN_MAIN spts-multi --profiles views; then
  echo "!!! spts-multi FAILED — aborting"; exit 1
fi

# ── STEP 8: paired ─────────────────────────────────────────────────
step 8 "paired --profile views"
if ! $PG_RUN_MAIN paired --profile views; then
  echo "!!! paired FAILED — aborting"; exit 1
fi

# ── STEP 9: verify ─────────────────────────────────────────────────
step 9 "verify Graz→Cph route"
docker compose restart api 2>&1 | tail -3
sleep 5
ls -la data/spt/views/paired_trunks.db 2>&1

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
fi

echo
echo "=== [$(date)] RECOVERY-B-WAYTAGS DONE ==="
