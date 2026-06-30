#!/usr/bin/env bash
# Recovery-B post-ingest chain: after CZ has been re-ingested with
# supplement off, propagate the new bikeable rows through every
# downstream stage so spts-multi can finish the views profile.
#
# Steps:
#   1. SANITY: confirm CZ now has bikeable rows in Prague (else abort)
#   2. RE-SNAP anchors against the bikeable subgraph (some snaps may
#      have been on now-superseded vertices)
#   3. recompute-cost-views CTAS — filters to NULL cost_views so it
#      only processes the new CZ bikeable rows (and any DK/DE gaps)
#   4. Rebuild ways_bike (full drop+recreate; cheap at ~15-25 min)
#   5. Force-rebuild graph_multi.npz cache (rm; spts-multi rebuilds
#      on next launch)
#   6. Clean up the 18 tiny garbage CZ npz files from the prior crash
#   7. Restart spts-multi (skips the 245 healthy AT npz; processes
#      ~2,950 remaining)
#   8. paired --profile views
#   9. Verify Graz→Cph route
#
# Launch:
#   nohup ./run_recovery_b_postingest.sh > logs/recovery-b-post-$(date +%Y%m%d-%H%M%S).log 2>&1 &
set -uo pipefail
cd /mnt/e/proj/bike
mkdir -p logs

PG_DB=bike_v2_test
PG_MOUNT="-v /mnt/e/proj/bike/pgrouting:/app"
PG_RUN="docker compose --profile preprocess run --rm $PG_MOUNT -e PGDATABASE=$PG_DB -e PYTHONUNBUFFERED=1 pgrouting"
PSQL="docker exec -i bike-postgres psql -U bike -d $PG_DB"

step() { echo; echo "=== [$(date)] STEP $1: $2 ==="; }

# ── STEP 1: SANITY CHECK ────────────────────────────────────────────
step 1 "sanity: CZ Prague bikeable count"
CZ_BIKEABLE=$($PSQL -qtA -c "
WITH cz_vid AS (
  SELECT id FROM ways_vertices_pgr
  WHERE the_geom && ST_MakeEnvelope(14.37, 50.03, 14.47, 50.13, 4326) LIMIT 200
)
SELECT COUNT(*) FROM cz_vid c JOIN ways w ON w.source = c.id
WHERE NOT w.bike_excluded;" | tail -1)
echo "[recovery-b] CZ Prague 5km bikeable = $CZ_BIKEABLE"
if [ "$CZ_BIKEABLE" -lt 50 ]; then
  echo "!!! CZ still has too few bikeable rows ($CZ_BIKEABLE) — aborting"
  exit 1
fi

# ── STEP 2: RE-SNAP anchors against bikeable subgraph ──────────────
step 2 "re-snap anchors against bikeable-only vertices"
$PSQL <<'SQL'
CREATE UNLOGGED TABLE _bikeable_vertex_geom AS
SELECT v.id, v.the_geom
  FROM ways_vertices_pgr v
 WHERE v.id IN (
   SELECT src_id FROM ways_bike UNION SELECT dst_id FROM ways_bike
 );
CREATE INDEX ON _bikeable_vertex_geom USING gist (the_geom);
ANALYZE _bikeable_vertex_geom;

UPDATE anchors a
   SET snap_vertex_id = nearest.id
  FROM (
    SELECT a2.id AS anchor_id,
           (SELECT v.id
              FROM _bikeable_vertex_geom v
             ORDER BY v.the_geom <-> a2.geom
             LIMIT 1) AS id
      FROM anchors a2
  ) AS nearest
 WHERE a.id = nearest.anchor_id;

DROP TABLE _bikeable_vertex_geom;
SQL

# ── STEP 3: recompute-cost-views for new bikeable rows ────────────
step 3 "recompute-cost-views CTAS (filter to NULL cost_views)"
if ! docker compose --profile preprocess run --rm $PG_MOUNT \
       -e PGDATABASE=$PG_DB -e PYTHONUNBUFFERED=1 \
       --entrypoint python3 pgrouting recompute_cost_views_ctas.py; then
  echo "!!! recompute-cost-views FAILED — aborting"; exit 1
fi

# ── STEP 4: rebuild ways_bike ──────────────────────────────────────
step 4 "rebuild ways_bike (drop + create + index)"
if ! docker compose --profile preprocess run --rm $PG_MOUNT \
       -e PGDATABASE=$PG_DB -e PYTHONUNBUFFERED=1 \
       --entrypoint python3 pgrouting build_ways_bike.py; then
  echo "!!! build-ways-bike FAILED — aborting"; exit 1
fi

# ── STEP 5: invalidate graph_multi.npz cache ───────────────────────
step 5 "delete stale graph cache so spts-multi rebuilds"
ls -la data/spt_cache/graph_multi.npz 2>&1 | tail -2
rm -f data/spt_cache/graph_multi.npz
echo "[recovery-b] cache deleted; spts-multi will rebuild on launch"

# ── STEP 6: clean up garbage CZ npz from prior crash ──────────────
step 6 "delete tiny (broken) CZ npz files (< 10 KB)"
N_TINY=$(find data/spt/views/spt -name "*.npz" -size -10k 2>/dev/null | wc -l)
echo "[recovery-b] found $N_TINY tiny npz files to delete"
find data/spt/views/spt -name "*.npz" -size -10k -delete 2>&1 | tail -5
N_REMAINING=$(ls data/spt/views/spt/*.npz 2>/dev/null | wc -l)
echo "[recovery-b] $N_REMAINING healthy AT npz files remain"

# ── STEP 7: restart spts-multi ─────────────────────────────────────
step 7 "spts-multi --profiles views (rebuilds cache + processes ~2,970 anchors)"
if ! $PG_RUN spts-multi --profiles views; then
  echo "!!! spts-multi FAILED — aborting"; exit 1
fi

# ── STEP 8: paired ─────────────────────────────────────────────────
step 8 "paired --profile views (build trunk SQLite DB)"
if ! $PG_RUN paired --profile views; then
  echo "!!! paired FAILED — aborting"; exit 1
fi

# ── STEP 9: verify Graz→Cph ────────────────────────────────────────
step 9 "verify Graz→Cph route on views profile"
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
else
  echo "!!! couldn't resolve Graz/Copenhagen anchor ids"
  exit 1
fi

echo
echo "=== [$(date)] RECOVERY-B POST-INGEST DONE ==="
