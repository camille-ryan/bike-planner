#!/usr/bin/env bash
# Full 4-country scenicness re-ingest, bake & publish.
#
# Steps:
#   0. Reset stale bake checkpoint + zero landcover-derived columns
#   1. Landcover ingest (AT/CZ/DE/DK)
#   2. Waterway ingest (AT/CZ/DE/DK)
#   3. Coastline ingest (one-shot)
#   4. DEM download + ingest
#   5. Canopy recompute
#   6. Scenicness bake + incremental overlay publishing (background watcher)
#   7. Recompute cost per multi-axis profile (forest_lover, vineyard_lover, views, water)
#   8. Propagate cost_<profile> columns onto ways_bike
#
# Resumable: every step is idempotent. If interrupted, just re-run.
# The bake (step 6) is the long one (hours); skips tiles already in
# _scenicness_tiles_done. The watcher exits cleanly when the bake exits.
#
# Launch unattended:
#   nohup ./run_4country_full.sh > logs/full-$(date +%Y%m%d-%H%M%S).log 2>&1 &
#
# WSL caveat: host sleep/hibernate kills this orchestrator. For multi-day
# runs, either keep the machine awake or run via systemd.

set -euo pipefail

cd /mnt/e/proj/bike
mkdir -p data/logs logs

LOG=/mnt/e/proj/bike/data/logs/pipeline.log
PG_MOUNT="-v /mnt/e/proj/bike/pgrouting:/app"
COUNTRIES="austria,czech-republic,germany,denmark"
PROFILES=(forest_lover vineyard_lover views water)
TS=$(date +%Y%m%d-%H%M%S)

step() {
  echo
  echo "=== [$(date)] STEP $1: $2 ===" | tee -a "$LOG"
}

# ── STEP 0 ─────────────────────────────────────────────────────────
# TRUNCATE only; skip the column-zeroing UPDATE — the bake covers
# every row's tile anyway (extent = ways_vertices_pgr min/max), so
# the UPDATE was 30+ min of wasted I/O on 131M rows.
step "0/8" "reset bake checkpoint"
docker exec -i bike-postgres psql -U bike -d bike <<'SQL' 2>&1 | tee -a "$LOG"
  TRUNCATE _scenicness_tiles_done;
SQL

# ── STEP 1 ─────────────────────────────────────────────────────────
step "1/8" "landcover-ingest ($COUNTRIES)"
docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
  landcover-ingest --countries "$COUNTRIES" 2>&1 | tee -a "$LOG"

# ── STEP 2 ─────────────────────────────────────────────────────────
step "2/8" "waterway-ingest ($COUNTRIES)"
docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
  waterway-ingest --countries "$COUNTRIES" 2>&1 | tee -a "$LOG"

# ── STEP 3 ─────────────────────────────────────────────────────────
step "3/8" "coastline-ingest (one-shot)"
docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
  coastline-ingest 2>&1 | tee -a "$LOG"

# ── STEP 4 ─────────────────────────────────────────────────────────
step "4/8" "dem-download + dem-ingest"
docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
  dem-download --countries "$COUNTRIES" 2>&1 | tee -a "$LOG"
docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
  dem-ingest 2>&1 | tee -a "$LOG"

# ── STEP 5 ─────────────────────────────────────────────────────────
step "5/8" "canopy-compute (refresh ways.canopy_frac from 4-country landcover)"
docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
  canopy-compute 2>&1 | tee -a "$LOG"

# ── STEP 6 ─────────────────────────────────────────────────────────
step "6/8" "scenicness-bake + incremental overlay publishing"

# Background watcher: rsyncs new XYZ tiles to web/public every 60s.
./watch_overlays.sh >> "$LOG" 2>&1 &
WATCHER_PID=$!
echo "[orchestrator] watcher pid=$WATCHER_PID" | tee -a "$LOG"

# Bounded auto-retry: bake is resumable (skips _scenicness_tiles_done
# entries), so on OOM-kill or container-level death we just relaunch.
BAKE_MAX_ATTEMPTS=${BAKE_MAX_ATTEMPTS:-5}
attempt=1
bake_ok=0
while [ "$attempt" -le "$BAKE_MAX_ATTEMPTS" ]; do
  echo "--- [$(date)] bake attempt $attempt/$BAKE_MAX_ATTEMPTS ---" | tee -a "$LOG"
  set +e
  docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
    scenicness-bake --export-rasters /data/scenicness 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -eq 0 ]; then
    echo "=== [$(date)] bake completed cleanly on attempt $attempt ===" | tee -a "$LOG"
    bake_ok=1
    break
  fi
  echo "!!! [$(date)] bake exited rc=$rc on attempt $attempt; resuming in 30s ===" | tee -a "$LOG"
  attempt=$((attempt + 1))
  sleep 30
done

# Stop the watcher regardless of bake outcome.
kill "$WATCHER_PID" 2>/dev/null || true
wait "$WATCHER_PID" 2>/dev/null || true

if [ "$bake_ok" -ne 1 ]; then
  echo "!!! [$(date)] bake failed $BAKE_MAX_ATTEMPTS times — aborting before publish ===" | tee -a "$LOG"
  exit 1
fi

# Final sync + manifest publish.
step "6/8" "final overlay sync + manifest publish"
mkdir -p web/public/data/scenicness/xyz
rsync -a --update \
  /mnt/e/proj/bike/data/scenicness/xyz/ \
  /mnt/e/proj/bike/web/public/data/scenicness/xyz/ 2>&1 | tee -a "$LOG"
cp -v /mnt/e/proj/bike/data/scenicness/manifest.json \
      /mnt/e/proj/bike/web/public/data/scenicness/manifest.json 2>&1 | tee -a "$LOG"

# ── STEP 7 ─────────────────────────────────────────────────────────
step "7/8" "recompute-cost for ${PROFILES[*]}"
for P in "${PROFILES[@]}"; do
  echo "--- [$(date)] recompute-cost --profile $P ---" | tee -a "$LOG"
  docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
    recompute-cost --profile "$P" 2>&1 | tee -a "$LOG"
done

# ── STEP 8 ─────────────────────────────────────────────────────────
step "8/8" "propagate cost_<profile> columns onto ways_bike"
docker exec -i bike-postgres psql -U bike -d bike <<'SQL' 2>&1 | tee -a "$LOG"
  DO $$
  DECLARE p text;
  BEGIN
    FOREACH p IN ARRAY ARRAY['forest_lover','vineyard_lover','views','water']
    LOOP
      EXECUTE format('ALTER TABLE ways_bike ADD COLUMN IF NOT EXISTS cost_%I real', p);
      EXECUTE format('ALTER TABLE ways_bike ADD COLUMN IF NOT EXISTS reverse_cost_%I real', p);
      EXECUTE format(
        'UPDATE ways_bike b SET cost_%I = w.cost_%I, reverse_cost_%I = w.reverse_cost_%I
           FROM ways w WHERE w.gid = b.gid', p, p, p, p);
    END LOOP;
  END$$;
SQL

echo
echo "=== [$(date)] PIPELINE DONE ($TS) ===" | tee -a "$LOG"
echo "    Verify: open the web app and toggle each scenicness signal." | tee -a "$LOG"
echo "    Verify: SELECT class, country, COUNT(*) FROM landcover GROUP BY class, country;" | tee -a "$LOG"
