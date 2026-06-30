#!/usr/bin/env bash
# Full scenic pipeline, resumable, safe to run for days.
#   1. scenicness-bake --export-rasters  (per-tile committed → resumable;
#      also emits PNG overlays for the web app)
#   2. publish PNGs + manifest to web/public/data/scenicness
#   3. recompute-cost for each scenic profile (per-profile cost columns on ways)
#   4. propagate cost_<profile> columns onto ways_bike
#
# Idempotent / resumable: if interrupted (incl. WSL restart), just re-run
# this script. The bake skips tiles already in _scenicness_tiles_done;
# recompute-cost re-runs its profiles (cheap-ish, tile-based); the
# propagation UPDATE is naturally idempotent.

set -euo pipefail

cd /mnt/e/proj/bike
LOG=/mnt/e/proj/bike/data/logs/pipeline.log
PG_MOUNT="-v /mnt/e/proj/bike/pgrouting:/app"
PROFILES=(forest_lover vineyard_lover views water)

echo "=== [$(date)] SCENIC 1/4 bake (+ raster export, resumable) ===" | tee -a "$LOG"
# Bounded auto-retry: the bake is resumable (skips tiles in
# _scenicness_tiles_done), so if a container-level death (e.g. OOM-kill)
# drops it while this script/host stays up, just relaunch and it picks up
# where it left off. NOTE: this does NOT survive a full WSL/host reboot,
# which kills this orchestrator process itself.
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
if [ "$bake_ok" -ne 1 ]; then
  echo "!!! [$(date)] bake failed $BAKE_MAX_ATTEMPTS times — aborting before publish ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] SCENIC 2/4 publish PNGs to web ===" | tee -a "$LOG"
mkdir -p web/public/data/scenicness
cp -v data/scenicness/*.png data/scenicness/manifest.json \
      web/public/data/scenicness/ 2>&1 | tee -a "$LOG"

echo "=== [$(date)] SCENIC 3/4 recompute-cost per scenic profile ===" | tee -a "$LOG"
for P in "${PROFILES[@]}"; do
  echo "--- [$(date)] recompute-cost --profile $P ---" | tee -a "$LOG"
  docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
    recompute-cost --profile "$P" 2>&1 | tee -a "$LOG"
done

echo "=== [$(date)] SCENIC 4/4 propagate cost_<profile> onto ways_bike ===" | tee -a "$LOG"
SQL=""
for P in "${PROFILES[@]}"; do
  SQL="$SQL
    ALTER TABLE ways_bike ADD COLUMN IF NOT EXISTS cost_${P} real;
    ALTER TABLE ways_bike ADD COLUMN IF NOT EXISTS reverse_cost_${P} real;
    UPDATE ways_bike b SET cost_${P} = w.cost_${P}, reverse_cost_${P} = w.reverse_cost_${P}
      FROM ways w WHERE w.gid = b.gid;
  "
done
docker exec bike-postgres psql -U bike -d bike -c "$SQL" 2>&1 | tee -a "$LOG"

echo "=== [$(date)] SCENIC PIPELINE DONE ===" | tee -a "$LOG"
echo "    ways scenicness columns populated; PNGs published; per-profile" | tee -a "$LOG"
echo "    cost columns on ways + ways_bike. Next: scenic-profile SPT compute" | tee -a "$LOG"
echo "    (needs compute_spts_polygon.py cost-column parameterization)." | tee -a "$LOG"
