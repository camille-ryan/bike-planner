#!/usr/bin/env bash
# Periodic z4-z11 overview rebuild + web publish, every 2 hours.
#
# Runs alongside the bake so the web app's wide-view shows fresh data as
# new z12 tiles are baked, instead of waiting for bake completion (which
# is when build_overviews normally fires).
#
# Safe to run concurrently with the bake: pure file I/O on /data/scenicness/xyz,
# no postgres lock contention.
#
# Launch:
#   nohup ./run_periodic_overviews.sh > logs/overviews-$(date +%Y%m%d-%H%M%S).log 2>&1 &

set -u
cd /mnt/e/proj/bike

LOG=/mnt/e/proj/bike/data/logs/overviews.log
PG_MOUNT="-v /mnt/e/proj/bike/pgrouting:/app"
INTERVAL="${OVERVIEW_INTERVAL:-7200}"  # 2 hours

# Quiet exit on Ctrl-C / kill from orchestrator cleanup
trap 'echo "[overviews] caught signal, exiting"; exit 0' INT TERM

mkdir -p data/logs

while true; do
  echo "" | tee -a "$LOG"
  echo "=== [$(date)] periodic overview rebuild ===" | tee -a "$LOG"

  docker compose --profile preprocess run --rm $PG_MOUNT \
    -e PYTHONUNBUFFERED=1 --entrypoint python3 pgrouting \
    /app/refresh_overviews.py 2>&1 | tee -a "$LOG"

  echo "[overviews] rsync xyz → web/public" | tee -a "$LOG"
  rsync -a --update \
    /mnt/e/proj/bike/data/scenicness/xyz/ \
    /mnt/e/proj/bike/web/public/data/scenicness/xyz/ 2>&1 | tee -a "$LOG"

  echo "[overviews] sleeping ${INTERVAL}s" | tee -a "$LOG"
  sleep "$INTERVAL"
done
