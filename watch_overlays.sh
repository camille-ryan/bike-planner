#!/usr/bin/env bash
# Background watcher: rsync newly-baked XYZ tiles from
# data/scenicness/xyz/ to web/public/data/scenicness/xyz/ every 60s.
#
# Pairs with run_4country_full.sh step 6 (scenicness bake). The web app's
# existing manifest already points to xyz/<signal>/{z}/{x}/{y}.png, so
# new tiles get picked up automatically as they appear.
#
# Foreground; intended to be backgrounded by the orchestrator with &.
# Trap term so kill from the orchestrator cleans up cleanly.

set -u
cd /mnt/e/proj/bike

SRC=/mnt/e/proj/bike/data/scenicness/xyz/
DST=/mnt/e/proj/bike/web/public/data/scenicness/xyz/
INTERVAL="${WATCH_INTERVAL:-60}"

mkdir -p "$DST"

trap 'echo "[watch] caught signal, exiting"; exit 0' INT TERM

echo "[watch] starting; src=$SRC dst=$DST interval=${INTERVAL}s"
while true; do
  if [ -d "$SRC" ]; then
    rsync -a --update "$SRC" "$DST" 2>/dev/null || true
  fi
  sleep "$INTERVAL"
done
