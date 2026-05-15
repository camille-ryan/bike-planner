#!/usr/bin/env bash
# Launch the corridor SPT preprocess with per-line timestamps in the log.
#
# Run with `nohup ./launch_corridor.sh </dev/null >/dev/null 2>&1 & disown`
# from the cron poller — this script blocks until docker compose exits.
#
# Each log line gets a `[HH:MM:SS]` prefix at write time via awk.
# PYTHONUNBUFFERED=1 keeps the python side flushing line-by-line so the
# timestamps reflect real activity, not buffer flush moments.
set -u

LOG="/tmp/corridor-$(date +%s).log"
COMPOSE_FILE=/mnt/e/proj/bike/docker-compose.yml

# Corridor priority polyline: Graz, Wien, Brno, Praha, Dresden, Berlin,
# Hamburg, Lübeck, København (lon,lat,lon,lat,...). Anchors close to
# this line are processed first so a usable on-route slice lands early.
PRIORITY_LINE="15.4395,47.0707,16.3725,48.2082,16.6068,49.1951,14.4378,50.0755,13.7373,51.0504,13.405,52.52,9.9937,53.5511,10.6866,53.8654,12.5683,55.6761"

docker compose -f "$COMPOSE_FILE" --profile preprocess run --rm \
    -e PYTHONUNBUFFERED=1 \
    -e "SPT_PRIORITY_LINE=$PRIORITY_LINE" \
    pgrouting spts --profile lht 2>&1 \
  | awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }' \
  > "$LOG" 2>&1
