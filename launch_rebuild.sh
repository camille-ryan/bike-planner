#!/usr/bin/env bash
# Rebuild SPTs (30 km uniform with is_frontier + topology + CSR) and
# the corridor-wide paired trunk DB. Logs with per-line timestamps.
#
# Run with:
#   nohup ./launch_rebuild.sh </dev/null >/dev/null 2>&1 & disown
set -u
LOG="/tmp/rebuild-$(date +%s).log"
COMPOSE_FILE=/mnt/e/proj/bike/docker-compose.yml
PRIORITY_LINE="15.4395,47.0707,16.3725,48.2082,16.6068,49.1951,14.4378,50.0755,13.7373,51.0504,13.405,52.52,9.9937,53.5511,10.6866,53.8654,12.5683,55.6761"

run_logged() {
  # Run a command, pipe through awk for [HH:MM:SS] timestamps, propagate
  # the underlying exit code from PIPESTATUS.
  local stage="$1"; shift
  set -o pipefail
  "$@" 2>&1 \
    | awk -v s="$stage" '{ print strftime("[%H:%M:%S]"), "["s"]", $0; fflush() }' \
    >> "$LOG"
  local rc=${PIPESTATUS[0]}
  set +o pipefail
  return "$rc"
}

# Stage 1: SPTs (30 km uniform, is_frontier, topology, city_graph with
# ferry edges). ~80 minutes.
echo "[stage1] starting at $(date)" > "$LOG"
run_logged stage1 \
    docker compose -f "$COMPOSE_FILE" --profile preprocess run --rm \
        -e PYTHONUNBUFFERED=1 \
        -e "SPT_PRIORITY_LINE=$PRIORITY_LINE" \
        pgrouting spts --profile lht
rc1=$?
if [ "$rc1" -ne 0 ]; then
    echo "[stage1] FAILED with exit code $rc1 — not running stage 2." >> "$LOG"
    echo "[done] $(date)" >> "$LOG"
    exit "$rc1"
fi

# Stage 2: corridor-wide paired SPTs into the trunk DB. ~2-3 hours.
echo "[stage2] starting at $(date)" >> "$LOG"
run_logged stage2 \
    docker compose -f "$COMPOSE_FILE" --profile preprocess run --rm \
        -e PYTHONUNBUFFERED=1 \
        pgrouting paired --profile lht
rc2=$?
if [ "$rc2" -ne 0 ]; then
    echo "[stage2] FAILED with exit code $rc2." >> "$LOG"
fi

echo "[done] $(date)" >> "$LOG"
exit "$rc2"
