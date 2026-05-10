#!/usr/bin/env bash
# Resume stage 2 only (paired trunk DB) after the bounded-LRU cache fix.
# Stage 1 artifacts (3,212 SPTs + city_graph.json) are already on disk
# and are reused; build_paired_corridor.py skips pairs already in
# paired_trunks.db.
#
# Run with:
#   nohup ./launch_resume_paired.sh </dev/null >/dev/null 2>&1 & disown
set -u
LOG="/tmp/resume-$(date +%s).log"
COMPOSE_FILE=/mnt/e/proj/bike/docker-compose.yml

echo "[stage2] resume starting at $(date)" > "$LOG"
set -o pipefail
docker compose -f "$COMPOSE_FILE" --profile preprocess run --rm \
    -e PYTHONUNBUFFERED=1 \
    pgrouting paired --profile lht 2>&1 \
  | awk '{ print strftime("[%H:%M:%S]"), "[stage2]", $0; fflush() }' \
  >> "$LOG"
rc=${PIPESTATUS[0]}
set +o pipefail
echo "[done] $(date) rc=$rc" >> "$LOG"
exit "$rc"
