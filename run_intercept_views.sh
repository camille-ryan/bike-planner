#!/usr/bin/env bash
# Watches the in-flight resume-v2 orchestrator log for the moment
# `recompute-cost --profile views` is about to start (i.e. vineyard_lover
# just finished). When it fires:
#   1. Kill the resume-v2 orchestrator + its child containers
#   2. Launch run_post_vineyard.sh which does:
#        - derive cost_views, cost_water from cost_vineyard_lover via SQL
#        - build ways_bike + propagate
#        - spts-multi + paired × 4 + verify
#
# Idea: the resume-v2 orchestrator would otherwise spend ~20h re-running
# views + water. Skipping them via algebraic derivation saves ~20h.

set -u

cd /mnt/e/proj/bike
RESUME_LOG=/mnt/e/proj/bike/logs/resume-v2-20260601-003951.log
TRIGGER_LINE="recompute-cost --profile views"
POST_LOG=/mnt/e/proj/bike/logs/post-vl-$(date +%Y%m%d-%H%M%S).log
INTERCEPT_LOG=/mnt/e/proj/bike/logs/intercept-$(date +%Y%m%d-%H%M%S).log

exec > >(tee -a "$INTERCEPT_LOG") 2>&1

echo "[intercept] starting; watching $RESUME_LOG for '$TRIGGER_LINE'"

# tail -n0 -F: only emit lines added after we start; -F survives log rotation
tail -n0 -F "$RESUME_LOG" | while IFS= read -r line; do
  if echo "$line" | grep -qF "$TRIGGER_LINE"; then
    echo "[intercept] $(date) TRIGGER: $line"
    echo "[intercept] killing resume-v2 orchestrator + bake/recompute containers..."
    pkill -9 -f run_promote_v2test_resume 2>&1 || true
    pkill -9 -f "main.py recompute-cost" 2>&1 || true
    pkill -9 -f watch_overlays 2>&1 || true
    # Stop any in-flight pgrouting container too
    docker ps --format '{{.Names}}' | grep -E 'pgrouting-run' | while read c; do
      echo "[intercept]   stopping container $c"
      docker stop "$c" >/dev/null 2>&1 || true
    done
    sleep 3
    echo "[intercept] launching run_post_vineyard.sh → $POST_LOG"
    nohup /mnt/e/proj/bike/run_post_vineyard.sh > "$POST_LOG" 2>&1 &
    disown
    echo "[intercept] done — tail script PID $!"
    exit 0
  fi
done
