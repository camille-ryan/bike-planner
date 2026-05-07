#!/usr/bin/env bash
# Live status display for the corridor SPT preprocess.
#
# Usage:
#   ./watch_corridor.sh              # refresh every 5 sec
#   ./watch_corridor.sh 10           # refresh every 10 sec
#
# Shows: container status, npz progress, recent rate, ETA, last log
# line, container memory + CPU.  Exits with a beep when city_graph.json
# appears (i.e. preprocess complete).

set -u

INTERVAL="${1:-5}"
TARGET=3212
SPT_DIR=/mnt/e/proj/bike/data/spt/lht
DONE_FILE="$SPT_DIR/city_graph.json"

# Rolling history: timestamp + npz count, last 10 samples.
HIST_TS=()
HIST_NPZ=()

while true; do
  now_ts=$(date +%s)
  npz=$(ls "$SPT_DIR"/spt/*.npz 2>/dev/null | wc -l)

  HIST_TS+=("$now_ts")
  HIST_NPZ+=("$npz")
  if [ "${#HIST_TS[@]}" -gt 10 ]; then
    HIST_TS=("${HIST_TS[@]:1}")
    HIST_NPZ=("${HIST_NPZ[@]:1}")
  fi

  # Compute rate over the rolling window.
  n=${#HIST_TS[@]}
  rate_per_min="?"
  eta_human="?"
  if [ "$n" -ge 2 ]; then
    dt=$(( ${HIST_TS[$((n-1))]} - ${HIST_TS[0]} ))
    dn=$(( ${HIST_NPZ[$((n-1))]} - ${HIST_NPZ[0]} ))
    if [ "$dt" -gt 0 ] && [ "$dn" -gt 0 ]; then
      rate_per_min=$(awk -v dn="$dn" -v dt="$dt" 'BEGIN { printf "%.1f", dn * 60 / dt }')
      remaining=$((TARGET - npz))
      if [ "$remaining" -gt 0 ]; then
        eta_sec=$(awk -v r="$remaining" -v dn="$dn" -v dt="$dt" \
          'BEGIN { printf "%d", r * dt / dn }')
        eta_human=$(printf '%dh%02dm' $((eta_sec/3600)) $(((eta_sec/60)%60)))
      fi
    fi
  fi

  # Container status.
  container=$(docker ps --filter "name=pgrouting-run" --format "{{.Names}}\t{{.Status}}" 2>/dev/null | head -1)
  postgres=$(docker stats --no-stream --format "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" bike-postgres 2>/dev/null | head -1)
  pgrouting_stats=""
  if [ -n "$container" ]; then
    cname=$(echo "$container" | cut -f1)
    pgrouting_stats=$(docker stats --no-stream --format "{{.CPUPerc}}\t{{.MemUsage}}" "$cname" 2>/dev/null | head -1)
  fi

  # Latest log line.
  latest_log=$(ls -t /tmp/corridor-*.log 2>/dev/null | head -1)
  log_tail=""
  if [ -n "$latest_log" ]; then
    log_tail=$(tail -3 "$latest_log" | sed 's/^/  /')
  fi

  clear
  echo "=== Corridor SPT preprocess monitor (refresh every ${INTERVAL}s, Ctrl-C to quit) ==="
  echo
  printf "  npz:   %d / %d   (%.1f%%)\n" "$npz" "$TARGET" \
    "$(awk -v n="$npz" -v t="$TARGET" 'BEGIN { printf "%.1f", n*100/t }')"
  printf "  rate:  %s SPTs/min   ETA: %s\n" "$rate_per_min" "$eta_human"
  echo
  if [ -n "$container" ]; then
    echo "  container: $container"
  else
    echo "  container: NOT RUNNING (cron should relaunch within 10 min)"
  fi
  if [ -n "$pgrouting_stats" ]; then
    echo "  pgrouting:  $pgrouting_stats" | tr '\t' ' '
  fi
  if [ -n "$postgres" ]; then
    echo "  postgres:   $postgres" | tr '\t' ' '
  fi
  echo
  echo "  recent log lines:"
  if [ -n "$log_tail" ]; then
    echo "$log_tail"
  else
    echo "  (no log file found at /tmp/corridor-*.log)"
  fi
  echo
  echo "  log file: ${latest_log:-none}"
  echo "  press Ctrl-C to exit; tail manually with: tail -f $latest_log"

  if [ -f "$DONE_FILE" ]; then
    echo
    echo "  *** DONE: $DONE_FILE exists ***"
    printf '\a'   # terminal bell
    sleep 5
    break
  fi

  sleep "$INTERVAL"
done
