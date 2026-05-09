#!/usr/bin/env bash
# Live status display for the SPT + paired-trunk preprocess pipeline.
#
# Usage:
#   ./watch_corridor.sh              # refresh every 5 sec
#   ./watch_corridor.sh 10           # refresh every 10 sec
#
# Detects which stage is active (Stage 1 = per-anchor SPTs, Stage 2 =
# corridor-wide paired-trunk DB) and shows the relevant counters, rate,
# and ETA. Picks up either /tmp/rebuild-*.log (full launch_rebuild.sh)
# or /tmp/corridor-*.log (legacy launch_corridor.sh). Exits when
# stage 2 finishes (paired_trunks.db present and stable).

set -u

INTERVAL="${1:-5}"
SPT_TARGET=3212
SPT_DIR=/mnt/e/proj/bike/data/spt/lht
SPT_NPZ_DIR="$SPT_DIR/spt"
TRUNK_DB="$SPT_DIR/paired_trunks.db"
CITY_GRAPH="$SPT_DIR/city_graph.json"

# Rolling history for rate / ETA. Two parallel rings — one for stage 1
# (npz count), one for stage 2 (trunk DB row count). Decided by which
# stage is active; stale rings get reset on stage transition.
HIST_TS=()
HIST_VAL=()
LAST_STAGE=""

reset_history() {
  HIST_TS=()
  HIST_VAL=()
}

while true; do
  now_ts=$(date +%s)

  # Determine current stage.
  spt_npz=$(ls "$SPT_NPZ_DIR"/*.npz 2>/dev/null | wc -l)
  has_city_graph=0
  [ -f "$CITY_GRAPH" ] && has_city_graph=1
  has_trunk=0
  [ -f "$TRUNK_DB" ] && has_trunk=1

  if [ "$spt_npz" -lt "$SPT_TARGET" ]; then
    stage=1
    stage_label="Stage 1: per-anchor SPTs"
    cur_val="$spt_npz"
    cur_target="$SPT_TARGET"
  elif [ "$has_trunk" -eq 0 ] || [ "$has_city_graph" -eq 0 ]; then
    stage=1
    stage_label="Stage 1: post-SPT (city_graph + ferry edges)"
    cur_val="$spt_npz"
    cur_target="$SPT_TARGET"
  else
    stage=2
    stage_label="Stage 2: corridor paired-trunk DB"
    if [ -f "$TRUNK_DB" ]; then
      cur_val=$(sqlite3 "$TRUNK_DB" "SELECT COUNT(*) FROM trunks" 2>/dev/null || echo 0)
      pair_count=$(sqlite3 "$TRUNK_DB" \
        "SELECT COUNT(*) FROM (SELECT DISTINCT src_city, dst_city FROM trunks)" \
        2>/dev/null || echo 0)
    else
      cur_val=0
      pair_count=0
    fi
    cur_target="?"   # not knowable up-front for stage 2
  fi

  if [ "$stage" != "$LAST_STAGE" ]; then
    reset_history
    LAST_STAGE="$stage"
  fi

  HIST_TS+=("$now_ts")
  HIST_VAL+=("$cur_val")
  if [ "${#HIST_TS[@]}" -gt 12 ]; then
    HIST_TS=("${HIST_TS[@]:1}")
    HIST_VAL=("${HIST_VAL[@]:1}")
  fi

  # Rate over the rolling window.
  n=${#HIST_TS[@]}
  rate_per_min="?"
  eta_human="?"
  if [ "$n" -ge 2 ]; then
    dt=$(( ${HIST_TS[$((n-1))]} - ${HIST_TS[0]} ))
    dn=$(( ${HIST_VAL[$((n-1))]} - ${HIST_VAL[0]} ))
    if [ "$dt" -gt 0 ] && [ "$dn" -gt 0 ]; then
      rate_per_min=$(awk -v dn="$dn" -v dt="$dt" 'BEGIN { printf "%.1f", dn * 60 / dt }')
      if [ "$stage" = "1" ] && [ "$cur_target" != "?" ]; then
        remaining=$((cur_target - cur_val))
        if [ "$remaining" -gt 0 ]; then
          eta_sec=$(awk -v r="$remaining" -v dn="$dn" -v dt="$dt" \
            'BEGIN { printf "%d", r * dt / dn }')
          eta_human=$(printf '%dh%02dm' $((eta_sec/3600)) $(((eta_sec/60)%60)))
        fi
      fi
    fi
  fi

  # Containers + stats.
  container=$(docker ps --filter "name=pgrouting-run" --format "{{.Names}}\t{{.Status}}" 2>/dev/null | head -1)
  postgres=$(docker stats --no-stream --format "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" bike-postgres 2>/dev/null | head -1)
  pgrouting_stats=""
  if [ -n "$container" ]; then
    cname=$(echo "$container" | cut -f1)
    pgrouting_stats=$(docker stats --no-stream --format "{{.CPUPerc}}\t{{.MemUsage}}" "$cname" 2>/dev/null | head -1)
  fi

  # Pick the freshest log file (rebuild-*.log preferred; corridor-*.log fallback).
  latest_log=$(ls -t /tmp/rebuild-*.log /tmp/corridor-*.log 2>/dev/null | head -1)
  log_tail=""
  log_mtime=""
  log_age=""
  if [ -n "$latest_log" ]; then
    log_mtime_epoch=$(stat -c %Y "$latest_log" 2>/dev/null)
    if [ -n "$log_mtime_epoch" ]; then
      log_mtime=$(date -d "@$log_mtime_epoch" '+%H:%M:%S')
      age_sec=$((now_ts - log_mtime_epoch))
      log_age=$(printf '%dm%02ds' $((age_sec/60)) $((age_sec%60)))
    fi
    prefix="  [${log_mtime:-??:??:??}~] "
    log_tail=$(tail -4 "$latest_log" \
      | awk -v p="$prefix" '/^\[[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\]/ { print "  " $0; next } { print p $0 }')
  fi

  # Latest npz mtime (Stage 1) — answers "is per-anchor Dijkstra still completing?"
  latest_npz=$(ls -t "$SPT_NPZ_DIR"/*.npz 2>/dev/null | head -1)
  npz_mtime=""
  npz_age=""
  if [ -n "$latest_npz" ]; then
    npz_mtime_epoch=$(stat -c %Y "$latest_npz" 2>/dev/null)
    if [ -n "$npz_mtime_epoch" ]; then
      npz_mtime=$(date -d "@$npz_mtime_epoch" '+%H:%M:%S')
      age_sec=$((now_ts - npz_mtime_epoch))
      npz_age=$(printf '%dm%02ds' $((age_sec/60)) $((age_sec%60)))
    fi
  fi

  # Trunk DB size (Stage 2 progress signal).
  trunk_size_mb=""
  if [ "$has_trunk" -eq 1 ]; then
    trunk_size_mb=$(du -m "$TRUNK_DB" 2>/dev/null | cut -f1)
  fi

  clear
  echo "=== Pipeline monitor — $stage_label (refresh ${INTERVAL}s, Ctrl-C to quit) ==="
  echo
  if [ "$stage" = "1" ]; then
    pct=$(awk -v n="$cur_val" -v t="$cur_target" 'BEGIN { printf "%.1f", n*100/t }')
    printf "  SPT npz:   %d / %d   (%s%%)\n" "$cur_val" "$cur_target" "$pct"
    printf "  rate:      %s SPTs/min   ETA stage 1: %s\n" "$rate_per_min" "$eta_human"
  else
    printf "  trunk pairs: %d   trunk rows: %d\n" "$pair_count" "$cur_val"
    printf "  rate:        %s rows/min\n" "$rate_per_min"
    if [ -n "$trunk_size_mb" ]; then
      printf "  DB size:     %s MB\n" "$trunk_size_mb"
    fi
  fi
  echo
  if [ -n "$container" ]; then
    echo "  container: $container"
  else
    echo "  container: NOT RUNNING"
  fi
  if [ -n "$pgrouting_stats" ]; then
    echo "  pgrouting:  $pgrouting_stats" | tr '\t' ' '
  fi
  if [ -n "$postgres" ]; then
    echo "  postgres:   $postgres" | tr '\t' ' '
  fi
  echo
  printf "  last log activity: %s  (%s ago)\n" "${log_mtime:-?}" "${log_age:-?}"
  if [ "$stage" = "1" ]; then
    printf "  last npz written:  %s  (%s ago)\n" "${npz_mtime:-?}" "${npz_age:-?}"
  fi
  echo
  echo "  recent log lines:"
  if [ -n "$log_tail" ]; then
    echo "$log_tail"
  else
    echo "  (no log file found at /tmp/rebuild-*.log or /tmp/corridor-*.log)"
  fi
  echo
  echo "  log file: ${latest_log:-none}"
  echo "  press Ctrl-C to exit; tail manually with: tail -f $latest_log"

  # Done detection: stage 2 has finished when the rebuild log shows
  # the [done] marker line OR the trunk DB stops growing for two
  # successive samples while the runner container is gone.
  done_marker=""
  if [ -n "$latest_log" ]; then
    done_marker=$(tail -1 "$latest_log" 2>/dev/null | grep "^\[done\]")
  fi
  if [ -n "$done_marker" ]; then
    echo
    echo "  *** DONE: $done_marker ***"
    printf '\a'
    sleep 5
    break
  fi

  sleep "$INTERVAL"
done
