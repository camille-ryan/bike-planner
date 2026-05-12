#!/usr/bin/env bash
# Quick reusable DB watcher.
#
# Edits the SQL block below for whatever you want to follow. Re-running
# this script just refreshes the same view; the body is kept in one
# place so the next thing you want to watch is a 30-second edit, not a
# new file. Override DB or interval inline:
#
#   ./watch.sh                       # bike_v2_test, 5s refresh
#   INTERVAL=2 ./watch.sh            # faster refresh
#   DB=bike ./watch.sh               # against production DB
set -u

DB="${DB:-bike_v2_test}"
INTERVAL="${INTERVAL:-5}"

# --- query block --------------------------------------------------------
# Current focus: canopy_frac sweep progress.
#
# Cheap single seq-scan of `ways` — no joins, no subqueries on
# ways_vertices_pgr, so it doesn't acquire row-level locks that can
# deadlock the canopy-compute UPDATE. The numbers it tracks:
#   any_canopy       — edges with any forest overlap recorded
#   majority/full    — edges mostly/fully under canopy
#   max_processed    — highest gid whose canopy_frac is > 0 right now
#   max_gid_overall  — gid_max of `ways`
#   pct_swept        — max_processed / max_gid_overall × 100
#                      (rough sweep estimate; ticks up as the cursor
#                      advances through in-corridor forest hits)
read -r -d '' SQL <<'EOF'
SELECT
  COUNT(*) FILTER (WHERE canopy_frac > 0)                          AS any_canopy,
  COUNT(*) FILTER (WHERE canopy_frac > 0.5)                        AS majority,
  COUNT(*) FILTER (WHERE canopy_frac >= 0.99)                      AS full,
  MAX(gid) FILTER (WHERE canopy_frac > 0)                          AS max_processed,
  MAX(gid)                                                          AS max_gid_overall,
  ROUND(100.0 * (MAX(gid) FILTER (WHERE canopy_frac > 0))::numeric
        / NULLIF(MAX(gid), 0), 1)                                  AS pct_swept
FROM ways;
EOF
# -----------------------------------------------------------------------

# Appends rather than clears so failures and history stay visible. To
# see only the latest row, pipe through tail -3.
trap 'echo "[watch] stopped"; exit 0' INT TERM
printf '[watch] starting — db=%s interval=%ss (Ctrl-C to stop)\n' "$DB" "$INTERVAL"
i=0
while true; do
  i=$((i + 1))
  printf '\n=== iter %d  %s ===\n' "$i" "$(date '+%H:%M:%S')"
  if ! docker exec bike-postgres psql -U bike -d "$DB" -c "$SQL"; then
    echo "[watch] psql failed (exit $?). Will retry in ${INTERVAL}s."
  fi
  sleep "$INTERVAL"
done
