#!/usr/bin/env bash
# Chain after scenicness bake: recompute per-edge cost for each scenic
# profile, then propagate the cost columns onto ways_bike for SPT use.
#
# This is the work the user gated before — gate is now removed.

set -euo pipefail

cd /mnt/e/proj/bike
LOG=/mnt/e/proj/bike/data/logs/pipeline.log
PG_MOUNT="-v /mnt/e/proj/bike/pgrouting:/app"

PROFILES=(forest_lover vineyard_lover views water)

# Wait for the scenicness-bake container to exit before starting.
while docker ps --format '{{.Names}}' | grep -q "^bike-pgrouting-run-"; do
  sleep 60
done
echo "=== [$(date)] scenicness bake container exited; chaining recompute-cost ===" | tee -a "$LOG"

for P in "${PROFILES[@]}"; do
  echo "=== [$(date)] recompute-cost --profile $P ===" | tee -a "$LOG"
  docker compose --profile preprocess run --rm $PG_MOUNT pgrouting \
    recompute-cost --profile "$P" 2>&1 | tee -a "$LOG"
done

echo "=== [$(date)] propagate cost_<profile> columns onto ways_bike ===" | tee -a "$LOG"
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

echo "=== [$(date)] scenic prep + cost propagation done. ===" | tee -a "$LOG"
echo "    Next: modify compute_spts_polygon.py to read cost_<profile> column" | tee -a "$LOG"
echo "    and run SPT compute per profile." | tee -a "$LOG"
