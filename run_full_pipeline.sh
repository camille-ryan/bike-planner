#!/usr/bin/env bash
# Full V3 SPT pipeline orchestrator. Runs the three preprocess stages
# in sequence, fail-fast: if one step errors out, downstream steps
# don't run on bad data.
#
# Stages:
#   1. scenic cost recomputes (~6 hours)  →  run_scenic_costs.sh
#   2. multi-profile per-anchor SPT compute (~50 min, after --force-cache)
#   3. paired SPT / trunk DB per profile (5 profiles)
#
# Each stage has its own per-step logs under logs/. The driver log
# (output of this script) is a high-level transcript with timestamps
# for each stage boundary.
#
# Total wall ≈ 7–9 hours unattended. nohup this and walk away:
#   nohup ./run_full_pipeline.sh > logs/full-pipeline-$(date +%Y%m%d-%H%M%S).log 2>&1 &
#
# To skip stages you've already run, comment them out below or run the
# per-stage scripts directly: run_scenic_costs.sh / run_spts_multi.sh /
# `docker compose run pgrouting paired --profile <P>`.

set -e
cd /mnt/e/proj/bike
mkdir -p logs

TS=$(date +%Y%m%d-%H%M%S)
echo "=== run_full_pipeline.sh start: $(date) ==="

# ── Stage 1/3: scenic cost recomputes ────────────────────────────────
echo
echo "=== [$(date +%H:%M:%S)] stage 1/3: scenic cost recomputes ==="
./run_scenic_costs.sh

# ── Stage 2/3: multi-profile SPT compute ─────────────────────────────
# --force-cache is required because the existing cache was built when
# the scenic cost columns were NULL — the inf sanitiser captured them
# as "no edge". After stage 1 the cost columns have real values, so
# the cache must be regenerated to pick them up.
echo
echo "=== [$(date +%H:%M:%S)] stage 2/3: spts-multi (--force-cache) ==="
LOG_SPTS="logs/spts-multi-${TS}.log"
echo "    log: $LOG_SPTS"
docker compose run --rm -e PGDATABASE=bike_v2_test pgrouting \
  spts-multi --force-cache 2>&1 | tee "$LOG_SPTS"

# ── Stage 3/3: paired SPT / trunk DB per profile ─────────────────────
# Each profile gets its own data/spt/<profile>/paired_trunks.db.
# Uses V1 defaults (Graz→Cph polyline, 80 km anchor inclusion radius,
# pruning on, no per-pair npzs). The trunk DB is the canonical
# routing artifact; per-pair npzs are inspection-only.
echo
echo "=== [$(date +%H:%M:%S)] stage 3/3: paired SPT / trunk DB per profile ==="
PROFILES="direct vineyard_lover forest_lover views water"
for profile in $PROFILES; do
  LOG_PAIRED="logs/paired-${profile}-${TS}.log"
  echo
  echo "=== [$(date +%H:%M:%S)] paired --profile $profile  →  $LOG_PAIRED ==="
  docker compose run --rm -e PGDATABASE=bike_v2_test pgrouting \
    paired --profile "$profile" 2>&1 | tee "$LOG_PAIRED"
done

echo
echo "=== run_full_pipeline.sh DONE: $(date) ==="
