#!/usr/bin/env bash
# Re-extract data/landcover/<country>-landuse.osm.pbf from the full
# country PBFs using `osmium tags-filter` (which correctly preserves
# multipolygon relation members so big multipolygon forests like
# Schwarzwald assemble correctly downstream).
#
# Replaces the under-extracted landuse PBFs in place (rename old to .bak
# first).
#
# Run from project root.

set -euo pipefail

cd /mnt/e/proj/bike

OSM_DIR=data/osm
LC_DIR=data/landcover
COUNTRIES=(austria czech-republic germany denmark)

for c in "${COUNTRIES[@]}"; do
  IN=$OSM_DIR/${c}-latest.osm.pbf
  OUT=$LC_DIR/${c}-landuse.osm.pbf

  if [ ! -f "$IN" ]; then
    echo "!!! missing input $IN — skipping" >&2
    continue
  fi

  if [ -f "$OUT" ]; then
    mv "$OUT" "${OUT}.bak"
    echo "[extract] backed up $OUT → ${OUT}.bak"
  fi

  echo "=== [$(date)] osmium tags-filter $c ==="
  # nwr/landuse — any node, way, or relation with landuse=*
  # nwr/natural — any node, way, or relation with natural=*
  # waterway too so future signals can leverage it (we currently ingest
  # waterways from full PBFs separately, but free to include here).
  docker compose --profile preprocess run --rm \
    -v /mnt/e/proj/bike/data:/data \
    --entrypoint osmium pgrouting \
    tags-filter \
    "/data/osm/${c}-latest.osm.pbf" \
    "nwr/landuse" \
    "nwr/natural" \
    -o "/data/landcover/${c}-landuse.osm.pbf" \
    --overwrite
done

echo
echo "=== new landuse PBF sizes ==="
ls -lah $LC_DIR/*-landuse.osm.pbf
