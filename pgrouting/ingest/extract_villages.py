"""Extract place=village (and place=hamlet for sparse countries) nodes
from a PBF into a geojsonseq sidecar, matching the format select_anchors
already consumes (/data/osm/<country>-villages.geojsonseq).

Run inside the pgrouting container:
    docker compose --profile preprocess run --rm \\
      -v /mnt/e/proj/bike/pgrouting:/app \\
      --entrypoint python3 pgrouting /app/extract_villages.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import osmium


PBFS = [
    "austria-latest.osm.pbf",
    "czech-republic-latest.osm.pbf",
    "germany-latest.osm.pbf",
    "denmark-latest.osm.pbf",
]
OSM_DIR = Path("/data/osm")


def _stem(pbf_name: str) -> str:
    # "austria-latest.osm.pbf" -> "austria"
    return pbf_name.split("-")[0]


def extract_one(pbf: Path, out: Path) -> int:
    t0 = time.time()
    fp = osmium.FileProcessor(str(pbf))
    n = 0
    with out.open("w") as f:
        for obj in fp:
            if not obj.is_node():
                continue
            tags = obj.tags
            if tags.get("place") != "village":
                continue
            try:
                lon = obj.location.lon
                lat = obj.location.lat
            except (osmium.InvalidLocationError, RuntimeError):
                continue
            props = {
                "name":       tags.get("name", "") or "",
                "place":      "village",
                "population": tags.get("population", "") or "",
                "wikidata":   tags.get("wikidata", "") or "",
            }
            # Strip empty strings to match the original austria file's style
            props = {k: v for k, v in props.items() if v}
            feature = {
                "type": "Feature",
                "id": f"node/{obj.id}",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": props,
            }
            f.write(json.dumps(feature, separators=(",", ":"), ensure_ascii=False))
            f.write("\n")
            n += 1
    print(f"[villages] {pbf.name}: wrote {n:,} village nodes -> {out.name} "
          f"in {time.time()-t0:.1f}s", flush=True)
    return n


def main() -> None:
    total = 0
    for pbf_name in PBFS:
        pbf = OSM_DIR / pbf_name
        if not pbf.exists():
            print(f"[villages] SKIP {pbf_name} (not found)", flush=True)
            continue
        out = OSM_DIR / f"{_stem(pbf_name)}-villages.geojsonseq"
        if out.exists() and out.stat().st_size > 0:
            print(f"[villages] SKIP {out.name} (already exists, "
                  f"{out.stat().st_size:,} bytes)", flush=True)
            total += sum(1 for _ in out.open())
            continue
        total += extract_one(pbf, out)
    print(f"[villages] DONE — {total:,} villages total", flush=True)


if __name__ == "__main__":
    main()
