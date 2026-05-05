"""Subset country PBFs into POI / protected-area / bike-route extracts via osmium tags-filter."""
import subprocess
from pathlib import Path

from config import POI_FILTERS, PROTECTED_AREA_FILTERS, BIKE_ROUTE_FILTERS, ANCHOR_FILTERS, DATA_DIR


def filter_pbf(input_pbf: Path, output_pbf: Path, filters: list[str]) -> Path:
    output_pbf.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["osmium", "tags-filter", "--overwrite", str(input_pbf), *filters,
           "-o", str(output_pbf)]
    print(f"[osmium] {input_pbf.name} -> {output_pbf.name}")
    subprocess.run(cmd, check=True)
    return output_pbf


def run(country_pbfs: list[Path]) -> list[dict]:
    out_dir = DATA_DIR / "pois"
    results = []
    for pbf in country_pbfs:
        name = pbf.stem.replace("-latest.osm", "")
        results.append({
            "country":   name,
            "pois":      filter_pbf(pbf, out_dir / f"{name}-pois.osm.pbf",      POI_FILTERS),
            "protected": filter_pbf(pbf, out_dir / f"{name}-protected.osm.pbf", PROTECTED_AREA_FILTERS),
            "routes":    filter_pbf(pbf, out_dir / f"{name}-routes.osm.pbf",    BIKE_ROUTE_FILTERS),
            "anchors":   filter_pbf(pbf, out_dir / f"{name}-anchors.osm.pbf",   ANCHOR_FILTERS),
        })
    return results
