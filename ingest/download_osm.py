"""Download Geofabrik country PBFs."""
from pathlib import Path

from config import COUNTRIES, GEOFABRIK_BASE, DATA_DIR
from download_util import fetch


def run(countries: list[str] | None = None) -> list[Path]:
    countries = countries or COUNTRIES
    out_dir = DATA_DIR / "osm"
    paths = []
    for c in countries:
        url = f"{GEOFABRIK_BASE}/{c}-latest.osm.pbf"
        out = out_dir / f"{c}-latest.osm.pbf"
        paths.append(fetch(url, out, label="osm"))
    return paths


if __name__ == "__main__":
    run()
