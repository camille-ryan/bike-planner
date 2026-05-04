"""Download BRouter segment (.rd5) tiles for the corridor."""
from pathlib import Path

from config import BROUTER_TILES, BROUTER_BASE, DATA_DIR
from download_util import fetch


def run(tiles: list[str] | None = None) -> list[Path]:
    tiles = tiles or BROUTER_TILES
    out_dir = DATA_DIR / "brouter"
    paths = []
    for t in tiles:
        url = f"{BROUTER_BASE}/{t}.rd5"
        out = out_dir / f"{t}.rd5"
        paths.append(fetch(url, out, label="brouter"))
    return paths


if __name__ == "__main__":
    run()
