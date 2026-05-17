"""Download national GTFS feeds for the railway-ingest pipeline.

Each entry in `_FEEDS` is a country → direct-download URL for the
canonical static GTFS zip. URLs occasionally rotate; if a fetch starts
returning 404 the upstream portal page is the source of truth — see
references in docstrings below.

Files land in `data/gtfs/<country>-gtfs.zip`. Re-runs are idempotent
unless `--force` is passed.
"""
from __future__ import annotations
import time
from pathlib import Path
from urllib.request import Request, urlopen

import config


# Direct GTFS zip URLs per country.
# Austria: ÖBB feed mirrored at https://data.oebb.at/de/datensaetze~soll-fahrplan-gtfs~
#   The portal page wraps the file behind a click-through but the static
#   asset URL is stable across reissues (the trailing year segment is the
#   first publication year of the validity window, NOT the download date).
_FEEDS: dict[str, str] = {
    "austria": "https://static.web.oebb.at/open-data/soll-fahrplan-gtfs/GTFS_OP_2025_obb.zip",
}


def feed_path(country: str) -> Path:
    return config.DATA_DIR / "gtfs" / f"{country}-gtfs.zip"


def download(country: str, force: bool = False) -> Path:
    """Fetch the GTFS zip for one country into `data/gtfs/`. Returns the
    local path. Skips download when the file already exists unless
    `force=True`."""
    if country not in _FEEDS:
        raise SystemExit(
            f"no GTFS feed configured for {country}; "
            f"available: {sorted(_FEEDS)}"
        )
    url = _FEEDS[country]
    out_path = feed_path(country)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / 1e6
        print(f"[gtfs] {country}: {out_path} exists ({size_mb:.1f} MB), "
              f"skipping (use --force to redownload)")
        return out_path
    print(f"[gtfs] {country}: downloading {url}")
    t0 = time.time()
    req = Request(url, headers={"User-Agent": "bike-routing-ingest/1.0"})
    with urlopen(req, timeout=120) as r:
        data = r.read()
    out_path.write_bytes(data)
    print(f"[gtfs] {country}: wrote {out_path} ({len(data) / 1e6:.1f} MB) "
          f"in {time.time()-t0:.1f}s")
    return out_path
