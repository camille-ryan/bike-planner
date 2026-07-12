"""Download national GTFS feeds for the railway-ingest pipeline.

Each entry in `_FEEDS` is a country → feed-descriptor. A descriptor can
be either:
  - a URL string (direct zip download, default UA);
  - a dict `{"url": ..., "ua": ...}` (direct zip, custom UA for servers
    that reject non-browser UAs); or
  - a dict `{"assemble": [(url, name), ...]}` (assemble a zip from
    loose files hosted individually — used for repos that publish GTFS
    as a directory tree rather than a packaged zip).

URLs occasionally rotate; if a fetch starts returning 404 the upstream
portal page is the source of truth — see references in docstrings below.

Files land in `data/gtfs/<country>-gtfs.zip`. Re-runs are idempotent
unless `--force` is passed.
"""
from __future__ import annotations
import gzip
import io
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import config


# Direct GTFS zip URLs per country. Each entry may be either:
#   - a URL string (uses default UA), or
#   - a dict {"url": ..., "ua": ...} when the upstream server sniffs
#     the User-Agent and only serves browser-like clients (Rejseplanen
#     rejects a plain "bike-routing-ingest/1.0" UA with 404).
#
# Sources:
#   Austria: ÖBB feed mirrored at https://data.oebb.at/de/datensaetze~soll-fahrplan-gtfs~
#     The portal page wraps the file behind a click-through but the
#     static asset URL is stable across reissues (the trailing year
#     segment is the first publication year of the validity window,
#     NOT the download date).
#   Germany: gtfs.de aggregated national feed (DB + regional operators).
#     Free tier is at /germany/free/latest.zip (~250 MB).
#   Denmark: Rejseplanen static GTFS at rejseplanen.info/labs. The
#     server 404s any non-browser UA — Mozilla is required.
#   Czech Republic: PID (Prague integrated transport) — covers Prague
#     area + regional trains reaching the corridor. Brno's IDSJMK feed
#     could be added later as its own country key ("czech-republic-brno")
#     if we want better coverage south of Prague.
_FEEDS: dict[str, object] = {
    "austria":        "https://static.web.oebb.at/open-data/soll-fahrplan-gtfs/GTFS_OP_2025_obb.zip",
    "germany":        "https://download.gtfs.de/germany/free/latest.zip",
    "denmark":       {"url": "https://www.rejseplanen.info/labs/GTFS.zip",
                      "ua":  "Mozilla/5.0"},
    # Czech Republic: aggregated NATIONAL feed (PID + IDSJMK + regional
    # operators) hosted as loose files in the tangero/jizdni-rady-czech-
    # republic GitHub repo. We stitch them into a single zip because the
    # ingest expects a zipped GTFS. PID alone would miss Brno; this
    # covers the whole corridor.
    "czech-republic": {"assemble": [
        (f"https://raw.githubusercontent.com/tangero/jizdni-rady-czech-republic/main/data/merged/{n}", n)
        for n in ("agency.txt", "calendar_dates.txt", "routes.txt",
                  "stops.txt", "trips.txt")
    ] + [
        ("https://raw.githubusercontent.com/tangero/jizdni-rady-czech-republic/main/data/merged/stop_times.txt.gz",
         "stop_times.txt"),  # gunzip on the way in
    ]},
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
    entry = _FEEDS[country]
    out_path = feed_path(country)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / 1e6
        print(f"[gtfs] {country}: {out_path} exists ({size_mb:.1f} MB), "
              f"skipping (use --force to redownload)")
        return out_path

    t0 = time.time()

    if isinstance(entry, dict) and "assemble" in entry:
        # Assemble a zip from N loose files, each hosted separately.
        # Members whose *source* URL ends in .gz but whose target
        # *name* doesn't are gunzipped on the way in — GTFS ingest
        # expects `stop_times.txt`, not `.txt.gz`.
        print(f"[gtfs] {country}: assembling from "
              f"{len(entry['assemble'])} files")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for url, name in entry["assemble"]:
                print(f"[gtfs] {country}:   fetch {url}")
                req = Request(url, headers={"User-Agent": "bike-routing-ingest/1.0"})
                with urlopen(req, timeout=300) as r:
                    raw = r.read()
                if url.endswith(".gz") and not name.endswith(".gz"):
                    raw = gzip.decompress(raw)
                zf.writestr(name, raw)
        data = buf.getvalue()
    else:
        if isinstance(entry, dict):
            url = entry["url"]
            ua = entry.get("ua", "bike-routing-ingest/1.0")
        else:
            url = entry
            ua = "bike-routing-ingest/1.0"
        print(f"[gtfs] {country}: downloading {url}")
        req = Request(url, headers={"User-Agent": ua})
        with urlopen(req, timeout=300) as r:
            data = r.read()

    out_path.write_bytes(data)
    print(f"[gtfs] {country}: wrote {out_path} ({len(data) / 1e6:.1f} MB) "
          f"in {time.time()-t0:.1f}s")
    return out_path
