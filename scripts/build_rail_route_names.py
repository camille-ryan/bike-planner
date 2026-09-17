"""Build data/rail_route_names.json — {route_id: route_short_name} for
rail (route_type=2) routes across all four GTFS feeds.

The rail-anchor graph (`api/app/rail_router.py`) builds edges from
`rail_station_routes.json` = {station_key: set(route_id)}. That works
for within-country pairs but disconnects cross-border trains: the
same Wien↔Brno Railjet is filed as different `route_id`s in the AT
and CZ feeds, so the shared-route_id test fails and no edge exists.

Trick: match on `route_short_name` (e.g. "R41", "RJ", "EC", "IC")
instead of raw `route_id`. Same InterCity/regional train usually
carries the same short_name in both national feeds, so unifying by
short_name lets the graph express cross-border direct service.

Output shape kept minimal — the router just needs to look up a
route_id's short_name, so it's a flat dict. Non-rail routes and
routes with no short_name are omitted.

Run standalone: `python3 scripts/build_rail_route_names.py`
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path


GTFS_DIR = Path("data/gtfs")
COUNTRIES = ["austria", "czech-republic", "germany", "denmark"]
OUT_PATH = Path("data/rail_route_names.json")


def _extract_rail_routes(zip_path: Path) -> list[tuple[str, str]]:
    """Return [(route_id, route_short_name)] for rail (route_type=2)
    routes in this feed."""
    with zipfile.ZipFile(zip_path) as z:
        with z.open("routes.txt") as f:
            # utf-8-sig strips the BOM some feeds include (the AT feed
            # has one — without stripping, DictReader keys the first
            # column as `﻿route_id` and row.get('route_id')
            # returns empty, so every row is silently skipped).
            text = f.read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    out: list[tuple[str, str]] = []
    for row in reader:
        rtype = (row.get("route_type") or "").strip()
        if rtype != "2":
            continue
        rid = (row.get("route_id") or "").strip()
        sn = (row.get("route_short_name") or "").strip()
        if not rid or not sn:
            continue
        out.append((rid, sn))
    return out


def main() -> None:
    combined: dict[str, str] = {}
    for c in COUNTRIES:
        zip_path = GTFS_DIR / f"{c}-gtfs.zip"
        if not zip_path.exists():
            print(f"skip {c}: {zip_path} missing")
            continue
        rows = _extract_rail_routes(zip_path)
        print(f"{c}: {len(rows)} rail routes with short_name")
        for rid, sn in rows:
            # Raw route_ids as keys — matches the shape of
            # `rail_station_routes.json` (station_key → set of raw
            # route_ids). Confirmed 0 collisions across AT/CZ/DE/DK
            # feeds (each uses a distinct prefix scheme: AT
            # "3-R93-...", CZ "RT_...", DE numeric, DK quoted numeric).
            combined[rid] = sn
    OUT_PATH.write_text(json.dumps(combined, ensure_ascii=False, indent=0))
    print(f"wrote {len(combined)} route_id → short_name entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
