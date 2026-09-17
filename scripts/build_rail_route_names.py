"""Build two rail-route sidecars from the raw GTFS feeds:

- `data/rail_route_names.json` = {route_id: route_short_name} for
  every rail (route_type=2) route across all four feeds. Used by
  `rail_router` to canonicalize cross-feed edges: the same Wien↔Brno
  Railjet is filed as different `route_id`s in AT and CZ feeds, but
  usually shares its short_name.

- `data/rail_route_excludes.json` = [route_id, ...] to treat as
  NOT-real-rail even though the feed marked them route_type=2.
  National feeds abuse the rail type for fare-integrated regional
  bus lines (Pražská integrovaná doprava, Moravskoslezský kraj,
  German Verkehrsverbünde). Their inclusion caused `direct_rail_
  service` to falsely report bus service as direct train service.

Both files are consumed at runtime by `rail_router` and by the
`_load_station_routes_cache` helper in `api/app/tools.py`.

Run standalone: `python3 scripts/build_rail_route_names.py`
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path


GTFS_DIR = Path("data/gtfs")
COUNTRIES = ["austria", "czech-republic", "germany", "denmark"]
OUT_NAMES = Path("data/rail_route_names.json")
OUT_EXCLUDES = Path("data/rail_route_excludes.json")


# Agencies that file route_type=2 for services that are NOT physical
# rail. Matched against agency_name (case-insensitive substring). The
# common pattern is fare-integration bodies that lump regional bus
# lines under a "rail-integrated" tag:
#   - CZ: "integrovaná doprava" (Prague integrated transport)
#   - CZ: "kraj" (regional fare integrators, e.g. Moravskoslezský kraj)
#   - DE: "Verkehrsverbund" (transport associations)
#   - DE: bare "Verkehrsverbund" / "verbund" / "KVV" abbreviations
FARE_INTEGRATOR_PATTERNS = [
    "integrovaná doprava",
    "verkehrsverbund",
    "moravskoslezský kraj",
    "středočeská",
]

# Short_name prefixes commonly used by CZ fare integrators for bus
# routes tagged as rail. Kept narrow to avoid false-positives.
CZ_BUS_PREFIXES = re.compile(r"^(Vlak|V\d|U\d|M\d|L\d)")


def _extract_routes(zip_path: Path):
    """Return list of (route_id, short_name, agency_id, agency_name)
    tuples for rail (route_type=2) routes."""
    with zipfile.ZipFile(zip_path) as z:
        # utf-8-sig strips the BOM some feeds include (AT feed has
        # one — without stripping, DictReader keys the first column
        # as `﻿route_id` and every row is silently skipped).
        routes_text  = z.read("routes.txt").decode("utf-8-sig")
        agency_text  = z.read("agency.txt").decode("utf-8-sig")
    agencies = {r["agency_id"]: (r.get("agency_name") or "")
                for r in csv.DictReader(io.StringIO(agency_text))}
    out: list[tuple[str, str, str, str]] = []
    for row in csv.DictReader(io.StringIO(routes_text)):
        rtype = (row.get("route_type") or "").strip()
        if rtype != "2":
            continue
        rid = (row.get("route_id") or "").strip().strip('"')
        sn  = (row.get("route_short_name") or "").strip().strip('"')
        aid = (row.get("agency_id") or "").strip().strip('"')
        aname = agencies.get(aid, "")
        if not rid:
            continue
        out.append((rid, sn, aid, aname))
    return out


def _looks_like_bus(country: str, short_name: str, agency_name: str) -> bool:
    """Heuristic: is this route type=2 actually a bus mislabeled as
    rail? Two signals, either one is enough:
      - agency name matches a known fare-integrator pattern
      - CZ-specific short_name prefix pattern (Vlak\\d+, V\\d+, U\\d+, ...)
    """
    an = (agency_name or "").lower()
    if any(pat in an for pat in FARE_INTEGRATOR_PATTERNS):
        return True
    if country == "czech-republic" and short_name:
        if CZ_BUS_PREFIXES.match(short_name):
            return True
    return False


def main() -> None:
    names: dict[str, str] = {}
    excludes: list[str] = []
    for c in COUNTRIES:
        zip_path = GTFS_DIR / f"{c}-gtfs.zip"
        if not zip_path.exists():
            print(f"skip {c}: {zip_path} missing")
            continue
        rows = _extract_routes(zip_path)
        kept = 0; skipped = 0
        for rid, sn, aid, aname in rows:
            if _looks_like_bus(c, sn, aname):
                excludes.append(rid)
                skipped += 1
                continue
            if sn:
                names[rid] = sn
                kept += 1
        print(f"{c}: {kept} rail routes with short_name kept, "
              f"{skipped} bus-pretending-to-be-rail excluded")

    OUT_NAMES.write_text(json.dumps(names, ensure_ascii=False, indent=0))
    print(f"wrote {len(names)} route_id → short_name entries to {OUT_NAMES}")

    OUT_EXCLUDES.write_text(json.dumps(sorted(excludes),
                                       ensure_ascii=False, indent=0))
    print(f"wrote {len(excludes)} excluded route_ids to {OUT_EXCLUDES}")


if __name__ == "__main__":
    main()
