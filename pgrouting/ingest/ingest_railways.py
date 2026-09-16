"""Ingest passenger rail stations + lines for the routing context layer.

Two-source hybrid:

  Stations  ←  GTFS  (definitive served-stop list, with route count per stop)
  Lines     ←  OSM   (better track geometry than GTFS shapes), filtered
                     spatially to lines within 200 m of any GTFS station —
                     drops freight-only mainlines that OSM still tags
                     `usage=main`.

The function `ingest(conn, country, gtfs_zip, pbf, ...)` runs all phases
for one country end-to-end and is idempotent (DELETE-then-INSERT keyed
on `country`).
"""
from __future__ import annotations
import csv
import io
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import osmium
import osmium.geom
import osmium.filter
import psycopg


# GTFS route_type values that count as "rail" for our purposes:
# 2 = legacy rail; 100-117 = extended rail subtypes (high-speed, intercity,
# regional, suburban, etc.). Excludes 3 (bus), 11 (trolleybus), etc.
_RAIL_ROUTE_TYPES: set[int] = {2} | set(range(100, 118))

# Bus (and bus-adjacent) route types. Tracked separately so we can
# expose `n_routes_bus` on rail_stations for stops that are honestly
# multimodal (e.g. a station forecourt with a bus terminal). Does not
# affect which stops are kept — stops are still filtered by
# _RAIL_ROUTE_TYPES membership. `700`-block is the "extended" GTFS
# bus family (city, regional, express); `11` is trolleybus.
_BUS_ROUTE_TYPES: set[int] = {3, 11} | set(range(700, 800))

# Per-country bbox for filtering GTFS stops to in-country only. Some
# national feeds include foreign termini (e.g. the ÖBB feed has
# Stuttgart Hbf, München Hbf, etc.); without a geographic filter those
# leak into the station layer.
_COUNTRY_BBOX: dict[str, tuple[float, float, float, float]] = {
    # (min_lon, min_lat, max_lon, max_lat) — chosen slightly wider than
    # the country border so stations at frontiers (e.g. Aachen for DE,
    # or Rødby for DK) aren't clipped.
    "austria":        ( 9.2, 46.3, 17.5, 49.2),
    "germany":        ( 5.5, 47.1, 15.2, 55.2),
    "denmark":        ( 7.9, 54.5, 15.4, 57.9),
    "czech-republic": (12.0, 48.5, 19.0, 51.1),
}

# Cluster radius (in degrees) for collapsing sibling-platform stops into
# one station. Stations within this radius AND sharing a normalized
# name are treated as the same physical station; the union of their
# route sets becomes the consolidated n_routes value. ~0.001° ≈ 110 m
# lat / 75 m lon at Austrian latitudes — wide enough to capture
# cross-platform sibling stops at any large station, narrow enough not
# to merge truly distinct nearby stops.
_DEDUP_CLUSTER_DEG: float = 0.001


# Strip the trailing platform-number suffix that some GTFS feeds append
# to sibling-stop names: " 2", " 5" (plain platform number) or " g1",
# " g2" (Gleis-prefixed in some Austrian feeds).
_TRAILING_PLATFORM_RE = re.compile(r"\s+(?:g)?\d+\s*$", re.IGNORECASE)

# Name abbreviation aliases — normalized to the long form so e.g.
# "Wörgl Hbf" and "Wörgl Hauptbahnhof" cluster together. Only includes
# unambiguous synonyms (Bahnhst = Bahnhaltestelle is a different concept
# from Bahnhof and is intentionally NOT mapped).
_NAME_ALIASES: dict[str, str] = {
    "hbf": "hauptbahnhof",
    "bhf": "bahnhof",
}


def _normalize_station_name(name: str) -> str:
    """Lower-case, strip trailing platform digits, expand a few
    standard abbreviations. Used as the grouping key for dedup; the
    display name comes from the cluster's first member unchanged."""
    s = _TRAILING_PLATFORM_RE.sub("", name).strip().lower()
    # Word-boundary substitutions so "bahnhof" inside a name stays.
    for short, long_ in _NAME_ALIASES.items():
        s = re.sub(rf"\b{short}\b", long_, s)
    return s

# OSM line filter — keep only main/branch passenger-grade rails.
_KEEP_RAILWAY = {"rail", "light_rail"}
_KEEP_USAGE   = {"main", "branch"}
_DROP_SERVICE = {"spur", "siding", "yard", "crossover"}

# OSM node tag values that mark a real rail stop. `station` = big
# staffed station; `halt` = flag stop / unstaffed; `stop` = generic
# stop position on a rail line; `tram_stop` explicitly excluded (we
# want heavy/regional rail, not urban tram). `station_site` and
# `service_station` are yard/depot markers — skip.
_KEEP_OSM_RAIL_STOP = {"station", "halt", "stop"}

# Radius for cross-checking GTFS-derived station centroids against
# nearby OSM railway stop nodes. Bigger than the ~50m typical GTFS/
# OSM disagreement, tighter than any two different stops in the same
# small town. Rescues legitimate rail stops whose GTFS coord drifts.
_OSM_MATCH_RADIUS_M = 250.0

# Additional filter: require every kept OSM rail-stop node to lie
# within this many meters of an OSM `railway=rail` way vertex. Drops
# standalone nodes tagged `railway=station` (or `halt`/`stop`) that
# aren't actually on a track — most commonly bus terminals mistagged
# by OSM contributors, or long-disused halts whose tracks were lifted
# but the POI persists. Combined with the tag-based filter below,
# this catches most of the false-positives that survive the plain
# `railway=station|halt|stop` extractor.
_OSM_STOP_NEAR_RAIL_M = 100.0

# Spatial filter: drop OSM lines that aren't within this many meters of
# any GTFS-served station (geographic distance via geography cast).
# 200 m is generous enough to absorb the slight lateral offset between
# OSM track centerlines and the GTFS station POI placement, while tight
# enough to drop freight-only lines that don't pass through any served
# station.
_LINE_STATION_DIST_M = 200.0

# Reverse filter: after `_filter_connected_to_stations` prunes the line
# set to just the connected passenger network, drop STATIONS that are
# more than this many meters from any surviving line. Empirically, real
# rail halts are ≤ ~50m from the tracks; anything at ≥100m is a nearby
# bus stop that survived the OSM cross-check because it happens to
# share a name (or be close to) a real station. In the CZ Kolín area,
# the actual `Kolín` station is at 0m; suburban bus stops named
# `Kolín,Tatradomy` / `Kolín,nám.Republiky` sit at 155-190m. 100m
# splits them cleanly.
_STATION_LINE_DIST_M = 100.0

# Batched insert size for OSM lines (similar to the waterway ingester).
BATCH_SIZE = 5_000


# ---------------------------------------------------------------------
# GTFS parsing
# ---------------------------------------------------------------------

def _gtfs_open(zf: zipfile.ZipFile, name: str):
    """Open one CSV file from the GTFS zip with utf-8-sig (some feeds
    ship a BOM on the first column header, e.g. `\\ufeffroute_id`)."""
    return io.TextIOWrapper(zf.open(name), encoding="utf-8-sig")


def _parse_gtfs(zip_paths: list[Path], country: str) -> list[dict]:
    """Parse one or more GTFS zips → list of stop dicts. Each dict has:
      gtfs_id, name, lat, lon, routes_rail, routes_bus (both sets).

    Only stops served by at least one *rail* route are kept — bus-only
    stops are dropped even when they're in the same feed. Stops served
    by both rail and bus keep both route sets. Passed-in zips are
    unioned during dedup (name + ~100 m position cluster), so calling
    with [tangero.zip, cd.zip] naturally merges Prague Hlavní etc. into
    one station with the union of route ids.

    route_id namespace across zips: we deliberately DON'T prefix ids
    with a source name — because a station clustered from multiple
    feeds gets the *union* of route ids, and identical ids across
    feeds (rare but possible) would spuriously deflate the count.
    Since we only ever compare route_id set membership within one
    stop's routes, and cardinality is what matters downstream, the
    tiny chance of a collision is acceptable.
    """
    bbox = _COUNTRY_BBOX.get(country)
    if bbox is None:
        print(f"[railways] WARN: no bbox configured for {country}; "
              f"keeping all stops including foreign termini")
    t0 = time.time()

    # Union of candidates across all provided feeds. Each feed
    # contributes its own set of route ids per stop.
    candidates: list[dict] = []
    n_dropped_bbox_total = 0

    for zip_path in zip_paths:
        print(f"[railways] gtfs: parsing {zip_path.name}")
        with zipfile.ZipFile(zip_path) as zf:
            # Phase 1: classify routes by mode
            rail_route_ids: set[str] = set()
            bus_route_ids: set[str] = set()
            with _gtfs_open(zf, "routes.txt") as f:
                for r in csv.DictReader(f):
                    try:
                        rt = int(r["route_type"])
                    except (KeyError, ValueError):
                        continue
                    if rt in _RAIL_ROUTE_TYPES:
                        rail_route_ids.add(r["route_id"])
                    elif rt in _BUS_ROUTE_TYPES:
                        bus_route_ids.add(r["route_id"])
            print(f"[railways]   {len(rail_route_ids):,} rail routes, "
                  f"{len(bus_route_ids):,} bus routes")

            # Phase 2: trip_id → (route_id, mode). Include both rail and
            # bus so we can attribute per-mode counts on stops served by
            # both.
            trip_to_route: dict[str, tuple[str, str]] = {}
            with _gtfs_open(zf, "trips.txt") as f:
                for r in csv.DictReader(f):
                    rid = r.get("route_id")
                    if rid in rail_route_ids:
                        trip_to_route[r["trip_id"]] = (rid, "rail")
                    elif rid in bus_route_ids:
                        trip_to_route[r["trip_id"]] = (rid, "bus")

            # Phase 3: per-stop route sets, split by mode.
            stop_rail_routes: dict[str, set[str]] = defaultdict(set)
            stop_bus_routes: dict[str, set[str]] = defaultdict(set)
            n_st_rows = 0
            with _gtfs_open(zf, "stop_times.txt") as f:
                for r in csv.DictReader(f):
                    n_st_rows += 1
                    tup = trip_to_route.get(r.get("trip_id"))
                    if tup is None:
                        continue
                    rid, mode = tup
                    sid = r["stop_id"]
                    if mode == "rail":
                        stop_rail_routes[sid].add(rid)
                    else:
                        stop_bus_routes[sid].add(rid)
            print(f"[railways]   scanned {n_st_rows:,} stop_times rows; "
                  f"{len(stop_rail_routes):,} rail-served stops, "
                  f"{len(stop_bus_routes):,} bus-served stops")

            # Phase 4: keep only rail-served stops. Bus routes at a
            # rail-served stop annotate n_routes_bus; bus-only stops
            # are ignored (they'd explode station cardinality without
            # helping the "rail-accessible overnight" query).
            n_dropped_bbox = 0
            with _gtfs_open(zf, "stops.txt") as f:
                for r in csv.DictReader(f):
                    sid = r.get("stop_id")
                    if sid not in stop_rail_routes:
                        continue
                    try:
                        loc_type = int(r.get("location_type") or 0)
                    except ValueError:
                        loc_type = 0
                    if loc_type not in (0, 1):
                        continue
                    try:
                        lat = float(r["stop_lat"])
                        lon = float(r["stop_lon"])
                    except (KeyError, ValueError):
                        continue
                    if bbox is not None and not (
                        bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]
                    ):
                        n_dropped_bbox += 1
                        continue
                    candidates.append({
                        "gtfs_id":     sid,
                        "name":        (r.get("stop_name") or sid).strip(),
                        "lat":         lat,
                        "lon":         lon,
                        "routes_rail": stop_rail_routes[sid],
                        "routes_bus":  stop_bus_routes.get(sid, set()),
                    })
            n_dropped_bbox_total += n_dropped_bbox

    if n_dropped_bbox_total:
        print(f"[railways] gtfs: dropped {n_dropped_bbox_total:,} stops "
              f"outside {country} bbox (foreign termini)")

    # Phase 5: dedup by (normalized name, ~100m position cluster).
    # Greedy clustering: each candidate stop attaches to an existing
    # cluster if names match AND it's within _DEDUP_CLUSTER_DEG of the
    # cluster's centroid. Grid bucketing was tried first but missed
    # clusters that straddled bucket boundaries (e.g. two 5-m-apart
    # stops landing in different cells); greedy clustering avoids that
    # at minor cost (O(N·K) ≈ 2M ops for Austria).
    clusters: list[dict] = []
    # Bucket cluster centroids to speed up the within-radius lookup.
    bucket_index: dict[tuple, list[int]] = defaultdict(list)

    def _bucket_keys(lat: float, lon: float):
        # Check the cell containing the point AND its 8 neighbours so a
        # centroid near a cell boundary doesn't get missed.
        bl = int(lat / _DEDUP_CLUSTER_DEG)
        bo = int(lon / _DEDUP_CLUSTER_DEG)
        for dl in (-1, 0, 1):
            for do in (-1, 0, 1):
                yield (bl + dl, bo + do)

    for s in candidates:
        nname = _normalize_station_name(s["name"])
        matched: dict | None = None
        for bk in _bucket_keys(s["lat"], s["lon"]):
            for ci in bucket_index.get(bk, ()):
                c = clusters[ci]
                if c["name_norm"] != nname:
                    continue
                if (abs(s["lat"] - c["lat"]) < _DEDUP_CLUSTER_DEG
                        and abs(s["lon"] - c["lon"]) < _DEDUP_CLUSTER_DEG):
                    matched = c
                    break
            if matched is not None:
                break
        if matched is None:
            c = {
                "name_norm": nname,
                # Display name strips any trailing platform digits so
                # "Leoben Hauptbahnhof 3" → "Leoben Hauptbahnhof". Many
                # feeds (incl. ÖBB) suffix every sibling platform with a
                # number and never emit a clean unsuffixed variant.
                "name":      _TRAILING_PLATFORM_RE.sub("", s["name"]).strip(),
                "lat":       s["lat"],
                "lon":       s["lon"],
                "members":   [s],
            }
            ci = len(clusters)
            clusters.append(c)
            for bk in _bucket_keys(c["lat"], c["lon"]):
                bucket_index[bk].append(ci)
        else:
            matched["members"].append(s)
            n = len(matched["members"])
            matched["lat"] = (matched["lat"] * (n - 1) + s["lat"]) / n
            matched["lon"] = (matched["lon"] * (n - 1) + s["lon"]) / n

    stations: list[dict] = []
    for c in clusters:
        union_rail: set[str] = set()
        union_bus: set[str] = set()
        for m in c["members"]:
            union_rail |= m["routes_rail"]
            union_bus  |= m["routes_bus"]
        n_rail = len(union_rail)
        n_bus  = len(union_bus)
        stations.append({
            "gtfs_id":       min(m["gtfs_id"] for m in c["members"]),
            "name":          c["name"],
            "lat":           c["lat"],
            "lon":           c["lon"],
            # `n_routes` retained as rail+bus total (backwards compat
            # for anything reading the column that predates the split).
            # New callers should prefer `n_routes_rail`.
            "n_routes":      n_rail + n_bus,
            "n_routes_rail": n_rail,
            "n_routes_bus":  n_bus,
            # Kept alongside the counts so `export-rail-routes` can
            # emit a sidecar without re-parsing GTFS; _ingest_stations
            # ignores this key.
            "route_ids_rail": sorted(union_rail),
        })
    n_collapsed = len(candidates) - len(stations)
    print(f"[railways] gtfs: deduped {len(candidates):,} stops "
          f"→ {len(stations):,} stations ({n_collapsed:,} sibling-"
          f"platform rows collapsed; parsed in {time.time()-t0:.1f}s)")
    return stations


def _ingest_stations(conn: psycopg.Connection,
                     country: str,
                     stations: list[dict]) -> None:
    """Insert stations into `rail_stations`. Ensures the per-mode
    columns exist (idempotent ALTER TABLE) so the first run after
    task #78 lands doesn't require a manual migration."""
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE rail_stations
              ADD COLUMN IF NOT EXISTS n_routes_rail INTEGER NOT NULL DEFAULT 0,
              ADD COLUMN IF NOT EXISTS n_routes_bus  INTEGER NOT NULL DEFAULT 0
        """)
        cur.execute("DELETE FROM rail_stations WHERE country = %s", (country,))
        if not stations:
            conn.commit()
            return
        cur.executemany(
            "INSERT INTO rail_stations "
            "(gtfs_id, name, n_routes, n_routes_rail, n_routes_bus, "
            " country, geom) VALUES "
            "(%s, %s, %s, %s, %s, %s, "
            "ST_SetSRID(ST_MakePoint(%s, %s), 4326))",
            [
                (s["gtfs_id"], s["name"], s["n_routes"],
                 s["n_routes_rail"], s["n_routes_bus"],
                 country, s["lon"], s["lat"])
                for s in stations
            ],
        )
    conn.commit()
    print(f"[railways] inserted {len(stations):,} rail_stations "
          f"for {country}")


# ---------------------------------------------------------------------
# OSM line parsing
# ---------------------------------------------------------------------

def _extract_osm_rail_nodes(pbf: Path) -> list[tuple[float, float]]:
    """Return coords of OSM nodes that plausibly represent a real
    passenger-rail stop, after two filters:

      (a) `railway` tag in {station, halt, stop} AND the node is NOT
          bus-only (drops nodes tagged bus=yes without train=yes —
          common misuse pattern where a bus terminal was tagged as a
          "station" for road-signage reasons).
      (b) The node's location is within `_OSM_STOP_NEAR_RAIL_M` of an
          OSM `railway=rail`/`light_rail` way vertex. Drops orphan
          nodes that survived (a) but aren't on any actual track —
          disused halts whose rails have been lifted, POIs mistagged
          as `station`, or a handful of tourism-related items.

    Two passes over the PBF: first collects way vertex coords (uses
    a sparse location index so the file processor can resolve node
    refs while streaming ways); second collects & filters candidate
    stop nodes. Rail-way vertex count in a country PBF is
    O(millions); we downsample to every 5th vertex before building
    the kdtree so memory stays bounded and the ~5-vertex-per-100m-of
    -track density still leaves gaps below the 100 m match radius.
    """
    import math
    import numpy as np
    from scipy.spatial import cKDTree

    t0 = time.time()

    # Pass 1: rail-way vertex coords (sampled). Uses the same sparse
    # index _ingest_osm_lines uses; osmium reuses the file if already
    # populated, so this is cheap the second time.
    rail_pts: list[tuple[float, float]] = []
    STRIDE = 5
    fp = (osmium.FileProcessor(str(pbf))
          .with_locations("sparse_file_array,/tmp/osmium-railway.idx")
          .with_filter(osmium.filter.KeyFilter("railway")))
    for obj in fp:
        if not obj.is_way():
            continue
        tags = dict(obj.tags)
        if tags.get("railway") not in _KEEP_RAILWAY:
            continue
        # Skip disused / abandoned / construction rails — the same
        # filter _accept_way applies to line ingest. A stop near a
        # disused track shouldn't count as rail-served.
        if (tags.get("disused") == "yes" or tags.get("abandoned") == "yes"
                or "construction" in tags):
            continue
        try:
            for i, node in enumerate(obj.nodes):
                if i % STRIDE != 0:
                    continue
                rail_pts.append((float(node.lon), float(node.lat)))
        except osmium.InvalidLocationError:
            # Way with a node ref outside the PBF extent — happens at
            # cross-border rails. Skip.
            continue
    print(f"[railways]   OSM: {len(rail_pts):,} rail-way vertex samples "
          f"(stride={STRIDE}) in {time.time()-t0:.1f}s")

    # Pass 2: candidate stop nodes + tag filter (a).
    t1 = time.time()
    candidates: list[tuple[float, float]] = []
    n_dropped_bus = 0
    fp = (osmium.FileProcessor(str(pbf))
          .with_filter(osmium.filter.KeyFilter("railway")))
    for obj in fp:
        if not obj.is_node():
            continue
        tags = dict(obj.tags)
        if tags.get("railway") not in _KEEP_OSM_RAIL_STOP:
            continue
        # Tag filter: reject bus-only nodes. Accepts if `train=yes`
        # is present (explicit rail) OR if `bus=yes` is absent (default
        # rail station). Rejects only when the node explicitly declares
        # `bus=yes` without also declaring `train=yes`.
        train_yes = tags.get("train") == "yes"
        bus_yes = tags.get("bus") == "yes"
        if bus_yes and not train_yes:
            n_dropped_bus += 1
            continue
        candidates.append((float(obj.location.lon), float(obj.location.lat)))
    print(f"[railways]   OSM: {len(candidates):,} candidate stop nodes "
          f"(dropped {n_dropped_bus:,} bus-only) in {time.time()-t1:.1f}s")

    # Filter (b): must be within _OSM_STOP_NEAR_RAIL_M of a rail-way
    # vertex. Uses a lat/lon kdtree with a degree-radius approximation;
    # for a 100 m radius at temperate latitudes the meridian/parallel
    # difference is <1%, so we just use the smaller degree-per-m to
    # get an upper-bound radius (rejects marginal matches, safe).
    if not rail_pts:
        print("[railways]   WARN: no rail-way vertices — skipping proximity "
              "filter (keeping all candidate stops)")
        out = candidates
    else:
        mean_lat = sum(p[1] for p in rail_pts) / len(rail_pts)
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(mean_lat))
        radius_deg = _OSM_STOP_NEAR_RAIL_M / min(m_per_deg_lat, m_per_deg_lon)
        tree = cKDTree(np.array(rail_pts, dtype=np.float64))
        out = []
        for c in candidates:
            d, _ = tree.query([c[0], c[1]], distance_upper_bound=radius_deg)
            if d < radius_deg:
                out.append(c)
        n_off_track = len(candidates) - len(out)
        print(f"[railways]   OSM: proximity filter kept {len(out):,}, "
              f"dropped {n_off_track:,} (no rail-way vertex within "
              f"{_OSM_STOP_NEAR_RAIL_M:.0f}m)")

    return out


def _cross_check_against_osm(stations: list[dict],
                             osm_rail: list[tuple[float, float]]
                             ) -> list[dict]:
    """Keep only GTFS-derived stations that have an OSM railway
    station/halt/stop node within _OSM_MATCH_RADIUS_M. Drops MHD bus
    and tram stops that GTFS misclassifies as `route_type=2`
    (dominant failure mode in the CZ tangero aggregate feed)."""
    if not osm_rail:
        print("[railways]   WARN: no OSM rail nodes — skipping cross-check "
              "(keeping all GTFS stations)")
        return stations
    import numpy as np
    from scipy.spatial import cKDTree
    arr = np.array(osm_rail, dtype=np.float64)
    tree = cKDTree(arr)
    # Radius in degrees at the mean latitude (small-angle approximation
    # — good to <1% at the scale we care about, and this is a proximity
    # test not a routing distance).
    mean_lat = float(arr[:, 1].mean())
    import math
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(mean_lat))
    # Use lon-scaled degrees so a distance_upper_bound in degrees is
    # roughly the same in meters regardless of latitude.
    radius_deg = _OSM_MATCH_RADIUS_M / min(m_per_deg_lat, m_per_deg_lon)
    kept = []
    for s in stations:
        d, _ = tree.query([s["lon"], s["lat"]],
                          distance_upper_bound=radius_deg)
        if d < radius_deg:
            kept.append(s)
    dropped = len(stations) - len(kept)
    pct = (dropped * 100 // max(len(stations), 1))
    print(f"[railways]   OSM cross-check: kept {len(kept):,}/{len(stations):,} "
          f"(dropped {dropped:,} = {pct}% with no OSM rail node "
          f"within {_OSM_MATCH_RADIUS_M:.0f}m)")
    return kept


def _accept_way(tags: dict) -> bool:
    """Filter OSM ways to passenger-grade rail trunks."""
    railway = tags.get("railway")
    if railway not in _KEEP_RAILWAY:
        return False
    if tags.get("usage") not in _KEEP_USAGE:
        return False
    if tags.get("disused") == "yes" or tags.get("abandoned") == "yes":
        return False
    if "construction" in tags or tags.get("railway") == "construction":
        return False
    if tags.get("service") in _DROP_SERVICE:
        return False
    return True


def _ingest_osm_lines(conn: psycopg.Connection,
                      country: str,
                      pbf: Path) -> int:
    """Stream `railway=*` ways from a country PBF, keep passenger trunks,
    insert as LineStrings into `rail_lines`."""
    t0 = time.time()
    fp = (osmium.FileProcessor(str(pbf))
          .with_locations("sparse_file_array,/tmp/osmium-railway.idx")
          .with_filter(osmium.filter.KeyFilter("railway")))
    wkt_fac = osmium.geom.WKTFactory()

    with conn.cursor() as cur:
        cur.execute("DELETE FROM rail_lines WHERE country = %s", (country,))
    conn.commit()

    rows: list[tuple] = []
    inserted = 0
    skipped_filter = 0
    skipped_geom = 0

    def _flush() -> None:
        nonlocal inserted
        if not rows:
            return
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO rail_lines "
                "(osm_id, name, operator, usage, electrified, country, geom) "
                "VALUES (%s, %s, %s, %s, %s, %s, "
                "ST_SetSRID(ST_GeomFromText(%s), 4326))",
                rows,
            )
        inserted += len(rows)
        conn.commit()
        rows.clear()

    for obj in fp:
        if not obj.is_way():
            continue
        tags = dict(obj.tags)
        if not _accept_way(tags):
            skipped_filter += 1
            continue
        try:
            wkt = wkt_fac.create_linestring(obj)
        except Exception:
            skipped_geom += 1
            continue
        rows.append((
            int(obj.id),
            tags.get("name"),
            tags.get("operator"),
            tags.get("usage"),
            tags.get("electrified"),
            country,
            wkt,
        ))
        if len(rows) >= BATCH_SIZE:
            _flush()

    _flush()
    print(f"[railways] osm: inserted {inserted:,} rail_lines "
          f"(skipped {skipped_filter:,} filter, {skipped_geom:,} geom) "
          f"in {time.time()-t0:.1f}s")
    return inserted


# ---------------------------------------------------------------------
# Spatial filter: drop lines that aren't near any GTFS station.
# ---------------------------------------------------------------------

def _filter_connected_to_stations(conn: psycopg.Connection,
                                  country: str,
                                  dist_m: float = _LINE_STATION_DIST_M) -> None:
    """Keep only the lines reachable (via shared endpoints) from any line
    that itself passes within `dist_m` meters of a GTFS station.

    OSM splits a long rail line into many short ways at every junction
    and station, so a naive "drop lines further than 200 m from a
    station" filter throws out the long inter-station segments. This
    function seeds with the station-adjacent segments and then
    transitively expands to every segment touching a kept one,
    producing the connected passenger rail network.
    """
    t0 = time.time()
    with conn.cursor() as cur:
        # Stage 1: seed — lines within dist_m of any station.
        cur.execute("DROP TABLE IF EXISTS _rail_kept")
        cur.execute("""
            CREATE TEMP TABLE _rail_kept AS
            SELECT l.id, l.geom FROM rail_lines l
            WHERE l.country = %s
              AND EXISTS (
                SELECT 1 FROM rail_stations s
                WHERE s.country = l.country
                  AND ST_DWithin(l.geom::geography, s.geom::geography, %s)
              )
        """, (country, dist_m))
        cur.execute("CREATE UNIQUE INDEX ON _rail_kept(id)")
        cur.execute("CREATE INDEX ON _rail_kept USING gist(geom)")
        cur.execute("ANALYZE _rail_kept")
        cur.execute("SELECT COUNT(*) FROM _rail_kept")
        n_seed = cur.fetchone()[0]
        print(f"[railways] connect: {n_seed:,} seed segments "
              f"(within {dist_m:.0f}m of a station)")

        # Stage 2: iteratively expand to lines touching the kept set.
        n_iter = 0
        while True:
            n_iter += 1
            cur.execute("""
                WITH new_rows AS (
                  SELECT DISTINCT l.id, l.geom
                    FROM rail_lines l
                    JOIN _rail_kept k ON ST_Intersects(l.geom, k.geom)
                   WHERE l.country = %s
                     AND l.id NOT IN (SELECT id FROM _rail_kept)
                )
                INSERT INTO _rail_kept (id, geom)
                SELECT id, geom FROM new_rows
            """, (country,))
            n_added = cur.rowcount
            if n_added == 0:
                break
            cur.execute("ANALYZE _rail_kept")
            print(f"[railways]   iter {n_iter}: +{n_added:,}")

        cur.execute("SELECT COUNT(*) FROM _rail_kept")
        n_kept = cur.fetchone()[0]

        # Stage 3: drop everything not in the connected set.
        cur.execute("""
            DELETE FROM rail_lines
             WHERE country = %s
               AND id NOT IN (SELECT id FROM _rail_kept)
        """, (country,))
        n_dropped = cur.rowcount
        cur.execute("DROP TABLE _rail_kept")
    conn.commit()
    print(f"[railways] connect: kept {n_kept:,} lines, dropped {n_dropped:,} "
          f"after {n_iter} iterations in {time.time()-t0:.1f}s")


def _filter_stations_by_line_proximity(conn: psycopg.Connection,
                                       country: str,
                                       dist_m: float = _STATION_LINE_DIST_M) -> None:
    """Drop stations that are more than `dist_m` meters from any
    passenger rail line. Runs after `_filter_connected_to_stations`
    has pruned the line set to the connected network.

    Rationale: the OSM cross-check drops obvious bus-only stops, but
    can't distinguish a real rail station from a bus stop that
    happens to be within 250m of one (Kolín area is the pathological
    case — many suburban bus stops share names with the real
    station). A hard "must be within 100m of a track" filter cuts
    those out cleanly. Real halts sit within 50m of the tracks by
    construction, so 100m is safely conservative.

    SAFEGUARD: if the country has fewer than 100 rail_lines rows OR
    total line length under 500 km, we skip the filter — that
    indicates the source PBF is missing node references (Denmark's
    Geofabrik cut is the concrete example: it produces 12 tiny
    fragments totalling <1 km, and applying the filter would drop
    ~95% of legitimate stations). In that case we keep the OSM-
    cross-checked station set as-is; downstream tools should still
    treat those results conservatively.
    """
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*), COALESCE(SUM(ST_Length(geom::geography))::bigint, 0) "
            "FROM rail_lines WHERE country = %s",
            (country,),
        )
        n_lines, total_len_m = cur.fetchone()
        if n_lines < 100 or total_len_m < 500_000:
            print(f"[railways] station-line proximity: SKIPPED for {country} "
                  f"(only {n_lines:,} lines, {(total_len_m or 0)/1000:.1f} km total "
                  f"— PBF likely missing node refs; filter would false-drop)")
            return
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH doomed AS (
              SELECT s.id
                FROM rail_stations s
               WHERE s.country = %s
                 AND NOT EXISTS (
                   SELECT 1 FROM rail_lines l
                    WHERE l.country = s.country
                      AND ST_DWithin(s.geom::geography, l.geom::geography, %s)
                 )
            )
            DELETE FROM rail_stations
             WHERE id IN (SELECT id FROM doomed)
            """,
            (country, dist_m),
        )
        n_dropped = cur.rowcount
        cur.execute(
            "SELECT COUNT(*) FROM rail_stations WHERE country = %s",
            (country,),
        )
        n_kept = cur.fetchone()[0]
    conn.commit()
    print(f"[railways] station-line proximity: kept {n_kept:,}, "
          f"dropped {n_dropped:,} (>{dist_m:.0f}m from any line) "
          f"in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------

def ingest(conn: psycopg.Connection,
           country: str,
           gtfs_zips: Path | list[Path],
           pbf: Path) -> None:
    """End-to-end ingest of one country's passenger rails.

    `gtfs_zips` may be a single Path or a list of Paths. Multi-feed
    is retained as a hook for future compositions (e.g. national +
    metro), but no country uses it today."""
    if isinstance(gtfs_zips, Path):
        gtfs_zips = [gtfs_zips]
    print(f"[railways] === {country} ({len(gtfs_zips)} feed"
          f"{'s' if len(gtfs_zips) != 1 else ''}) ===")
    stations = _parse_gtfs(gtfs_zips, country)
    # Cross-check against OSM `railway=station|halt|stop` nodes to
    # filter out MHD bus/tram/trolley stops that some aggregated GTFS
    # feeds (notably CZ tangero) mistag as `route_type=2`. Uses the
    # same PBF that _ingest_osm_lines reads a step later — parsed
    # in-memory here since the OSM stop count is small (~10k per
    # country) and doesn't need Postgres.
    osm_rail_nodes = _extract_osm_rail_nodes(pbf)
    stations = _cross_check_against_osm(stations, osm_rail_nodes)
    _ingest_stations(conn, country, stations)
    _ingest_osm_lines(conn, country, pbf)
    _filter_connected_to_stations(conn, country)
    # Reverse filter: kill any remaining stations that aren't ON the
    # connected network. Runs AFTER _filter_connected_to_stations so
    # `rail_lines` already contains only the connected passenger set.
    _filter_stations_by_line_proximity(conn, country)
    with conn.cursor() as cur:
        cur.execute("ANALYZE rail_stations")
        cur.execute("ANALYZE rail_lines")
