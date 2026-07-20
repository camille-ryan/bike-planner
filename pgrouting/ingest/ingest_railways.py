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

# Spatial filter: drop OSM lines that aren't within this many meters of
# any GTFS-served station (geographic distance via geography cast).
# 200 m is generous enough to absorb the slight lateral offset between
# OSM track centerlines and the GTFS station POI placement, while tight
# enough to drop freight-only lines that don't pass through any served
# station.
_LINE_STATION_DIST_M = 200.0

# Batched insert size for OSM lines (similar to the waterway ingester).
BATCH_SIZE = 5_000


# ---------------------------------------------------------------------
# GTFS parsing
# ---------------------------------------------------------------------

def _gtfs_open(zf: zipfile.ZipFile, name: str):
    """Open one CSV file from the GTFS zip with utf-8-sig (some feeds
    ship a BOM on the first column header, e.g. `\\ufeffroute_id`)."""
    return io.TextIOWrapper(zf.open(name), encoding="utf-8-sig")


def _parse_gtfs(zip_path: Path, country: str) -> list[dict]:
    """Parse one GTFS zip → list of stop dicts (only stops served by at
    least one rail route, inside the country's bbox, and deduped by
    name + ~100 m position cluster so sibling-platform stops become one
    station with a unioned route set).

    Each dict has: gtfs_id, name, lat, lon, n_routes.
    """
    bbox = _COUNTRY_BBOX.get(country)
    if bbox is None:
        print(f"[railways] WARN: no bbox configured for {country}; "
              f"keeping all stops including foreign termini")
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        # Phase 1: identify rail route_ids
        with _gtfs_open(zf, "routes.txt") as f:
            rail_route_ids: set[str] = set()
            for r in csv.DictReader(f):
                try:
                    rt = int(r["route_type"])
                except (KeyError, ValueError):
                    continue
                if rt in _RAIL_ROUTE_TYPES:
                    rail_route_ids.add(r["route_id"])
        print(f"[railways] gtfs: {len(rail_route_ids):,} rail routes")

        # Phase 2: trip_id → route_id (only for rail trips)
        with _gtfs_open(zf, "trips.txt") as f:
            trip_to_route: dict[str, str] = {}
            for r in csv.DictReader(f):
                rid = r.get("route_id")
                if rid in rail_route_ids:
                    trip_to_route[r["trip_id"]] = rid
        print(f"[railways] gtfs: {len(trip_to_route):,} rail trips")

        # Phase 3: stop_routes — for each stop, the set of distinct rail
        # routes that serve it. stop_times.txt is the big file (millions
        # of rows); stream it.
        stop_routes: dict[str, set[str]] = defaultdict(set)
        n_st_rows = 0
        with _gtfs_open(zf, "stop_times.txt") as f:
            for r in csv.DictReader(f):
                n_st_rows += 1
                tid = r.get("trip_id")
                rid = trip_to_route.get(tid)
                if rid is not None:
                    stop_routes[r["stop_id"]].add(rid)
        print(f"[railways] gtfs: scanned {n_st_rows:,} stop_times rows; "
              f"{len(stop_routes):,} rail-served stops")

        # Phase 4: stops.txt — keep only rail-served stops at location_type
        # 0 (stop) or 1 (station). The "platform" / "entrance" rows
        # (location_type 2-4) are dropped — they're sub-features of a
        # parent station, and we want one POI per station.
        candidates: list[dict] = []
        n_dropped_bbox = 0
        with _gtfs_open(zf, "stops.txt") as f:
            for r in csv.DictReader(f):
                sid = r.get("stop_id")
                if sid not in stop_routes:
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
                    "gtfs_id":  sid,
                    "name":     (r.get("stop_name") or sid).strip(),
                    "lat":      lat,
                    "lon":      lon,
                    "routes":   stop_routes[sid],   # set, for unioning
                })
    if n_dropped_bbox:
        print(f"[railways] gtfs: dropped {n_dropped_bbox:,} stops outside "
              f"{country} bbox (foreign termini)")

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
        union_routes: set[str] = set()
        for m in c["members"]:
            union_routes |= m["routes"]
        stations.append({
            "gtfs_id":  min(m["gtfs_id"] for m in c["members"]),
            "name":     c["name"],
            "lat":      c["lat"],
            "lon":      c["lon"],
            "n_routes": len(union_routes),
        })
    n_collapsed = len(candidates) - len(stations)
    print(f"[railways] gtfs: deduped {len(candidates):,} stops "
          f"→ {len(stations):,} stations ({n_collapsed:,} sibling-"
          f"platform rows collapsed; parsed in {time.time()-t0:.1f}s)")
    return stations


def _ingest_stations(conn: psycopg.Connection,
                     country: str,
                     stations: list[dict]) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM rail_stations WHERE country = %s", (country,))
        if not stations:
            conn.commit()
            return
        cur.executemany(
            "INSERT INTO rail_stations "
            "(gtfs_id, name, n_routes, country, geom) VALUES "
            "(%s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))",
            [
                (s["gtfs_id"], s["name"], s["n_routes"], country,
                 s["lon"], s["lat"])
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
    """Return every OSM node tagged `railway ∈ _KEEP_OSM_RAIL_STOP` in
    the PBF, as a list of (lon, lat). Streams the file without
    building the location index (nodes carry their own coords)."""
    t0 = time.time()
    out: list[tuple[float, float]] = []
    fp = (osmium.FileProcessor(str(pbf))
          .with_filter(osmium.filter.KeyFilter("railway")))
    for obj in fp:
        if obj.is_node() and obj.tags.get("railway") in _KEEP_OSM_RAIL_STOP:
            out.append((float(obj.location.lon), float(obj.location.lat)))
    print(f"[railways]   OSM: {len(out):,} `railway=station|halt|stop` nodes "
          f"in {time.time()-t0:.1f}s")
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


# ---------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------

def ingest(conn: psycopg.Connection,
           country: str,
           gtfs_zip: Path,
           pbf: Path) -> None:
    """End-to-end ingest of one country's passenger rails."""
    print(f"[railways] === {country} ===")
    stations = _parse_gtfs(gtfs_zip, country)
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
    with conn.cursor() as cur:
        cur.execute("ANALYZE rail_stations")
        cur.execute("ANALYZE rail_lines")
