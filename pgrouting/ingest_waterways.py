"""OSM waterway ingest -> landcover (class='waterway').

Streams the full country PBFs (NOT the landuse extract — that drops
waterway lines; see Austria survey: 978 streams vs ~100k actual) and
emits buffered polygons for `waterway in (river, canal, stream)`.

The `landcover` table is polygon-only, so we buffer each centerline by
~10 m on the server (ST_Buffer over geography to get meters) before
insert. 10 m matches the typical real-world half-width of small
streams; large rivers tagged as `waterway=river` *lines* (where their
bank polygons are not separately mapped) end up under-buffered, but
those are rare — the polygon mapping for sizable rivers is good
enough that this stays an acceptable approximation.

Filters applied per OSM feature:
  - waterway in {river, canal, stream}                   (kind)
  - tunnel IS NULL                                       (skip culverts)
  - layer IS NULL or castable to int >= 0                (skip underground)
  - intermittent != 'yes'                                (skip seasonal)

The Austria-PBF survey showed 24% of streams are tunnel/culvert and
~7% intermittent — applying these cuts removes ~25% of mapped lines
that the rider doesn't actually experience.

Per-country idempotent: each invocation deletes the rows for the
countries it's processing (matching country='<name>' AND class='waterway')
then re-ingests. Forest/water/wetland rows for the same country are
unaffected because we filter the DELETE by class too.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import osmium
import osmium.geom
import osmium.filter
import psycopg

import config


# Per-edge buffer width in meters on the server side. Small enough that
# adjacent road and stream don't merge erroneously; large enough that
# a 200 m blur kernel later picks up the geometry meaningfully.
BUFFER_M = 10.0

# Lines per COPY batch. Each row carries a small WKT (a few hundred bytes
# typical for a stream segment) so we can afford bigger batches than the
# polygon ingest.
BATCH_SIZE = 2000

ACCEPTED_KINDS = ("river", "canal", "stream")


def _accept(tags: dict) -> bool:
    """All filters applied here so the WKT generation step only runs
    for rows that will survive."""
    wt = tags.get("waterway")
    if wt not in ACCEPTED_KINDS:
        return False
    if tags.get("tunnel"):
        return False
    layer = tags.get("layer")
    if layer is not None:
        try:
            if int(layer) < 0:
                return False
        except ValueError:
            # malformed layer tag — be conservative and skip
            return False
    if tags.get("intermittent") == "yes":
        return False
    return True


def _stage_pbf(conn: psycopg.Connection,
               pbf: Path,
               country: str,
               bbox: tuple[float, float, float, float] | None = None) -> int:
    """Stream waterway LineStrings from one full-country PBF into
    landcover, buffering each line to a thin polygon on the server.
    """
    fp = (osmium.FileProcessor(str(pbf))
          .with_locations(
              "sparse_file_array,/tmp/osmium-waterway.idx")
          .with_filter(osmium.filter.KeyFilter("waterway")))
    wkt_fac = osmium.geom.WKTFactory()

    rows: list[tuple[int, str]] = []   # (osm_id, wkt) — country is constant per file
    inserted = 0
    skipped_filter = 0
    skipped_geom = 0
    skipped_bbox = 0

    def _flush() -> None:
        nonlocal inserted
        if not rows:
            return
        # Buffer the line in geography space (meters) then cast back to
        # geometry(MultiPolygon, 4326). ST_MakeValid + ST_CollectionExtract
        # clean up self-intersecting buffers and unwrap any GeometryCollection
        # the validation step might emit. psycopg 3 has no mogrify; use
        # executemany.
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO landcover (osm_id, country, class, geom) VALUES "
                "(%s, %s, 'waterway', "
                "ST_Multi(ST_CollectionExtract(ST_MakeValid("
                "ST_Buffer("
                "ST_SetSRID(ST_GeomFromText(%s), 4326)::geography, %s"
                ")::geometry"
                "), 3)))",
                [(osm_id, country, wkt, BUFFER_M) for osm_id, wkt in rows],
            )
            inserted += len(rows)
        conn.commit()
        rows.clear()

    for obj in fp:
        if not obj.is_way():
            continue
        tags = dict(obj.tags)
        if not _accept(tags):
            skipped_filter += 1
            continue
        # bbox filter on the way's node locations (cheap)
        if bbox is not None:
            in_box = False
            for n in obj.nodes:
                if not n.location.valid():
                    continue
                x, y = n.location.lon, n.location.lat
                if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    in_box = True
                    break
            if not in_box:
                skipped_bbox += 1
                continue
        try:
            wkt = wkt_fac.create_linestring(obj)
        except Exception:
            skipped_geom += 1
            continue
        rows.append((int(obj.id), wkt))
        if len(rows) >= BATCH_SIZE:
            _flush()

    _flush()

    print(f"[waterway]   {pbf.name}: inserted {inserted:,} waterway buffers "
          f"(skipped {skipped_filter:,} filter, {skipped_geom:,} geom, "
          f"{skipped_bbox:,} bbox)")
    return inserted


def ingest(conn: psycopg.Connection,
           pbfs: Iterable[Path],
           countries: Iterable[str],
           bbox: tuple[float, float, float, float] | None = None) -> None:
    """Populate landcover with waterway buffers for the given countries.

    `pbfs` should be the *full* country PBFs (e.g.
    `austria-latest.osm.pbf`), not the landuse extracts — those drop
    most waterway lines.
    """
    pbfs = list(pbfs)
    countries = list(countries)
    assert len(pbfs) == len(countries), \
        "pbfs and countries must be parallel lists"

    # The landcover table itself is created by ingest_landcover.ingest
    # or ingest_coastline.ingest. We assume it already exists; if not,
    # the COPY will fail loudly — fine, the caller should run
    # landcover-ingest first.

    with conn.cursor() as cur:
        # Only wipe waterway-class rows for these countries so forest /
        # water / wetland from the polygon ingest are preserved.
        cur.execute(
            "DELETE FROM landcover "
            "WHERE country = ANY(%s) AND class = 'waterway'",
            (countries,),
        )
        deleted = cur.rowcount
    conn.commit()
    if deleted:
        print(f"[waterway] cleared {deleted:,} prior waterway rows for "
              f"countries={countries}")

    total = 0
    for pbf, country in zip(pbfs, countries):
        if not pbf.exists():
            raise SystemExit(f"missing country PBF: {pbf}")
        print(f"[waterway] streaming {pbf.name} (country={country})"
              + (f" bbox={bbox}" if bbox else ""))
        total += _stage_pbf(conn, pbf, country, bbox=bbox)
    print(f"[waterway] {total:,} waterway polygons total")

    print("[waterway] ANALYZE landcover")
    with conn.cursor() as cur:
        cur.execute("ANALYZE landcover")
    conn.commit()
