"""OSM landcover ingest -> landcover table.

Streams `*-landuse.osm.pbf` extracts, assembles every area tagged
`landuse=forest` or `natural=wood`, and stages them as multipolygons
in the `landcover` table with class='forest'.

The table is per-country (the `country` column tags each row with
its source PBF) so a country can be re-ingested in isolation: each
invocation deletes the rows belonging to the countries it's about
to process, then re-inserts. Running Austria first and Czech later
preserves Austria.

Run order:
    ingest -> snap -> boundaries -> dem-* -> landcover-ingest ->
    canopy-compute -> recompute-cost -> spts -> paired

Source files live at config.OSM_DIR.parent/"landcover"/{country}-landuse.osm.pbf
(separate landuse extracts produced upstream so we don't have to
re-scan the full ~7 GB country PBFs to pick out polygons).
"""
from pathlib import Path
from typing import Iterable

import osmium
import osmium.geom
import osmium.filter
import psycopg

import config


BATCH_SIZE = 1000

# Tag combinations we accept as "tree cover." `landuse=forest` is the
# canonical land-use designation; `natural=wood` is the natural-feature
# variant — both correspond to "this polygon is wooded."
def _classify(tags: dict) -> str | None:
    if tags.get("landuse") == "forest":
        return "forest"
    if tags.get("natural") == "wood":
        return "forest"
    return None


def _area_overlaps_bbox(obj, bbox: tuple[float, float, float, float]) -> bool:
    """Cheap bbox-vs-area overlap test using the area's outer rings'
    node locations. Avoids the (much more expensive) WKT assembly
    upstream of the spatial join.

    bbox: (min_lon, min_lat, max_lon, max_lat).

    False positives (area's bounding-box overlaps but the area itself
    doesn't) are fine — they cost a wasted WKT + COPY for a polygon
    that the downstream ST_Intersects in canopy-compute will
    correctly ignore.
    """
    min_lon = min_lat = float("inf")
    max_lon = max_lat = float("-inf")
    for ring in obj.outer_rings():
        for n in ring:
            loc = n.location
            if not loc.valid():
                continue
            x, y = loc.lon, loc.lat
            if x < min_lon: min_lon = x
            if x > max_lon: max_lon = x
            if y < min_lat: min_lat = y
            if y > max_lat: max_lat = y
    if min_lon == float("inf"):
        return False  # no valid nodes; skip
    return not (
        min_lon > bbox[2] or
        max_lon < bbox[0] or
        min_lat > bbox[3] or
        max_lat < bbox[1]
    )


def _stage_pbf(conn: psycopg.Connection,
               pbf: Path,
               country: str,
               bbox: tuple[float, float, float, float] | None = None) -> int:
    """Stream forest/wood multipolygons from one PBF into landcover.

    When `bbox` is supplied, skip any area whose envelope falls
    entirely outside the bbox. Polygons that straddle the boundary
    are kept whole — clipping happens later, in the canopy SQL.
    """
    fp = (osmium.FileProcessor(str(pbf))
          .with_locations(
              "sparse_file_array,/tmp/osmium-landcover.idx")
          .with_areas()
          # KeyFilter accepts any of the given keys — captures both
          # landuse=* and natural=* features, then `_classify` does the
          # final filter on the value.
          .with_filter(osmium.filter.KeyFilter("landuse", "natural")))
    wkt_fac = osmium.geom.WKTFactory()

    rows: list[tuple[int | None, str, str, str]] = []
    inserted = 0
    skipped_invalid = 0
    skipped_bbox = 0
    seen_areas = 0

    for obj in fp:
        if not obj.is_area():
            continue
        seen_areas += 1
        tags = dict(obj.tags)
        cls = _classify(tags)
        if cls is None:
            continue
        if bbox is not None and not _area_overlaps_bbox(obj, bbox):
            skipped_bbox += 1
            continue
        try:
            wkt = wkt_fac.create_multipolygon(obj)
        except Exception:
            skipped_invalid += 1
            continue
        # osmium areas can be backed by either a closed way (osm_id is
        # the way id) or a relation (osm_id is the relation id with a
        # transformation: id*2 for ways, id*2+1 for relations in
        # osmium's area ids). We store the area's `id` directly — it's
        # only used for debugging, not for joins.
        rows.append((int(obj.id), country, cls, wkt))
        if len(rows) >= BATCH_SIZE:
            inserted += _flush(conn, rows)
            rows.clear()
    if rows:
        inserted += _flush(conn, rows)
    print(f"[landcover]   {pbf.name}: scanned {seen_areas:,} areas, "
          f"staged {inserted:,} forest polygons "
          f"(skipped {skipped_invalid:,} invalid, "
          f"{skipped_bbox:,} outside bbox)")
    return inserted


def _flush(conn: psycopg.Connection, rows: list) -> int:
    with conn.cursor() as cur:
        with cur.copy(
            "COPY landcover (osm_id, country, class, geom) FROM STDIN"
        ) as cp:
            for osm_id, country, cls, wkt in rows:
                cp.write_row((osm_id, country, cls, f"SRID=4326;{wkt}"))
    conn.commit()
    return len(rows)


def ingest(conn: psycopg.Connection,
           pbfs: Iterable[Path],
           countries: Iterable[str],
           bbox: tuple[float, float, float, float] | None = None) -> None:
    """Top-level: populate `landcover` for the given countries.

    `pbfs` and `countries` are positionally paired. Each country's
    existing rows are deleted before its PBF is re-ingested so the
    step is idempotent at the country level.

    `bbox` (min_lon, min_lat, max_lon, max_lat), if supplied, skips
    any polygon whose envelope falls outside the box — much faster for
    iteration runs over a known pathfinding corridor. NB: a country
    re-ingested with a narrower bbox effectively *shrinks* that
    country's coverage in `landcover`, since the per-country DELETE
    runs first. A subsequent full-country re-ingest is needed if you
    later want full coverage.
    """
    pbfs = list(pbfs)
    countries = list(countries)
    assert len(pbfs) == len(countries), \
        "pbfs and countries must be parallel lists"

    with conn.cursor() as cur:
        # Schema declares these in fresh setups, but the ALTERs let an
        # older DB pick up the new table without manual intervention.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS landcover (
                id      bigserial PRIMARY KEY,
                osm_id  bigint,
                country text NOT NULL,
                class   text NOT NULL,
                geom    geometry(MultiPolygon, 4326) NOT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS landcover_geom_idx "
            "ON landcover USING gist(geom)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS landcover_class_idx "
            "ON landcover(class)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS landcover_country_idx "
            "ON landcover(country)"
        )
    conn.commit()

    # Per-country wipe & re-ingest. Avoid TRUNCATE — leaves other
    # countries' rows intact.
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM landcover WHERE country = ANY(%s)",
            (countries,),
        )
        deleted = cur.rowcount
    conn.commit()
    if deleted:
        print(f"[landcover] cleared {deleted:,} existing rows for "
              f"countries={countries}")

    total = 0
    for pbf, country in zip(pbfs, countries):
        if not pbf.exists():
            raise SystemExit(f"missing landuse PBF: {pbf}")
        print(f"[landcover] streaming {pbf.name} (country={country})"
              + (f" bbox={bbox}" if bbox else ""))
        total += _stage_pbf(conn, pbf, country, bbox=bbox)
    print(f"[landcover] {total:,} forest polygons total")

    print("[landcover] ANALYZE landcover")
    with conn.cursor() as cur:
        cur.execute("ANALYZE landcover")
    conn.commit()
