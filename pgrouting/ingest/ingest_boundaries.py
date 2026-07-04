"""OSM admin polygon ingest -> anchors.geom_boundary.

For each PBF, stream all administrative areas via osmium.FileProcessor
(with area assembly + a key-filter on `boundary`), filter to admin_level
in {6, 7, 8} (the corridor's "city" / municipality range across AT, CZ,
DE, DK), and stage as multipolygons. After all PBFs are processed,
spatially-join each anchor to the smallest containing polygon and copy
the boundary into anchors.geom_boundary.

Anchors with no enclosing polygon keep geom_boundary=NULL — the SPT
seed step falls back to single-vertex seeding for those.

Why "smallest containing": a city anchor near a state border can be
inside both the municipality (admin_level=8) and the district
(admin_level=6) polygon; we want the most specific (smallest) one as
the city's boundary.
"""
from pathlib import Path
from typing import Iterable

import osmium
import osmium.geom
import osmium.filter
import psycopg


# Admin levels broadly corresponding to "city / town" boundaries across
# the corridor. 8 = Gemeinde (AT, DE), 7-8 = obec (CZ), 7 = kommune (DK).
ADMIN_LEVELS = {"6", "7", "8"}
BATCH_SIZE = 1000


def _stage_pbf(conn: psycopg.Connection, pbf: Path) -> int:
    """Stream admin multipolygons from one PBF into tmp_boundaries."""
    fp = (osmium.FileProcessor(str(pbf))
          .with_locations(
              # Disk-backed to keep RAM bounded on corridor-scale PBFs.
              "sparse_file_array,/tmp/osmium-boundaries.idx")
          .with_areas()
          .with_filter(osmium.filter.KeyFilter("boundary")))
    wkt_fac = osmium.geom.WKTFactory()

    rows: list[tuple[str | None, str, str]] = []
    inserted = 0
    skipped_invalid = 0
    seen_areas = 0

    for obj in fp:
        if not obj.is_area():
            continue
        seen_areas += 1
        tags = dict(obj.tags)
        if tags.get("boundary") != "administrative":
            continue
        if tags.get("admin_level") not in ADMIN_LEVELS:
            continue
        try:
            wkt = wkt_fac.create_multipolygon(obj)
        except Exception:
            skipped_invalid += 1
            continue
        rows.append((tags.get("name"), tags["admin_level"], wkt))
        if len(rows) >= BATCH_SIZE:
            inserted += _flush(conn, rows)
            rows.clear()
    if rows:
        inserted += _flush(conn, rows)
    print(f"[boundaries]   {pbf.name}: scanned {seen_areas:,} areas, "
          f"staged {inserted:,} admin polygons "
          f"(skipped {skipped_invalid:,} with invalid geometry)")
    return inserted


def _flush(conn: psycopg.Connection, rows: list) -> int:
    with conn.cursor() as cur:
        with cur.copy(
            "COPY tmp_boundaries (name, admin_level, geom) FROM STDIN"
        ) as cp:
            for name, lvl, wkt in rows:
                cp.write_row((name, lvl, f"SRID=4326;{wkt}"))
    conn.commit()
    return len(rows)


def ingest(conn: psycopg.Connection, pbfs: Iterable[Path]) -> None:
    """Top-level: build anchors.geom_boundary from corridor PBFs."""
    pbfs = list(pbfs)

    with conn.cursor() as cur:
        # Idempotent: schema.sql declares this column for fresh setups,
        # but the ALTER lets `boundaries` run against an older DB too.
        cur.execute("""
            ALTER TABLE anchors
            ADD COLUMN IF NOT EXISTS geom_boundary geometry(MultiPolygon, 4326)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS anchors_geom_boundary_idx
            ON anchors USING gist(geom_boundary)
        """)
        cur.execute("DROP TABLE IF EXISTS tmp_boundaries")
        cur.execute("""
            CREATE UNLOGGED TABLE tmp_boundaries (
                name        text,
                admin_level text NOT NULL,
                geom        geometry(MultiPolygon, 4326) NOT NULL
            )
        """)
    conn.commit()

    total = 0
    for pbf in pbfs:
        if not pbf.exists():
            raise SystemExit(f"missing PBF: {pbf}")
        print(f"[boundaries] streaming {pbf.name}")
        total += _stage_pbf(conn, pbf)
    print(f"[boundaries] {total:,} admin polygons total in tmp_boundaries")

    print("[boundaries] indexing tmp_boundaries.geom for spatial join...")
    with conn.cursor() as cur:
        cur.execute("CREATE INDEX ON tmp_boundaries USING gist(geom)")
        cur.execute("ANALYZE tmp_boundaries")
    conn.commit()

    print("[boundaries] matching anchors to smallest containing polygon...")
    with conn.cursor() as cur:
        cur.execute("UPDATE anchors SET geom_boundary = NULL")
        cur.execute("""
            UPDATE anchors a
            SET geom_boundary = m.geom
            FROM (
                SELECT DISTINCT ON (a2.id) a2.id AS aid, tb.geom
                FROM   anchors a2
                JOIN   tmp_boundaries tb ON ST_Contains(tb.geom, a2.geom)
                ORDER  BY a2.id, ST_Area(tb.geom) ASC
            ) m
            WHERE a.id = m.aid
        """)
        matched = cur.rowcount
        cur.execute("SELECT COUNT(*) FROM anchors WHERE snap_vertex_id IS NOT NULL")
        total_anchors = int(cur.fetchone()[0])
        print(f"[boundaries]   matched {matched:,} of {total_anchors:,} anchors "
              f"({total_anchors - matched:,} fall back to snap-vertex seeding)")
        cur.execute("DROP TABLE tmp_boundaries")
    conn.commit()
