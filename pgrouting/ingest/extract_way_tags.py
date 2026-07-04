"""Build the way_tags sidecar table from source OSM PBFs.

osm2pgrouting drops `name`, `ref`, `highway`, `access`, `bicycle`,
`surface` during ingest, but downstream features need them:

  * way-graph flood-fill: extend the primary chain along same-named
    lower-class segments (uses name + ref).
  * way-graph chain builder (this file's primary consumer): filter `ways`
    to the drivable subset by JOINing on osm_way_id and checking highway
    (since osm2pgrouting + the existing 121M-row ways table no longer
    carry the highway column).
  * future turn-cost penalty: cheap when the route stays on the same
    named road across a vertex, expensive when it turns onto a
    different one.

Schema:

    way_tags(
      osm_way_id BIGINT PK,
      name       TEXT NOT NULL DEFAULT '',
      ref        TEXT NOT NULL DEFAULT '',
      highway    TEXT NOT NULL DEFAULT '',
      access     TEXT NOT NULL DEFAULT '',
      bicycle    TEXT NOT NULL DEFAULT '',
      surface    TEXT NOT NULL DEFAULT ''
    )

Rows are stored for any way with `highway` set, OR with at least one of
name/ref present. The chain graph filters on `highway IN (...)` so we
need every drivable way represented even if it's unnamed.

Run inside the pgrouting container:
    docker compose run --rm -T pgrouting python3 /app/extract_way_tags.py
"""
from __future__ import annotations

import time
from pathlib import Path

import osmium
import psycopg

import config


PBFS = [
    "austria-latest.osm.pbf",
    "czech-republic-latest.osm.pbf",
    "germany-latest.osm.pbf",
    "denmark-latest.osm.pbf",
]
BATCH_SIZE = 50_000


def _ensure_table(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS way_tags (
                osm_way_id BIGINT PRIMARY KEY,
                name       TEXT NOT NULL DEFAULT '',
                ref        TEXT NOT NULL DEFAULT '',
                highway    TEXT NOT NULL DEFAULT '',
                access     TEXT NOT NULL DEFAULT '',
                bicycle    TEXT NOT NULL DEFAULT '',
                surface    TEXT NOT NULL DEFAULT ''
            );
        """)
        cur.execute("ALTER TABLE way_tags ADD COLUMN IF NOT EXISTS highway TEXT NOT NULL DEFAULT '';")
        cur.execute("ALTER TABLE way_tags ADD COLUMN IF NOT EXISTS access  TEXT NOT NULL DEFAULT '';")
        cur.execute("ALTER TABLE way_tags ADD COLUMN IF NOT EXISTS bicycle TEXT NOT NULL DEFAULT '';")
        cur.execute("ALTER TABLE way_tags ADD COLUMN IF NOT EXISTS surface TEXT NOT NULL DEFAULT '';")
        cur.execute("CREATE INDEX IF NOT EXISTS way_tags_name_idx    ON way_tags (name)    WHERE name <> '';")
        cur.execute("CREATE INDEX IF NOT EXISTS way_tags_ref_idx     ON way_tags (ref)     WHERE ref  <> '';")
        cur.execute("CREATE INDEX IF NOT EXISTS way_tags_highway_idx ON way_tags (highway) WHERE highway <> '';")
    conn.commit()


def _stage_pbf(conn: psycopg.Connection, pbf: Path) -> tuple[int, int]:
    """Stream ways from one PBF, stage (osm_way_id, name, ref, highway,
    access, bicycle, surface) for any way with `highway` set OR
    name/ref present. Returns (seen, written)."""
    t0 = time.time()
    fp = osmium.FileProcessor(str(pbf))

    rows: list[tuple[int, str, str, str, str, str, str]] = []
    seen_ways = 0
    written = 0

    with conn.cursor() as cur:
        for obj in fp:
            if not obj.is_way():
                continue
            seen_ways += 1
            tags = obj.tags
            highway = tags.get("highway", "") or ""
            if not highway and "name" not in tags and "ref" not in tags:
                continue
            name    = tags.get("name", "") or ""
            ref     = tags.get("ref",  "") or ""
            if not highway and not name and not ref:
                continue
            access  = tags.get("access",  "") or ""
            bicycle = tags.get("bicycle", "") or ""
            surface = tags.get("surface", "") or ""
            rows.append((int(obj.id), name, ref, highway, access, bicycle, surface))
            if len(rows) >= BATCH_SIZE:
                written += _flush(cur, rows)
                rows.clear()
                if seen_ways % 500_000 == 0:
                    print(f"[way-tags]   {pbf.name}: scanned {seen_ways:,} ways, "
                          f"written {written:,} ({time.time()-t0:.1f}s)",
                          flush=True)
        if rows:
            written += _flush(cur, rows)
    conn.commit()
    print(f"[way-tags]   {pbf.name}: scanned {seen_ways:,} ways, "
          f"wrote {written:,} tagged ways in {time.time()-t0:.1f}s",
          flush=True)
    return seen_ways, written


def _flush(cur, rows: list[tuple[int, str, str, str, str, str, str]]) -> int:
    with cur.copy(
        "COPY way_tags_staging (osm_way_id, name, ref, highway, access, bicycle, surface) FROM STDIN"
    ) as cp:
        for row in rows:
            cp.write_row(row)
    return len(rows)


def main() -> None:
    t0 = time.time()
    print(f"[way-tags] extracting from {len(PBFS)} PBF(s) into way_tags table",
          flush=True)

    with psycopg.connect(config.PG_DSN) as conn:
        _ensure_table(conn)
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS way_tags_staging;")
            cur.execute("""
                CREATE UNLOGGED TABLE way_tags_staging (
                    osm_way_id BIGINT,
                    name       TEXT,
                    ref        TEXT,
                    highway    TEXT,
                    access     TEXT,
                    bicycle    TEXT,
                    surface    TEXT
                );
            """)
        conn.commit()

        total_seen = 0
        total_written = 0
        for pbf_name in PBFS:
            pbf = Path("/data/osm") / pbf_name
            if not pbf.exists():
                print(f"[way-tags] SKIP {pbf_name} (not found)", flush=True)
                continue
            seen, written = _stage_pbf(conn, pbf)
            total_seen += seen
            total_written += written

        with conn.cursor() as cur:
            print(f"[way-tags] merging staging → way_tags (last-write-wins on osm_way_id)",
                  flush=True)
            cur.execute("""
                INSERT INTO way_tags (osm_way_id, name, ref, highway, access, bicycle, surface)
                SELECT DISTINCT ON (osm_way_id)
                       osm_way_id, name, ref, highway, access, bicycle, surface
                FROM way_tags_staging
                ORDER BY osm_way_id
                ON CONFLICT (osm_way_id) DO UPDATE
                SET name = EXCLUDED.name, ref = EXCLUDED.ref,
                    highway = EXCLUDED.highway, access = EXCLUDED.access,
                    bicycle = EXCLUDED.bicycle, surface = EXCLUDED.surface;
            """)
            cur.execute("DROP TABLE way_tags_staging;")
            cur.execute(
                "SELECT count(*), "
                "count(*) FILTER (WHERE name <> '') AS with_name, "
                "count(*) FILTER (WHERE ref  <> '') AS with_ref, "
                "count(*) FILTER (WHERE highway <> '') AS with_hw "
                "FROM way_tags;"
            )
            n_total, n_name, n_ref, n_hw = cur.fetchone()
        conn.commit()

    print(f"[way-tags] DONE in {time.time()-t0:.1f}s — "
          f"way_tags has {n_total:,} rows "
          f"({n_name:,} with name, {n_ref:,} with ref, {n_hw:,} with highway) "
          f"from {total_seen:,} ways scanned",
          flush=True)


if __name__ == "__main__":
    main()
