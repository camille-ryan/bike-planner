"""Denormalize the paved-subgraph JOIN into a single table with a GIST
spatial index, so per-tile chain-graph queries become indexed range
scans instead of 4-table nested-loop joins.

Source query (the slow one):
    SELECT w.source, w.target, w.length_m,
           ST_X(vs.the_geom), ST_Y(vs.the_geom),
           ST_X(vt.the_geom), ST_Y(vt.the_geom)
    FROM ways w
    JOIN way_tags wt ON wt.osm_way_id = w.osm_way_id
    JOIN ways_vertices_pgr vs ON vs.id = w.source
    JOIN ways_vertices_pgr vt ON vt.id = w.target
    WHERE wt.highway IN (...) AND w.length_m > 0

Pre-baked table `ways_paved` carries everything inline:
    gid bigint PK
    src_id, dst_id bigint           -- original ways.source/target
    length_m double precision
    src_lon, src_lat double precision
    dst_lon, dst_lat double precision
    src_pt geometry(Point, 4326)    -- ST_MakePoint(src_lon, src_lat)
                                    -- GIST-indexed for tile lookups

Per-tile chain-graph query then drops to a single GIST range probe.
"""
import time
import psycopg

import config


PAVED_HIGHWAYS = (
    "motorway", "trunk", "primary", "secondary",
    "tertiary", "unclassified", "residential",
    "living_street",
    "primary_link", "secondary_link", "tertiary_link",
    "trunk_link", "motorway_link", "road",
)


def _step(cur, label: str, sql: str, params=None) -> None:
    t = time.time()
    print(f"[ways_paved] {label}…", flush=True)
    if params is None:
        cur.execute(sql)
    else:
        cur.execute(sql, params)
    print(f"[ways_paved]   {label}: {cur.rowcount:,} rows  "
          f"({time.time()-t:.1f}s)", flush=True)


def main() -> None:
    t0 = time.time()
    placeholders = ",".join(["%s"] * len(PAVED_HIGHWAYS))
    with psycopg.connect(config.PG_DSN, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute("SET work_mem = '512MB'")
            cur.execute("SET maintenance_work_mem = '2GB'")

            _step(cur, "drop existing ways_paved", """
                DROP TABLE IF EXISTS ways_paved;
            """)
            _step(cur, "create ways_paved (no indexes)", """
                CREATE TABLE ways_paved (
                    gid       bigint PRIMARY KEY,
                    src_id    bigint NOT NULL,
                    dst_id    bigint NOT NULL,
                    length_m  double precision NOT NULL,
                    src_lon   double precision NOT NULL,
                    src_lat   double precision NOT NULL,
                    dst_lon   double precision NOT NULL,
                    dst_lat   double precision NOT NULL
                );
            """)

            # No ORDER BY. The prior ORDER BY ST_GeoHash spatial sort
            # was the multi-hour bottleneck (~10-27 h on 4-country data).
            # GIST bbox lookups from connect_anchors_pairs.py walk the
            # index directly and don't care about heap-page clustering,
            # so dropping the sort brings build to ~30-60 min with no
            # runtime query regression.
            _step(cur, "INSERT from JOIN (one-time bulk, unordered)", f"""
                INSERT INTO ways_paved
                  (gid, src_id, dst_id, length_m,
                   src_lon, src_lat, dst_lon, dst_lat)
                SELECT w.gid, w.source, w.target, w.length_m,
                       ST_X(vs.the_geom), ST_Y(vs.the_geom),
                       ST_X(vt.the_geom), ST_Y(vt.the_geom)
                FROM ways w
                JOIN way_tags wt ON wt.osm_way_id = w.osm_way_id
                JOIN ways_vertices_pgr vs ON vs.id = w.source
                JOIN ways_vertices_pgr vt ON vt.id = w.target
                WHERE wt.highway IN ({placeholders})
                  AND w.length_m > 0.0;
            """, list(PAVED_HIGHWAYS))

            _step(cur, "add src_pt geometry column", """
                ALTER TABLE ways_paved
                ADD COLUMN src_pt geometry(Point, 4326)
                GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(src_lon, src_lat), 4326)) STORED;
            """)

            _step(cur, "CREATE GIST INDEX ways_paved_src_pt_idx", """
                CREATE INDEX ways_paved_src_pt_idx
                ON ways_paved USING gist (src_pt);
            """)
            _step(cur, "ANALYZE ways_paved", """
                ANALYZE ways_paved;
            """)
        conn.commit()

    print(f"[ways_paved] DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
