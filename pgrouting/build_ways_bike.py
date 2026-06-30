"""Bike-routing equivalent of ways_paved: denormalize the per-edge data
the bike SPT compute needs into one table with a GIST spatial index on
the source vertex.

What bike SPT needs per edge:
  - source/target vertex ids                  (graph topology)
  - cost / reverse_cost / length_m            (Dijkstra weights)
  - src_lon / src_lat / dst_lon / dst_lat     (for path geom reconstruction)
  - src_pt geometry, GIST-indexed             (for ST_Contains polygon test)

Filter: WHERE NOT bike_excluded — the drivable-only supplementary rows
(motorways, bike-banned tunnels) don't participate in bike routing.

Build once (~15-25 min), then per-anchor polygon-bounded SPT compute
runs each query as an indexed range scan against this purpose-built
table instead of a 4-table join over ways/way_tags/ways_vertices_pgr.
"""
import time
import psycopg

import config


def _step(cur, label: str, sql: str) -> None:
    t = time.time()
    print(f"[ways_bike] {label}…", flush=True)
    cur.execute(sql)
    print(f"[ways_bike]   {label}: {cur.rowcount:,} rows  "
          f"({time.time()-t:.1f}s)", flush=True)


def main() -> None:
    t0 = time.time()
    with psycopg.connect(config.PG_DSN, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute("SET work_mem = '512MB'")
            cur.execute("SET maintenance_work_mem = '2GB'")

            _step(cur, "drop existing ways_bike", """
                DROP TABLE IF EXISTS ways_bike;
            """)
            _step(cur, "create ways_bike (no indexes)", """
                CREATE TABLE ways_bike (
                    gid                 bigint PRIMARY KEY,
                    src_id              bigint NOT NULL,
                    dst_id              bigint NOT NULL,
                    cost                double precision NOT NULL,
                    reverse_cost        double precision NOT NULL,
                    length_m            double precision NOT NULL,
                    src_lon             double precision NOT NULL,
                    src_lat             double precision NOT NULL,
                    dst_lon             double precision NOT NULL,
                    dst_lat             double precision NOT NULL,
                    cost_views          real,
                    reverse_cost_views  real
                );
            """)

            # Include cost_views columns in the bulk INSERT — avoids a
            # separate ~114M-row UPDATE post-build that would otherwise
            # MVCC-thrash the heap (failure mode 13). Adding the cols
            # is essentially free during the seq scan.
            # ORDER BY ST_GeoHash removed — at 4-country scale (127M
            # JOIN'd rows) the external sort with 512 MB work_mem ran
            # 27+ hours on WSL ext4 before being killed. We do a
            # separate CLUSTER USING the GIST index after this completes
            # to achieve the same spatial heap layout in a cleaner
            # two-step process. Mirrors the cluster_vertices_ctas pattern.
            _step(cur, "INSERT bikeable rows from JOIN (one-time bulk)", """
                INSERT INTO ways_bike
                  (gid, src_id, dst_id, cost, reverse_cost, length_m,
                   src_lon, src_lat, dst_lon, dst_lat,
                   cost_views, reverse_cost_views)
                SELECT w.gid, w.source, w.target,
                       w.cost, w.reverse_cost, w.length_m,
                       ST_X(vs.the_geom), ST_Y(vs.the_geom),
                       ST_X(vt.the_geom), ST_Y(vt.the_geom),
                       w.cost_views, w.reverse_cost_views
                FROM ways w
                JOIN ways_vertices_pgr vs ON vs.id = w.source
                JOIN ways_vertices_pgr vt ON vt.id = w.target
                WHERE NOT w.bike_excluded
                  AND w.length_m > 0.0;
            """)

            _step(cur, "add src_pt generated geometry column", """
                ALTER TABLE ways_bike
                ADD COLUMN src_pt geometry(Point, 4326)
                GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(src_lon, src_lat), 4326)) STORED;
            """)
            _step(cur, "CREATE GIST INDEX ways_bike_src_pt_idx", """
                CREATE INDEX ways_bike_src_pt_idx
                ON ways_bike USING gist (src_pt);
            """)
            _step(cur, "ANALYZE ways_bike", """
                ANALYZE ways_bike;
            """)
        conn.commit()
    print(f"[ways_bike] DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
