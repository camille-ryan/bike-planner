"""Dedup duplicate (osm_way_id, source, target) rows in ways while
preserving scenicness signals (water/sea/waterway/wetland).

Approach:
  1. Build dup_canon: one row per duplicate triple, holding keep_gid (MIN)
     and MAX of every scenicness column across the dup group.
  2. UPDATE the kept row with the aggregated scenicness so we don't
     lose any signal carried by a non-canonical duplicate.
  3. Materialize the drop set (gids to delete) into a small indexed
     table — needed because we have no index on (osm_way_id, source,
     target) and DELETE-by-gid uses the PK.
  4. DELETE the drop set.
  5. CREATE UNIQUE INDEX ways_dedup_idx ON ways(osm_way_id, source, target).
     Now the original ingest's incremental INSERT ... ON CONFLICT path
     works.
"""
import time
import psycopg

import config


SCENIC_COLS = [
    "water_local", "water_wide",
    "sea_local", "sea_wide",
    "waterway_along_edge", "waterway_local",
    "wetland_local",
]


def _step(cur, label: str, sql: str) -> None:
    t = time.time()
    print(f"[dedup] {label}…", flush=True)
    cur.execute(sql)
    print(f"[dedup]   {label}: {cur.rowcount:,} rows  ({time.time()-t:.1f}s)",
          flush=True)


def main() -> None:
    t0 = time.time()
    max_sel = ",\n  ".join(f"MAX({c}) AS max_{c}" for c in SCENIC_COLS)
    set_clause = ",\n  ".join(f"{c} = c.max_{c}" for c in SCENIC_COLS)

    with psycopg.connect(config.PG_DSN, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute("SET work_mem = '512MB'")
            cur.execute("SET maintenance_work_mem = '2GB'")

            _step(cur, "build dup_canon (GROUP BY only kept dups)", f"""
                DROP TABLE IF EXISTS dup_canon;
                CREATE UNLOGGED TABLE dup_canon AS
                SELECT
                  MIN(gid) AS keep_gid,
                  osm_way_id, source, target,
                  count(*) AS n_dups,
                  array_agg(gid) AS all_gids,
                  {max_sel}
                FROM ways
                GROUP BY osm_way_id, source, target
                HAVING count(*) > 1;
            """)
            cur.execute("ANALYZE dup_canon")

            _step(cur, "update keep_gid with aggregated scenicness", f"""
                UPDATE ways w SET
                  {set_clause}
                FROM dup_canon c
                WHERE w.gid = c.keep_gid;
            """)

            _step(cur, "build drop_gids table (PK-indexed)", """
                DROP TABLE IF EXISTS drop_gids;
                CREATE UNLOGGED TABLE drop_gids (gid bigint PRIMARY KEY);
                INSERT INTO drop_gids
                  SELECT unnest(all_gids) FROM dup_canon
                  EXCEPT
                  SELECT keep_gid FROM dup_canon;
            """)
            cur.execute("ANALYZE drop_gids")

            _step(cur, "DELETE non-canonical rows via PK", """
                DELETE FROM ways w
                USING drop_gids d
                WHERE w.gid = d.gid;
            """)

            _step(cur, "drop scratch tables", """
                DROP TABLE dup_canon;
                DROP TABLE drop_gids;
            """)

            _step(cur, "CREATE UNIQUE INDEX ways_dedup_idx", """
                CREATE UNIQUE INDEX ways_dedup_idx
                ON ways (osm_way_id, source, target);
            """)
            conn.commit()

    print(f"[dedup] DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
