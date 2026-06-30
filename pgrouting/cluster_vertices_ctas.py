"""Spatially reorder ways_vertices_pgr via CTAS.

The native `CLUSTER ways_vertices_pgr USING ways_vertices_pgr_geom_idx`
ran at 0.9% per 2h on the 4-country (111M-row, 12 GB) table because
its index-driven scan is itself bottlenecked by the random-IO heap
fetches we're trying to fix. ETA was projected at 9+ days.

CTAS does the same job differently: SEQUENTIAL read of the source
heap + in-memory/on-disk sort by ST_GeoHash, then sequential write
of the new heap. ~10-30 min for the read+sort+write vs 9+ days for
CLUSTER. Same end result: heap physically ordered by spatial
proximity, so subsequent ST_DWithin queries do sequential page reads
instead of scattered random reads.

Strategy mirrors ingest_dem_ctas.py — LOGGED throughout, atomic swap,
FK rebuild.
"""
import time
import psycopg
import config


def main() -> None:
    print("[cluster-vertices] starting CTAS spatial reorder ...", flush=True)
    with psycopg.connect(config.PG_DSN) as conn:

        # 1. Drop stale ways_vertices_pgr_new if any.
        print("[cluster-vertices] step 1: drop stale ways_vertices_pgr_new ...",
              flush=True)
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS ways_vertices_pgr_new CASCADE")
        conn.commit()

        # 2. CREATE LOGGED ways_vertices_pgr_new with same structure +
        # populated by SELECT ORDER BY ST_GeoHash. Skip GENERATED column
        # (the_geom) in column list — it will be re-generated on insert
        # via INCLUDING GENERATED in LIKE.
        print("[cluster-vertices] step 2: CREATE LOGGED + INSERT "
              "(seq-scan + sort by ST_GeoHash) ...", flush=True)
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE ways_vertices_pgr_new "
                "(LIKE ways_vertices_pgr INCLUDING DEFAULTS INCLUDING GENERATED)"
            )
        conn.commit()

        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute("SET work_mem = '1GB'")
            cur.execute("SET maintenance_work_mem = '4GB'")
            cur.execute("SET max_parallel_workers_per_gather = 4")
            cur.execute("""
                INSERT INTO ways_vertices_pgr_new (id, osm_id, lon, lat, elev_m)
                SELECT id, osm_id, lon, lat, elev_m
                  FROM ways_vertices_pgr
                 ORDER BY ST_GeoHash(the_geom, 12)
            """)
            rc = cur.rowcount
        conn.commit()
        print(f"[cluster-vertices]   INSERT done: {rc:,} rows in "
              f"{time.time()-t0:.1f}s", flush=True)

        # 3. Build indexes on new table.
        print("[cluster-vertices] step 3: build indexes ...", flush=True)
        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute("SET maintenance_work_mem = '4GB'")
            cur.execute("ALTER TABLE ways_vertices_pgr_new ADD PRIMARY KEY (id)")
            cur.execute(
                "ALTER TABLE ways_vertices_pgr_new "
                "ADD CONSTRAINT ways_vertices_pgr_new_osm_id_key UNIQUE (osm_id)"
            )
            cur.execute(
                "CREATE INDEX ways_vertices_pgr_new_geom_idx "
                "ON ways_vertices_pgr_new USING gist (the_geom)"
            )
            cur.execute(
                "CREATE INDEX ways_vertices_null_elev_new_idx "
                "ON ways_vertices_pgr_new (lat, lon) WHERE elev_m IS NULL"
            )
        conn.commit()
        print(f"[cluster-vertices]   indexes built in {time.time()-t0:.1f}s",
              flush=True)

        # 4. Atomic swap + FK rebuild (anchors.snap_vertex_id → ways_vertices_pgr.id).
        print("[cluster-vertices] step 4: swap tables + rebuild FK ...",
              flush=True)
        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute("SELECT last_value FROM ways_vertices_pgr_id_seq")
            seq_val = cur.fetchone()[0]
            cur.execute("DROP TABLE ways_vertices_pgr CASCADE")
            cur.execute(
                "ALTER INDEX ways_vertices_pgr_new_pkey "
                "RENAME TO ways_vertices_pgr_pkey"
            )
            cur.execute(
                "ALTER INDEX ways_vertices_pgr_new_osm_id_key "
                "RENAME TO ways_vertices_pgr_osm_id_key"
            )
            cur.execute(
                "ALTER INDEX ways_vertices_pgr_new_geom_idx "
                "RENAME TO ways_vertices_pgr_geom_idx"
            )
            cur.execute(
                "ALTER INDEX ways_vertices_null_elev_new_idx "
                "RENAME TO ways_vertices_null_elev_idx"
            )
            cur.execute(
                "ALTER TABLE ways_vertices_pgr_new RENAME TO ways_vertices_pgr"
            )
            cur.execute("CREATE SEQUENCE ways_vertices_pgr_id_seq")
            cur.execute(f"SELECT setval('ways_vertices_pgr_id_seq', {seq_val})")
            cur.execute(
                "ALTER TABLE ways_vertices_pgr ALTER COLUMN id "
                "SET DEFAULT nextval('ways_vertices_pgr_id_seq')"
            )
            cur.execute(
                "ALTER SEQUENCE ways_vertices_pgr_id_seq "
                "OWNED BY ways_vertices_pgr.id"
            )
            cur.execute(
                "ALTER TABLE anchors "
                "ADD CONSTRAINT anchors_snap_vertex_id_fkey "
                "FOREIGN KEY (snap_vertex_id) "
                "REFERENCES ways_vertices_pgr(id)"
            )
        conn.commit()
        print(f"[cluster-vertices]   swap + FK done in {time.time()-t0:.1f}s",
              flush=True)

        # 5. ANALYZE to refresh stats.
        print("[cluster-vertices] step 5: ANALYZE ...", flush=True)
        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute("ANALYZE ways_vertices_pgr")
        conn.commit()
        print(f"[cluster-vertices]   ANALYZE done in {time.time()-t0:.1f}s",
              flush=True)

    print("[cluster-vertices] DONE", flush=True)


if __name__ == "__main__":
    main()
