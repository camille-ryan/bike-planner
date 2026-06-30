"""Backfill highway/surface/access/bicycle on `ways` from `way_tags`.

After supplement-off ingest, the new bikeable rows in `ways` have empty
tag columns (the ingest writes only 8 cols for speed). `recompute-cost`
needs them populated (returns None for empty `highway` → excludes the
edge from routing).

Strategy: **CTAS + swap** (CREATE TABLE AS SELECT, then DROP+RENAME).
A single UPDATE on 92M rows of a 14GB heap with shared_buffers=4GB
thrashes at ~1 MB/sec because MVCC creates new tuple versions that
spill across pages, triggering checkpoint amplification (~15× write
amp observed: 233 GB written to commit ~5 GB of new tuple data).

CTAS writes sequentially to a fresh heap (no dead tuples, no
checkpoint amplification), then a tiny metadata-only DROP+RENAME
swaps the table. Total: ~10-15 min for the heap write + ~10-20 min
for index rebuilds.

NOT restart-survivable: the CTAS is a single transaction. If
interrupted, ways_new is dropped on rollback and we start over.
But each attempt is ~30-45 min vs ~hours for UPDATE.
"""
from __future__ import annotations

import time

import psycopg

import config


_INSERT_COLS_AND_SELECT = """
INSERT INTO ways_new (
    gid, osm_way_id, source, target, cost, reverse_cost,
    length_m, is_ferry,
    highway, surface, tracktype, oneway, bicycle, cycleway, bicycle_road, access,
    curv_fwd, curv_rev, grade_pct,
    canopy_frac, canopy_frac_polygon,
    forest_local, forest_wide,
    view_dominance, local_relief, regional_relief, distance_to_drama,
    water_local, water_wide, sea_local, sea_wide,
    waterway_along_edge, waterway_local, wetland_local,
    vineyard_local, viewpoint_local, viewpoint_regional,
    cost_direct, reverse_cost_direct,
    cost_vineyard_lover, reverse_cost_vineyard_lover,
    cost_forest_lover, reverse_cost_forest_lover,
    cost_views, reverse_cost_views,
    cost_water, reverse_cost_water,
    bike_excluded)
SELECT
    w.gid, w.osm_way_id, w.source, w.target, w.cost, w.reverse_cost,
    w.length_m, w.is_ferry,
    COALESCE(NULLIF(wt.highway, ''), w.highway),
    COALESCE(NULLIF(wt.surface, ''), w.surface),
    w.tracktype, w.oneway,
    COALESCE(NULLIF(wt.bicycle, ''), w.bicycle),
    w.cycleway, w.bicycle_road,
    COALESCE(NULLIF(wt.access,  ''), w.access),
    w.curv_fwd, w.curv_rev, w.grade_pct,
    w.canopy_frac, w.canopy_frac_polygon,
    w.forest_local, w.forest_wide,
    w.view_dominance, w.local_relief, w.regional_relief, w.distance_to_drama,
    w.water_local, w.water_wide, w.sea_local, w.sea_wide,
    w.waterway_along_edge, w.waterway_local, w.wetland_local,
    w.vineyard_local, w.viewpoint_local, w.viewpoint_regional,
    w.cost_direct, w.reverse_cost_direct,
    w.cost_vineyard_lover, w.reverse_cost_vineyard_lover,
    w.cost_forest_lover, w.reverse_cost_forest_lover,
    w.cost_views, w.reverse_cost_views,
    w.cost_water, w.reverse_cost_water,
    w.bike_excluded
FROM ways w
LEFT JOIN way_tags wt ON wt.osm_way_id = w.osm_way_id
"""


def backfill() -> None:
    print("[backfill-ctas] starting CTAS-based backfill ...", flush=True)
    with psycopg.connect(config.PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM ways WHERE highway = ''")
            n_pending = cur.fetchone()[0]
        if n_pending == 0:
            print("[backfill-ctas] no rows to backfill, exiting", flush=True)
            return
        print(f"[backfill-ctas] {n_pending:,} rows need backfill, "
              "running full CTAS+swap", flush=True)

        # 1. Drop ways_new if it exists from a prior interrupted run.
        print("[backfill-ctas] step 1: drop stale ways_new ...", flush=True)
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS ways_new CASCADE")
        conn.commit()

        # 2. Create ways_new with same column types but no indexes/defaults.
        # Using LIKE INCLUDING DEFAULTS preserves the gid sequence default;
        # we'll handle the sequence specially during swap.
        print("[backfill-ctas] step 2: create ways_new (LIKE ways) ...",
              flush=True)
        with conn.cursor() as cur:
            cur.execute("CREATE UNLOGGED TABLE ways_new (LIKE ways INCLUDING DEFAULTS)")
        conn.commit()

        # 3. Bulk INSERT joined values. UNLOGGED skips WAL — huge win
        # for sequential write. We'll switch to LOGGED after.
        print("[backfill-ctas] step 3: INSERT joined data (UNLOGGED, no WAL) ...",
              flush=True)
        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute("SET work_mem = '1GB'")
            cur.execute("SET maintenance_work_mem = '2GB'")
            cur.execute("SET max_parallel_workers_per_gather = 4")
            cur.execute(_INSERT_COLS_AND_SELECT)
            rc = cur.rowcount
        conn.commit()
        print(f"[backfill-ctas]   INSERT done: {rc:,} rows in "
              f"{time.time()-t0:.1f}s", flush=True)

        # 4. Convert to LOGGED (forces a single big WAL write for the table).
        t0 = time.time()
        print("[backfill-ctas] step 4: ALTER TABLE ways_new SET LOGGED ...",
              flush=True)
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE ways_new SET LOGGED")
        conn.commit()
        print(f"[backfill-ctas]   LOGGED in {time.time()-t0:.1f}s", flush=True)

        # 5. Build indexes on ways_new (one-shot, fast on full table).
        t0 = time.time()
        print("[backfill-ctas] step 5: build indexes on ways_new ...",
              flush=True)
        with conn.cursor() as cur:
            cur.execute("SET maintenance_work_mem = '2GB'")
            cur.execute("ALTER TABLE ways_new ADD PRIMARY KEY (gid)")
            cur.execute(
                "CREATE UNIQUE INDEX ways_new_dedup_idx "
                "ON ways_new (osm_way_id, source, target)"
            )
            cur.execute("CREATE INDEX ways_new_source_idx ON ways_new(source)")
            cur.execute("CREATE INDEX ways_new_target_idx ON ways_new(target)")
        conn.commit()
        print(f"[backfill-ctas]   indexes built in {time.time()-t0:.1f}s",
              flush=True)

        # 6. Atomic swap: drop ways, rename ways_new, fix sequence ownership,
        # restart sequence from max(gid)+1.
        t0 = time.time()
        print("[backfill-ctas] step 6: swap tables ...", flush=True)
        with conn.cursor() as cur:
            # Save current sequence value (for the new ways)
            cur.execute("SELECT last_value FROM ways_gid_seq")
            seq_val = cur.fetchone()[0]
            # Drop indexes on the old table by their default names
            cur.execute("DROP TABLE ways CASCADE")
            # Rename the indexes on ways_new to original names
            cur.execute("ALTER INDEX ways_new_pkey RENAME TO ways_pkey")
            cur.execute("ALTER INDEX ways_new_dedup_idx RENAME TO ways_dedup_idx")
            cur.execute("ALTER INDEX ways_new_source_idx RENAME TO ways_source_idx")
            cur.execute("ALTER INDEX ways_new_target_idx RENAME TO ways_target_idx")
            cur.execute("ALTER TABLE ways_new RENAME TO ways")
            # Recreate the sequence (DROP CASCADE removed it) and link to gid.
            cur.execute("CREATE SEQUENCE ways_gid_seq")
            cur.execute(f"SELECT setval('ways_gid_seq', {seq_val})")
            cur.execute(
                "ALTER TABLE ways ALTER COLUMN gid "
                "SET DEFAULT nextval('ways_gid_seq')"
            )
            cur.execute("ALTER SEQUENCE ways_gid_seq OWNED BY ways.gid")
        conn.commit()
        print(f"[backfill-ctas]   swap done in {time.time()-t0:.1f}s",
              flush=True)

        # 7. Final check.
        with conn.cursor() as cur:
            cur.execute("SELECT count(*), "
                        "count(*) FILTER (WHERE highway = '') AS empty_hw "
                        "FROM ways")
            total, empty = cur.fetchone()
        print(f"[backfill-ctas] DONE: ways has {total:,} rows, "
              f"{empty:,} still empty highway", flush=True)


if __name__ == "__main__":
    backfill()
