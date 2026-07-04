"""CTAS-based recompute-cost for the `views` profile, scoped to the
NEW bikeable rows whose `cost_views IS NULL`.

Why this exists (vs the generic recompute_cost.py):

  1. **Filter to NULL cost_views.** Of the 114M bikeable rows in `ways`,
     ~22M (Austria from prior runs) already have cost_views populated
     against unchanged signal data — their source/target vertex
     elev_m didn't change in the recent dem-ingest, so re-computing
     them would yield the same value. The other ~92M (DE-north, plus
     newly-added CZ/DK gaps) need cost_views computed from scratch.
     4× less Python work.

  2. **CTAS swap instead of per-tile UPDATE.** The per-tile UPDATE on
     the 14 GB ways heap thrashes MVCC pages — this is failure mode
     13. CTAS sequentially writes a fresh heap (no dead tuples, no
     checkpoint amplification) then atomic DROP+RENAME. Empirically
     ~4-5× faster than UPDATE on this scale.

  3. **LOGGED staging table from the start + resume detection.** No
     SET LOGGED step, survives WSL crashes, can re-run to pick up
     where it left off.

Estimated total: ~3-5 hours (vs ~12 hours for plain recompute_cost).

Per-tile Python compute is still single-threaded (multiprocessing
parallelism left for a future iteration).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import psycopg

import config
from .cost import _PROFILES
from .recompute_cost import (
    _recompute_one,
    _iter_tiles,
    _bake_extent,
    _BATCH,
    _TILE_SIZE_DEG,
)


_PROFILE = "views"
_COST_COL = "cost_views"
_REV_COL  = "reverse_cost_views"
_STAGING_TABLE = "_views_cost_staging"


# ── Step 1: staging table (LOGGED, resume-aware) ──────────────────────

def _setup_staging(conn: psycopg.Connection) -> int:
    """Return existing row count in staging if it has substantial data,
    else drop+recreate and return 0."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*) FROM pg_class WHERE relname = %s AND relkind = 'r'",
            (_STAGING_TABLE,),
        )
        exists = cur.fetchone()[0] > 0
        if exists:
            cur.execute(f"SELECT count(*) FROM {_STAGING_TABLE}")
            n_existing = cur.fetchone()[0]
        else:
            n_existing = 0

    if exists and n_existing > 50_000_000:
        print(f"[views-ctas] step 1: RESUME — {_STAGING_TABLE} has "
              f"{n_existing:,} rows already, skipping compute step",
              flush=True)
        return n_existing

    print(f"[views-ctas] step 1: drop+create LOGGED {_STAGING_TABLE} ...",
          flush=True)
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {_STAGING_TABLE}")
        cur.execute(f"""
            CREATE TABLE {_STAGING_TABLE} (
                gid           bigint PRIMARY KEY,
                cost          real,
                reverse_cost  real
            )
        """)
    conn.commit()
    return 0


# ── Step 2: single seq-scan compute → COPY to staging ────────────────

def _scan_and_compute_all(
    conn_scan: psycopg.Connection,
    conn_copy: psycopg.Connection,
) -> tuple[int, int]:
    """Single seq scan of all NULL-cost_views bikeable edges. Streams
    rows, computes via cost.bike_edge_cost, COPYs to staging every
    100K rows. Returns (n_costed, n_skipped).

    Why not tiled: a per-tile WHERE bbox forces postgres to full-scan
    each Austrian tile to find that 0 rows match (no partial index
    on cost_views IS NULL). 500+ wasted tile scans dominate runtime.
    Single seq scan + filter is ~5 min for the 14 GB ways heap; the
    rest is Python compute on the matched rows."""
    n_costed = 0
    n_skipped = 0
    buf: list[tuple] = []
    t_start = time.time()
    t_last_log = t_start

    def _flush():
        nonlocal n_costed
        if not buf:
            return
        with conn_copy.cursor() as cur:
            with cur.copy(
                f"COPY {_STAGING_TABLE} (gid, cost, reverse_cost) FROM STDIN"
            ) as cp:
                for t in buf:
                    cp.write_row(t)
        conn_copy.commit()
        n_costed += len(buf)
        buf.clear()

    # Force the planner to seq-scan the heap rather than try an index.
    # Bitmap-index on a hypothetical (cost_views) NULL partial would
    # be faster than seq scan, but we don't have one and building it
    # costs ~30 min — comparable to the seq scan itself.
    with conn_scan.cursor() as setup:
        setup.execute("SET enable_bitmapscan = off")
        setup.execute("SET enable_indexscan = off")
        setup.execute("SET max_parallel_workers_per_gather = 4")
    with conn_scan.cursor(name="views_ctas_full_scan") as scan:
        scan.itersize = _BATCH
        scan.execute(
            "SELECT w.gid, w.length_m, w.is_ferry, w.reverse_cost, "
            "w.highway, w.surface, w.tracktype, w.oneway, "
            "w.bicycle, w.cycleway, w.bicycle_road, w.access, "
            "w.curv_fwd, w.curv_rev, w.canopy_frac, "
            "w.forest_local, w.forest_wide, w.vineyard_local, "
            "w.water_local, w.water_wide, "
            "w.sea_local, w.sea_wide, "
            "w.waterway_along_edge, w.waterway_local, w.wetland_local, "
            "w.view_dominance, w.local_relief, "
            "w.regional_relief, w.distance_to_drama, "
            "w.viewpoint_local, w.viewpoint_regional, "
            "vs.elev_m AS elev_src, vt.elev_m AS elev_dst "
            "FROM ways w "
            "JOIN ways_vertices_pgr vs ON vs.id = w.source "
            "JOIN ways_vertices_pgr vt ON vt.id = w.target "
            f"WHERE w.{_COST_COL} IS NULL "
            "  AND NOT w.bike_excluded"
        )
        for row in scan:
            gid, new_cost, new_rev = _recompute_one(row, _PROFILE)
            if new_cost is None:
                n_skipped += 1
                continue
            buf.append((gid, float(new_cost), float(new_rev)))
            if len(buf) >= _BATCH:
                _flush()
                # Log every ~5M rows
                if (n_costed % 5_000_000) < _BATCH:
                    now = time.time()
                    rate = n_costed / (now - t_start) if now > t_start else 0
                    print(f"[views-ctas]   computed+staged {n_costed:,} "
                          f"rows ({n_skipped:,} skipped), "
                          f"{rate:.0f} rows/s avg",
                          flush=True)
                    t_last_log = now
        _flush()

    return n_costed, n_skipped


# ── Step 3: CTAS swap ─────────────────────────────────────────────────

# All ways columns in their on-disk order, for the CTAS INSERT.
_ALL_COLS = [
    "gid", "osm_way_id", "source", "target", "cost", "reverse_cost",
    "length_m", "is_ferry",
    "highway", "surface", "tracktype", "oneway", "bicycle", "cycleway",
    "bicycle_road", "access",
    "curv_fwd", "curv_rev", "grade_pct",
    "canopy_frac", "canopy_frac_polygon",
    "forest_local", "forest_wide",
    "view_dominance", "local_relief", "regional_relief", "distance_to_drama",
    "water_local", "water_wide", "sea_local", "sea_wide",
    "waterway_along_edge", "waterway_local", "wetland_local",
    "vineyard_local", "viewpoint_local", "viewpoint_regional",
    "cost_direct", "reverse_cost_direct",
    "cost_vineyard_lover", "reverse_cost_vineyard_lover",
    "cost_forest_lover", "reverse_cost_forest_lover",
    "cost_views", "reverse_cost_views",
    "cost_water", "reverse_cost_water",
    "bike_excluded",
]


def _ctas_swap(conn: psycopg.Connection) -> None:
    """CTAS rebuild of ways with COALESCEd cost_views/reverse_cost_views,
    then atomic DROP + RENAME. All other columns passthrough from old ways."""
    # Resume detection: if ways_new exists with full row count, skip
    # to indexes + swap.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_class WHERE relname='ways_new' AND relkind='r'"
        )
        new_exists = cur.fetchone()[0] > 0
        n_new = 0
        if new_exists:
            cur.execute("SELECT count(*) FROM ways_new")
            n_new = cur.fetchone()[0]

    if new_exists and n_new > 100_000_000:
        print(f"[views-ctas] step 3a: RESUME — ways_new already has "
              f"{n_new:,} rows, skipping CTAS INSERT", flush=True)
    else:
        print("[views-ctas] step 3a: drop stale ways_new ...", flush=True)
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS ways_new CASCADE")
        conn.commit()

        print("[views-ctas] step 3b: CREATE LOGGED ways_new (LIKE) ...",
              flush=True)
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE ways_new (LIKE ways INCLUDING DEFAULTS)"
            )
        conn.commit()

        # Build the SELECT — explicit per-column, with COALESCE on the
        # two target columns. Generated in code so this script tracks
        # the schema cleanly.
        select_cols = []
        for col in _ALL_COLS:
            if col == "cost_views":
                select_cols.append(
                    "COALESCE(s.cost, w.cost_views) AS cost_views"
                )
            elif col == "reverse_cost_views":
                select_cols.append(
                    "COALESCE(s.reverse_cost, w.reverse_cost_views) "
                    "AS reverse_cost_views"
                )
            else:
                select_cols.append(f"w.{col}")
        select_sql = ",\n                       ".join(select_cols)
        col_list = ", ".join(_ALL_COLS)

        print("[views-ctas] step 3c: INSERT joined data ...", flush=True)
        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute("SET work_mem = '1GB'")
            cur.execute("SET maintenance_work_mem = '2GB'")
            cur.execute("SET max_parallel_workers_per_gather = 4")
            cur.execute(f"""
                INSERT INTO ways_new ({col_list})
                SELECT {select_sql}
                  FROM ways w
                  LEFT JOIN {_STAGING_TABLE} s ON s.gid = w.gid
            """)
            rc = cur.rowcount
        conn.commit()
        print(f"[views-ctas]   INSERT done: {rc:,} rows in {time.time()-t0:.1f}s",
              flush=True)

    # Indexes (idempotent — skip if already built).
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_indexes WHERE tablename='ways_new'"
        )
        n_idx = cur.fetchone()[0]
    if n_idx >= 4:
        print(f"[views-ctas] step 3d: RESUME — {n_idx} indexes already on "
              f"ways_new, skipping build", flush=True)
    else:
        print("[views-ctas] step 3d: build indexes on ways_new ...",
              flush=True)
        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute("SET maintenance_work_mem = '2GB'")
            cur.execute("ALTER TABLE ways_new ADD PRIMARY KEY (gid)")
            cur.execute(
                "ALTER TABLE ways_new "
                "ADD CONSTRAINT ways_new_dedup UNIQUE (osm_way_id, source, target)"
            )
            cur.execute("CREATE INDEX ways_new_source_idx ON ways_new(source)")
            cur.execute("CREATE INDEX ways_new_target_idx ON ways_new(target)")
        conn.commit()
        print(f"[views-ctas]   indexes built in {time.time()-t0:.1f}s",
              flush=True)

    # Atomic swap. DROP CASCADE removes the sequence too; recreate it.
    print("[views-ctas] step 3e: swap tables ...", flush=True)
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute("SELECT last_value FROM ways_gid_seq")
        seq_val = cur.fetchone()[0]
        cur.execute("DROP TABLE ways CASCADE")
        cur.execute("ALTER INDEX ways_new_pkey RENAME TO ways_pkey")
        cur.execute("ALTER INDEX ways_new_dedup RENAME TO ways_dedup_idx")
        cur.execute("ALTER INDEX ways_new_source_idx RENAME TO ways_source_idx")
        cur.execute("ALTER INDEX ways_new_target_idx RENAME TO ways_target_idx")
        cur.execute("ALTER TABLE ways_new RENAME TO ways")
        cur.execute("CREATE SEQUENCE ways_gid_seq")
        cur.execute(f"SELECT setval('ways_gid_seq', {seq_val})")
        cur.execute(
            "ALTER TABLE ways ALTER COLUMN gid "
            "SET DEFAULT nextval('ways_gid_seq')"
        )
        cur.execute("ALTER SEQUENCE ways_gid_seq OWNED BY ways.gid")
    conn.commit()
    print(f"[views-ctas]   swap done in {time.time()-t0:.1f}s", flush=True)


# ── Top level ────────────────────────────────────────────────────────

def main() -> None:
    if _PROFILE not in _PROFILES:
        raise SystemExit(f"unknown profile {_PROFILE!r}")
    print(f"[views-ctas] starting CTAS recompute for profile={_PROFILE}",
          flush=True)
    t_total = time.time()

    with psycopg.connect(config.PG_DSN) as conn:
        n_pre_staged = _setup_staging(conn)

        if n_pre_staged > 0:
            # Resume: staging is already populated, skip compute step.
            n_total_costed = n_pre_staged
        else:
            print("[views-ctas] step 2: single seq scan of all bikeable "
                  "edges with NULL cost_views, compute+stage ...",
                  flush=True)
            t0 = time.time()
            # Use a second connection for COPYs (dual-conn pattern from
            # dem-ctas — keeps the cursor's transaction open).
            conn_copy = psycopg.connect(config.PG_DSN)
            n_total_costed, n_total_skipped = _scan_and_compute_all(
                conn, conn_copy
            )
            conn_copy.close()
            print(f"[views-ctas] step 2 DONE: {n_total_costed:,} costed, "
                  f"{n_total_skipped:,} skipped in {time.time()-t0:.1f}s",
                  flush=True)

        # Step 3: CTAS swap.
        _ctas_swap(conn)

        # Step 4: cleanup staging.
        print("[views-ctas] step 4: drop staging ...", flush=True)
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {_STAGING_TABLE}")
        conn.commit()

        # Final verification.
        with conn.cursor() as cur:
            cur.execute("SET max_parallel_workers_per_gather = 0")
            cur.execute(
                f"SELECT count(*), count(*) FILTER (WHERE {_COST_COL} IS NULL) "
                f"FROM ways WHERE NOT bike_excluded"
            )
            total, null_cost = cur.fetchone()

    print(f"[views-ctas] DONE in {(time.time()-t_total)/60:.1f} min — "
          f"ways has {total:,} bikeable rows, {null_cost:,} still "
          f"NULL cost_views (these have unroutable highway tags)",
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    args = p.parse_args()
    main()
