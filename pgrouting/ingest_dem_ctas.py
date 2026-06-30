"""CTAS-based DEM ingest with single-scan sampling.

v1 issue: per-tile SELECT (`WHERE lat in box AND elev_m IS NULL`) was
slow because it used Bitmap Heap Scan on the 2.5 GB partial index, then
random heap reads on an insert-ordered (not spatial) heap. Each tile
needed 78+ min of scattered cold reads — failure mode 6 territory.

v2 strategy: ONE sequential scan reads all NULL-elev rows at ~50 MB/sec
disk speed. Stream via server cursor, bucket by tile (lat, lon) in
Python, sample each bucket's DEM tile, COPY all results to staging,
then CTAS swap.

  Single seq scan: ~2-5 min (5-7 GB heap at 50 MB/sec)
  Bucket + sample: ~5-15 min (per-tile bilinear is fast)
  CTAS rebuild + swap: ~60-90 min (heap copy + 4 indexes)
  Total: ~75-110 min vs ~92 hours per-tile.

Python memory: 71M rows × ~80 bytes = ~6 GB. Pgrouting container has
no mem_limit so this fits. If it ever doesn't, switch to per-batch
bucket-and-flush.
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

import config
from ingest_dem import _enumerate_tiles, _sample_bilinear


def ingest_ctas(conn: psycopg.Connection, dem_dir: Path | None = None) -> None:
    dem_dir = dem_dir or config.DEM_DIR
    if not dem_dir.exists():
        raise SystemExit(f"DEM dir does not exist: {dem_dir}")
    tiles = _enumerate_tiles(dem_dir)
    if not tiles:
        raise SystemExit(f"no DEM tiles found in {dem_dir}")
    print(f"[dem-ctas] {len(tiles)} tiles on disk", flush=True)

    # ── Step 1: staging table (LOGGED for crash safety) ──────────────
    # Resume-friendly: if _vertex_elev_all already has the full sample
    # set from a previous crashed run, skip step 2+3 (the 55 min
    # stream+sample phase). LOGGED so the samples survive WSL crashes.
    skip_sampling = False
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_class "
            "WHERE relname='_vertex_elev_all' AND relkind='r'"
        )
        exists = cur.fetchone()[0] > 0
        if exists:
            cur.execute("SELECT count(*) FROM _vertex_elev_all")
            n_existing = cur.fetchone()[0]
        else:
            n_existing = 0

    if exists and n_existing > 50_000_000:
        # Plausible resume — we expect ~80M samples for 4-country.
        # 50M is the floor to count as "previous run actually progressed".
        print(f"[dem-ctas] step 1: RESUME — _vertex_elev_all already has "
              f"{n_existing:,} samples, skipping stream+sample",
              flush=True)
        skip_sampling = True
    else:
        print("[dem-ctas] step 1: drop+create LOGGED staging _vertex_elev_all ...",
              flush=True)
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS _vertex_elev_all")
            cur.execute("""
                CREATE TABLE _vertex_elev_all (
                    id bigint PRIMARY KEY,
                    elev_m real NOT NULL
                )
            """)
        conn.commit()

    # ── Step 2+3: chunked stream → bucket → sample → flush ──────────
    # Bucket-and-flush per batch. A previous attempt bucketed the entire
    # stream first (~80M rows × ~120 bytes Python overhead = ~10 GB),
    # OOM'd the 12 GB WSL VM. Per-batch flush keeps peak Python heap at
    # ~500 MB regardless of total row count.
    total_samples = n_existing  # may be 0 (fresh) or large (resume skip)

    if skip_sampling:
        print(f"[dem-ctas] step 2+3: SKIPPED (resuming with "
              f"{total_samples:,} pre-staged samples)", flush=True)
    else:
        print("[dem-ctas] step 2+3: chunked stream + sample + flush ...",
              flush=True)
        BATCH_SIZE = 5_000_000
        t0 = time.time()
        total_streamed = 0
        batch_ids:  dict = defaultdict(list)
        batch_lons: dict = defaultdict(list)
        batch_lats: dict = defaultdict(list)

        # Use a SECOND connection for COPYs so the main conn keeps the
        # server cursor in its open transaction (no commit needed = cursor
        # stays alive without WITH HOLD materialization cost).
        copy_conn = psycopg.connect(config.PG_DSN)

        def _flush_batch():
            """Sample each tile-bucket in the current batch via rasterio,
            COPY (id, elev_m) into staging on the SECONDARY connection,
            drop the bucket. The secondary's commits don't affect the
            primary's cursor."""
            nonlocal total_samples
            for key in list(batch_ids.keys()):
                tile_path = tiles.get(key)
                ids_list  = batch_ids.pop(key)
                lons_list = batch_lons.pop(key)
                lats_list = batch_lats.pop(key)
                if tile_path is None:
                    continue  # vertex outside DEM coverage
                ids  = np.asarray(ids_list, dtype=np.int64)
                lons = np.asarray(lons_list, dtype=np.float64)
                lats = np.asarray(lats_list, dtype=np.float64)
                elevs = _sample_bilinear(tile_path, lons, lats)
                valid = np.isfinite(elevs)
                n_valid = int(valid.sum())
                if n_valid:
                    with copy_conn.cursor() as cur:
                        with cur.copy(
                            "COPY _vertex_elev_all (id, elev_m) FROM STDIN"
                        ) as cp:
                            for vid, ve in zip(ids[valid], elevs[valid]):
                                cp.write_row((int(vid), float(ve)))
                    copy_conn.commit()
                    total_samples += n_valid

        # Force seq scan: bitmap heap scan on the 2.5 GB partial index
        # does scattered cold reads against the insert-ordered heap
        # (failure mode 6). Plain seq scan on the 5-7 GB heap with
        # partial-elev filter is ~3 min.
        with conn.cursor() as setup:
            setup.execute("SET enable_bitmapscan = off")
            setup.execute("SET enable_indexscan = off")
            setup.execute("SET max_parallel_workers_per_gather = 4")
        with conn.cursor(name="null_elev_stream") as scan:
            scan.itersize = 200_000
            scan.execute(
                "SELECT id, lon, lat FROM ways_vertices_pgr "
                "WHERE elev_m IS NULL"
            )
            for vid, lon, lat in scan:
                key = (int(lat // 1), int(lon // 1))
                batch_ids[key].append(vid)
                batch_lons[key].append(lon)
                batch_lats[key].append(lat)
                total_streamed += 1
                if total_streamed % BATCH_SIZE == 0:
                    _flush_batch()
                    print(f"[dem-ctas]   streamed+flushed {total_streamed:,}"
                          f" rows, {total_samples:,} samples staged",
                          flush=True)

        _flush_batch()
        copy_conn.close()
        print(f"[dem-ctas]   stream+sample DONE: {total_streamed:,} rows "
              f"scanned, {total_samples:,} samples staged in "
              f"{time.time()-t0:.1f}s", flush=True)

    if total_samples == 0:
        print("[dem-ctas] no samples to apply, exiting before CTAS",
              flush=True)
        return

    # ── Step 4: CTAS (LOGGED for crash safety + resume) ──────────────
    # Resume: if ways_vertices_pgr_new already has the full row count,
    # the previous run's INSERT completed; skip ahead to indexes/swap.
    skip_ctas_insert = False
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_class "
            "WHERE relname='ways_vertices_pgr_new' AND relkind='r'"
        )
        new_exists = cur.fetchone()[0] > 0
        n_new = 0
        if new_exists:
            cur.execute("SELECT count(*) FROM ways_vertices_pgr_new")
            n_new = cur.fetchone()[0]

    # Expect ~111M rows for 4-country. Floor: 100M as "actually inserted".
    if new_exists and n_new > 100_000_000:
        print(f"[dem-ctas] step 4: RESUME — ways_vertices_pgr_new already "
              f"has {n_new:,} rows, skipping CTAS INSERT", flush=True)
        skip_ctas_insert = True

    if not skip_ctas_insert:
        print("[dem-ctas] step 4a: drop stale ways_vertices_pgr_new ...",
              flush=True)
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS ways_vertices_pgr_new CASCADE")
        conn.commit()

        print("[dem-ctas] step 4b: CREATE LOGGED ways_vertices_pgr_new "
              "(LIKE) ...", flush=True)
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE ways_vertices_pgr_new
                (LIKE ways_vertices_pgr INCLUDING DEFAULTS INCLUDING GENERATED)
            """)
        conn.commit()

        print("[dem-ctas] step 4c: INSERT joined data (LOGGED, "
              "WAL-streamed) ...", flush=True)
        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute("SET work_mem = '1GB'")
            cur.execute("SET maintenance_work_mem = '2GB'")
            cur.execute("SET max_parallel_workers_per_gather = 4")
            cur.execute("""
                INSERT INTO ways_vertices_pgr_new (id, osm_id, lon, lat, elev_m)
                SELECT v.id, v.osm_id, v.lon, v.lat,
                       COALESCE(v.elev_m, e.elev_m)
                  FROM ways_vertices_pgr v
                  LEFT JOIN _vertex_elev_all e ON e.id = v.id
            """)
            rc = cur.rowcount
        conn.commit()
        print(f"[dem-ctas]   INSERT done: {rc:,} rows in {time.time()-t0:.1f}s",
              flush=True)

    # Resume: skip index build if all 4 indexes already exist.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_indexes WHERE tablename='ways_vertices_pgr_new'"
        )
        n_idx = cur.fetchone()[0]
    if n_idx >= 4:
        print(f"[dem-ctas] step 4e: RESUME — {n_idx} indexes already on "
              f"ways_vertices_pgr_new, skipping build", flush=True)
    else:
        print("[dem-ctas] step 4e: build indexes on ways_vertices_pgr_new ...",
              flush=True)
        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute("SET maintenance_work_mem = '2GB'")
            # IF NOT EXISTS on each so a partial prior run can pick up
            cur.execute("DROP INDEX IF EXISTS ways_vertices_pgr_new_pkey")
            cur.execute("ALTER TABLE ways_vertices_pgr_new ADD PRIMARY KEY (id)")
            cur.execute(
                "ALTER TABLE ways_vertices_pgr_new DROP CONSTRAINT IF EXISTS "
                "ways_vertices_pgr_new_osm_id_key"
            )
            cur.execute(
                "ALTER TABLE ways_vertices_pgr_new "
                "ADD CONSTRAINT ways_vertices_pgr_new_osm_id_key UNIQUE (osm_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS ways_vertices_pgr_new_geom_idx "
                "ON ways_vertices_pgr_new USING gist (the_geom)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS ways_vertices_null_elev_new_idx "
                "ON ways_vertices_pgr_new (lat, lon) WHERE elev_m IS NULL"
            )
        conn.commit()
        print(f"[dem-ctas]   indexes built in {time.time()-t0:.1f}s", flush=True)

    # ── Step 5: atomic swap + FK rebuild ─────────────────────────────
    print("[dem-ctas] step 5: swap tables + rebuild FK ...", flush=True)
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
    print(f"[dem-ctas]   swap + FK done in {time.time()-t0:.1f}s",
          flush=True)

    # ── Step 6: cleanup ──────────────────────────────────────────────
    print("[dem-ctas] step 6: drop staging _vertex_elev_all ...", flush=True)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS _vertex_elev_all")
    conn.commit()

    # ── Final summary ────────────────────────────────────────────────
    with conn.cursor() as cur:
        cur.execute(
            "SET max_parallel_workers_per_gather=0; "
            "SELECT count(*), count(*) FILTER (WHERE elev_m IS NULL) "
            "FROM ways_vertices_pgr"
        )
        total, null_elev = cur.fetchone()
    print(f"[dem-ctas] DONE: ways_vertices_pgr has {total:,} rows, "
          f"{null_elev:,} still NULL elev_m", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dem-dir", type=Path, default=None)
    args = p.parse_args()
    with psycopg.connect(config.PG_DSN) as conn:
        ingest_ctas(conn, args.dem_dir)
