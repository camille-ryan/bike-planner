"""Export bikeable edges to per-cell .npz files for polygon-SPT compute.

Replaces the painful `ways_bike` global materialization (4+ hour CTAS)
with a one-pass in-numpy JOIN that streams both tables sequentially
and writes per-cell .npz files directly.

Why this exists:
  - `compute_spts_polygon.py` previously read from `ways_bike` (the
    denormalized bike-edge table) via a GIST bbox query.
  - Rebuilding `ways_bike` after a new ingest required a 131M-row
    JOIN + ORDER BY ST_GeoHash + GIST build = days on WSL ext4.
  - The polygon SPT compute only ever reads per-tile bbox slices —
    we never need the global table in postgres.

Strategy:
  1. SELECT id, lon, lat FROM ways_vertices_pgr (~111M rows, sequential
     seq scan ~5-10 min on WSL). Stream into a sorted-by-id numpy array.
     Memory: ~3.5 GB (id int64 + lon float64 + lat float64).
  2. SELECT source, target, cost_{profile}, reverse_cost_{profile},
     length_m FROM ways WHERE NOT bike_excluded AND cost_{profile}
     IS NOT NULL (~127M rows, sequential seq scan ~10-20 min).
  3. Per batch: searchsorted to map source/target → coords, compute
     cell index from src_lon/src_lat, bucket into per-cell lists.
  4. Write each cell to data/cells/<cx>_<cy>.npz with columns:
     src_id, dst_id, src_lon, src_lat, dst_lon, dst_lat,
     cost, reverse_cost.

Cell size: 1° lat × 1° lon (cell key = (floor(src_lon), floor(src_lat))).
~80-200 cell files for 4-country extent (lat 46-58, lon 6-19).

Profile-aware: SPT_PROFILE=views writes data/cells_views/, etc.
Cross-profile: re-running with a different profile rewrites a different
cell directory; postgres scan is identical, only the cost column changes.

Total runtime: ~30-60 min for one-time pass over the 4-country graph.
Memory peak: ~5-7 GB during the per-batch processing (vertex array +
batch buffers + per-cell accumulators).
"""
from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

import config


PROFILE = os.environ.get("SPT_PROFILE", "direct")
if not re.fullmatch(r"[a-z_][a-z0-9_]*", PROFILE):
    raise SystemExit(f"unsafe profile name: {PROFILE!r}")

CELL_DEG  = 1.0                                         # 1° × 1° cells
OUT_DIR   = Path("/data/cells") / PROFILE
BATCH     = 500_000                                     # ways stream batch
COST_COL  = f"cost_{PROFILE}"
REV_COL   = f"reverse_cost_{PROFILE}"


def _load_vertices(conn: psycopg.Connection) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream ways_vertices_pgr into sorted-by-id numpy arrays.

    Critical: NO `ORDER BY id` in the SQL — the heap is geo-clustered
    after cluster_vertices_ctas (not id-clustered), so an id-ordered
    index scan triggers random heap reads (~100 KB/s on WSL ext4).
    Instead we do a seq scan (sequential heap reads, ~50 MB/s) and
    sort by id in numpy after the load (~5-10 sec for 123M rows).

    Batch parsing via list-unzip + numpy avoids per-row Python overhead.
    """
    t0 = time.time()
    print(f"[export-cells] step 1: streaming ways_vertices_pgr (seq scan + numpy sort) ...",
          flush=True)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM ways_vertices_pgr")
        n_total = cur.fetchone()[0]
    print(f"[export-cells]   {n_total:,} vertices to load", flush=True)

    vid_raw  = np.empty(n_total, dtype=np.int64)
    vlon_raw = np.empty(n_total, dtype=np.float64)
    vlat_raw = np.empty(n_total, dtype=np.float64)

    i = 0
    last_log = t0
    with conn.cursor(name="vertex_stream") as cur:
        cur.itersize = 500_000
        # NO ORDER BY — seq scan
        cur.execute("SELECT id, lon, lat FROM ways_vertices_pgr")
        while True:
            batch = cur.fetchmany(500_000)
            if not batch:
                break
            n = len(batch)
            # Batch unzip is ~10× faster than per-row Python loop.
            ids, lons, lats = zip(*batch)
            vid_raw[i:i+n]  = ids
            vlon_raw[i:i+n] = lons
            vlat_raw[i:i+n] = lats
            i += n
            if time.time() - last_log >= 30:
                pct = 100.0 * i / n_total
                rate = i / max(1e-9, time.time()-t0)
                print(f"[export-cells]   loaded {i:,}/{n_total:,} ({pct:.0f}%) "
                      f"in {time.time()-t0:.0f}s ({rate:,.0f} rows/s)",
                      flush=True)
                last_log = time.time()

    assert i == n_total, f"vertex count mismatch: streamed {i} expected {n_total}"
    t_stream = time.time() - t0
    print(f"[export-cells]   raw stream DONE in {t_stream:.0f}s "
          f"({i/max(1e-9,t_stream):,.0f} rows/s)", flush=True)

    # Sort by id (needed for searchsorted later). ~5-10 sec for 123M.
    t1 = time.time()
    order = np.argsort(vid_raw, kind="stable")
    vid  = vid_raw[order]
    vlon = vlon_raw[order]
    vlat = vlat_raw[order]
    del vid_raw, vlon_raw, vlat_raw, order
    print(f"[export-cells]   sorted-by-id in {time.time()-t1:.1f}s", flush=True)
    return vid, vlon, vlat


def _stream_and_partition(
    conn: psycopg.Connection,
    vid: np.ndarray, vlon: np.ndarray, vlat: np.ndarray,
) -> dict[tuple[int, int], list[np.ndarray]]:
    """Stream ways, JOIN via searchsorted, partition by cell. Returns
    {cell_key: [array_chunks]} ready to concat + save.

    Each chunk is a structured array with columns:
      src_id, dst_id, src_lon, src_lat, dst_lon, dst_lat, cost, reverse_cost.
    """
    t0 = time.time()
    print(f"[export-cells] step 2: streaming ways and partitioning by cell ...",
          flush=True)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT COUNT(*) FROM ways "
            f"WHERE NOT bike_excluded AND {COST_COL} IS NOT NULL "
            f"AND length_m > 0"
        )
        n_total = cur.fetchone()[0]
    print(f"[export-cells]   {n_total:,} bikeable rows to stream", flush=True)

    # dtype for the final cell .npz row layout
    edge_dtype = np.dtype([
        ("src_id",       np.int64),
        ("dst_id",       np.int64),
        ("src_lon",      np.float64),
        ("src_lat",      np.float64),
        ("dst_lon",      np.float64),
        ("dst_lat",      np.float64),
        ("cost",         np.float32),
        ("reverse_cost", np.float32),
    ])

    cell_chunks: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    rows_streamed = 0
    rows_skipped_no_vertex = 0

    with conn.cursor(name="ways_stream") as cur:
        cur.itersize = BATCH
        cur.execute(
            f"SELECT source, target, {COST_COL}, {REV_COL}, length_m "
            f"FROM ways "
            f"WHERE NOT bike_excluded AND {COST_COL} IS NOT NULL "
            f"AND length_m > 0"
        )
        while True:
            batch = cur.fetchmany(BATCH)
            if not batch:
                break
            n = len(batch)

            # Batch-unzip (~10× faster than np.fromiter(generator)).
            # The cur.execute is filtering NOT NULL on the cost cols, so
            # rcost can still be NULL (reverse direction), default -1.
            srcs, dsts, costs, rcosts, _lens = zip(*batch)
            src = np.array(srcs, dtype=np.int64)
            dst = np.array(dsts, dtype=np.int64)
            cost = np.array(costs, dtype=np.float32)
            # rcost may have Nones; replace with -1.0 sentinel
            rcost = np.array([(-1.0 if r is None else r) for r in rcosts],
                             dtype=np.float32)

            # In-memory JOIN via searchsorted (vid is sorted ASC).
            src_pos = np.searchsorted(vid, src)
            dst_pos = np.searchsorted(vid, dst)

            # Validate: searchsorted can return len(vid) for too-large
            # values, and may return a position whose id != source.
            src_ok = (src_pos < len(vid)) & (vid[np.clip(src_pos, 0, len(vid)-1)] == src)
            dst_ok = (dst_pos < len(vid)) & (vid[np.clip(dst_pos, 0, len(vid)-1)] == dst)
            keep = src_ok & dst_ok
            if not keep.all():
                rows_skipped_no_vertex += int((~keep).sum())
            if not keep.any():
                rows_streamed += n
                continue

            src       = src[keep]
            dst       = dst[keep]
            cost      = cost[keep]
            rcost     = rcost[keep]
            src_pos_k = src_pos[keep]
            dst_pos_k = dst_pos[keep]

            src_lon = vlon[src_pos_k]
            src_lat = vlat[src_pos_k]
            dst_lon = vlon[dst_pos_k]
            dst_lat = vlat[dst_pos_k]

            # Cell index from source vertex location.
            cx = np.floor(src_lon / CELL_DEG).astype(np.int32)
            cy = np.floor(src_lat / CELL_DEG).astype(np.int32)

            # Build structured array for this batch.
            batch_arr = np.empty(len(src), dtype=edge_dtype)
            batch_arr["src_id"]       = src
            batch_arr["dst_id"]       = dst
            batch_arr["src_lon"]      = src_lon
            batch_arr["src_lat"]      = src_lat
            batch_arr["dst_lon"]      = dst_lon
            batch_arr["dst_lat"]      = dst_lat
            batch_arr["cost"]         = cost
            batch_arr["reverse_cost"] = rcost

            # Partition by cell. Group rows by (cx, cy) using argsort on
            # a single composite key for speed.
            composite = cx.astype(np.int64) * 1_000_000 + cy.astype(np.int64)
            order = np.argsort(composite, kind="stable")
            comp_sorted = composite[order]
            batch_sorted = batch_arr[order]
            cx_sorted = cx[order]; cy_sorted = cy[order]

            # Find run-boundaries in sorted composite.
            boundaries = np.flatnonzero(np.diff(comp_sorted)) + 1
            starts = np.concatenate([[0], boundaries])
            ends   = np.concatenate([boundaries, [len(comp_sorted)]])
            for s, e in zip(starts, ends):
                key = (int(cx_sorted[s]), int(cy_sorted[s]))
                cell_chunks[key].append(batch_sorted[s:e].copy())

            rows_streamed += n
            if rows_streamed % (BATCH * 10) == 0:
                pct = 100.0 * rows_streamed / n_total
                print(f"[export-cells]   streamed {rows_streamed:,}/{n_total:,} "
                      f"({pct:.0f}%) in {time.time()-t0:.0f}s, "
                      f"{len(cell_chunks)} cells active, skipped="
                      f"{rows_skipped_no_vertex:,}", flush=True)

    print(f"[export-cells]   stream DONE: {rows_streamed:,} rows in "
          f"{time.time()-t0:.0f}s, {len(cell_chunks)} cells, "
          f"{rows_skipped_no_vertex:,} skipped (missing vertex)",
          flush=True)
    return cell_chunks


def _write_cells(cell_chunks: dict[tuple[int, int], list[np.ndarray]]) -> None:
    """Concat per-cell chunks and write each cell to its own .npz."""
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[export-cells] step 3: writing {len(cell_chunks)} cells to {OUT_DIR}",
          flush=True)
    total_rows = 0
    for (cx, cy), chunks in sorted(cell_chunks.items()):
        arr = np.concatenate(chunks)
        out_path = OUT_DIR / f"{cx}_{cy}.npz"
        np.savez(out_path, edges=arr)
        total_rows += len(arr)
    print(f"[export-cells]   wrote {total_rows:,} edges across "
          f"{len(cell_chunks)} cells in {time.time()-t0:.0f}s", flush=True)


def main() -> None:
    t_start = time.time()
    print(f"[export-cells] starting cell export, profile={PROFILE}, "
          f"cell_deg={CELL_DEG}, out={OUT_DIR}", flush=True)
    with psycopg.connect(config.PG_DSN) as conn:
        vid, vlon, vlat = _load_vertices(conn)
        cell_chunks = _stream_and_partition(conn, vid, vlon, vlat)
    _write_cells(cell_chunks)
    print(f"[export-cells] DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
