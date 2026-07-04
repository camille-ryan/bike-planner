"""Polygon-bounded SPT compute — tiled + multiprocessing edition.

Architecture:
  1. Pre-compute each anchor's polygon bbox + assign it to a 3° tile
     whose (tile + 3° buffer) extent fully contains the polygon.
  2. For each tile: ONE postgres query loads every ways_bike row whose
     source vertex lies in (tile + buffer). Build CSR + vertex kdtree
     ONCE for the tile.
  3. multiprocessing.Pool(8) forks 8 worker processes that share the
     loaded tile data via copy-on-write. Each worker pops anchors off a
     queue, runs:
         a. matplotlib.Path.contains_points on tile verts → in-polygon mask
         b. slice tile CSR by mask → per-anchor subgraph CSR
         c. multi-source Dijkstra from 1 km seed → SPT
         d. save <i>.npz
  4. After the tile's anchors are done, the pool is closed and tile
     data is freed before loading the next tile.

Idempotent: existing <i>.npz files are skipped, so this can resume
from any partial state.

Why this layout: the previous per-anchor pattern rebuilt CSR + kdtree
1943× and used 1/12 cores. Tiling amortizes the build across all
anchors in a tile; multiprocessing fills the remaining cores. Wall
time drops from ~9 hr to ~1 hr.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import psycopg
import shapely
from shapely.geometry import Polygon
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import config


PROFILE              = os.environ.get("SPT_PROFILE", "direct")
# Whitelist profile names so we can safely interpolate into SQL column
# names (e.g. cost_views, cost_forest_lover). Reject anything else.
import re as _re
if not _re.fullmatch(r"[a-z_][a-z0-9_]*", PROFILE):
    raise SystemExit(f"unsafe profile name: {PROFILE!r}")
SEED_BBOX_RADIUS_M   = 1000.0
POLYGONS_IN          = Path("/data/way_city_spt_polygons.json")
ANCHORS_IN           = Path("/data/way_city_anchors.geojson")
OUT_DIR              = Path("/data/spt") / f"{PROFILE}_polygon"

TILE_SIZE_DEG = float(os.environ.get("SPT_TILE_DEG", "1.5"))
BUFFER_DEG    = float(os.environ.get("SPT_BUFFER_DEG", "2.0"))
N_WORKERS     = int(os.environ.get("SPT_WORKERS", "8"))

R_EARTH_M = 6_371_000.0


# ---------------------------------------------------------------------------
# Module-level state shared with worker processes via fork-COW. Populated by
# _process_tile before forking the Pool; each worker reads only.
# ---------------------------------------------------------------------------
_TILE: dict = {}


def _chord_for_radius_m(radius_m: float) -> float:
    return 2.0 * math.sin(radius_m / R_EARTH_M / 2.0)


def _load_anchors() -> list[dict]:
    fc = json.loads(ANCHORS_IN.read_text())
    out = []
    for f in fc["features"]:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        out.append({
            "ref": p["ref"],
            "name": p["name"],
            "lon": float(lon),
            "lat": float(lat),
            "in_graph": bool(p.get("in_graph", True)),
        })
    return out


def _polygon_bbox(ring: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def _tile_loaded_bbox(tx: int, ty: int) -> tuple[float, float, float, float]:
    lon_min = tx * TILE_SIZE_DEG - BUFFER_DEG
    lat_min = ty * TILE_SIZE_DEG - BUFFER_DEG
    lon_max = (tx + 1) * TILE_SIZE_DEG + BUFFER_DEG
    lat_max = (ty + 1) * TILE_SIZE_DEG + BUFFER_DEG
    return (lon_min, lat_min, lon_max, lat_max)


def _assign_tiles(anchors: list[dict], polygons: dict
                  ) -> tuple[dict[tuple[int, int], list[int]], list[int]]:
    """Bucket anchors by the integer tile that contains their anchor lon/lat.
    Drop any anchor whose polygon bbox isn't fully inside that tile's
    loaded bbox (tile + buffer) into a fallback list — those need a
    bigger buffer or per-anchor processing.
    """
    tiles: dict[tuple[int, int], list[int]] = {}
    fallback: list[int] = []
    for i, a in enumerate(anchors):
        ring = polygons.get(a["ref"])
        if not ring:
            fallback.append(i)
            continue
        tx = int(math.floor(a["lon"] / TILE_SIZE_DEG))
        ty = int(math.floor(a["lat"] / TILE_SIZE_DEG))
        l_min, b_min, l_max, b_max = _tile_loaded_bbox(tx, ty)
        pb = _polygon_bbox(ring)
        if (pb[0] >= l_min and pb[2] <= l_max
            and pb[1] >= b_min and pb[3] <= b_max):
            tiles.setdefault((tx, ty), []).append(i)
        else:
            fallback.append(i)
    return tiles, fallback


CELL_DEG = 1.0   # must match export_cells.py
CELL_DIR = Path("/data/cells") / PROFILE


def _enumerate_cells(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """Cell keys (cx, cy) = (floor(lon), floor(lat)) whose 1°×1° square
    intersects `bbox` (minx, miny, maxx, maxy). Inclusive on both edges."""
    cx_lo = int(np.floor(bbox[0] / CELL_DEG))
    cx_hi = int(np.floor(bbox[2] / CELL_DEG))
    cy_lo = int(np.floor(bbox[1] / CELL_DEG))
    cy_hi = int(np.floor(bbox[3] / CELL_DEG))
    return [(cx, cy)
            for cx in range(cx_lo, cx_hi + 1)
            for cy in range(cy_lo, cy_hi + 1)]


def _load_tile_subgraph(conn: psycopg.Connection,
                        bbox: tuple[float, float, float, float]):
    """Load every bikeable edge in cells that intersect `bbox`. Build:
      - src_local, dst_local: int32 arrays of edge endpoints (interned)
      - fwd, rev:             float32 forward/reverse costs
      - verts (lon, lat):     float64 (n_verts, 2)
      - gid_of_local:         int64 global vertex id for each local id

    Reads from per-cell .npz files at CELL_DIR (written by export_cells.py).
    No postgres query — pure mmap of pre-exported numpy arrays. Each
    cell file is 1° × 1° lat/lon, edges keyed by source vertex location.

    Cross-cell edges (src in cell A, dst in cell B) are stored once
    in cell A's file with both endpoint coords inlined, so loading
    cell A alone resolves the edge correctly. The polygon SPT loop
    typically loads multiple adjacent cells for buffer coverage, which
    happens here via _enumerate_cells over (tile + buffer) bbox.

    `conn` is unused — kept for signature compatibility with callers.
    """
    cells = _enumerate_cells(bbox)
    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    fwd_chunks: list[np.ndarray] = []
    rev_chunks: list[np.ndarray] = []
    sx_chunks:  list[np.ndarray] = []
    sy_chunks:  list[np.ndarray] = []
    tx_chunks:  list[np.ndarray] = []
    ty_chunks:  list[np.ndarray] = []
    for (cx, cy) in cells:
        path = CELL_DIR / f"{cx}_{cy}.npz"
        if not path.exists():
            continue
        with np.load(path, mmap_mode="r") as z:
            arr = z["edges"]
            n = len(arr)
            if n == 0:
                continue
            src_chunks.append(np.asarray(arr["src_id"]))
            dst_chunks.append(np.asarray(arr["dst_id"]))
            fwd_chunks.append(np.asarray(arr["cost"], dtype=np.float32))
            rev_chunks.append(np.asarray(arr["reverse_cost"], dtype=np.float32))
            sx_chunks.append(np.asarray(arr["src_lon"]))
            sy_chunks.append(np.asarray(arr["src_lat"]))
            tx_chunks.append(np.asarray(arr["dst_lon"]))
            ty_chunks.append(np.asarray(arr["dst_lat"]))
    total = sum(c.shape[0] for c in src_chunks)
    if total == 0:
        return None

    src_gid = np.concatenate(src_chunks)
    dst_gid = np.concatenate(dst_chunks)
    fwd     = np.concatenate(fwd_chunks)
    rev     = np.concatenate(rev_chunks)
    sx = np.concatenate(sx_chunks); sy = np.concatenate(sy_chunks)
    tx = np.concatenate(tx_chunks); ty = np.concatenate(ty_chunks)

    all_gid = np.concatenate([src_gid, dst_gid])
    unique_gid, inverse = np.unique(all_gid, return_inverse=True)
    src_local = inverse[:total].astype(np.int32)
    dst_local = inverse[total:].astype(np.int32)
    n_verts = len(unique_gid)
    verts = np.empty((n_verts, 2), dtype=np.float64)
    verts[dst_local, 0] = tx; verts[dst_local, 1] = ty
    verts[src_local, 0] = sx; verts[src_local, 1] = sy

    return {
        "src": src_local, "dst": dst_local,
        "fwd": fwd, "rev": rev, "verts": verts,
        "gid_of_local": unique_gid.astype(np.int64),
        "n_edges": total, "n_verts": n_verts,
    }


def _build_tile_csr(tile) -> csr_matrix:
    fwd = tile["fwd"]; rev = tile["rev"]
    src = tile["src"]; dst = tile["dst"]
    fwd_mask = fwd >= 0
    rev_mask = rev >= 0
    e_src = np.concatenate([src[fwd_mask], dst[rev_mask]])
    e_dst = np.concatenate([dst[fwd_mask], src[rev_mask]])
    e_cost = np.concatenate([fwd[fwd_mask], rev[rev_mask]])
    return csr_matrix(
        (e_cost, (e_src, e_dst)),
        shape=(tile["n_verts"], tile["n_verts"]), dtype=np.float32,
    )


def _pool_init():
    """Worker initializer. The tile data was set in the parent's
    `_TILE` module-global BEFORE forking; workers inherit it via
    Linux copy-on-write. As long as workers only read from `_TILE`
    (slicing csr, indexing verts), pages stay shared and total
    memory is ~O(tile) regardless of worker count. Don't write to
    `_TILE` from workers — that would trigger COW and balloon RAM."""
    pass


def _spt_one_anchor(args) -> tuple[int, str]:
    """Worker entry point. Compute one anchor's SPT against the
    pre-loaded tile CSR. Returns (city_idx, status).
    """
    import os, traceback
    city_idx, anchor, polygon_ring = args
    out_path = OUT_DIR / f"{city_idx}.npz"
    if out_path.exists():
        return (city_idx, "exists")
    try:
        return _spt_one_anchor_inner(city_idx, anchor, polygon_ring)
    except Exception as e:
        print(f"[worker {os.getpid()}] anchor {city_idx} ({anchor.get('name')}) "
              f"FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return (city_idx, f"error:{type(e).__name__}")


def _spt_one_anchor_inner(city_idx, anchor, polygon_ring):
    out_path = OUT_DIR / f"{city_idx}.npz"

    verts = _TILE["verts"]
    csr = _TILE["csr"]

    # Polygon containment over tile vertices. Shapely 2.0's
    # contains_xy is vectorized and lives in C — comparable speed to
    # matplotlib's Path.contains_points without the extra dep.
    poly = Polygon(np.asarray(polygon_ring, dtype=np.float64))
    in_poly = shapely.contains_xy(poly, verts[:, 0], verts[:, 1])
    sub_idx = np.flatnonzero(in_poly)
    if len(sub_idx) == 0:
        return (city_idx, "no-verts-in-poly")

    sub_csr = csr[sub_idx, :][:, sub_idx]

    # Seeds: tile-local indices within 1 km of the anchor location.
    chord = _chord_for_radius_m(SEED_BBOX_RADIUS_M)
    chord_sq = chord * chord
    a_lon = anchor["lon"]; a_lat = anchor["lat"]
    # cheap planar distance in chord-on-unit-sphere; close enough at 1 km
    lat_r = math.radians(a_lat); lon_r = math.radians(a_lon)
    coslat = math.cos(lat_r)
    a_x = coslat * math.cos(lon_r)
    a_y = coslat * math.sin(lon_r)
    a_z = math.sin(lat_r)
    sub_verts = verts[sub_idx]
    v_lat_r = np.radians(sub_verts[:, 1])
    v_lon_r = np.radians(sub_verts[:, 0])
    v_cl = np.cos(v_lat_r)
    vx = v_cl * np.cos(v_lon_r)
    vy = v_cl * np.sin(v_lon_r)
    vz = np.sin(v_lat_r)
    d2 = (vx - a_x) ** 2 + (vy - a_y) ** 2 + (vz - a_z) ** 2
    seeds = np.flatnonzero(d2 <= chord_sq).astype(np.int32)
    if len(seeds) == 0:
        return (city_idx, "no-seeds")

    cost, predecessors, _ = dijkstra(
        csgraph=sub_csr, indices=seeds,
        return_predecessors=True, directed=True, min_only=True,
    )
    reachable = np.isfinite(cost)
    if not reachable.any():
        return (city_idx, "unreachable")
    keep = np.flatnonzero(reachable)
    local_to_kept = np.full(sub_csr.shape[0], -9999, dtype=np.int32)
    local_to_kept[keep] = np.arange(len(keep), dtype=np.int32)
    pred = predecessors[keep]
    pred_safe = np.where(pred >= 0, pred, 0)
    parent_kept = np.where(pred >= 0, local_to_kept[pred_safe], -9999).astype(np.int32)
    cost_kept = cost[keep].astype(np.float32)

    # Map sub-indices back to global gids via tile.gid_of_local
    keep_tile_local = sub_idx[keep]
    keep_global = _TILE["gid_of_local"][keep_tile_local].astype(np.int64)
    coords_kept = verts[keep_tile_local].astype(np.float32)

    # is_frontier: vertex has at least one out-edge in the full cell
    # graph that lands OUTSIDE the polygon. Interior dead-end spurs
    # (cul-de-sacs, hilltop finger roads) will have sub_out_degree ==
    # full_out_degree and thus is_frontier=False, letting the paired-
    # SPT builder drop them during ancestor-of-frontier-leaves pruning.
    full_deg = csr.indptr[keep_tile_local + 1] - csr.indptr[keep_tile_local]
    sub_deg  = sub_csr.indptr[keep + 1] - sub_csr.indptr[keep]
    is_frontier_kept = (full_deg > sub_deg).astype(np.uint8)

    np.savez_compressed(
        out_path,
        node_global=keep_global,
        parent=parent_kept,
        cost=cost_kept,
        coords_lonlat=coords_kept,
        is_frontier=is_frontier_kept,
    )
    return (city_idx, "ok")


def _process_tile(conn: psycopg.Connection, tile_key: tuple[int, int],
                  home_anchors: list[int], anchors: list[dict],
                  polygons: dict, scratch_dir: Path) -> dict:
    """Load tile subgraph, set as module global, then fork workers
    that inherit it via copy-on-write. Workers must only READ from
    `_TILE` so pages stay shared (no per-worker copy)."""
    global _TILE
    bbox = _tile_loaded_bbox(*tile_key)
    t_load = time.time()
    tile = _load_tile_subgraph(conn, bbox)
    if tile is None:
        return {"tile": tile_key, "skip_reason": "empty-bbox",
                "done": 0, "skipped": 0}
    csr = _build_tile_csr(tile)
    load_s = time.time() - t_load
    print(f"[tile {tile_key}] loaded {tile['n_edges']:,} edges, "
          f"{tile['n_verts']:,} verts in {load_s:.1f}s", flush=True)

    # Drop the per-direction arrays we no longer need before forking —
    # cuts the parent's heap by ~1.5 GB so workers inherit a tighter
    # footprint via COW.
    del tile["fwd"], tile["rev"], tile["src"], tile["dst"]

    # Set module global so the forked workers can read tile data
    # without copying it. Cleared after the pool closes.
    _TILE = {
        "verts": tile["verts"],
        "gid_of_local": tile["gid_of_local"],
        "csr": csr,
        "n_verts": tile["n_verts"],
    }

    # Build task list. Skip anchors whose npz already exists so resume
    # doesn't repeat work.
    tasks = []
    pre_skipped = 0
    for i in home_anchors:
        a = anchors[i]
        ring = polygons.get(a["ref"])
        if not ring:
            pre_skipped += 1
            continue
        if (OUT_DIR / f"{i}.npz").exists():
            pre_skipped += 1
            continue
        tasks.append((i, a, ring))

    if not tasks:
        _TILE = {}
        print(f"[tile {tile_key}]   all {len(home_anchors)} anchors already done",
              flush=True)
        return {"tile": tile_key, "done": 0, "skipped": pre_skipped,
                "load_s": load_s}

    print(f"[tile {tile_key}]   {len(tasks)} tasks, "
          f"{pre_skipped} already done — spawning {N_WORKERS} workers "
          f"(fork-COW share)", flush=True)

    t_work = time.time()
    try:
        if N_WORKERS <= 1:
            # Run sequentially in the main process (debug path)
            results = [_spt_one_anchor(t) for t in tasks]
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(N_WORKERS, initializer=_pool_init) as pool:
                results = []
                for r in pool.imap_unordered(_spt_one_anchor, tasks,
                                              chunksize=2):
                    results.append(r)
                    if len(results) % 10 == 0:
                        print(f"[tile {tile_key}]     {len(results)}/{len(tasks)} "
                              f"({time.time()-t_work:.1f}s)", flush=True)
    except Exception as e:
        import traceback
        print(f"[tile {tile_key}]   POOL FAILED: {type(e).__name__}: {e}",
              flush=True)
        traceback.print_exc()
        raise
    work_s = time.time() - t_work

    # Release tile data so the next tile can load fresh.
    _TILE = {}
    n_ok = sum(1 for _, s in results if s == "ok")
    n_skip = sum(1 for _, s in results if s != "ok")
    print(f"[tile {tile_key}]   done in {work_s:.1f}s "
          f"({n_ok} ok, {n_skip} non-ok, pre_skipped {pre_skipped})",
          flush=True)
    return {
        "tile": tile_key, "done": n_ok, "skipped": n_skip + pre_skipped,
        "load_s": load_s, "work_s": work_s,
    }


def main() -> None:
    t0 = time.time()
    print(f"[polygon-spt] profile={PROFILE} out={OUT_DIR}", flush=True)
    print(f"[polygon-spt] tile {TILE_SIZE_DEG:.1f}° + buffer "
          f"{BUFFER_DEG:.1f}°  workers={N_WORKERS}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scratch_dir = OUT_DIR / "_scratch"
    scratch_dir.mkdir(exist_ok=True)

    polygons = json.loads(POLYGONS_IN.read_text())
    anchors = _load_anchors()
    print(f"[polygon-spt] {len(anchors):,} anchors, "
          f"{len(polygons):,} polygons loaded", flush=True)

    cities = [{
        "city_idx":   i, "ref": a["ref"], "name": a["name"],
        "lon": a["lon"], "lat": a["lat"], "in_graph": a["in_graph"],
    } for i, a in enumerate(anchors)]
    (OUT_DIR / "cities.json").write_text(json.dumps(cities, ensure_ascii=False))

    tiles, fallback = _assign_tiles(anchors, polygons)
    n_tiled = sum(len(v) for v in tiles.values())
    print(f"[polygon-spt] {len(tiles)} non-empty tiles covering {n_tiled:,} "
          f"anchors; {len(fallback)} fallback (polygon too big for tile)",
          flush=True)

    n_done_total = 0
    n_skipped_total = 0
    with psycopg.connect(config.PG_DSN) as conn:
        for ti, (tkey, home_anchors) in enumerate(sorted(tiles.items())):
            print(f"[polygon-spt] tile {ti+1}/{len(tiles)} {tkey} "
                  f"({len(home_anchors)} home anchors)…", flush=True)
            result = _process_tile(conn, tkey, home_anchors, anchors,
                                    polygons, scratch_dir)
            n_done_total += result.get("done", 0)
            n_skipped_total += result.get("skipped", 0)
            elapsed = time.time() - t0
            print(f"[polygon-spt]   running total: done={n_done_total} "
                  f"skipped={n_skipped_total}  elapsed={elapsed/60:.1f}min",
                  flush=True)

    if fallback:
        print(f"[polygon-spt] {len(fallback)} fallback anchors — running "
              f"each as its own tile (polygon too big for the 3+3° tile)",
              flush=True)
        global _TILE
        with psycopg.connect(config.PG_DSN) as conn:
            for i in fallback:
                a = anchors[i]
                ring = polygons.get(a["ref"])
                if not ring:
                    continue
                pb = _polygon_bbox(ring)
                bbox = (pb[0] - 0.05, pb[1] - 0.05, pb[2] + 0.05, pb[3] + 0.05)
                tile = _load_tile_subgraph(conn, bbox)
                if tile is None:
                    continue
                csr = _build_tile_csr(tile)
                _TILE = {
                    "verts": tile["verts"],
                    "gid_of_local": tile["gid_of_local"],
                    "csr": csr,
                    "n_verts": tile["n_verts"],
                }
                _spt_one_anchor((i, a, ring))
                _TILE = {}
                n_done_total += 1

    if scratch_dir.exists():
        # scratch_dir was used by an earlier disk-mediated worker init;
        # not needed in the fork-COW path. Best-effort cleanup.
        try:
            scratch_dir.rmdir()
        except OSError:
            pass
    print(f"[polygon-spt] DONE in {time.time()-t0:.1f}s — "
          f"{n_done_total} written, {n_skipped_total} skipped", flush=True)


if __name__ == "__main__":
    main()
