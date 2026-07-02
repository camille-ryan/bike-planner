"""Build paired_trunks.db from polygon-SPT npz files + city_graph.json.

The existing API (api/app/trunk_router.py) reads paired_trunks.db (a
sqlite blob store) with schema:

    trunk_blobs(src_city, dst_city, n_rows, blob)

Each `blob` is a numpy structured array of (vid: i8, succ: i8,
lat: f4, lon: f4), sorted by vid. The router uses it to walk
A→B along B's SPT subtree.

build_paired_corridor.py is the canonical builder for that DB, but
it expects compute_spts_multi NPZ format (node_global, parent_local,
edge_indptr/indices, is_frontier). Polygon-SPT NPZs use a leaner
schema (node_global, parent, cost, coords_lonlat). This script builds
the same DB from the polygon NPZs.

For each directed edge (A → B) in city_graph.json:
  1. Open B's polygon SPT npz.
  2. Find A's snap_vertex_id in B's node_global array. That's the
     entry point.
  3. Walk B's parent[] pointers from entry → seed (parent[entry],
     parent[parent[entry]], …) until parent = -1 or out-of-range.
     This yields the A→B polyline as vertex indices in B's SPT.
  4. Pack the path's (vid, succ, lat, lon) as TRUNK_DTYPE bytes.

We store ONLY the trunk path (~100-1000 vertices per pair) rather
than the entire reachable subtree — much smaller DB (~110 MB est.
vs ~22 GB if we stored everything), and the API only needs the
path entry+walk anyway.

Usage:
  SPT_PROFILE=views python3 build_polygon_paired_db.py
Output:
  data/spt/<profile>/paired_trunks.db
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np


PROFILE = os.environ.get("SPT_PROFILE", "views")
if not re.fullmatch(r"[a-z_][a-z0-9_]*", PROFILE):
    raise SystemExit(f"unsafe profile name: {PROFILE!r}")

DATA_DIR    = Path(os.environ.get("DATA_DIR", "/data"))
POLY_SPT_IN = DATA_DIR / "spt" / f"{PROFILE}_polygon"
PAIRED_DIR  = DATA_DIR / "spt" / PROFILE
DB_PATH     = PAIRED_DIR / os.environ.get("PAIRED_DB_NAME", "paired_trunks.db")

TRUNK_DTYPE = np.dtype([
    ("vid",  "<i8"),
    ("succ", "<i8"),
    ("lat",  "<f4"),
    ("lon",  "<f4"),
])
NULL_SENTINEL = np.int64(-1)


def _synth_vid(city_idx: int) -> int:
    """Synthetic 'anchor-center' vid for city_idx — disjoint from
    postgres positive vids. Used as the canonical entry/exit point
    for every trunk so consecutive legs join cleanly regardless of
    which multi-seed Dijkstra root the trunk physically walked to."""
    return -2 - city_idx   # -2, -3, … avoids NULL_SENTINEL (-1)


def _open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()   # rebuild from scratch
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA synchronous = NORMAL")
    db.execute(
        """
        CREATE TABLE trunk_blobs (
            src_city  INTEGER NOT NULL,
            dst_city  INTEGER NOT NULL,
            n_rows    INTEGER NOT NULL,
            blob      BLOB    NOT NULL,
            PRIMARY KEY (src_city, dst_city)
        ) WITHOUT ROWID
        """
    )
    return db


def _walk_to_root(parent: np.ndarray, start_idx: int,
                  max_steps: int = 100_000) -> list[int]:
    """Walk parent[] from start_idx toward the root (B's seed) and
    return the list of indices traversed (start … root inclusive).
    Stops at parent < 0 or out-of-range or revisit (cycle guard)."""
    n = len(parent)
    out = []
    cur = start_idx
    seen: set[int] = set()
    for _ in range(max_steps):
        if cur < 0 or cur >= n or cur in seen:
            break
        out.append(cur)
        seen.add(cur)
        p = int(parent[cur])
        if p == cur or p < 0 or p >= n:
            break
        cur = p
    return out


def _pack_trunk(node_global: np.ndarray, parent: np.ndarray,
                coords: np.ndarray, path_idxs: list[int],
                a_ci: int, a_lon: float, a_lat: float,
                b_ci: int, b_lon: float, b_lat: float) -> tuple[int, bytes]:
    """Pack `path_idxs` into a TRUNK_DTYPE blob, sorted by vid.

    Bracketed with two SYNTHETIC vertices so chain walks join cleanly:
      [ SYNTH_A (vid=-2-a_ci, succ=real_first_vid, lat/lon=a center),
        real_first_vid (= a's snap),
        ...
        real_last_vid (= some b-seed; multi-seed Dijkstra root for this path),
        SYNTH_B (vid=-2-b_ci, succ=NULL, lat/lon=b center) ]

    Successor chain: SYNTH_A → real[0] → real[1] → … → real[-1] → SYNTH_B → END.
    Each anchor's SYNTH vid is canonical, so the (A, B) trunk's tail
    matches the (B, C) trunk's head deterministically.
    """
    K = len(path_idxs)
    if K == 0:
        return 0, b""
    path_idxs_arr = np.asarray(path_idxs, dtype=np.int64)
    real_vids  = node_global[path_idxs_arr].astype(np.int64)
    real_lons  = coords[path_idxs_arr, 0].astype(np.float32)
    real_lats  = coords[path_idxs_arr, 1].astype(np.float32)
    real_succs = np.full(K, NULL_SENTINEL, dtype=np.int64)
    if K > 1:
        real_succs[:-1] = real_vids[1:]
    # Last real vertex links to SYNTH_B
    synth_b_vid = np.int64(_synth_vid(b_ci))
    real_succs[-1] = synth_b_vid

    synth_a_vid = np.int64(_synth_vid(a_ci))
    # SYNTH_A succ = first real vid
    vids_total = np.concatenate([
        np.array([synth_a_vid], dtype=np.int64),
        real_vids,
        np.array([synth_b_vid], dtype=np.int64),
    ])
    succs_total = np.concatenate([
        np.array([real_vids[0]], dtype=np.int64),  # SYNTH_A → first real
        real_succs,
        np.array([NULL_SENTINEL], dtype=np.int64),  # SYNTH_B is root
    ])
    lats_total = np.concatenate([
        np.array([np.float32(a_lat)]),
        real_lats,
        np.array([np.float32(b_lat)]),
    ])
    lons_total = np.concatenate([
        np.array([np.float32(a_lon)]),
        real_lons,
        np.array([np.float32(b_lon)]),
    ])
    N = len(vids_total)
    # Sort by vid for searchsorted-friendly storage
    order = np.argsort(vids_total, kind="stable")
    arr = np.empty(N, dtype=TRUNK_DTYPE)
    arr["vid"]  = vids_total[order]
    arr["succ"] = succs_total[order]
    arr["lat"]  = lats_total[order]
    arr["lon"]  = lons_total[order]
    return N, arr.tobytes()


def main() -> None:
    t_start = time.time()
    print(f"[paired-db] profile={PROFILE}", flush=True)
    print(f"[paired-db]   polygon SPT input: {POLY_SPT_IN}", flush=True)
    print(f"[paired-db]   output db: {DB_PATH}", flush=True)

    # Load cities + city_graph
    cities = json.load(open(PAIRED_DIR / "cities.json"))
    by_idx = {c["city_idx"]: c for c in cities}
    cg = json.load(open(PAIRED_DIR / "city_graph.json"))
    edges = list(zip(cg["from_city"], cg["to_city"], cg["weight"]))
    # Drop negative-weight sentinels (one-way edges, can't route on)
    edges = [(int(fa), int(tb), float(w)) for fa, tb, w in edges if w >= 0]
    print(f"[paired-db] {len(cities):,} cities, {len(edges):,} edges to pack",
          flush=True)

    # Group edges by ci_to so each B-NPZ is opened exactly once.
    edges_by_b: dict[int, list[int]] = {}
    for fa, tb, _w in edges:
        edges_by_b.setdefault(tb, []).append(fa)
    print(f"[paired-db] grouped into {len(edges_by_b):,} target-anchor "
          f"NPZ loads", flush=True)

    db = _open_db(DB_PATH)
    n_packed = 0
    n_missing_npz = 0
    n_missing_snap = 0
    n_empty_path = 0
    total_vertices = 0
    last_log = t_start
    batch: list[tuple[int, int, int, bytes]] = []
    BATCH_FLUSH = 5_000

    for b_ci, src_cis in sorted(edges_by_b.items()):
        path_npz = POLY_SPT_IN / f"{b_ci}.npz"
        if not path_npz.exists():
            n_missing_npz += len(src_cis)
            continue
        with np.load(path_npz) as z:
            ng = np.asarray(z["node_global"], dtype=np.int64)
            par = np.asarray(z["parent"], dtype=np.int32)
            coords = np.asarray(z["coords_lonlat"], dtype=np.float32)
            cost = np.asarray(z["cost"], dtype=np.float32)
        if len(ng) == 0:
            n_missing_snap += len(src_cis)
            continue
        # node_global may not be sorted; need sorted for searchsorted lookup
        if len(ng) > 1 and ng[1] < ng[0]:
            order = np.argsort(ng, kind="stable")
            ng = ng[order]; par = par[order]; coords = coords[order]
            cost = cost[order]
            # parent[] uses original indices — re-index via inverse permutation
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            valid = (par >= 0) & (par < len(par))
            new_par = par.copy()
            new_par[valid] = inv[par[valid]]
            par = new_par.astype(np.int32)

        b_info = by_idx.get(b_ci, {})
        b_lon = b_info.get("lon"); b_lat = b_info.get("lat")
        for a_ci in src_cis:
            a = by_idx.get(a_ci)
            if a is None:
                n_missing_snap += 1
                continue
            a_lon = a.get("lon"); a_lat = a.get("lat")
            if a_lon is None or a_lat is None:
                n_missing_snap += 1
                continue
            # MULTI-SEED entry: try every one of A's seeds in B's NPZ,
            # pick the one with the lowest cost (the cheapest entry
            # point from A's region into B's region).
            a_seeds = a.get("snap_vids") or []
            if not a_seeds:
                a_seeds = [a["snap_vertex_id"]] if a.get("snap_vertex_id") else []
            best_pos = -1; best_cost = float("inf")
            if a_seeds:
                seeds_arr = np.array(a_seeds, dtype=np.int64)
                idx = np.searchsorted(ng, seeds_arr)
                in_range = idx < len(ng)
                matched = np.zeros_like(in_range)
                matched[in_range] = ng[idx[in_range]] == seeds_arr[in_range]
                if matched.any():
                    matched_pos = idx[matched]
                    cv = par.dtype  # noqa: unused
                    # Pick min-cost match. par[matched_pos] is parent indices,
                    # not cost — we need cost array from the NPZ; we already
                    # have it as a NoneType here. Read from outer scope.
                    # (cost is loaded above as `cost` variable)
                    best_local_idx = int(np.argmin(cost[matched_pos]))
                    best_pos = int(matched_pos[best_local_idx])
                    best_cost = float(cost[best_pos])
            if best_pos < 0:
                # No seed of A is in B's NPZ (typical for ferry edges where
                # the ferry endpoint isn't reachable from B's polygon).
                # Emit a 2-vertex synthetic-only trunk: SYNTH_A → SYNTH_B.
                # API will render this leg as a single straight segment
                # between anchor centers — visible but well-defined.
                if b_lon is None or b_lat is None:
                    n_missing_snap += 1
                    continue
                synth_a = np.int64(_synth_vid(a_ci))
                synth_b = np.int64(_synth_vid(b_ci))
                degen = np.empty(2, dtype=TRUNK_DTYPE)
                degen["vid"]  = np.array([synth_a, synth_b], dtype=np.int64)
                degen["succ"] = np.array([synth_b, NULL_SENTINEL], dtype=np.int64)
                degen["lat"]  = np.array([float(a_lat), float(b_lat)], dtype=np.float32)
                degen["lon"]  = np.array([float(a_lon), float(b_lon)], dtype=np.float32)
                order = np.argsort(degen["vid"], kind="stable")
                degen = degen[order]
                batch.append((a_ci, b_ci, 2, degen.tobytes()))
                n_packed += 1
                total_vertices += 2
                if len(batch) >= BATCH_FLUSH:
                    db.executemany(
                        "INSERT OR IGNORE INTO trunk_blobs VALUES (?, ?, ?, ?)", batch
                    )
                    db.commit()
                    batch.clear()
                continue
            path_idxs = _walk_to_root(par, best_pos)
            if not path_idxs:
                n_empty_path += 1
                continue
            n_rows, blob = _pack_trunk(
                ng, par, coords, path_idxs,
                a_ci, float(a_lon), float(a_lat),
                b_ci, float(b_lon), float(b_lat),
            )
            if n_rows == 0:
                n_empty_path += 1
                continue
            batch.append((a_ci, b_ci, n_rows, blob))
            n_packed += 1
            total_vertices += n_rows
            if len(batch) >= BATCH_FLUSH:
                db.executemany(
                    "INSERT OR IGNORE INTO trunk_blobs VALUES (?, ?, ?, ?)", batch
                )
                db.commit()
                batch.clear()
        if time.time() - last_log >= 30:
            pct = 100.0 * n_packed / max(1, len(edges))
            print(f"[paired-db]   {n_packed:,}/{len(edges):,} ({pct:.0f}%) "
                  f"trunks packed, {total_vertices:,} vertices total, "
                  f"{time.time()-t_start:.0f}s", flush=True)
            last_log = time.time()

    if batch:
        db.executemany(
            "INSERT OR IGNORE INTO trunk_blobs VALUES (?, ?, ?, ?)", batch
        )
    db.commit()

    # Add index for (src, dst) lookups (PK already covers this, just verify)
    db.execute("ANALYZE")
    db.commit()
    db.close()

    size_mb = DB_PATH.stat().st_size / 1e6
    print(f"[paired-db] DONE in {(time.time()-t_start)/60:.1f} min", flush=True)
    print(f"[paired-db]   {n_packed:,} trunks packed, {total_vertices:,} "
          f"total vertices, db size {size_mb:.1f} MB", flush=True)
    print(f"[paired-db]   skipped: {n_missing_npz} missing NPZ, "
          f"{n_missing_snap} missing-snap, {n_empty_path} empty-path",
          flush=True)


if __name__ == "__main__":
    main()
