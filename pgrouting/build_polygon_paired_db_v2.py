"""V1-style paired-SPT builder using is_frontier for precise pruning.

Per directed chain edge (A → B), stores a SLICE of A.SPT that lets
the router walk A.parent from any starting vertex and terminate
naturally at a B-frontier vertex (city-center bypass).

Kept set:
  F-only-ancestors = A.SPT vertices reachable via A.parent from any
                     is_frontier leaf that's also in F-only. These are
                     the "backbone" paths from A-seed out to A's true
                     geographic polygon boundary.
  B-frontier       = A vertices in A∩B whose A.parent is F-only. Entry
                     gates where A.SPT.parent first crosses into B.
  kept             = F-only-ancestors ∪ B-frontier

Router walk: succ (= remapped A.parent) points down the A.SPT gradient
toward A-seed. Walk terminates when succ = NULL_SENTINEL — either
because we're at a B-frontier vertex (parent in B-interior, excluded)
or at a true A-seed (parent < 0). The terminating vid is looked up in
the next pair (B, C) to continue.

Requires: NPZs written by compute_spts_polygon.py with the
`is_frontier` field (per-vertex bool, True iff the vertex has an
out-edge in the full cell graph that lands OUTSIDE A's polygon —
i.e., a real geographic frontier, not an interior dead end).

Per-pair work is O(N_A + N_B), touching ONLY A's and B's NPZ.
Scalable: adding more anchors linearly increases work.

Usage:
  SPT_PROFILE=views python3 build_polygon_paired_db_v2.py
Output:
  data/spt/<profile>/paired_trunks.db      (replaces single-path DB)
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from functools import lru_cache
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
    """Only used for degenerate ferry pairs with no A∩B overlap."""
    return -2 - city_idx


def _open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
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
    # Precomputed termination-vid list per trunk. Feeds the iterative
    # entry-point pruner (task #40): entry points of (A, B) = union of
    # termination vids across (X, A) for chain-neighbors X, intersected
    # with (A, B)'s kept set. Storing the list here avoids scanning the
    # full trunk blob at pruning time.
    #
    # `n_termini` = number of vids where succ == NULL_SENTINEL in the
    # trunk blob. `vids` = little-endian int64 array of those vids,
    # sorted ascending for searchsorted-friendly membership checks.
    db.execute(
        """
        CREATE TABLE trunk_termini (
            src_city   INTEGER NOT NULL,
            dst_city   INTEGER NOT NULL,
            n_termini  INTEGER NOT NULL,
            vids       BLOB    NOT NULL,
            PRIMARY KEY (src_city, dst_city)
        ) WITHOUT ROWID
        """
    )
    return db


def _sort_and_reindex(ng: np.ndarray, par: np.ndarray, *arrays):
    """Ensure node_global is ascending, reindex parent[] via inverse
    permutation, and permute companion arrays. Returns (ng, par, *arrays).
    """
    if len(ng) <= 1 or (ng[1:] > ng[:-1]).all():
        return (ng, par, *arrays)
    order = np.argsort(ng, kind="stable")
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    valid = (par >= 0) & (par < len(par))
    new_par = par.copy()
    new_par[valid] = inv[par[valid]]
    return (ng[order], new_par[order].astype(par.dtype),
            *(a[order] for a in arrays))


def _load_a_spt(path: Path) -> dict | None:
    """Load A's SPT + is_frontier. Sorted by node_global ascending
    for searchsorted-friendly lookups."""
    if not path.exists():
        return None
    with np.load(path) as z:
        ng    = np.asarray(z["node_global"], dtype=np.int64)
        par   = np.asarray(z["parent"], dtype=np.int32)
        cost  = np.asarray(z["cost"], dtype=np.float32)
        coord = np.asarray(z["coords_lonlat"], dtype=np.float32)
        if "is_frontier" not in z.files:
            raise SystemExit(
                f"{path.name}: missing `is_frontier` field. "
                f"Re-run compute_spts_polygon.py to regenerate NPZs."
            )
        isf = np.asarray(z["is_frontier"], dtype=bool)
    if len(ng) == 0:
        return None
    ng, par, cost, coord, isf = _sort_and_reindex(ng, par, cost, coord, isf)
    return {"ng": ng, "par": par, "cost": cost, "coord": coord, "isf": isf}


@lru_cache(maxsize=512)
def _load_b_ng(path_str: str) -> np.ndarray | None:
    """B-side needs only node_global. Sorted ascending. LRU-cached."""
    path = Path(path_str)
    if not path.exists():
        return None
    with np.load(path) as z:
        ng = np.asarray(z["node_global"], dtype=np.int64)
    if len(ng) == 0:
        return None
    if len(ng) > 1 and not (ng[1:] > ng[:-1]).all():
        ng = np.sort(ng)
    return ng


def _build_pair(a: dict, b_ng: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray] | None:
    """V1's _build_pair adapted to polygon NPZs with is_frontier.

    Returns (kept_idx_in_A, succ_global_vids) or None if kept is empty.

    A-seed(s) are intentionally NOT in kept. The router routes into
    a trunk via inside-polygon Dijkstra (task #37) rather than a
    synthetic A-seed-in-trunk pointer chain. (An earlier experiment
    added such a chain; it turned the router's "check the second pair,
    then the first pair" fallback into a no-op.)
    """
    a_ng = a["ng"]; a_par = a["par"]; a_isf = a["isf"]
    n = len(a_ng)

    # in_b[i] = A vertex i is also in B.SPT
    pos = np.searchsorted(b_ng, a_ng)
    in_range = pos < len(b_ng)
    in_b = np.zeros(n, dtype=bool)
    in_b[in_range] = b_ng[pos[in_range]] == a_ng[in_range]

    f_only = ~in_b
    if not f_only.any():
        return None

    valid_par = a_par >= 0

    # B-frontier: A vertices in A∩B whose A.parent is F-only. These are
    # the entry gates where A.SPT.parent chains cross from F-only into B.
    f_only_parents = a_par[f_only]
    fp_valid = f_only_parents >= 0
    fp_candidates = f_only_parents[fp_valid]
    b_frontier = np.unique(fp_candidates[in_b[fp_candidates]])

    # F-only frontier leaves: is_frontier AND in F-only. Ancestors of
    # these (via A.parent, staying inside F-only) form the pruned
    # backbone kept in the trunk.
    f_frontier_leaves = a_isf & f_only

    in_ancestors = f_frontier_leaves.copy()
    prev = -1
    for _ in range(8192):
        cur = int(in_ancestors.sum())
        if cur == prev:
            break
        prev = cur
        marked = np.flatnonzero(in_ancestors & valid_par)
        if len(marked) == 0:
            break
        in_ancestors[a_par[marked]] = True

    kept_mask = f_only & in_ancestors
    kept_mask[b_frontier] = True

    kept_idx = np.flatnonzero(kept_mask)
    if len(kept_idx) == 0:
        return None

    # Remap parent → global vid of parent-in-kept, or NULL_SENTINEL
    # if parent falls outside kept (= walk terminates here).
    remap = np.full(n, -1, dtype=np.int32)
    remap[kept_idx] = np.arange(len(kept_idx), dtype=np.int32)
    parent_in_a = a_par[kept_idx]
    parent_kept_local = np.where(
        parent_in_a >= 0, remap[np.maximum(parent_in_a, 0)], -1,
    ).astype(np.int32)
    succ_global = np.full(len(kept_idx), NULL_SENTINEL, dtype=np.int64)
    valid_succ = parent_kept_local >= 0
    succ_global[valid_succ] = a_ng[kept_idx[parent_kept_local[valid_succ]]]

    return kept_idx, succ_global


def _pack_blob(a: dict, kept_idx: np.ndarray,
               succ_global: np.ndarray
               ) -> tuple[int, bytes, int, bytes]:
    """Pack (vid, succ, lat, lon) trunk blob + termini blob.

    Returns (n_rows, trunk_blob, n_termini, termini_blob). The termini
    blob is a sorted little-endian int64 array of the vids where
    succ == NULL_SENTINEL — the vertices where a walk in this trunk
    terminates. Feeds the entry-point pruner in task #40.
    """
    a_ng = a["ng"]; a_coord = a["coord"]
    K = len(kept_idx)
    if K == 0:
        return 0, b"", 0, b""
    vids = a_ng[kept_idx]
    lons = a_coord[kept_idx, 0]
    lats = a_coord[kept_idx, 1]
    order = np.argsort(vids, kind="stable")
    arr = np.empty(K, dtype=TRUNK_DTYPE)
    arr["vid"]  = vids[order]
    arr["succ"] = succ_global[order]
    arr["lat"]  = lats[order]
    arr["lon"]  = lons[order]
    termini_mask = arr["succ"] == NULL_SENTINEL
    termini_vids = arr["vid"][termini_mask].astype(np.int64)
    # termini_vids is already sorted (arr is sorted by vid).
    return K, arr.tobytes(), int(len(termini_vids)), termini_vids.tobytes()


def _degenerate_blob(a_ci: int, a_lon: float, a_lat: float,
                     b_ci: int, b_lon: float, b_lat: float
                     ) -> tuple[int, bytes, int, bytes]:
    synth_a = np.int64(_synth_vid(a_ci))
    synth_b = np.int64(_synth_vid(b_ci))
    arr = np.empty(2, dtype=TRUNK_DTYPE)
    arr["vid"]  = np.array([synth_a, synth_b], dtype=np.int64)
    arr["succ"] = np.array([synth_b, NULL_SENTINEL], dtype=np.int64)
    arr["lat"]  = np.array([np.float32(a_lat), np.float32(b_lat)])
    arr["lon"]  = np.array([np.float32(a_lon), np.float32(b_lon)])
    order = np.argsort(arr["vid"], kind="stable")
    arr = arr[order]
    # Only synth_b has succ=NULL (walk terminus).
    termini_mask = arr["succ"] == NULL_SENTINEL
    termini_vids = arr["vid"][termini_mask].astype(np.int64)
    return 2, arr.tobytes(), int(len(termini_vids)), termini_vids.tobytes()


def main() -> None:
    t_start = time.time()
    print(f"[paired-db-v2] profile={PROFILE}", flush=True)
    print(f"[paired-db-v2]   polygon SPT input: {POLY_SPT_IN}", flush=True)
    print(f"[paired-db-v2]   output db: {DB_PATH}", flush=True)

    cities = json.load(open(PAIRED_DIR / "cities.json"))
    by_idx = {c["city_idx"]: c for c in cities}
    cg = json.load(open(PAIRED_DIR / "city_graph.json"))
    edges = list(zip(cg["from_city"], cg["to_city"], cg["weight"]))
    edges = [(int(fa), int(tb), float(w)) for fa, tb, w in edges if w >= 0]
    print(f"[paired-db-v2] {len(cities):,} cities, {len(edges):,} directed "
          f"edges to pack", flush=True)

    edges_by_a: dict[int, list[int]] = {}
    for fa, tb, _w in edges:
        edges_by_a.setdefault(fa, []).append(tb)
    print(f"[paired-db-v2] grouped into {len(edges_by_a):,} source-anchor "
          f"NPZ loads", flush=True)

    db = _open_db(DB_PATH)
    n_packed = n_missing_npz = n_empty_pair = n_degenerate = 0
    total_vertices = 0
    total_termini = 0
    total_a_vertices = 0
    last_log = t_start
    batch: list[tuple[int, int, int, bytes]] = []
    termini_batch: list[tuple[int, int, int, bytes]] = []
    BATCH_FLUSH = 5_000

    for a_ci in sorted(edges_by_a):
        a = _load_a_spt(POLY_SPT_IN / f"{a_ci}.npz")
        if a is None:
            n_missing_npz += len(edges_by_a[a_ci])
            continue
        a_info = by_idx.get(a_ci, {})
        a_lon = a_info.get("lon"); a_lat = a_info.get("lat")
        a_size = len(a["ng"])

        for b_ci in edges_by_a[a_ci]:
            b_ng = _load_b_ng(str(POLY_SPT_IN / f"{b_ci}.npz"))
            b_info = by_idx.get(b_ci, {})
            b_lon = b_info.get("lon"); b_lat = b_info.get("lat")
            if b_ng is None:
                if a_lon is None or b_lon is None:
                    n_empty_pair += 1
                    continue
                n_rows, blob, n_term, term_blob = _degenerate_blob(
                    a_ci, a_lon, a_lat, b_ci, b_lon, b_lat,
                )
                n_degenerate += 1
            else:
                pair = _build_pair(a, b_ng)
                if pair is None:
                    if a_lon is None or b_lon is None:
                        n_empty_pair += 1
                        continue
                    n_rows, blob, n_term, term_blob = _degenerate_blob(
                        a_ci, a_lon, a_lat, b_ci, b_lon, b_lat,
                    )
                    n_degenerate += 1
                else:
                    kept_idx, succ_global = pair
                    n_rows, blob, n_term, term_blob = _pack_blob(
                        a, kept_idx, succ_global,
                    )
                    total_a_vertices += a_size

            if n_rows == 0:
                n_empty_pair += 1
                continue
            batch.append((a_ci, b_ci, n_rows, blob))
            termini_batch.append((a_ci, b_ci, n_term, term_blob))
            n_packed += 1
            total_vertices += n_rows
            total_termini += n_term
            if len(batch) >= BATCH_FLUSH:
                db.executemany(
                    "INSERT OR IGNORE INTO trunk_blobs VALUES (?, ?, ?, ?)",
                    batch,
                )
                db.executemany(
                    "INSERT OR IGNORE INTO trunk_termini VALUES (?, ?, ?, ?)",
                    termini_batch,
                )
                db.commit()
                batch.clear()
                termini_batch.clear()

        if time.time() - last_log >= 30:
            pct = 100.0 * n_packed / max(1, len(edges))
            print(f"[paired-db-v2]   {n_packed:,}/{len(edges):,} ({pct:.0f}%) "
                  f"trunks packed, {total_vertices:,} vertices, "
                  f"{time.time()-t_start:.0f}s", flush=True)
            last_log = time.time()

    if batch:
        db.executemany(
            "INSERT OR IGNORE INTO trunk_blobs VALUES (?, ?, ?, ?)", batch,
        )
        db.executemany(
            "INSERT OR IGNORE INTO trunk_termini VALUES (?, ?, ?, ?)",
            termini_batch,
        )
    db.commit()
    db.execute("ANALYZE")
    db.commit()
    db.close()

    size_mb = DB_PATH.stat().st_size / 1e6
    avg = total_vertices / max(1, n_packed)
    avg_term = total_termini / max(1, n_packed)
    retention = 100.0 * total_vertices / max(1, total_a_vertices)
    print(f"[paired-db-v2] DONE in {(time.time()-t_start)/60:.1f} min",
          flush=True)
    print(f"[paired-db-v2]   {n_packed:,} trunks packed, {total_vertices:,} "
          f"total vertices, db {size_mb:.1f} MB", flush=True)
    print(f"[paired-db-v2]   avg vertices/trunk: {avg:.0f}, "
          f"avg termini/trunk: {avg_term:.1f}", flush=True)
    print(f"[paired-db-v2]   retention (kept/A summed): {retention:.1f}%",
          flush=True)
    print(f"[paired-db-v2]   skipped: {n_missing_npz} missing NPZ, "
          f"{n_empty_pair} empty-pair; {n_degenerate} degenerate ferries",
          flush=True)


if __name__ == "__main__":
    main()
