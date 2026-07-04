"""Task #40: iterative entry-point pruning for the v2 paired-trunks DB.

Given a source paired_trunks_v2*.db, keep only the vertices in each
(A, B) trunk that a router walk starting from a chain-neighbor's
termination vid can actually reach via succ chain. Everything else is
dead weight — the router never enters those vertices.

**Iterative** (2026-07-03 revision): after a first pruning pass, each
trunk's termini set shrinks — which further shrinks the `entries_A`
set for downstream trunks. Re-prune with the smaller entries; termini
shrink again; continue until nothing changes. Convergence is
typically 2-3 passes; each pass is fast because everything runs
in-memory over the packed blobs (no NPZ decompression needed).

Border safety: anchors within ~25 km of the outer convex hull of
all anchor coords are held frozen — their outbound trunks are copied
verbatim (chain-neighbors may enter them from outside the 4-country
region and we can't observe those termini). Their termini feed
downstream pruning like any other trunk.

Inputs (all under /data/spt/<profile>/):
  * paired_trunks.db (or PAIRED_DB_NAME override) — source, read-only.
  * city_graph.json — chain neighbors.
  * cities.json — border detection.

Output:
  * /data/spt/<profile>/paired_trunks_v2d.db  (OUT_DB_NAME override).
    Built in RAM, then written to /tmp/<name>.db (native ext4), then
    copied to /data at the end — WSL bind mount is ~750 KB/s under
    sqlite WAL contention, so a single sequential-write copy is
    much faster than incremental writes.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


PROFILE = os.environ.get("SPT_PROFILE", "views")
if not re.fullmatch(r"[a-z_][a-z0-9_]*", PROFILE):
    raise SystemExit(f"unsafe profile name: {PROFILE!r}")

DATA_DIR    = Path(os.environ.get("DATA_DIR", "/data"))
PAIRED_DIR  = DATA_DIR / "spt" / PROFILE
IN_DB_NAME  = os.environ.get("PAIRED_DB_NAME", "paired_trunks.db")
OUT_DB_NAME = os.environ.get("OUT_DB_NAME",   "paired_trunks_v2d.db")
IN_DB       = PAIRED_DIR / IN_DB_NAME
OUT_DB      = PAIRED_DIR / OUT_DB_NAME
WORK_DB     = Path("/tmp") / OUT_DB_NAME

BORDER_KM   = float(os.environ.get("BORDER_KM", "25"))
MAX_PASSES  = int(os.environ.get("MAX_PASSES", "10"))

TRUNK_DTYPE = np.dtype([
    ("vid",  "<i8"),
    ("succ", "<i8"),
    ("lat",  "<f4"),
    ("lon",  "<f4"),
])
NULL_SENTINEL = np.int64(-1)


# ---------------------------------------------------------------------
# Border detection
# ---------------------------------------------------------------------

def _convex_hull(points_xy: np.ndarray) -> np.ndarray:
    pts = points_xy[np.lexsort((points_xy[:, 1], points_xy[:, 0]))]
    lower = []
    for p in pts:
        while (len(lower) >= 2
               and (lower[-1][0] - lower[-2][0]) * (p[1] - lower[-2][1])
                 - (lower[-1][1] - lower[-2][1]) * (p[0] - lower[-2][0]) <= 0):
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while (len(upper) >= 2
               and (upper[-1][0] - upper[-2][0]) * (p[1] - upper[-2][1])
                 - (upper[-1][1] - upper[-2][1]) * (p[0] - upper[-2][0]) <= 0):
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _min_km_to_hull_edge(lon: float, lat: float,
                         hull: np.ndarray) -> float:
    best = float("inf")
    n = len(hull)
    cos_lat = math.cos(math.radians(lat))
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        dx1 = (x1 - lon) * cos_lat; dy1 = (y1 - lat)
        dx2 = (x2 - lon) * cos_lat; dy2 = (y2 - lat)
        seg_dx = dx2 - dx1;         seg_dy = dy2 - dy1
        seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len2 <= 0:
            dist_deg = math.hypot(dx1, dy1)
        else:
            t = max(0.0, min(1.0, -(dx1 * seg_dx + dy1 * seg_dy) / seg_len2))
            px = dx1 + t * seg_dx; py = dy1 + t * seg_dy
            dist_deg = math.hypot(px, py)
        best = min(best, dist_deg * 111.0)
    return best


# ---------------------------------------------------------------------
# In-memory pruning core
# ---------------------------------------------------------------------

def _next_idx_array(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    vid = arr["vid"]
    pos = np.searchsorted(vid, arr["succ"])
    pos_c = np.clip(pos, 0, n - 1)
    matched = (pos < n) & (vid[pos_c] == arr["succ"])
    return np.where(matched & (arr["succ"] != -1), pos, -1).astype(np.int64)


def _prune_trunk(arr: np.ndarray, entries_A: np.ndarray
                 ) -> np.ndarray | None:
    """Walk succ forward from every entry vid; keep only reached vertices.
    Return the pruned array (sorted by vid), or None if empty."""
    n = len(arr)
    if n == 0 or len(entries_A) == 0:
        return None
    ni = _next_idx_array(arr)
    pos_e = np.searchsorted(arr["vid"], entries_A)
    pos_c = np.clip(pos_e, 0, n - 1)
    hit = (pos_e < n) & (arr["vid"][pos_c] == entries_A)
    starts = pos_e[hit]
    if len(starts) == 0:
        return None
    reached = np.zeros(n, dtype=bool)
    reached[starts] = True
    frontier = starts
    while len(frontier):
        nxt = ni[frontier]
        nxt = nxt[nxt >= 0]
        nxt = nxt[~reached[nxt]]
        if not len(nxt):
            break
        reached[nxt] = True
        frontier = nxt
    if not reached.any():
        return None
    kept_idx = np.flatnonzero(reached)
    kept = arr[kept_idx]
    return kept[np.argsort(kept["vid"], kind="stable")]


def main() -> None:
    t_start = time.time()
    print(f"[prune] profile={PROFILE}", flush=True)
    print(f"[prune]   in  db: {IN_DB}", flush=True)
    print(f"[prune]   work db: {WORK_DB}", flush=True)
    print(f"[prune]   out db: {OUT_DB}", flush=True)
    if not IN_DB.exists():
        raise SystemExit(f"input DB not found: {IN_DB}")

    cities = json.loads((PAIRED_DIR / "cities.json").read_text())
    pts = np.array([(c["lon"], c["lat"]) for c in cities], dtype=np.float64)
    hull = _convex_hull(pts)
    print(f"[prune] convex hull: {len(hull)} vertices", flush=True)
    is_border = np.zeros(len(cities), dtype=bool)
    for c in cities:
        d = _min_km_to_hull_edge(c["lon"], c["lat"], hull)
        if d < BORDER_KM:
            is_border[int(c["city_idx"])] = True
    print(f"[prune] {int(is_border.sum()):,} border anchors "
          f"(< {BORDER_KM:.0f} km from hull) — verbatim", flush=True)

    cg = json.loads((PAIRED_DIR / "city_graph.json").read_text())
    edges_by_a: dict[int, list[int]] = defaultdict(list)
    edges_by_b: dict[int, list[int]] = defaultdict(list)
    for fa, tb in zip(cg["from_city"], cg["to_city"]):
        a = int(fa); b = int(tb)
        edges_by_a[a].append(b)
        edges_by_b[b].append(a)

    # --- Load all blobs into memory ----------------------------------
    # Peak RAM ≈ source DB size (~6 GB). Fits in the 11 GB WSL VM.
    print("[prune] loading all trunks into RAM…", flush=True)
    t = time.time()
    conn_in = sqlite3.connect(f"file:{IN_DB}?mode=ro&immutable=1", uri=True)
    trunks: dict[tuple[int, int], np.ndarray] = {}
    for src, dst, n_rows, blob in conn_in.execute(
            "SELECT src_city, dst_city, n_rows, blob FROM trunk_blobs"):
        # np.frombuffer creates a view over the sqlite bytes — no copy.
        # But sqlite blobs are backed by a temporary buffer; force a
        # copy so we own the memory across queries.
        arr = np.frombuffer(blob, dtype=TRUNK_DTYPE).copy()
        trunks[(int(src), int(dst))] = arr
    conn_in.close()
    print(f"[prune]   {len(trunks):,} trunks, "
          f"{sum(len(a) for a in trunks.values()):,} total verts, "
          f"in {time.time()-t:.1f}s", flush=True)

    # Termini = per-trunk cache of succ==-1 vids. Rebuilt on change.
    termini: dict[tuple[int, int], np.ndarray] = {
        ab: arr["vid"][arr["succ"] == -1].astype(np.int64)
        for ab, arr in trunks.items()
    }

    # Termini snapshot from the FIRST pass, used as the border-safe
    # entry set. Border anchors' termini stay constant (we don't
    # observe how external routes end at them), so downstream anchors
    # can always feed off these seed termini even if the interior
    # iteration collapses.
    initial_termini = dict(termini)

    def compute_entries(a_ci: int) -> np.ndarray:
        lst = []
        for x in edges_by_b.get(a_ci, []):
            # Border anchors' termini are always the initial ones;
            # for interior anchors we use the current iteration state.
            src_termini = initial_termini if is_border[x] else termini
            t = src_termini.get((x, a_ci))
            if t is not None and len(t):
                lst.append(t)
        if not lst:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(lst))

    # --- Iterate until fixed point -----------------------------------
    total_a_verts_source = sum(len(a) for a in trunks.values())
    for pass_idx in range(1, MAX_PASSES + 1):
        t_pass = time.time()
        changed = 0
        shrunk_verts = 0
        for a_ci in sorted(edges_by_a):
            if is_border[a_ci]:
                continue  # border: keep unchanged
            entries_A = compute_entries(a_ci)
            if len(entries_A) == 0:
                continue
            for b_ci in edges_by_a[a_ci]:
                key = (a_ci, b_ci)
                arr = trunks.get(key)
                if arr is None or len(arr) == 0:
                    continue
                pruned = _prune_trunk(arr, entries_A)
                if pruned is None or len(pruned) == 0:
                    continue
                # Only replace if actually smaller / different.
                if len(pruned) < len(arr):
                    shrunk_verts += (len(arr) - len(pruned))
                    trunks[key] = pruned
                    new_termini = pruned["vid"][pruned["succ"] == -1].astype(np.int64)
                    if not np.array_equal(new_termini, termini[key]):
                        termini[key] = new_termini
                        changed += 1
        elapsed = time.time() - t_pass
        total_dst = sum(len(a) for a in trunks.values())
        pct_kept = 100.0 * total_dst / max(1, total_a_verts_source)
        print(f"[prune] pass {pass_idx}: {changed:,} trunks with "
              f"termini changes, "
              f"{shrunk_verts:,} verts dropped this pass, "
              f"{total_dst:,} verts total ({pct_kept:.1f}% of source) "
              f"in {elapsed:.1f}s",
              flush=True)
        if changed == 0:
            print(f"[prune] converged at pass {pass_idx}", flush=True)
            break

    # --- Write final result to /tmp then copy to /data ---------------
    print(f"[prune] writing {WORK_DB}…", flush=True)
    if WORK_DB.exists():
        WORK_DB.unlink()
    for suf in ("-wal", "-shm"):
        p = WORK_DB.with_name(WORK_DB.name + suf)
        if p.exists():
            p.unlink()
    conn_out = sqlite3.connect(WORK_DB)
    conn_out.execute("PRAGMA journal_mode = WAL")
    conn_out.execute("PRAGMA synchronous = NORMAL")
    conn_out.execute("""
        CREATE TABLE trunk_blobs (
            src_city  INTEGER NOT NULL,
            dst_city  INTEGER NOT NULL,
            n_rows    INTEGER NOT NULL,
            blob      BLOB    NOT NULL,
            PRIMARY KEY (src_city, dst_city)
        ) WITHOUT ROWID
    """)
    conn_out.execute("""
        CREATE TABLE trunk_termini (
            src_city   INTEGER NOT NULL,
            dst_city   INTEGER NOT NULL,
            n_termini  INTEGER NOT NULL,
            vids       BLOB    NOT NULL,
            PRIMARY KEY (src_city, dst_city)
        ) WITHOUT ROWID
    """)
    t = time.time()
    for (src, dst), arr in trunks.items():
        if len(arr) == 0:
            continue
        conn_out.execute(
            "INSERT INTO trunk_blobs VALUES (?, ?, ?, ?)",
            (src, dst, len(arr), arr.tobytes()),
        )
        tvs = termini[(src, dst)]
        conn_out.execute(
            "INSERT INTO trunk_termini VALUES (?, ?, ?, ?)",
            (src, dst, int(len(tvs)), tvs.astype(np.int64).tobytes()),
        )
    conn_out.commit()
    print(f"[prune]   inserts in {time.time()-t:.1f}s", flush=True)
    print("[prune] VACUUM…", flush=True)
    t = time.time()
    conn_out.execute("VACUUM")
    conn_out.commit()
    conn_out.close()
    print(f"[prune]   vacuum in {time.time()-t:.1f}s", flush=True)

    print(f"[prune] copying {WORK_DB} → {OUT_DB}", flush=True)
    t = time.time()
    OUT_DB.parent.mkdir(parents=True, exist_ok=True)
    if OUT_DB.exists():
        OUT_DB.unlink()
    shutil.copyfile(WORK_DB, OUT_DB)
    print(f"[prune]   copy in {time.time()-t:.1f}s", flush=True)

    src_mb = IN_DB.stat().st_size / 1e6
    dst_mb = OUT_DB.stat().st_size / 1e6
    total_dst = sum(len(a) for a in trunks.values())
    print(f"[prune] DONE in {(time.time()-t_start)/60:.1f} min", flush=True)
    print(f"[prune]   source db: {src_mb:.1f} MB", flush=True)
    print(f"[prune]   dest   db: {dst_mb:.1f} MB "
          f"({100.0 * dst_mb / src_mb:.1f}% of source)", flush=True)
    print(f"[prune]   vertices  source: {total_a_verts_source:,}  "
          f"dest: {total_dst:,}", flush=True)


if __name__ == "__main__":
    main()
