"""Build pruned paired SPTs corridor-wide as a SQLite trunk DB.

Selects all anchors within `max_km` of a polyline, then builds
paired_(A, B) for every directed city_graph edge connecting two such
anchors. With Phase A's is_frontier byproduct, pruning is precise: kept
= F-only vertices whose A.SPT.parent chain leads to a true geographic
frontier leaf, plus the B-frontier vertices themselves.

Output: a SQLite database at `data/spt/<profile>/paired_trunks.db` with
one row per pair, each row holding a packed numpy structured array:

    CREATE TABLE trunk_blobs (
      src_city  INTEGER NOT NULL,
      dst_city  INTEGER NOT NULL,
      n_rows    INTEGER NOT NULL,
      blob      BLOB    NOT NULL,        -- TRUNK_DTYPE * n_rows
      PRIMARY KEY (src_city, dst_city)
    ) WITHOUT ROWID;

    TRUNK_DTYPE = [('vid', i8), ('succ', i8), ('lat', f4), ('lon', f4)]
    succ == -1 means trunk root (B-frontier)
    rows within a blob are sorted by vid ascending

Routing clients load each pair's trunk with np.frombuffer (zero-copy
view) and walk via a precomputed next_idx pointer array. A typical
chain (Graz→Cph, 71 legs) preloads in ~200 ms warm, ~30 MB resident,
and walks in <30 ms thereafter — see test_graz_cph.py for the
reference loader/walker.

If the user's start vertex isn't in the first trunk, fall back to the
unpruned CSR data in the per-anchor SPT npz (Phase C: edge_indptr,
edge_indices, edge_cost) for a small local Dijkstra to bridge.
"""
from __future__ import annotations
import argparse
import json
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import psycopg

import config
from build_paired_spts import (
    _build_pair, _build_ferry_pair_fake,
    _identify_ferry_chain_pairs, _load_spt, _load_topology, _fetch_coords,
)


SPT_CACHE_MAX = 64        # ~80 MB/SPT loaded × 64 ≈ 5 GB cap
TOPOLOGY_CACHE_MAX = 64   # ~10 MB each, much smaller


def _parse_polyline(spec: str) -> list[tuple[float, float]]:
    parts = [float(x) for x in spec.split(",")]
    if len(parts) < 4 or len(parts) % 2 != 0:
        raise ValueError(f"polyline needs lon,lat,lon,lat,...; got {len(parts)}")
    return list(zip(parts[0::2], parts[1::2]))


DEFAULT_POLYLINE = (
    "15.4395,47.0707,16.3725,48.2082,16.6068,49.1951,"
    "14.4378,50.0755,13.7373,51.0504,13.405,52.52,"
    "9.9937,53.5511,10.6866,53.8654,12.5683,55.6761"
)


def _corridor_anchor_ids(
    conn: psycopg.Connection, polyline: list[tuple[float, float]],
    max_km: float,
) -> set[int]:
    pts_sql = ", ".join(f"ST_MakePoint({lon}, {lat})" for lon, lat in polyline)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            WITH route AS (
                SELECT ST_SetSRID(ST_MakeLine(ARRAY[{pts_sql}]), 4326) AS line
            )
            SELECT a.id
            FROM   anchors a, route r
            WHERE  a.snap_vertex_id IS NOT NULL
              AND  ST_Distance(a.geom::geography, r.line::geography) <= %s
            """,
            (max_km * 1000,),
        )
        return {int(r[0]) for r in cur.fetchall()}


# Packed-blob trunk schema. Each pair = one row containing a numpy
# structured array of (vid, succ, lat, lon) tuples, sorted by vid.
# Routing clients load via np.frombuffer (zero-copy) and walk via
# precomputed next_idx pointer arrays.
TRUNK_DTYPE = np.dtype([
    ("vid",  "<i8"),
    ("succ", "<i8"),   # NULL_SENTINEL = -1 means trunk root (B-frontier)
    ("lat",  "<f4"),
    ("lon",  "<f4"),
])
NULL_SENTINEL = np.int64(-1)


def _open_trunk_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA synchronous = NORMAL")
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS trunk_blobs (
            src_city  INTEGER NOT NULL,
            dst_city  INTEGER NOT NULL,
            n_rows    INTEGER NOT NULL,
            blob      BLOB    NOT NULL,
            PRIMARY KEY (src_city, dst_city)
        ) WITHOUT ROWID
        """
    )
    return db


def _pack_trunk_blob(
    pair: dict, lon: np.ndarray, lat: np.ndarray,
) -> tuple[int, bytes]:
    """Pack a pair's kept vertices into a TRUNK_DTYPE byte buffer.

    `pair` fields:
        node_global   int32[K]   global vertex IDs
        parent_local  int32[K]   kept-local successor index, or <0 / >=K
                                 for trunk roots (B-frontier)

    Returns (n_rows, blob_bytes). Rows are sorted by vid ascending so
    clients can use searchsorted on `arr['vid']` without rebuilding.
    """
    ng  = pair["node_global"]
    par = pair["parent_local"]
    K   = len(ng)
    if K == 0:
        return 0, b""

    # Resolve each vertex's successor (kept-local index) to a global vid,
    # or to NULL_SENTINEL if it's a trunk root.
    valid_par = (par >= 0) & (par < K)
    succ = np.where(valid_par, ng[np.clip(par, 0, K - 1)], NULL_SENTINEL)

    # Order by vid so loaders can searchsorted on arr['vid'].
    order = np.argsort(ng.astype(np.int64), kind="stable")
    arr = np.empty(K, dtype=TRUNK_DTYPE)
    arr["vid"]  = ng[order].astype(np.int64)
    arr["succ"] = succ[order].astype(np.int64)
    arr["lat"]  = lat[order].astype(np.float32)
    arr["lon"]  = lon[order].astype(np.float32)
    return K, arr.tobytes()


def run(
    out_dir: Path,
    polyline: str | None = None,
    max_km: float = 80.0,
    prune: bool = True,
    keep_npzs: bool = False,
) -> None:
    """Build trunk DB for all corridor pairs.

    Args:
      out_dir: e.g. data/spt/lht/
      polyline: lon,lat,... string. Defaults to Graz→Cph corridor.
      max_km: anchor inclusion radius around polyline.
      prune: enable frontier-leaf-ancestor pruning (much smaller trunk).
      keep_npzs: also write per-pair npzs alongside the DB.
    """
    polyline_str = polyline or DEFAULT_POLYLINE
    polyline_pts = _parse_polyline(polyline_str)

    spt_dir       = out_dir / "spt"
    paired_dir    = out_dir / "paired"
    topology_dir  = out_dir.parent.parent / "road_topology"
    db_path       = out_dir / "paired_trunks.db"
    if keep_npzs:
        paired_dir.mkdir(parents=True, exist_ok=True)

    print(f"[corridor] polyline ({len(polyline_pts)} waypoints), max_km={max_km}",
          flush=True)

    # 1. Identify corridor anchors.
    with psycopg.connect(config.PG_DSN) as conn:
        corridor_ids = _corridor_anchor_ids(conn, polyline_pts, max_km)
    print(f"[corridor] {len(corridor_ids):,} anchors within {max_km:.0f} km of polyline",
          flush=True)

    with open(out_dir / "cities.json") as fh:
        cities = json.load(fh)
    aid_to_idx = {c["anchor_id"]: c["city_idx"] for c in cities}
    corridor_idxs = {aid_to_idx[aid] for aid in corridor_ids if aid in aid_to_idx}

    # 2. Pairs to build: directed (a, b) where the routing-chain step
    # uses A's SPT to reach B-frontier. city_graph stores forward edges
    # (X, Y) meaning X.SPT covers Y's polygon → routing pair = (Y, X).
    with open(out_dir / "city_graph.json") as fh:
        cg = json.load(fh)
    pairs_to_build: list[tuple[int, int]] = list({
        (int(tc), int(fc))
        for fc, tc in zip(cg["from_city"], cg["to_city"])
        if int(tc) in corridor_idxs and int(fc) in corridor_idxs
    })
    print(f"[corridor] {len(pairs_to_build):,} directed pairs to build", flush=True)

    # 3. Caches — bounded LRU. SPT npzs decompress to ~80 MB each;
    # 927 corridor anchors × all-cached would OOM a 16 GB host.
    spt_cache: OrderedDict[int, dict] = OrderedDict()
    topology_cache: OrderedDict[int, dict | None] = OrderedDict()

    def get_spt(idx: int) -> dict:
        if idx in spt_cache:
            spt_cache.move_to_end(idx)
            return spt_cache[idx]
        v = _load_spt(spt_dir, idx)
        spt_cache[idx] = v
        if len(spt_cache) > SPT_CACHE_MAX:
            spt_cache.popitem(last=False)
        return v

    def get_topology(idx: int) -> dict | None:
        if idx in topology_cache:
            topology_cache.move_to_end(idx)
            return topology_cache[idx]
        v = _load_topology(topology_dir, idx)
        topology_cache[idx] = v
        if len(topology_cache) > TOPOLOGY_CACHE_MAX:
            topology_cache.popitem(last=False)
        return v

    # 4. Build + write trunk rows.
    db = _open_trunk_db(db_path)
    with psycopg.connect(config.PG_DSN) as conn:
        # Identify ferry chain edges once.
        ferry_pairs = _identify_ferry_chain_pairs(
            conn, cities, spt_dir, sorted(corridor_idxs)
        )

        REGULAR_DEGENERATE_THRESHOLD = 200
        written = 0
        skipped_existing = 0
        skipped_degenerate = 0
        total_kept = 0
        total_a    = 0
        total_rows_inserted = 0
        t_start = time.time()
        last_report = t_start

        BATCH = 500              # blobs per executemany
        blob_buffer: list[tuple[int, int, int, bytes]] = []

        def flush_buffer():
            if not blob_buffer:
                return
            db.executemany(
                "INSERT OR REPLACE INTO trunk_blobs "
                "(src_city, dst_city, n_rows, blob) VALUES (?, ?, ?, ?)",
                blob_buffer,
            )
            blob_buffer.clear()

        # Resume support: skip pairs already in the DB.
        existing_pairs = set()
        for r in db.execute(
            "SELECT src_city, dst_city FROM trunk_blobs"
        ).fetchall():
            existing_pairs.add((int(r[0]), int(r[1])))
        if existing_pairs:
            print(f"[corridor] {len(existing_pairs):,} pairs already in DB "
                  f"(skipping)", flush=True)

        for i, (a, b) in enumerate(pairs_to_build):
            if (a, b) in existing_pairs:
                skipped_existing += 1
                continue

            a_spt = get_spt(a)
            b_spt = get_spt(b)
            pair = _build_pair(a_spt, b_spt, prune=prune)
            if (pair is None or pair["kept_size"] < REGULAR_DEGENERATE_THRESHOLD) \
               and (a, b) in ferry_pairs:
                pair = _build_ferry_pair_fake(
                    a_spt, b_spt, ferry_pairs[(a, b)],
                )

            if pair is None or pair["kept_size"] == 0:
                skipped_degenerate += 1
                continue

            # Coords for kept vertices: prefer topology file, fall back
            # to postgres if absent.
            kept_globals = pair["node_global"]
            topology = get_topology(a)
            if topology is not None:
                pos = np.searchsorted(
                    topology["node_global"], kept_globals.astype(np.int64),
                )
                in_range = pos < len(topology["node_global"])
                matched = np.zeros_like(in_range)
                matched[in_range] = (
                    topology["node_global"][pos[in_range]]
                    == kept_globals[in_range]
                )
                if matched.all():
                    lon = topology["lon"][pos]
                    lat = topology["lat"][pos]
                else:
                    # Fallback: postgres for any missing.
                    lon, lat = _fetch_coords(conn, kept_globals)
            else:
                lon, lat = _fetch_coords(conn, kept_globals)

            n, blob = _pack_trunk_blob(pair, lon, lat)
            if n == 0:
                skipped_degenerate += 1
                continue
            blob_buffer.append((a, b, n, blob))
            total_rows_inserted += n

            written += 1
            total_kept += pair["kept_size"]
            total_a    += pair["a_size"]

            if keep_npzs:
                npz_path = paired_dir / f"{a}_{b}.npz"
                np.savez(
                    npz_path,
                    node_global=pair["node_global"],
                    parent_local=pair["parent_local"],
                    cost=pair["cost"],
                    lon=lon, lat=lat,
                )

            if len(blob_buffer) >= BATCH:
                flush_buffer()

            now = time.time()
            if now - last_report >= 5.0 or i == len(pairs_to_build) - 1:
                flush_buffer()
                db.commit()
                elapsed = now - t_start
                done    = written + skipped_existing + skipped_degenerate
                remaining_pairs = len(pairs_to_build) - done
                rate = done / max(elapsed, 1e-3)
                eta  = remaining_pairs / max(rate, 1e-3)
                print(
                    f"[corridor] {done:,}/{len(pairs_to_build):,}  "
                    f"written={written:,} deg={skipped_degenerate:,}  "
                    f"trunk_rows={total_rows_inserted:,}  "
                    f"avg_kept_ratio={100*total_kept/max(1,total_a):.1f}% of A  "
                    f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                    flush=True,
                )
                last_report = now

        flush_buffer()
        db.execute("ANALYZE trunk_blobs")
        db.commit()
        db.close()

    db_size = db_path.stat().st_size if db_path.exists() else 0
    print(f"[corridor] DONE in {time.time()-t_start:.1f}s", flush=True)
    print(f"[corridor]   pairs written:     {written:,}", flush=True)
    print(f"[corridor]   pairs skipped (existing):   {skipped_existing:,}",
          flush=True)
    print(f"[corridor]   pairs skipped (degenerate): {skipped_degenerate:,}",
          flush=True)
    print(f"[corridor]   trunk rows inserted: {total_rows_inserted:,}",
          flush=True)
    print(f"[corridor]   trunk DB size:       {db_size/1e6:.1f} MB", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--polyline", default=DEFAULT_POLYLINE,
        help="lon,lat,lon,lat,... — corridor polyline")
    p.add_argument("--max-km", type=float, default=80.0,
        help="anchors within this distance of polyline qualify")
    p.add_argument("--profile", default="lht")
    p.add_argument("--no-prune", action="store_true",
        help="Disable pruning (kept = F-only ∪ B-frontier, ~3× larger DB)")
    p.add_argument("--keep-npzs", action="store_true",
        help="Also write per-pair npzs in paired/ (for inspection)")
    args = p.parse_args()

    out_dir = config.SPT_DIR / args.profile
    run(
        out_dir,
        polyline=args.polyline,
        max_km=args.max_km,
        prune=not args.no_prune,
        keep_npzs=args.keep_npzs,
    )


if __name__ == "__main__":
    main()
