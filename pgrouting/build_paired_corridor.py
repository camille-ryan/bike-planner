"""Build pruned paired SPTs corridor-wide as a SQLite trunk DB.

Selects all anchors within `max_km` of a polyline, then builds
paired_(A, B) for every directed city_graph edge connecting two such
anchors. With Phase A's is_frontier byproduct, pruning is precise: kept
= F-only vertices whose A.SPT.parent chain leads to a true geographic
frontier leaf, plus the B-frontier vertices themselves.

Output: a SQLite database at `data/spt/<profile>/paired_trunks.db` with
a single table:

    CREATE TABLE trunks (
      src_city  INTEGER NOT NULL,
      dst_city  INTEGER NOT NULL,
      vertex_id INTEGER NOT NULL,
      successor INTEGER,                -- NULL at trunk root (B-frontier)
      lat       REAL NOT NULL,
      lon       REAL NOT NULL,
      PRIMARY KEY (src_city, dst_city, vertex_id)
    ) WITHOUT ROWID;

Each trunk row gives the next vertex toward the B-frontier (`successor`)
and the row's own coords. A routing client snaps the user's start, finds
the chain via city_graph Dijkstra, then for each chain edge (A, B):

    SELECT successor, lat, lon FROM trunks
     WHERE src_city = A AND dst_city = B AND vertex_id = current

…follows successor until NULL (= reached B-frontier handoff to next
chain edge's trunk). No npz loading per leg, no postgres roundtrips.

If the user's start vertex isn't in the first trunk, fall back to the
unpruned CSR data in the per-anchor SPT npz (Phase C: edge_indptr,
edge_indices, edge_cost) for a small local Dijkstra to bridge.
"""
from __future__ import annotations
import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import psycopg

import config
from build_paired_spts import (
    _build_pair, _build_ferry_pair_fake, _filter_spt_to_radius,
    _identify_ferry_chain_pairs, _load_spt, _load_topology, _fetch_coords,
    LARGE_SPT_THRESHOLD, FILTER_RADIUS_M,
)


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


def _open_trunk_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA synchronous = NORMAL")
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS trunks (
            src_city  INTEGER NOT NULL,
            dst_city  INTEGER NOT NULL,
            vertex_id INTEGER NOT NULL,
            successor INTEGER,
            lat       REAL NOT NULL,
            lon       REAL NOT NULL,
            PRIMARY KEY (src_city, dst_city, vertex_id)
        ) WITHOUT ROWID
        """
    )
    return db


def _trunk_rows_for_pair(
    a: int, b: int, pair: dict, lon: np.ndarray, lat: np.ndarray,
):
    """Yield (src, dst, vertex_id, successor, lat, lon) tuples for a
    paired SPT's kept vertices.

    `pair` is the output of `_build_pair(a_spt, b_spt, prune=True)`:
        node_global   int32[K]   global vertex IDs of kept vertices
        parent_local  int32[K]   kept-local index of each vertex's
                                 successor (= next on walk to B-frontier),
                                 or -9999 for trunk roots (B-frontier)
        cost          float32[K] (unused in DB)

    `lon`, `lat` are float arrays length K, one per kept vertex.
    """
    ng       = pair["node_global"]
    par      = pair["parent_local"]
    K        = len(ng)
    for i in range(K):
        succ_local = int(par[i])
        if succ_local < 0 or succ_local >= K:
            successor = None
        else:
            successor = int(ng[succ_local])
        yield (
            a, b,
            int(ng[i]),
            successor,
            float(lat[i]),
            float(lon[i]),
        )


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

    # 3. Caches.
    spt_cache: dict[int, dict] = {}
    topology_cache: dict[int, dict | None] = {}
    filtered_cache: dict[int, dict] = {}

    def get_spt(idx: int) -> dict:
        if idx not in spt_cache:
            spt_cache[idx] = _load_spt(spt_dir, idx)
        return spt_cache[idx]

    def get_topology(idx: int) -> dict | None:
        if idx not in topology_cache:
            topology_cache[idx] = _load_topology(topology_dir, idx)
        return topology_cache[idx]

    # 4. Build + write trunk rows.
    db = _open_trunk_db(db_path)
    with psycopg.connect(config.PG_DSN) as conn:
        # Identify ferry chain edges once.
        ferry_pairs = _identify_ferry_chain_pairs(
            conn, cities, spt_dir, sorted(corridor_idxs)
        )

        def get_filtered_spt(idx: int) -> dict:
            spt = get_spt(idx)
            if len(spt["node_global"]) < LARGE_SPT_THRESHOLD:
                return spt
            if idx not in filtered_cache:
                anchor = cities[idx]
                filtered_cache[idx] = _filter_spt_to_radius(
                    spt, anchor["lon"], anchor["lat"],
                    FILTER_RADIUS_M, conn,
                )
            return filtered_cache[idx] or spt

        REGULAR_DEGENERATE_THRESHOLD = 200
        written = 0
        skipped_existing = 0
        skipped_degenerate = 0
        total_kept = 0
        total_a    = 0
        total_rows_inserted = 0
        t_start = time.time()
        last_report = t_start

        BATCH = 5000
        row_buffer: list[tuple] = []

        def flush_buffer():
            nonlocal total_rows_inserted
            if not row_buffer:
                return
            db.executemany(
                "INSERT OR REPLACE INTO trunks "
                "(src_city, dst_city, vertex_id, successor, lat, lon) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                row_buffer,
            )
            total_rows_inserted += len(row_buffer)
            row_buffer.clear()

        # Resume support: skip pairs already in the DB.
        existing_pairs = set()
        for r in db.execute(
            "SELECT DISTINCT src_city, dst_city FROM trunks"
        ).fetchall():
            existing_pairs.add((int(r[0]), int(r[1])))
        if existing_pairs:
            print(f"[corridor] {len(existing_pairs):,} pairs already in DB "
                  f"(skipping)", flush=True)

        for i, (a, b) in enumerate(pairs_to_build):
            if (a, b) in existing_pairs:
                skipped_existing += 1
                continue

            a_filt = get_filtered_spt(a)
            b_filt = get_filtered_spt(b)
            pair = _build_pair(a_filt, b_filt, prune=prune)
            if (pair is None or pair["kept_size"] < REGULAR_DEGENERATE_THRESHOLD) \
               and (a, b) in ferry_pairs:
                pair = _build_ferry_pair_fake(
                    a_filt, get_spt(b), ferry_pairs[(a, b)],
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

            # Insert rows.
            for row in _trunk_rows_for_pair(a, b, pair, lon, lat):
                row_buffer.append(row)
                if len(row_buffer) >= BATCH:
                    flush_buffer()

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
                    f"db_rows={total_rows_inserted + len(row_buffer):,}  "
                    f"avg_kept_ratio={100*total_kept/max(1,total_a):.1f}% of A  "
                    f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                    flush=True,
                )
                last_report = now

        flush_buffer()
        # Vacuum + index after all inserts (SQLite indexes the PK by
        # default for WITHOUT ROWID tables, so just ANALYZE here).
        db.execute("ANALYZE trunks")
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
