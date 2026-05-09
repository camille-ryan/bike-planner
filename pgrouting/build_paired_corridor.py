"""Build paired SPTs for the entire Graz→Cph corridor.

Selects all anchors within `--max-km` of the priority polyline, then
builds paired_(A, B) for every directed city_graph edge where the
required relation holds (chain_adj sense: B.SPT covers A's polygon).
Produces one npz per directed pair under data/spt/<profile>/paired/.

Skips degenerate pairs (kept_size == 0). Reports build-time and disk
totals for the full set.
"""
from __future__ import annotations
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

import config
from build_paired_spts import _build_pair, _load_spt


def _parse_polyline(spec: str) -> list[tuple[float, float]]:
    parts = [float(x) for x in spec.split(",")]
    if len(parts) < 4 or len(parts) % 2 != 0:
        raise ValueError(f"polyline needs lon,lat,lon,lat,... got {len(parts)}")
    return list(zip(parts[0::2], parts[1::2]))


def _corridor_anchor_ids(
    conn: psycopg.Connection, polyline: list[tuple[float, float]],
    max_km: float,
) -> set[int]:
    """Return anchor.id values within max_km of the polyline."""
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--polyline",
        default="15.4395,47.0707,16.3725,48.2082,16.6068,49.1951,"
                "14.4378,50.0755,13.7373,51.0504,13.405,52.52,"
                "9.9937,53.5511,10.6866,53.8654,12.5683,55.6761",
        help="lon,lat,lon,lat,... — default = Graz→Cph priority line",
    )
    p.add_argument("--max-km", type=float, default=80.0,
        help="anchors within this distance of polyline qualify (default 80 km)")
    p.add_argument("--profile", default="lht")
    args = p.parse_args()

    polyline = _parse_polyline(args.polyline)
    out_dir = config.SPT_DIR / args.profile
    spt_dir = out_dir / "spt"
    paired_dir = out_dir / "paired"
    paired_dir.mkdir(parents=True, exist_ok=True)

    print(f"[corridor] polyline ({len(polyline)} waypoints), max_km={args.max_km}")

    # 1. Identify corridor anchors via SQL.
    with psycopg.connect(config.PG_DSN) as conn:
        corridor_ids = _corridor_anchor_ids(conn, polyline, args.max_km)
    print(f"[corridor] {len(corridor_ids):,} anchors within {args.max_km:.0f} km of polyline")

    # 2. Map anchor.id → city_idx (cities.json index = anchor.id - 1).
    with open(out_dir / "cities.json") as fh:
        cities = json.load(fh)
    aid_to_idx = {c["anchor_id"]: c["city_idx"] for c in cities}
    corridor_idxs = {aid_to_idx[aid] for aid in corridor_ids if aid in aid_to_idx}

    # 3. Read city_graph and filter to edges where BOTH endpoints are
    # corridor anchors. city_graph stores forward edges (X, Y) meaning
    # X.SPT covers Y's polygon. The corresponding routing-chain pair
    # is (Y, X) — when at Y heading next, we use X.SPT, so we build
    # paired_(Y, X) with Y as A and X as B.
    with open(out_dir / "city_graph.json") as fh:
        cg = json.load(fh)
    pairs_to_build: list[tuple[int, int]] = []
    for fc, tc in zip(cg["from_city"], cg["to_city"]):
        a, b = int(tc), int(fc)   # routing-chain order: Y is A, X is B
        if a in corridor_idxs and b in corridor_idxs:
            pairs_to_build.append((a, b))
    pairs_to_build = list(set(pairs_to_build))
    print(f"[corridor] {len(pairs_to_build):,} directed pairs to build")

    # 4. Build them. Cache loaded SPTs since each anchor appears in many
    # pairs (~32 outgoing edges on average).
    spt_cache: dict[int, dict] = {}
    def get_spt(idx: int) -> dict:
        if idx not in spt_cache:
            spt_cache[idx] = _load_spt(spt_dir, idx)
        return spt_cache[idx]

    written = 0
    skipped_existing = 0
    skipped_degenerate = 0
    total_kept = 0
    total_a = 0
    total_bytes = 0
    t_start = time.time()
    last_report = t_start
    for i, (a, b) in enumerate(pairs_to_build):
        out_path = paired_dir / f"{a}_{b}.npz"
        if out_path.exists():
            skipped_existing += 1
            continue
        a_spt = get_spt(a); b_spt = get_spt(b)
        pair = _build_pair(a_spt, b_spt)
        if pair is None or pair["kept_size"] == 0:
            skipped_degenerate += 1
            # Mark as built (empty) so resume skips. Actually skip writing.
            continue
        np.savez(out_path,
                 node_global=pair["node_global"],
                 parent_local=pair["parent_local"],
                 cost=pair["cost"])
        sz = out_path.stat().st_size
        written += 1
        total_kept += pair["kept_size"]
        total_a += pair["a_size"]
        total_bytes += sz

        now = time.time()
        if now - last_report >= 5.0 or i == len(pairs_to_build) - 1:
            elapsed = now - t_start
            done = written + skipped_existing + skipped_degenerate
            rate = done / max(elapsed, 1e-3)
            remaining = (len(pairs_to_build) - done) / max(rate, 1e-3)
            print(f"[corridor] {done:,}/{len(pairs_to_build):,}  "
                  f"written={written:,} skipped_deg={skipped_degenerate:,}  "
                  f"disk={total_bytes/1e6:.1f} MB  "
                  f"avg_kept_ratio={100*total_kept/max(1,total_a):.1f}% of A  "
                  f"elapsed={elapsed:.0f}s  ETA={remaining:.0f}s",
                  flush=True)
            last_report = now

    print(f"[corridor] DONE in {time.time()-t_start:.1f}s")
    print(f"[corridor]   written:           {written:,}")
    print(f"[corridor]   skipped existing:  {skipped_existing:,}")
    print(f"[corridor]   skipped degenerate:{skipped_degenerate:,}")
    print(f"[corridor]   total disk:        {total_bytes/1e6:.1f} MB")
    print(f"[corridor]   avg kept/A:        {100*total_kept/max(1,total_a):.1f}%")


if __name__ == "__main__":
    main()
