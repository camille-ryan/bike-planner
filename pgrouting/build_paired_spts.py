"""Experimental paired-SPT builder for a single chain.

For each chain edge (A, B) starting from chain index 1 (i.e., NOT
including the start city's outgoing edge), we slice A.SPT to keep:
    F-only:    vertices in A.SPT but NOT in B.SPT (A's territory
               outside B's reach)
    B-frontier vertices in A: vertices in A∩B whose A.SPT.parent
               points to an F-only vertex — they're entry points
               into B's reach when walking A.SPT.parent from A's
               outer territory.

Routing through paired_(A, B): walk A.SPT.parent_local within the
kept set. Each step either advances toward A-seeds (deeper into A's
territory) or terminates at a B-frontier vertex (its parent is in
B's interior, which we excluded). Termination vertex is the handoff
to paired_(B, C).

Why we skip the start city's pair: the chain Graz→F→B→...→Wien starts
at s_vid which lives in Graz's polygon (= in F.SPT due to chain edge
Graz→F). Walking F.SPT.parent from s_vid heads NORTH toward F-seed —
exactly the direction we want toward Bruck. Graz.SPT would walk us
south back into Graz's center (the U-turn we explicitly avoid).
"""
from __future__ import annotations
import heapq
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

import config


def _load_chain_graph(out_dir: Path):
    cities = json.load(open(out_dir / "cities.json"))
    cg = json.load(open(out_dir / "city_graph.json"))
    chain_adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for fa, tb, w in zip(cg["from_city"], cg["to_city"], cg["weight"]):
        chain_adj[int(tb)].append((int(fa), float(w)))
    return cities, chain_adj


def _chain_dijkstra(adj, src, dst):
    if src == dst: return [src]
    dist = {src: 0.0}; parent = {}
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            path = [u]
            while u != src:
                u = parent[u]; path.append(u)
            return list(reversed(path))
        if d > dist.get(u, float("inf")): continue
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd; parent[v] = u
                heapq.heappush(heap, (nd, v))
    raise RuntimeError(f"no chain-feasible path from city {src} to {dst}")


def _resolve_chain_by_name(cities, chain_adj, names: list[str]) -> list[int]:
    name_to_idx = {c["name"]: c["city_idx"] for c in cities}
    waypoints = [name_to_idx[n] for n in names]
    chain: list[int] = []
    for i in range(len(waypoints) - 1):
        leg = _chain_dijkstra(chain_adj, waypoints[i], waypoints[i + 1])
        chain.extend(leg if i == 0 else leg[1:])
    return chain


def _load_spt(spt_dir: Path, city_idx: int) -> dict[str, np.ndarray]:
    with np.load(spt_dir / f"{city_idx}.npz") as d:
        return {
            "node_global":  np.asarray(d["node_global"]),
            "parent_local": np.asarray(d["parent_local"]),
            "cost":         np.asarray(d["cost"]),
        }


def _fetch_coords(conn: psycopg.Connection, vids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fetch (lon, lat) for a sorted-ascending int32/64 vid array.

    Returns parallel float32 arrays, same length as vids. Postgres
    returns rows ORDER BY id; since vids is already sorted, we just
    align positionally — no extra dict map.
    """
    vid_list = vids.astype(np.int64).tolist()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, ST_X(the_geom), ST_Y(the_geom)
            FROM   ways_vertices_pgr
            WHERE  id = ANY(%s::bigint[])
            ORDER  BY id
            """,
            (vid_list,),
        )
        rows = cur.fetchall()
    if len(rows) != len(vids):
        # Defensive: build a map and fill missing with NaN.
        m = {int(r[0]): (float(r[1]), float(r[2])) for r in rows}
        lon = np.array([m.get(int(v), (float("nan"),))[0] for v in vids], dtype=np.float32)
        lat = np.array([m.get(int(v), (0.0, float("nan")))[1] for v in vids], dtype=np.float32)
    else:
        lon = np.array([r[1] for r in rows], dtype=np.float32)
        lat = np.array([r[2] for r in rows], dtype=np.float32)
    return lon, lat


def _build_pair(a_spt: dict, b_spt: dict) -> dict | None:
    """Slice A.SPT to keep F-only vertices plus B's frontier in A.

    F-only:      v ∈ A.SPT \\ B.SPT
    B-frontier:  v ∈ A.SPT ∩ B.SPT, and ∃ u ∈ F-only with
                 A.SPT.parent_local[u] == v's local position in A.

    Kept = F-only ∪ B-frontier. Routing walks A.SPT.parent_local
    within the kept set: parent_local is remapped so that any kept
    vertex whose A-parent is in B's interior (which we excluded) gets
    parent_local = -9999, naturally terminating the walk at the
    B-frontier vertex on the way out.
    """
    a_ng = a_spt["node_global"]
    a_par = a_spt["parent_local"]
    a_cost = a_spt["cost"]
    n = len(a_ng)

    b_ng = b_spt["node_global"].astype(np.int64)

    pos = np.searchsorted(b_ng, a_ng)
    in_range = pos < len(b_ng)
    in_b = np.zeros(n, dtype=bool)
    in_b[in_range] = b_ng[pos[in_range]] == a_ng[in_range]

    f_only = ~in_b
    if not f_only.any():
        return None

    f_only_parents = a_par[f_only]
    valid = f_only_parents >= 0
    candidates = f_only_parents[valid]
    b_frontier = np.unique(candidates[in_b[candidates]])

    kept_mask = f_only.copy()
    kept_mask[b_frontier] = True
    kept_idx = np.flatnonzero(kept_mask)
    if len(kept_idx) == 0:
        return None

    kept_node_global = a_ng[kept_idx].astype(np.int32)
    kept_cost = a_cost[kept_idx].astype(np.float32)

    remap = np.full(n, -1, dtype=np.int32)
    remap[kept_idx] = np.arange(len(kept_idx), dtype=np.int32)

    parent_in_a = a_par[kept_idx]
    parent_kept = remap[parent_in_a.clip(0)]
    parent_local = np.where(
        (parent_in_a >= 0) & (parent_kept >= 0),
        parent_kept, np.int32(-9999),
    ).astype(np.int32)

    return {
        "node_global":   kept_node_global,
        "parent_local":  parent_local,
        "cost":          kept_cost,
        "kept_size":     int(len(kept_idx)),
        "a_size":        n,
        "f_only_size":   int(f_only.sum()),
        "frontier_size": int(len(b_frontier)),
    }


def main(chain_names: list[str], profile: str = "lht") -> None:
    out_dir = config.SPT_DIR / profile
    spt_dir = out_dir / "spt"
    paired_dir = out_dir / "paired"
    paired_dir.mkdir(parents=True, exist_ok=True)

    print(f"[paired] chain spec: {' → '.join(chain_names)}")
    print(f"[paired] output: {paired_dir}")

    cities, chain_adj = _load_chain_graph(out_dir)
    chain = _resolve_chain_by_name(cities, chain_adj, chain_names)
    print(f"[paired] resolved chain ({len(chain)} cities):")
    print("        " + " → ".join(cities[c]["name"] for c in chain))
    print(f"[paired] start city ({cities[chain[0]]['name']}) is NOT used in any "
          f"paired SPT — its SPT would walk us backward.")

    total_kept = 0
    total_a = 0
    total_bytes = 0
    t_total = time.time()

    spt_cache: dict[int, dict] = {}
    def get_spt(idx: int) -> dict:
        if idx not in spt_cache:
            spt_cache[idx] = _load_spt(spt_dir, idx)
        return spt_cache[idx]

    # Open postgres for fetching (lon, lat) of kept vertices. Embedding
    # coords directly in the paired SPT npz eliminates the postgres
    # roundtrip during routing — ~300 ms saved per Graz→Cph query.
    with psycopg.connect(config.PG_DSN) as conn:
        # Build paired_(chain[i], chain[i+1]) for i = 1..N-2 (skip i=0).
        for i in range(1, len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            t0 = time.time()
            a_spt = get_spt(a); b_spt = get_spt(b)
            t_load = time.time() - t0

            t1 = time.time()
            pair = _build_pair(a_spt, b_spt)
            t_build = time.time() - t1

            if pair is None:
                print(f"[paired] {cities[a]['name']} → {cities[b]['name']}: degenerate (skipped)")
                continue

            t2 = time.time()
            lon, lat = _fetch_coords(conn, pair["node_global"])
            t_coords = time.time() - t2

            path = paired_dir / f"{a}_{b}.npz"
            np.savez(path,
                     node_global=pair["node_global"],
                     parent_local=pair["parent_local"],
                     cost=pair["cost"],
                     lon=lon,
                     lat=lat)
            sz = path.stat().st_size
            total_kept += pair["kept_size"]
            total_a += pair["a_size"]
            total_bytes += sz
            print(
                f"[paired] {cities[a]['name']:<22} → {cities[b]['name']:<22} "
                f"|A|={pair['a_size']:>9,}  "
                f"f_only={pair['f_only_size']:>9,}  "
                f"frontier={pair['frontier_size']:>5,}  "
                f"|kept|={pair['kept_size']:>9,} "
                f"({100*pair['kept_size']/max(1,pair['a_size']):4.1f}% of A) "
                f"{sz/1024:>7.1f} KB  "
                f"build={t_load+t_build:.2f}s coords={t_coords:.2f}s"
            )

    print(
        f"[paired] DONE in {time.time()-t_total:.1f}s. "
        f"total kept: {total_kept:,} "
        f"({100*total_kept/max(1,total_a):.1f}% of A-set sum). "
        f"total disk: {total_bytes/1e6:.2f} MB"
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--chain", default="Graz,Wien")
    p.add_argument("--profile", default="lht")
    args = p.parse_args()
    names = [s.strip() for s in args.chain.split(",")]
    main(names, profile=args.profile)
