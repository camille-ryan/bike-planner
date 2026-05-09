"""Experimental paired-SPT builder for a single chain.

For each consecutive pair (A, B) along the chain we compute the
"trunk" subset of B's SPT — the union of parent-chain traces starting
from every A-seed and walking B.SPT.parent_local until hitting a
B-seed. That subset is exactly the vertices on optimal paths from
A-seeds to B-seeds in B's existing gradient.

No new Dijkstra needed: B.SPT already encodes those gradients.

Per pair we save `data/spt/lht/paired/{a_idx}_{b_idx}.npz` with
node_global (sorted), parent_local (kept-set local indices), cost
(B-side cost-from-B-seed). The walker semantics are the same as
production npzs: searchsorted to find current vertex, walk parent_local
toward cost==0.
"""
from __future__ import annotations
import heapq
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import config


def _load_chain_graph(out_dir: Path):
    cities = json.load(open(out_dir / "cities.json"))
    cg = json.load(open(out_dir / "city_graph.json"))
    chain_adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for fa, tb, w in zip(cg["from_city"], cg["to_city"], cg["weight"]):
        chain_adj[int(tb)].append((int(fa), float(w)))
    return cities, chain_adj


def _chain_dijkstra(adj: dict[int, list[tuple[int, float]]], src: int, dst: int) -> list[int]:
    if src == dst:
        return [src]
    dist = {src: 0.0}
    parent: dict[int, int] = {}
    heap: list[tuple[float, int]] = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            path = [u]
            while u != src:
                u = parent[u]
                path.append(u)
            return list(reversed(path))
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                parent[v] = u
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


def _build_pair(a_spt: dict, b_spt: dict) -> dict | None:
    """Trace B.SPT.parent_local from each A-seed back to a B-seed; the
    union of visited B-local indices is the paired-SPT vertex set.
    Construct (kept_node_global, kept_parent_local, kept_cost) from
    that union.
    """
    a_ng = a_spt["node_global"].astype(np.int64)
    a_seeds = a_ng[a_spt["cost"] == 0]

    b_ng = b_spt["node_global"].astype(np.int64)
    b_parent = b_spt["parent_local"]
    b_cost = b_spt["cost"]

    # Map A-seeds to their position in B.SPT (skip A-seeds not in B).
    pos = np.searchsorted(b_ng, a_seeds)
    in_range = pos < len(b_ng)
    matched = np.zeros_like(in_range)
    matched[in_range] = b_ng[pos[in_range]] == a_seeds[in_range]
    start_locals = pos[matched].astype(np.int64)
    if len(start_locals) == 0:
        return None

    # Trace each A-seed's path through B.SPT.parent_local until reaching
    # a B-seed (cost == 0). Mark all touched B-local indices.
    visited = np.zeros(len(b_ng), dtype=bool)
    for s in start_locals:
        cur = int(s)
        while cur >= 0 and not visited[cur]:
            visited[cur] = True
            if b_cost[cur] == 0.0:
                break  # at a B-seed; trace done
            nxt = int(b_parent[cur])
            if nxt < 0 or nxt == cur:
                break  # broken chain (shouldn't happen for well-formed SPT)
            cur = nxt

    if not visited.any():
        return None

    # Build the kept-set view. node_global and cost simply slice; parent
    # has to be remapped from b-local indices to kept-local indices.
    kept_idx = np.flatnonzero(visited)              # ascending b-local positions
    kept_node_global = b_ng[kept_idx].astype(np.int32)
    kept_cost = b_cost[kept_idx].astype(np.float32)

    # remap[b_local] = kept-local index, or -1 if not kept.
    remap = np.full(len(b_ng), -1, dtype=np.int32)
    remap[kept_idx] = np.arange(len(kept_idx), dtype=np.int32)

    parent_b = b_parent[kept_idx]
    parent_local = np.where(
        parent_b >= 0, remap[parent_b.clip(0)], np.int32(-9999),
    ).astype(np.int32)
    # If parent_b indexed a non-kept vertex (shouldn't happen for the
    # closed traces above), remap returns -1 → also treat as -9999.
    parent_local[parent_b < 0] = -9999
    parent_local[(parent_b >= 0) & (remap[parent_b.clip(0)] < 0)] = -9999

    return {
        "node_global":      kept_node_global,
        "parent_local":     parent_local,
        "cost":             kept_cost,
        "kept_size":        int(len(kept_idx)),
        "b_size":           int(len(b_ng)),
        "a_seed_count":     int(len(a_seeds)),
        "a_seed_in_b":      int(matched.sum()),
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

    total_kept = 0
    total_b = 0
    total_bytes = 0
    t_total = time.time()

    # Cache loaded SPTs to avoid re-reading consecutive pairs' shared
    # endpoint twice.
    spt_cache: dict[int, dict] = {}
    def get_spt(idx: int) -> dict:
        if idx not in spt_cache:
            spt_cache[idx] = _load_spt(spt_dir, idx)
        return spt_cache[idx]

    for i in range(len(chain) - 1):
        a, b = chain[i], chain[i + 1]
        t0 = time.time()
        a_spt = get_spt(a); b_spt = get_spt(b)
        t_load = time.time() - t0

        t1 = time.time()
        pair = _build_pair(a_spt, b_spt)
        t_trace = time.time() - t1

        if pair is None:
            print(f"[paired] {cities[a]['name']} → {cities[b]['name']}: NO INTERSECTION (skipped)")
            continue

        path = paired_dir / f"{a}_{b}.npz"
        np.savez(
            path,
            node_global=pair["node_global"],
            parent_local=pair["parent_local"],
            cost=pair["cost"],
        )
        sz = path.stat().st_size
        total_kept += pair["kept_size"]
        total_b += pair["b_size"]
        total_bytes += sz
        print(
            f"[paired] {cities[a]['name']:<22} → {cities[b]['name']:<22} "
            f"a_seeds={pair['a_seed_count']:>5,} (in B: {pair['a_seed_in_b']:>5,})  "
            f"|B|={pair['b_size']:>9,}  "
            f"|kept|={pair['kept_size']:>7,} "
            f"({100*pair['kept_size']/max(1,pair['b_size']):4.1f}% of B) "
            f"{sz/1024:>6.1f} KB  "
            f"load={t_load:.2f}s trace={t_trace:.2f}s"
        )

    print(
        f"[paired] DONE in {time.time()-t_total:.1f}s. "
        f"total kept: {total_kept:,} vs B-set sum {total_b:,} "
        f"({100*total_kept/max(1,total_b):.1f}%). "
        f"total disk: {total_bytes/1e6:.2f} MB"
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--chain", default="Graz,Wien",
                   help="comma-separated waypoint names")
    p.add_argument("--profile", default="lht")
    args = p.parse_args()
    names = [s.strip() for s in args.chain.split(",")]
    main(names, profile=args.profile)
