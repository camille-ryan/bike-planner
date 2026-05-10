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
    """Load a per-anchor SPT npz. Newer npzs (Phase A+) include
    `is_frontier` and CSR edge arrays; older npzs lack them and the
    consumer must fall back to heuristics (e.g. cost-percentile)."""
    with np.load(spt_dir / f"{city_idx}.npz") as d:
        out = {
            "node_global":  np.asarray(d["node_global"]),
            "parent_local": np.asarray(d["parent_local"]),
            "cost":         np.asarray(d["cost"]),
        }
        for opt in ("is_frontier", "edge_indptr", "edge_indices", "edge_cost"):
            if opt in d.files:
                out[opt] = np.asarray(d[opt])
        return out


def _load_topology(topology_dir: Path, city_idx: int) -> dict | None:
    """Load shared topology (lon/lat per kept vertex) for an anchor.
    Returns None if the topology file doesn't exist (older builds)."""
    p = topology_dir / f"{city_idx}.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        return {
            "node_global": np.asarray(d["node_global"]),
            "lon":         np.asarray(d["lon"]),
            "lat":         np.asarray(d["lat"]),
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


def _identify_ferry_chain_pairs(
    conn: psycopg.Connection, cities: list[dict], spt_dir: Path,
    chain: list[int],
) -> dict[tuple[int, int], int]:
    """Find chain edges that cross a long ferry edge.

    Returns dict {(a_idx, b_idx): b_terminal_vid}, where b_terminal_vid
    is the vertex ID of the B-side ferry terminal — the endpoint where
    a fake paired SPT walk should terminate.

    Method: query postgres for long ferry edges (length ≥ 5 km — same
    threshold compute_spts.py uses to flag a ferry-touching anchor).
    For each ferry's two endpoints, find which chain anchor's SPT
    covers it with minimum cost — that anchor "owns" the terminal.
    Distinct owners → a ferry chain pair, in both directions.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT source, target, length_m
            FROM   ways
            WHERE  is_ferry AND length_m >= 5000
              AND  (cost >= 0 OR reverse_cost >= 0)
        """)
        ferries = [(int(r[0]), int(r[1]), float(r[2])) for r in cur.fetchall()]
    if not ferries:
        return {}

    ferry_endpoints = set()
    for s, t, _ in ferries:
        ferry_endpoints.add(s); ferry_endpoints.add(t)

    # For each chain anchor, scan its SPT for ferry-endpoint membership.
    # Keep the (anchor, cost) with the lowest cost per endpoint.
    owners: dict[int, tuple[int, float]] = {}
    for c in chain:
        path = spt_dir / f"{c}.npz"
        if not path.exists():
            continue
        with np.load(path) as d:
            ng = np.asarray(d["node_global"])
            cost = np.asarray(d["cost"])
        for vt in ferry_endpoints:
            pos = int(np.searchsorted(ng, vt))
            if pos < len(ng) and int(ng[pos]) == vt:
                cv = float(cost[pos])
                if vt not in owners or cv < owners[vt][1]:
                    owners[vt] = (c, cv)

    pairs: dict[tuple[int, int], int] = {}
    for s, t, _ in ferries:
        a_owner = owners.get(s, (None,))[0]
        b_owner = owners.get(t, (None,))[0]
        if a_owner is None or b_owner is None or a_owner == b_owner:
            continue
        pairs[(a_owner, b_owner)] = t   # walking from a → terminate at t
        pairs[(b_owner, a_owner)] = s
    return pairs


def _build_ferry_pair_fake(
    a_spt: dict, b_spt_full: dict, b_terminal_vid: int,
) -> dict | None:
    """Build a 'fake' paired SPT for a ferry chain edge.

    Uses B's full (unfiltered, 100 km) SPT — which contains the ferry
    edge in its parent_local because B is ferry-touching — restricted
    to vertices in A's filtered region plus the B-side terminal.
    Walking parent_local from any A-region vertex traverses the ferry
    and lands at b_terminal_vid, where the walk terminates because
    b_terminal's parent in B.SPT lives in B's interior (which we
    cut out).

    The result is a slice of B.SPT, walked using B.SPT.parent — a
    different gradient direction from the regular paired_(A, B) which
    is a slice of A.SPT walked using A.SPT.parent. For ferry crossings
    A.SPT.parent leads toward A's center, away from the ferry, so the
    regular construction can't help.
    """
    a_ng = a_spt["node_global"].astype(np.int64)
    b_ng = b_spt_full["node_global"]
    b_par = b_spt_full["parent_local"]
    b_cost = b_spt_full["cost"]

    pos = np.searchsorted(b_ng, a_ng)
    in_range = pos < len(b_ng)
    in_a = np.zeros(len(b_ng), dtype=bool)
    matched = pos[in_range]
    in_a[matched] = (b_ng[matched].astype(np.int64) == a_ng[in_range])

    # Explicitly include the B-side terminal (it might already be in A's
    # region for short ferries, but safe to add).
    bt_pos = int(np.searchsorted(b_ng, b_terminal_vid))
    if bt_pos < len(b_ng) and int(b_ng[bt_pos]) == b_terminal_vid:
        in_a[bt_pos] = True

    kept_idx = np.flatnonzero(in_a)
    if len(kept_idx) == 0:
        return None

    n = len(b_ng)
    remap = np.full(n, -1, dtype=np.int32)
    remap[kept_idx] = np.arange(len(kept_idx), dtype=np.int32)
    parent_in_b = b_par[kept_idx]
    parent_kept = remap[parent_in_b.clip(0)]
    parent_local = np.where(
        (parent_in_b >= 0) & (parent_kept >= 0),
        parent_kept, np.int32(-9999),
    ).astype(np.int32)

    return {
        "node_global":   b_ng[kept_idx].astype(np.int32),
        "parent_local":  parent_local,
        "cost":          b_cost[kept_idx].astype(np.float32),
        "kept_size":     int(len(kept_idx)),
        "a_size":        int(len(a_ng)),
        "f_only_size":   int(len(a_ng)),  # all of A is kept
        "frontier_size": 1,                # just b_terminal
        "ferry":         True,
    }


def _build_pair(a_spt: dict, b_spt: dict, prune: bool = False) -> dict | None:
    """Slice A.SPT to keep F-only vertices plus B's frontier in A.

    F-only:      v ∈ A.SPT \\ B.SPT
    B-frontier:  v ∈ A.SPT ∩ B.SPT, and ∃ u ∈ F-only with
                 A.SPT.parent_local[u] == v's local position in A.

    Kept = F-only ∪ B-frontier (unpruned, default).

    If `prune=True`, kept is further restricted to vertices whose
    A.SPT.parent chain eventually reaches a B-frontier vertex — i.e.,
    the optimal-path subset from F-only-leaves to B-frontier. F-only
    vertices whose parent chain leads only to A-seed without crossing
    B-frontier are excluded. Computed by vectorized fixpoint upward
    propagation along parent_local; converges in O(SPT depth)
    iterations.

    Routing walks A.SPT.parent_local within the kept set: parent_local
    is remapped so that any kept vertex whose A-parent is in B's
    interior (which we excluded) gets parent_local = -9999, naturally
    terminating the walk at the B-frontier vertex on the way out.
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

    if prune:
        valid_par_mask = a_par >= 0
        # Geographic frontier leaves on F-only side. PRECISE if A.SPT
        # has the `is_frontier` byproduct (Phase A onwards): a vertex
        # is at the geographic frontier iff its original road-graph
        # out-degree exceeds its sub_csr out-degree (= some road edge
        # leaves the 30 km region). HEURISTIC otherwise: top-20% A.cost.
        if "is_frontier" in a_spt:
            f_frontier_leaves = a_spt["is_frontier"].astype(bool) & f_only
        else:
            has_child = np.zeros(n, dtype=bool)
            has_child[a_par[valid_par_mask]] = True
            all_leaves = ~has_child & valid_par_mask
            f_only_costs = a_cost[f_only]
            if len(f_only_costs) == 0:
                return None
            cost_threshold = float(np.percentile(f_only_costs, 80))
            f_frontier_leaves = all_leaves & f_only & (a_cost >= cost_threshold)

        # Ancestors-of-frontier-leaves via vectorized fixpoint.
        in_ancestors = f_frontier_leaves.copy()
        prev_count = -1
        for _ in range(2000):
            cur_count = int(in_ancestors.sum())
            if cur_count == prev_count:
                break
            prev_count = cur_count
            marked = np.flatnonzero(in_ancestors & valid_par_mask)
            if len(marked) == 0:
                break
            in_ancestors[a_par[marked]] = True

        kept_mask = f_only & in_ancestors
        kept_mask[b_frontier] = True
    else:
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


def main(chain_names: list[str], profile: str = "lht", prune: bool = False) -> None:
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
        # Identify ferry chain edges up-front. These get a different
        # construction (slice of B's full SPT walked via B.parent)
        # because A.SPT.parent leads away from the ferry toward A-seed.
        ferry_pairs = _identify_ferry_chain_pairs(conn, cities, spt_dir, chain)
        if ferry_pairs:
            print(f"[paired] ferry chain pairs detected: {len(ferry_pairs)//2}")
            seen = set()
            for (a, b), bt in ferry_pairs.items():
                key = tuple(sorted((a, b)))
                if key in seen: continue
                seen.add(key)
                print(f"[paired]   {cities[a]['name']} ↔ {cities[b]['name']}  "
                      f"(B-terminal vid={bt})")

        # Build paired_(chain[i], chain[i+1]) for i = 1..N-2 (skip i=0).
        # Strategy: always try regular construction first; only fall
        # back to the ferry-fake when (a) regular is degenerate
        # (A and B don't meaningfully overlap) AND (b) the pair was
        # flagged as a ferry by the auto-detector. This avoids false
        # positives from coastal anchors that happen to own ferry
        # endpoints but are connected by land.
        REGULAR_DEGENERATE_THRESHOLD = 200    # |kept| below this triggers ferry fallback
        for i in range(1, len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            t0 = time.time()
            a_spt = get_spt(a)
            b_spt = get_spt(b)
            t_load = time.time() - t0

            t1 = time.time()
            pair = _build_pair(a_spt, b_spt, prune=prune)
            is_ferry = False
            if (pair is None or pair["kept_size"] < REGULAR_DEGENERATE_THRESHOLD) \
               and (a, b) in ferry_pairs:
                pair = _build_ferry_pair_fake(
                    a_spt, b_spt, ferry_pairs[(a, b)],
                )
                is_ferry = pair is not None
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
            kind = "FERRY" if is_ferry else "     "
            print(
                f"[paired] {kind} {cities[a]['name']:<22} → {cities[b]['name']:<22} "
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
    p.add_argument("--prune", action="store_true",
        help="Prune kept set to vertices on optimal A-leaf → B-frontier paths")
    args = p.parse_args()
    names = [s.strip() for s in args.chain.split(",")]
    main(names, profile=args.profile, prune=args.prune)
