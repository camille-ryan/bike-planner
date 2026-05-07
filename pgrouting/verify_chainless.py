"""Compare a chainless gradient-walk route against full-graph Dijkstra.

Walks the path from S using the *farthest-reachable city's gradient*
in the chain, switching cities as we cross SPT boundaries. Once the
current vertex falls inside the END city's SPT, switch to A* (a
plain scipy single-source Dijkstra over the full graph) to find the
exact path to E.

Reports:
  - True optimum: scipy single-source Dijkstra over the full graph.
  - Chainless walk: actual cost of the gradient-walked path.
  - Slack: how much extra the chainless walk pays vs. the optimum.

Edge costs are derived from SPT cost differences along parent chains:
edge(parent[v], v) = npz.cost[v] - npz.cost[parent[v]].

Future optimizations (not yet applied):
  2. Vectorize the entire per-city walk in one numpy pass — build the
     full parent chain via repeated `parent_local` indexing, then
     `np.isin` against end_node_global to find the crossover. Sub-ms
     for any practical walk length.
  3. Pre-compute a global "(vid -> [city_idx, local_idx]) coverage"
     table so the per-step "farthest reachable" check is O(1) instead
     of O(K) cities × O(log N) searchsorted. Memory ~5-10 GB at
     corridor scale (each vid in ~5-10 SPTs).
  4. Cython/numba JIT the inner walk loop. ~1 ns/step. Combined with
     opt 1 below, a 1100 km gradient walk becomes <1 ms.
  5. mmap the npz files for the routing API hot path so per-query
     memory footprint reflects only touched pages.
"""
import heapq
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
from scipy.sparse.csgraph import dijkstra

import compute_spts
import config


SPT_DIR = Path(os.environ.get("DATA_DIR", "/data")) / "spt" / "lht"


def city_graph_dijkstra(cg: dict, src: int, dst: int) -> tuple[float, list[int]]:
    """Plain Dijkstra on the directed city graph for chain planning."""
    adj: dict[int, list[tuple[int, float]]] = {}
    for fc, tc, w in zip(cg["from_city"], cg["to_city"], cg["weight"]):
        adj.setdefault(int(fc), []).append((int(tc), float(w)))
    dist = {src: 0.0}
    parent = {src: -1}
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            break
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    if dst not in dist:
        return float("inf"), []
    chain = []
    cur = dst
    while cur != -1:
        chain.append(cur)
        cur = parent[cur]
    chain.reverse()
    return dist[dst], chain


def gradient_walk(
    chain: list[int], npzs: dict, end_npz_set: set, S_vid: int,
) -> tuple[int, list[int], float]:
    """Walk the farthest-reachable city's gradient from S until current
    vertex enters the END city's SPT (`end_npz_set`).

    Returns `(V_star_vid, walked_path, walked_cost)`.
    `V_star_vid` is the first vertex on the gradient walk that's in
    end_npz_set. Stops early at the start city's polygon (parent=-9999)
    if we never reach end's SPT.

    Optimization 1: locate current vid via searchsorted only when we
    enter a new city segment; thereafter, `parent_local[i]` IS the next
    vertex's local index — no per-step searchsorted needed.
    """
    current = S_vid
    path = [current]
    total_cost = 0.0
    chain = list(chain)

    while chain:
        # Find farthest Xi reachable from current (one searchsorted per
        # candidate city; runs once per city segment, not per step).
        best_ci = None
        best_idx = None
        for ci in reversed(chain):
            arr = npzs[ci]["node_global"]
            i = int(np.searchsorted(arr, current))
            if i < len(arr) and int(arr[i]) == current:
                best_ci = ci
                best_idx = i
                break
        if best_ci is None:
            return -1, path, total_cost  # stuck

        # Walk this city's gradient with O(1) per step. The membership
        # check against end_npz_set runs every step; everything else is
        # an array index.
        npz = npzs[best_ci]
        node_global = npz["node_global"]
        parent_local = npz["parent_local"]
        cost_arr = npz["cost"]
        local_idx = best_idx

        while True:
            cur_vid = int(node_global[local_idx])
            if cur_vid in end_npz_set:
                return cur_vid, path, total_cost
            par = int(parent_local[local_idx])
            if par == -9999:
                # Reached this city's polygon. Drop it and re-evaluate.
                chain = [c for c in chain if c != best_ci]
                break
            edge_cost = float(cost_arr[local_idx]) - float(cost_arr[par])
            total_cost += edge_cost
            local_idx = par
            current = int(node_global[par])
            path.append(current)

    return current, path, total_cost


def main():
    a_name = sys.argv[1] if len(sys.argv) > 1 else "Pöllau"
    b_name = sys.argv[2] if len(sys.argv) > 2 else "Deutschlandsberg"

    with psycopg.connect(config.PG_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, snap_vertex_id FROM anchors "
            "WHERE name = ANY(%s) ORDER BY id",
            ([a_name, b_name],),
        )
        rows = {r[1]: (int(r[0]), int(r[2])) for r in cur.fetchall()}
        if a_name not in rows or b_name not in rows:
            sys.exit(f"missing anchor(s): have={list(rows)}")
        a_id, a_vid = rows[a_name]
        b_id, b_vid = rows[b_name]
        print(f"[verify] {a_name} (vid={a_vid}) -> {b_name} (vid={b_vid})")

        node_global, csr = compute_spts._load_graph(conn)

    a_idx = a_id - 1
    b_idx = b_id - 1

    # Plan chain on city_graph.
    cg = json.load(open(SPT_DIR / "city_graph.json"))
    chain_cost, chain = city_graph_dijkstra(cg, a_idx, b_idx)
    if not chain:
        sys.exit(f"no city_graph chain from {a_name} to {b_name}")
    print(f"[verify] chain {chain}  city_graph cost={chain_cost:.0f} (lower bound)")

    # Load the npzs for every city in the chain.
    npzs = {ci: dict(np.load(SPT_DIR / "spt" / f"{ci}.npz")) for ci in chain}
    end_npz_set = set(int(v) for v in npzs[b_idx]["node_global"])

    # Ground truth: scipy Dijkstra from a_vid.
    s_local = int(np.searchsorted(node_global, a_vid))
    e_local = int(np.searchsorted(node_global, b_vid))
    if s_local >= len(node_global) or int(node_global[s_local]) != a_vid:
        sys.exit("a_vid not in graph")
    if e_local >= len(node_global) or int(node_global[e_local]) != b_vid:
        sys.exit("b_vid not in graph")
    t0 = time.time()
    cost_arr_S = dijkstra(csgraph=csr, indices=s_local,
                          directed=True, return_predecessors=False)
    true_cost = float(cost_arr_S[e_local])
    print(f"[verify] true Dijkstra(full graph from S)  cost={true_cost:.0f}  "
          f"({time.time() - t0:.1f}s)")

    # Phase 1: gradient-walk from S, switching cities, until current
    # vertex enters end city's SPT.
    t0 = time.time()
    V_star_vid, walk_path, walk_cost = gradient_walk(
        chain[:-1], npzs, end_npz_set, a_vid,
    )
    if V_star_vid == -1:
        sys.exit("gradient walk got stuck")
    print(f"[verify] phase 1: walked {len(walk_path)-1} edges, "
          f"cost={walk_cost:.0f}, hit end-SPT at vid={V_star_vid} "
          f"({time.time() - t0:.1f}s)")

    # Phase 2: A* (scipy Dijkstra) from V_star to E.
    t0 = time.time()
    v_local = int(np.searchsorted(node_global, V_star_vid))
    cost_arr_V = dijkstra(csgraph=csr, indices=v_local,
                          directed=True, return_predecessors=False)
    phase2_cost = float(cost_arr_V[e_local])
    print(f"[verify] phase 2: A* V_star -> E  cost={phase2_cost:.0f}  "
          f"({time.time() - t0:.1f}s)")

    chainless_total = walk_cost + phase2_cost
    slack = (chainless_total - true_cost) / true_cost * 100
    print(f"[verify] CHAINLESS WALK total = {walk_cost:.0f} + {phase2_cost:.0f} "
          f"= {chainless_total:.0f}")
    print(f"[verify] TRUE OPTIMUM        = {true_cost:.0f}")
    print(f"[verify] SLACK               = {slack:+.2f}%")


if __name__ == "__main__":
    main()
