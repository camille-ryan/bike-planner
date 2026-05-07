"""Hand-picked chain verification: route between two anchors using
gradient-walk along an explicit chain of intermediate cities.

Useful when city_graph.json doesn't exist yet (e.g. mid-preprocess).
Pass start city, end city, and the intermediate hops as extra args:

  python verify_chain.py Graz "København" \
      Wien Brno Praha Dresden Berlin Hamburg "Lübeck"

Algorithm: for each leg of the chain (X_i -> X_{i+1}), walk the
parent chain in X_{i+1}'s SPT from current vertex back to X_{i+1}'s
polygon. Then continue from there using the next leg. After the
final city's polygon is reached, run scipy single-source Dijkstra
from there to the end snap-vertex (the "A*" final mile).

Compares to true full-graph Dijkstra from the start vertex to the end.
"""
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


def walk_to_polygon(node_global, parent_local, cost_arr,
                    start_local_idx, polygon_set):
    """Walk parent chain from `start_local_idx` until current vertex is
    in `polygon_set` (a Python set of vids). Returns the (final_vid,
    cost_walked, edges_walked).
    """
    local_idx = start_local_idx
    cost_walked = 0.0
    edges_walked = 0
    while True:
        cur_vid = int(node_global[local_idx])
        if cur_vid in polygon_set:
            return cur_vid, cost_walked, edges_walked
        par = int(parent_local[local_idx])
        if par == -9999:
            # Reached this SPT's polygon (a source vertex).
            return cur_vid, cost_walked, edges_walked
        edge_cost = float(cost_arr[local_idx]) - float(cost_arr[par])
        cost_walked += edge_cost
        edges_walked += 1
        local_idx = par


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} START END [HOP1 HOP2 ...]")
    start_name = sys.argv[1]
    end_name = sys.argv[2]
    intermediates = sys.argv[3:]
    chain_names = [start_name] + intermediates + [end_name]

    print(f"[verify-chain] chain: {' -> '.join(chain_names)}")

    with psycopg.connect(config.PG_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, snap_vertex_id FROM anchors "
            "WHERE name = ANY(%s)",
            (chain_names,),
        )
        rows = {r[1]: (int(r[0]), int(r[2])) for r in cur.fetchall()}
        for n in chain_names:
            if n not in rows:
                sys.exit(f"missing anchor: {n}")
        chain = [(rows[n][0], rows[n][1]) for n in chain_names]
        # Polygon vertex sets for each chain city (need them to detect
        # "we've entered this city's polygon").
        polygon_sets: dict[int, set[int]] = {}
        for aid, _vid in chain:
            cur.execute("""
                SELECT v.id FROM ways_vertices_pgr v
                JOIN anchors a ON a.id = %s
                WHERE a.geom_boundary IS NOT NULL
                  AND ST_Contains(a.geom_boundary, v.the_geom)
            """, (aid,))
            polygon_sets[aid] = set(int(r[0]) for r in cur.fetchall())
            # If polygonless, fall back to {snap_vertex_id}.
            if not polygon_sets[aid]:
                polygon_sets[aid] = {_vid}
        # Load CSR for ground-truth Dijkstra.
        node_global, csr = compute_spts._load_graph(conn)

    start_vid = chain[0][1]
    end_vid = chain[-1][1]

    # Ground truth.
    s_local = int(np.searchsorted(node_global, start_vid))
    e_local = int(np.searchsorted(node_global, end_vid))
    if int(node_global[s_local]) != start_vid:
        sys.exit("start vid not in graph")
    if int(node_global[e_local]) != end_vid:
        sys.exit("end vid not in graph")
    t0 = time.time()
    cost_arr = dijkstra(csgraph=csr, indices=s_local,
                        directed=True, return_predecessors=False)
    true_cost = float(cost_arr[e_local])
    print(f"[verify-chain] true Dijkstra full graph: cost={true_cost:.0f}  "
          f"({time.time() - t0:.1f}s)")

    # Walk through chain. For leg X_i -> X_{i+1}, we use X_{i+1}'s SPT
    # to walk from current vertex back to its polygon.
    current_vid = start_vid
    total_cost = 0.0
    total_edges = 0
    for i in range(1, len(chain)):
        next_aid, next_vid = chain[i]
        next_ci = next_aid - 1
        spt_path = SPT_DIR / "spt" / f"{next_ci}.npz"
        if not spt_path.exists():
            sys.exit(f"missing SPT: {spt_path}")
        spt = np.load(spt_path)
        ng = spt["node_global"]; pl = spt["parent_local"]; ca = spt["cost"]
        idx = int(np.searchsorted(ng, current_vid))
        if idx >= len(ng) or int(ng[idx]) != current_vid:
            print(f"[verify-chain] FAIL: vid {current_vid} not in {chain_names[i]}'s SPT "
                  f"— chain broken at hop {i}")
            sys.exit(1)
        landed_vid, leg_cost, leg_edges = walk_to_polygon(
            ng, pl, ca, idx, polygon_sets[next_aid],
        )
        total_cost += leg_cost
        total_edges += leg_edges
        print(f"[verify-chain]   leg {i} {chain_names[i-1]}->{chain_names[i]}: "
              f"{leg_edges} edges, cost {leg_cost:.0f}, landed at vid {landed_vid}")
        current_vid = landed_vid

    # Final A* (full-graph Dijkstra) from the polygon entry to the end snap.
    if current_vid != end_vid:
        v_local = int(np.searchsorted(node_global, current_vid))
        t1 = time.time()
        cost_arr_v = dijkstra(csgraph=csr, indices=v_local,
                              directed=True, return_predecessors=False)
        last_leg_cost = float(cost_arr_v[e_local])
        total_cost += last_leg_cost
        print(f"[verify-chain]   final A* {current_vid} -> {end_vid}: "
              f"cost {last_leg_cost:.0f}  ({time.time() - t1:.1f}s)")

    slack = (total_cost - true_cost) / true_cost * 100
    print(f"[verify-chain] CHAIN total = {total_cost:.0f}  ({total_edges} gradient edges)")
    print(f"[verify-chain] TRUE OPTIMUM = {true_cost:.0f}")
    print(f"[verify-chain] SLACK        = {slack:+.2f}%")


if __name__ == "__main__":
    main()
