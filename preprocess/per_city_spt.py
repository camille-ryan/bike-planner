"""Per-city subgraph SPTs.

For each city C, build an SPT rooted at C's anchor node, computed only
over the subgraph of (cell(C) ∪ adjacent_cells(C)). Each SPT covers a
few-cell-wide region around C, with parent pointers everywhere in that
region pointing toward C's anchor.

Why this layout:
  - Subgraphs overlap: cell B appears in B's own SPT *and* in every SPT
    rooted at a city adjacent to B. So at query time, "in cell B,
    follow gradient toward C" reduces to looking up C's SPT and
    walking parents from the current node.
  - Bounded-memory build: each city's subgraph is small (tens to
    hundreds of MB worth of edges); we load, Dijkstra, save, drop.
    Memory peak is dominated by the global graph held once at startup.
  - Lazy load at query: a Graz->Linz query only mmaps the SPTs along
    its city-graph path (5-10 of them), not the corridor's full set.

Output: `data/spt/<profile>/spt/<city_idx>.npz` containing
  - node_global  : int32, sorted ascending — global graph node indices
                   in this subgraph
  - parent_local : int32 — parent index within this same array
                   (-9999 for the source itself or unreachable)
  - cost         : float32 — shortest-path cost from city to each node

`np.searchsorted(node_global, N)` gives the local index for global N.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def _neighbors_from_city_graph(city_graph) -> dict[int, list[int]]:
    """Return {city_idx: [adjacent_city_idx, ...]} from the city graph."""
    nbrs: dict[int, set[int]] = defaultdict(set)
    for fa, tb in zip(city_graph.from_city, city_graph.to_city):
        nbrs[int(fa)].add(int(tb))
        nbrs[int(tb)].add(int(fa))   # symmetric — even though the graph is directed,
                                     # adjacency for routing data is set-valued
    return {c: sorted(s) for c, s in nbrs.items()}


def build_per_city_spts(
    out_dir: Path,
    graph,
    fwd_spt,
    city_graph,
    cities: list[dict],
) -> None:
    """Compute and save one SPT per city. Sequential — one city at a time."""
    spt_dir = out_dir / "spt"
    spt_dir.mkdir(parents=True, exist_ok=True)

    n_global = len(graph.node_lon)
    assigned = fwd_spt.city_idx          # int32, length n_global, -1 if unreachable
    src      = graph.edge_src
    dst      = graph.edge_dst
    cost     = graph.edge_cost
    nbrs_of  = _neighbors_from_city_graph(city_graph)

    # Save the global cell assignment once for the API to use as the
    # cell-crossing detection signal at query time.
    np.savez(out_dir / "global_assignment.npz", assigned_city=assigned)

    n_cities = len(cities)
    print(f"[per-city-spt] computing SPTs for {n_cities:,} cities")

    # Precompute the dst's assigned cell as a parallel array — used to
    # filter edges per city. Doing it once avoids per-city array indexing
    # of the same data.
    src_assigned = assigned[src]
    dst_assigned = assigned[dst]

    for ci, city in enumerate(cities):
        city_node_idx = int(city["node_idx"])
        # Subgraph node mask: in C's cell or in an adjacent cell.
        own_set = {ci}
        own_set.update(nbrs_of.get(ci, ()))
        node_mask = np.isin(assigned, np.fromiter(own_set, dtype=np.int32))
        node_global = np.flatnonzero(node_mask).astype(np.int32)

        # Edges where both endpoints are in our subgraph. The src/dst
        # mask uses *cell membership* of endpoints which we already have.
        edge_mask = node_mask[src] & node_mask[dst]
        e_src = src[edge_mask]
        e_dst = dst[edge_mask]
        e_cost = cost[edge_mask]

        # Local indices via searchsorted on the sorted global ids.
        local_src = np.searchsorted(node_global, e_src).astype(np.int32)
        local_dst = np.searchsorted(node_global, e_dst).astype(np.int32)
        n_local = len(node_global)
        csr = csr_matrix(
            (e_cost, (local_src, local_dst)),
            shape=(n_local, n_local),
            dtype=np.float32,
        )

        # Source's local index. The city's anchor node was snapped to a
        # graph node during the global SPT pass, so it must be in its
        # own cell (or its adjacent cells via Voronoi placement).
        local_source = int(np.searchsorted(node_global, city_node_idx))
        if local_source >= n_local or node_global[local_source] != city_node_idx:
            print(f"[per-city-spt] WARN city {ci} ({city['name']}): "
                  f"anchor node {city_node_idx} not in own subgraph; skipping")
            continue

        cost_arr, predecessors = dijkstra(
            csgraph=csr,
            indices=local_source,
            return_predecessors=True,
            directed=True,
        )
        # scipy's predecessor sentinel for "no predecessor" is -9999.
        np.savez(
            spt_dir / f"{ci}.npz",
            node_global=node_global,
            parent_local=predecessors.astype(np.int32),
            cost=cost_arr.astype(np.float32),
        )
        if (ci + 1) % 25 == 0 or ci + 1 == n_cities:
            print(f"[per-city-spt] {ci+1:,}/{n_cities:,}  last: {city['name']} "
                  f"(local nodes={n_local:,})")

    print(f"[per-city-spt] wrote {n_cities:,} SPTs to {spt_dir}")
