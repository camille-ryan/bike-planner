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
    With edges pre-sharded by source cell on disk, an iteration only
    reads ~7-8 cells' worth of edges instead of scanning the global
    240M-edge array. Peak per-iteration memory drops from ~5 GB
    (when scanning globals) to <500 MB.
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


def shard_edges_by_cell(graph, assigned_city: np.ndarray, out_dir: Path) -> Path:
    """Partition graph edges by source cell, one file per cell.

    Loaded by `build_per_city_spts` to avoid scanning the global
    240M-edge arrays per iteration. The shard step peaks ~8 GB once
    (sort indirection on the corridor); the per-city loop afterward
    stays under 1 GB.

    Resume: skips re-sharding if the output dir already contains files.
    """
    cell_dir = out_dir / "_edges_by_cell"
    sentinel = cell_dir / "_DONE"
    if sentinel.exists():
        print(f"[shard] resume: shard complete, skipping")
        return cell_dir
    # Partial state from a crashed prior shard — clear and redo. Each
    # per-cell .npz is small but a partial shard is worse than no shard.
    if cell_dir.exists():
        import shutil
        shutil.rmtree(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)

    src  = graph.edge_src
    dst  = graph.edge_dst
    cost = graph.edge_cost
    print(f"[shard] sorting {len(src):,} edges by source cell...")

    # src_cell tells us which file each edge belongs in. Stable sort by
    # cell makes per-cell ranges contiguous; we then write each range.
    src_cell = assigned_city[src]
    order = np.argsort(src_cell, kind="stable")

    src_sorted      = src[order]
    dst_sorted      = dst[order]
    cost_sorted     = cost[order]
    src_cell_sorted = src_cell[order]
    del order, src_cell

    # Boundaries between cells in the sorted array.
    breaks = np.concatenate(
        [[0], np.flatnonzero(np.diff(src_cell_sorted)) + 1, [len(src_sorted)]]
    ).astype(np.int64)

    n_with_edges = 0
    for i in range(len(breaks) - 1):
        s = int(breaks[i]); e = int(breaks[i + 1])
        ci = int(src_cell_sorted[s])
        if ci < 0:
            continue  # unreachable nodes (source unassigned to any city)
        np.savez(
            cell_dir / f"{ci}.npz",
            src=src_sorted[s:e], dst=dst_sorted[s:e], cost=cost_sorted[s:e],
        )
        n_with_edges += 1
    sentinel.touch()
    print(f"[shard] wrote {n_with_edges:,} per-cell edge files to {cell_dir.name}/")
    return cell_dir


def build_per_city_spts(
    out_dir: Path,
    graph,                 # unused; kept for signature stability
    fwd_spt,
    city_graph,
    cities: list[dict],
) -> None:
    """Compute and save one SPT per city. Sequential — one city at a time.

    Edges are read from `out_dir/_edges_by_cell/<ci>.npz` (produced by
    `shard_edges_by_cell` upstream). Each iteration reads ~7-8 small
    files (own + adjacent cells), assembles a local subgraph, and runs
    a single-source Dijkstra. Memory is bounded by subgraph size.
    """
    spt_dir = out_dir / "spt"
    spt_dir.mkdir(parents=True, exist_ok=True)
    cell_dir = out_dir / "_edges_by_cell"
    if not cell_dir.exists():
        raise RuntimeError(
            f"missing {cell_dir} — call shard_edges_by_cell before this step"
        )

    assigned = fwd_spt.city_idx
    nbrs_of  = _neighbors_from_city_graph(city_graph)

    # Save the global cell assignment once for the API to use as the
    # cell-crossing detection signal at query time.
    np.savez(out_dir / "global_assignment.npz", assigned_city=assigned)

    n_cities = len(cities)
    existing = sum(1 for ci in range(n_cities) if (spt_dir / f"{ci}.npz").exists())
    if existing:
        print(f"[per-city-spt] resume: {existing:,}/{n_cities:,} already on disk")
    print(f"[per-city-spt] computing SPTs for {n_cities:,} cities")

    own_arr_buf = np.empty(16, dtype=np.int32)  # reused across iterations
    n_processed = 0
    for ci, city in enumerate(cities):
        out_path = spt_dir / f"{ci}.npz"
        if out_path.exists():
            continue
        city_node_idx = int(city["node_idx"])

        # Subgraph cells: own + adjacent. Load each cell's edges from
        # disk; concat. Outbound edges from these cells include some
        # whose dst is in a non-subgraph cell — filter those out.
        own_set = {ci}
        own_set.update(nbrs_of.get(ci, ()))
        own_arr = np.fromiter(own_set, dtype=np.int32, count=len(own_set))

        e_src_list, e_dst_list, e_cost_list = [], [], []
        for n_ci in own_set:
            f = cell_dir / f"{n_ci}.npz"
            if not f.exists():
                continue
            with np.load(f) as cell_data:
                e_src_list.append(np.asarray(cell_data["src"]))
                e_dst_list.append(np.asarray(cell_data["dst"]))
                e_cost_list.append(np.asarray(cell_data["cost"]))
        if not e_src_list:
            continue
        e_src  = np.concatenate(e_src_list)
        e_dst  = np.concatenate(e_dst_list)
        e_cost = np.concatenate(e_cost_list)
        del e_src_list, e_dst_list, e_cost_list

        # Drop edges whose dst lies outside the subgraph.
        dst_cell = assigned[e_dst]
        keep = np.isin(dst_cell, own_arr)
        if not keep.all():
            e_src  = e_src[keep]
            e_dst  = e_dst[keep]
            e_cost = e_cost[keep]
        del dst_cell, keep

        # Local indexing: union of edge endpoints, sorted ascending.
        node_global = np.unique(np.concatenate([e_src, e_dst])).astype(np.int32)
        n_local = len(node_global)

        local_src = np.searchsorted(node_global, e_src).astype(np.int32)
        local_dst = np.searchsorted(node_global, e_dst).astype(np.int32)

        csr = csr_matrix(
            (e_cost, (local_src, local_dst)),
            shape=(n_local, n_local),
            dtype=np.float32,
        )

        local_source = int(np.searchsorted(node_global, city_node_idx))
        if local_source >= n_local or int(node_global[local_source]) != city_node_idx:
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
            out_path,
            node_global=node_global,
            parent_local=predecessors.astype(np.int32),
            cost=cost_arr.astype(np.float32),
        )
        n_processed += 1
        if n_processed % 25 == 0 or ci + 1 == n_cities:
            print(f"[per-city-spt] {ci + 1:,}/{n_cities:,}  last: {city['name']} "
                  f"(local nodes={n_local:,})")

    print(f"[per-city-spt] wrote {n_cities:,} SPTs to {spt_dir}")
