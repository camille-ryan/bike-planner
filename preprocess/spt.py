"""Multi-source shortest-path tree on the road graph.

The Voronoi labeling we discussed in design: each road node ends up
tagged with `(assigned_city, parent_node, cost_to_city)`. We get this
from one pass of `scipy.sparse.csgraph.dijkstra(min_only=True)` where
the source list is the snap-to-graph ids of every anchor city.

`min_only=True` is critical: it avoids materializing a (cities × nodes)
distance matrix (which would be ~120 GB at corridor scale). Instead
scipy maintains a single best-source-so-far label per node and updates
in-place during the pop. Output arrays are all O(N).

Forward SPT (city → everywhere): graph as-built.
Reverse SPT (everywhere → city): same Dijkstra on the transpose. Bike
graphs are *mostly* symmetric (oneways are <5% of edges) but the few
that exist are precisely the cases — pedestrian zones, contraflow lanes —
where ingress and egress paths diverge meaningfully.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


@dataclass
class SPTResult:
    """One direction of the multi-source SPT.

    Indices are dense node ids. `city_idx` is into the `city_node_ids`
    array passed to `compute_spt`, NOT a global anchor id — the caller
    keeps the mapping.
    """
    cost:       np.ndarray  # float32, shape (N,) — min cost to any city
    parent:     np.ndarray  # int32,   shape (N,) — predecessor node, -9999 if no path
    city_idx:   np.ndarray  # int32,   shape (N,) — index into city_node_ids


def build_csr(graph) -> csr_matrix:
    """Wrap the (src, dst, cost) edge arrays as a scipy CSR matrix."""
    n = len(graph.node_lon)
    return csr_matrix(
        (graph.edge_cost, (graph.edge_src, graph.edge_dst)),
        shape=(n, n),
        dtype=np.float32,
    )


def snap_cities_to_nodes(
    city_lons: np.ndarray,
    city_lats: np.ndarray,
    node_lon: np.ndarray,
    node_lat: np.ndarray,
) -> np.ndarray:
    """Return the dense graph-node index nearest each city by haversine
    distance. Anchors (`place=city|town`) are typically standalone OSM
    nodes that are *not* part of the road graph, so we have to snap.
    """
    # Approximate as flat-Earth in degrees for the KDTree; fine since we
    # only need nearest-neighbor and the candidate set is the entire graph.
    pts = np.column_stack([node_lon, node_lat])
    tree = cKDTree(pts)
    qry = np.column_stack([city_lons.astype(np.float32), city_lats.astype(np.float32)])
    _, idx = tree.query(qry, k=1)
    return idx.astype(np.int32)


def compute_spt(graph, city_node_ids: np.ndarray,
                directions: tuple[str, ...] = ("forward", "reverse"),
                free_edges_after_csr: bool = False,
                ) -> tuple[SPTResult, SPTResult | None]:
    """Compute multi-source SPTs over the road graph.

    `city_node_ids` is an int32 array of length C — the dense node id for
    each city, in the order the caller wants preserved as `city_idx`.

    `directions` controls which SPTs to build. At corridor scale the CSR
    + working state for the reverse Dijkstra adds ~5 GB peak; until any
    routing code actually consumes the reverse SPT, building it costs
    memory we don't have. Pass `("forward",)` to skip it (`rev` will be
    None in the return tuple).

    `free_edges_after_csr=True` mutates `graph` by setting its edge_*
    arrays to None right after CSR construction. At corridor scale this
    is required to fit the Dijkstra in 14 GB — the original arrays
    (~3 GB) and the CSR (~3 GB) are otherwise alive simultaneously and
    leave too little headroom for scipy's working state. Caller must
    reload from disk before any later step that needs the edge arrays.
    """
    import gc

    csr_fwd = build_csr(graph)
    n = csr_fwd.shape[0]
    if free_edges_after_csr:
        graph.edge_src = None
        graph.edge_dst = None
        graph.edge_cost = None
        graph.edge_length_m = None
        gc.collect()

    fwd = _one_direction(csr_fwd, city_node_ids, n, label="forward")
    # Free the forward CSR before constructing the reverse one — at
    # corridor scale these are ~2.5 GB each and overlapping them is
    # what tipped the 14 GB cap on the prior run.
    del csr_fwd
    gc.collect()

    if "reverse" in directions:
        csr_rev = build_csr(graph).transpose().tocsr()
        rev = _one_direction(csr_rev, city_node_ids, n, label="reverse")
        del csr_rev
        gc.collect()
    else:
        rev = None
    return fwd, rev


def _one_direction(csr, city_node_ids, n, *, label: str) -> SPTResult:
    print(f"[spt] {label}: dijkstra over {n:,} nodes from {len(city_node_ids):,} sources...")
    dist, predecessors, sources = dijkstra(
        csgraph=csr,
        indices=city_node_ids.astype(np.int32),
        return_predecessors=True,
        min_only=True,
        directed=True,
    )
    # `sources[i]` is the *node id* of the source that won node i, not its
    # position in city_node_ids. Map to the position so callers (cells,
    # routing) can index the city array directly.
    src_to_pos = {int(nid): i for i, nid in enumerate(city_node_ids)}
    city_idx = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        s = int(sources[i])
        if s >= 0:
            city_idx[i] = src_to_pos.get(s, -1)
    reachable = (city_idx >= 0).sum()
    print(f"[spt] {label}: reachable nodes = {reachable:,} / {n:,}")
    return SPTResult(
        cost=dist.astype(np.float32),
        parent=predecessors.astype(np.int32),
        city_idx=city_idx,
    )
