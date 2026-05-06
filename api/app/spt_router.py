"""SPT-based routing for long-distance queries.

Uses the per-city SPTs produced by `preprocess/per_city_spt.py`:

  data/spt/<profile>/
    cities.json
    city_graph.json
    cells.geojson
    graph_nodes.npz            (lon, lat, osm_id) for global graph
    global_assignment.npz      (assigned_city per node)
    spt/<city_idx>.npz         per-city subgraph SPT

Algorithm:
  1. Snap the requested start/end lon/lats to global graph nodes (KDTree).
  2. Look up each end's assigned city via `assigned_city[node_idx]`.
  3. Run Dijkstra on the city graph from start_city to end_city -> sequence
     A, B, C, ..., G.
  4. Walk gradients: while not yet in the next city's cell, walk the
     next city's SPT parent pointers from the current node. When the
     assigned-city of the current node flips, advance to the next leg.
  5. Final leg: within G's cell, find current_node -> end_node via LCA
     in G's SPT (lowest common ancestor in the SPT tree gives the
     shortest path between two nodes that share the same root).

Per-city SPTs are mmap-loaded on demand and cached in-process. A
typical Austria-scale query touches 5-10 SPTs out of 236.
"""
from __future__ import annotations

import heapq
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .settings import SPT_DIR


# ---------------------------------------------------------------------
# Profile-scoped singletons. Loaded once per profile via _load_profile.
# ---------------------------------------------------------------------

class _ProfileData:
    def __init__(self, profile: str):
        self.profile = profile
        base = SPT_DIR / profile
        if not (base / "graph_nodes.npz").exists():
            raise FileNotFoundError(
                f"no SPT data for profile '{profile}' at {base}"
            )

        nodes = np.load(base / "graph_nodes.npz", mmap_mode="r")
        self.node_lon: np.ndarray = nodes["lon"]
        self.node_lat: np.ndarray = nodes["lat"]
        self.kdtree = cKDTree(np.column_stack([self.node_lon, self.node_lat]))

        ga = np.load(base / "global_assignment.npz", mmap_mode="r")
        self.assigned_city: np.ndarray = ga["assigned_city"]

        with open(base / "cities.json") as fh:
            self.cities: list[dict] = json.load(fh)
        with open(base / "city_graph.json") as fh:
            cg = json.load(fh)
        # Build a sparse adjacency for the city Dijkstra. Keep both
        # directions explicitly even though the build pass kept them.
        self.city_adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for fa, tb, w in zip(cg["from_city"], cg["to_city"], cg["weight"]):
            self.city_adj[int(fa)].append((int(tb), float(w)))

        self.spt_dir = base / "spt"

    @lru_cache(maxsize=64)
    def spt(self, city_idx: int) -> dict[str, np.ndarray]:
        """Mmap-load a per-city SPT. Caches up to 64 SPTs per process."""
        path = self.spt_dir / f"{city_idx}.npz"
        if not path.exists():
            raise FileNotFoundError(f"no SPT for city_idx={city_idx}")
        f = np.load(path, mmap_mode="r")
        return {
            "node_global":  f["node_global"],
            "parent_local": f["parent_local"],
            "cost":         f["cost"],
        }


@lru_cache(maxsize=4)
def _load_profile(profile: str) -> _ProfileData:
    return _ProfileData(profile)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _local_idx(spt: dict, global_node: int) -> int:
    """Look up the local index of a global node in an SPT's `node_global`.

    Returns -1 if the global node isn't in this SPT's subgraph.
    """
    arr = spt["node_global"]
    pos = int(np.searchsorted(arr, global_node))
    if pos >= len(arr) or int(arr[pos]) != int(global_node):
        return -1
    return pos


def _city_graph_dijkstra(adj: dict[int, list[tuple[int, float]]],
                         src: int, dst: int) -> list[int] | None:
    """Plain Dijkstra on a sparse city graph (~few hundred nodes).

    Returns the city-id sequence src ... dst, or None if no path.
    """
    if src == dst:
        return [src]
    dist = {src: 0.0}
    parent: dict[int, int] = {}
    heap: list[tuple[float, int]] = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            # reconstruct
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
    return None


def _walk_until_cell(spt: dict, assigned_city: np.ndarray,
                     start_global: int, target_city: int) -> np.ndarray:
    """Walk parent_local pointers in `spt` from `start_global` until the
    walked-into node's globally-assigned cell becomes `target_city`.

    Returns the global-node sequence as an int32 array, starting with
    `start_global` and ending with the first node whose assigned cell
    equals `target_city`. Raises RuntimeError on stall or budget exhaust.

    Hot path optimization: rather than checking the cell after every
    parent hop in Python (which dominated the original ~400 µs/step
    loop), we walk the entire chain locally as a tight numpy index
    chase, then resolve global ids and cell labels in two batched
    numpy operations. Cell-crossing is found by argmax on the boolean
    mask, all in C.
    """
    node_global = spt["node_global"]
    parent_local = spt["parent_local"]

    pos = int(np.searchsorted(node_global, start_global))
    if pos >= len(node_global) or int(node_global[pos]) != int(start_global):
        raise RuntimeError(
            f"node {start_global} not in SPT subgraph (target city {target_city})"
        )

    # Walk parent_local in a tight Python loop. Each step is one mmap
    # read + one comparison + one assignment — about 1 µs in CPython,
    # which is good enough; the dominant cost in the prior version was
    # the per-step Python-side cell check, not the parent walk itself.
    chain = np.empty(_MAX_LEG_STEPS, dtype=np.int32)
    chain[0] = pos
    n = 1
    cur = pos
    while n < _MAX_LEG_STEPS:
        nxt = int(parent_local[cur])
        if nxt < 0 or nxt == cur:
            break
        chain[n] = nxt
        cur = nxt
        n += 1
    chain = chain[:n]

    # Vectorized: local -> global, then global -> assigned_city.
    globals_arr = np.asarray(node_global)[chain]
    cells = np.asarray(assigned_city)[globals_arr]

    # First index where the cell matches the target. globals_arr[0] is
    # `start_global` whose cell is the *previous* city, so the crossing
    # is always at i >= 1.
    mask = (cells == target_city)
    if not mask.any():
        raise RuntimeError(
            f"gradient walk from {start_global} never crossed into "
            f"target cell (city_idx={target_city})"
        )
    cross_idx = int(mask.argmax())
    return globals_arr[: cross_idx + 1]


def _walk_to_root(spt: dict, start_local: int) -> list[int]:
    """Walk parent_local pointers from start_local until we hit the
    SPT source. Returns the list [start_local, ..., source_local]."""
    chain: list[int] = []
    cur = int(start_local)
    parent = spt["parent_local"]
    seen = set()
    while cur >= 0 and cur not in seen:
        chain.append(cur)
        seen.add(cur)
        nxt = int(parent[cur])
        if nxt < 0 or nxt == cur:
            break
        cur = nxt
    return chain


def _lca_path(spt: dict, a_local: int, b_local: int) -> list[int] | None:
    """Lowest-common-ancestor path between two nodes in an SPT (tree).

    Returns local indices [a_local, ..., LCA, ..., b_local].
    """
    a_chain = _walk_to_root(spt, a_local)
    if not a_chain:
        return None
    a_pos = {n: i for i, n in enumerate(a_chain)}
    parent = spt["parent_local"]
    b_chain: list[int] = []
    cur = int(b_local)
    while cur >= 0:
        b_chain.append(cur)
        if cur in a_pos:
            cut = a_pos[cur]
            return a_chain[: cut + 1] + list(reversed(b_chain[:-1]))
        nxt = int(parent[cur])
        if nxt < 0 or nxt == cur:
            return None  # disconnected from a's chain — shouldn't happen
                         # within one SPT; signal failure
        cur = nxt
    return None


def _coords_for(prof: _ProfileData, global_nodes: list[int]) -> list[list[float]]:
    """Materialize [[lon, lat], ...] for a list of global node indices."""
    return [
        [float(prof.node_lon[i]), float(prof.node_lat[i])]
        for i in global_nodes
    ]


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

# Maximum gradient-walk steps per leg. Bounded so a malformed SPT
# can't spin forever; Austria's furthest empirical chain is 2,049 hops.
_MAX_LEG_STEPS = 200_000


def route(start: tuple[float, float], end: tuple[float, float],
          profile: str) -> dict:
    """Plan a route from `start` to `end` using the SPT data.

    Returns a dict with: type=Feature, geometry=LineString of coords,
    properties.cities = list of city names traversed,
    properties.track-length = approximate total length.
    """
    prof = _load_profile(profile)

    # Snap to graph
    s_node = int(prof.kdtree.query([start[0], start[1]])[1])
    e_node = int(prof.kdtree.query([end[0], end[1]])[1])
    s_city = int(prof.assigned_city[s_node])
    e_city = int(prof.assigned_city[e_node])
    if s_city < 0 or e_city < 0:
        raise RuntimeError("start or end snapped to a node with no assigned city")

    # If both endpoints share a cell, do the LCA inside that cell.
    if s_city == e_city:
        spt = prof.spt(e_city)
        sl = _local_idx(spt, s_node)
        el = _local_idx(spt, e_node)
        if sl < 0 or el < 0:
            raise RuntimeError("endpoint not present in city's own SPT")
        local_path = _lca_path(spt, sl, el)
        if not local_path:
            raise RuntimeError("LCA path failed within cell")
        global_path = [int(spt["node_global"][i]) for i in local_path]
        return _build_feature(prof, global_path, [prof.cities[s_city]["name"]])

    # Multi-cell route: plan city sequence, walk gradients between adjacent
    # entries in the sequence.
    city_path = _city_graph_dijkstra(prof.city_adj, s_city, e_city)
    if not city_path:
        raise RuntimeError(
            f"no path on city graph from "
            f"{prof.cities[s_city]['name']} to {prof.cities[e_city]['name']}"
        )

    full_path: list[int] = [s_node]
    current = s_node
    # Walk through legs 0..len-2 (transitions between cities). Each leg
    # is a parent-pointer walk on the *next city's* SPT until the
    # current node's globally-assigned cell flips to that next city.
    for i in range(len(city_path) - 1):
        next_city = city_path[i + 1]
        spt = prof.spt(next_city)
        leg_globals = _walk_until_cell(spt, prof.assigned_city, current, next_city)
        # leg_globals[0] is `current`; skip it to avoid duplicate point.
        full_path.extend(leg_globals[1:].tolist())
        current = int(leg_globals[-1])

    # Final leg: we're now in e_city's cell. Connect current -> e_node
    # via LCA in e_city's SPT.
    if current != e_node:
        spt = prof.spt(e_city)
        cl = _local_idx(spt, current)
        el = _local_idx(spt, e_node)
        if cl < 0 or el < 0:
            raise RuntimeError("final-leg endpoints not in destination SPT")
        local_tail = _lca_path(spt, cl, el)
        if local_tail is None:
            raise RuntimeError("LCA path failed in destination cell")
        # local_tail starts at cl (== current) so skip its first element
        for i in local_tail[1:]:
            full_path.append(int(spt["node_global"][i]))

    city_names = [prof.cities[c]["name"] for c in city_path]
    return _build_feature(prof, full_path, city_names)


def _build_feature(prof: _ProfileData, global_path: list[int],
                   city_names: list[str]) -> dict:
    coords = _coords_for(prof, global_path)
    # Approximate track length via haversine on consecutive points.
    total_m = 0.0
    if len(coords) >= 2:
        lons = np.array([c[0] for c in coords], dtype=np.float64)
        lats = np.array([c[1] for c in coords], dtype=np.float64)
        dlat = np.deg2rad(np.diff(lats))
        dlon = np.deg2rad(np.diff(lons))
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.deg2rad(lats[:-1])) * np.cos(np.deg2rad(lats[1:])) *
             np.sin(dlon / 2) ** 2)
        total_m = float(2 * 6_371_000.0 * np.arcsin(np.sqrt(a)).sum())
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {
            "creator": "spt-router",
            "cities": city_names,
            "track-length": int(total_m),
            "node-count": len(global_path),
        },
    }
