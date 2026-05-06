"""Per-city Voronoi cell polygons + the city-graph adjacency table.

Inputs are the SPT labels (one city assignment per road node) plus the
graph itself. We compute:

  - `polygons`: one concave hull per city, in EPSG:4326. Frontend
    overlays these on click-to-show.
  - `city_graph`: weighted adjacency where edge A↔B exists iff the
    road graph has at least one edge crossing from A's cell to B's,
    weighted by the minimum-cost crossing path. This is what the
    upper-layer Dijkstra runs on at query time.
"""
import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from shapely import concave_hull
from shapely.geometry import MultiPoint, mapping


@dataclass
class CityGraph:
    """Sparse city-to-city adjacency.

    Stored as `(from_city, to_city, weight)` triples; we keep both
    directions explicitly to handle asymmetric forward/reverse SPTs
    later. For v1 we build it from the forward SPT only.
    """
    from_city: np.ndarray  # int32
    to_city:   np.ndarray  # int32
    weight:    np.ndarray  # float32


def build_city_graph(graph, fwd_spt) -> CityGraph:
    """Walk all road edges and emit cross-cell crossings."""
    src = graph.edge_src
    dst = graph.edge_dst
    cost = graph.edge_cost
    cell_src = fwd_spt.city_idx[src]
    cell_dst = fwd_spt.city_idx[dst]
    crossing = (cell_src != cell_dst) & (cell_src >= 0) & (cell_dst >= 0)

    # For each crossing edge, the candidate weight is:
    #   cost_to_src_city + this_edge_cost + cost_from_dst_city.
    # We then keep the minimum per (cell_src, cell_dst).
    src_cost = fwd_spt.cost[src][crossing]
    dst_cost = fwd_spt.cost[dst][crossing]
    edge_w   = cost[crossing]
    total    = src_cost + edge_w + dst_cost
    a = cell_src[crossing]
    b = cell_dst[crossing]

    # Reduce by min over (a, b). Build a dict keyed on packed int64
    # so this is O(M) regardless of city count.
    best: dict[tuple[int, int], float] = {}
    for i in range(len(a)):
        k = (int(a[i]), int(b[i]))
        v = float(total[i])
        prev = best.get(k)
        if prev is None or v < prev:
            best[k] = v

    print(f"[cells] city-graph edges: {len(best):,}")
    fa = np.empty(len(best), dtype=np.int32)
    fb = np.empty(len(best), dtype=np.int32)
    fw = np.empty(len(best), dtype=np.float32)
    for i, ((u, v), w) in enumerate(best.items()):
        fa[i] = u; fb[i] = v; fw[i] = w
    return CityGraph(from_city=fa, to_city=fb, weight=fw)


def build_polygons(graph, fwd_spt, city_names: list[str]) -> dict:
    """Concave hull (alpha-shape) per city. Returns a GeoJSON FeatureCollection.

    Uses shapely 2.0's `concave_hull` with a tunable ratio. Cells with
    too few points (<4) fall back to a small buffered point — these are
    typically rural anchors with sparse local road density.

    Implementation note: at corridor scale (114M nodes) the obvious
    `defaultdict(list)` of (lon, lat) tuples eats ~6 GB of pure Python
    object overhead and OOMs the container. Instead we group node
    indices by city via a single argsort and process each city's slice
    with numpy — only the few-thousand-element coordinate lists for the
    current city are ever Python-side at once.
    """
    city_idx = fwd_spt.city_idx
    n = len(city_idx)

    # Index of every node that has a city assignment, sorted by which
    # city. Memory: int32 arrays of length n_valid, plus argsort temp.
    valid_mask = city_idx >= 0
    valid_count = int(valid_mask.sum())
    print(f"[cells] {valid_count:,}/{n:,} nodes have a city assignment")
    if valid_count == 0:
        return {"type": "FeatureCollection", "features": []}

    valid_node_idx = np.flatnonzero(valid_mask).astype(np.int32)
    sort_perm = np.argsort(city_idx[valid_node_idx], kind="stable")
    nodes_by_city = valid_node_idx[sort_perm]
    cities_sorted = city_idx[nodes_by_city]
    del valid_mask, valid_node_idx, sort_perm

    # Boundaries between cities in the sorted array.
    breaks = np.concatenate(
        [[0], np.flatnonzero(np.diff(cities_sorted)) + 1, [len(nodes_by_city)]]
    ).astype(np.int64)
    n_cells = len(breaks) - 1
    print(f"[cells] cells with >= 1 node: {n_cells:,}")

    features = []
    node_lon = graph.node_lon
    node_lat = graph.node_lat
    for i in range(n_cells):
        s = int(breaks[i]); e = int(breaks[i + 1])
        c = int(cities_sorted[s])
        idx_slice = nodes_by_city[s:e]
        # Materialize coordinates only for this city's nodes. For an
        # average corridor cell of ~35k nodes this is ~280 KB.
        lons = node_lon[idx_slice]
        lats = node_lat[idx_slice]
        n_pts = len(idx_slice)
        # MultiPoint takes an iterable of (x, y) — pass the column-stack.
        pts = np.column_stack([lons, lats])
        mp = MultiPoint(pts)
        if n_pts < 4:
            geom = mp.buffer(0.01)
        else:
            # ratio=0.4 gives a moderately tight hull. Higher = smoother
            # (more like convex), lower = tighter (more concave).
            geom = concave_hull(mp, ratio=0.4)
            if geom.is_empty or geom.geom_type == "Point":
                geom = mp.buffer(0.005)
        name = city_names[c] if c < len(city_names) else f"city_{c}"
        features.append({
            "type": "Feature",
            "properties": {
                "city_idx": c,
                "name": name,
                "node_count": n_pts,
            },
            "geometry": mapping(geom),
        })
    return {"type": "FeatureCollection", "features": features}
