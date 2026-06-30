"""Transitive-closure indirect-edge filter for chain graph edges.

The original filter in `connect_anchors_pairs.py` only caught 3-hop
indirection (A→B drop if path passes near a C that's a mutual sector
neighbor of A and B). It misses long-corridor edges like A→G where
A→B→C→D→E→F→G chain exists — the intermediates are sector neighbors
of A or G but not BOTH, so the mutual rule doesn't fire.

This filter is graph-theoretic: for each chain edge A→B, check whether
there's a multi-hop chain path A→…→B with total cost ≤ RATIO × direct
cost. If so, A→B is redundant — the multi-hop covers it.

Process edges longest-first, greedily (so we can't fully disconnect a
pair — if all their alternative paths get dropped, the remaining
direct edge is kept). Safety net: never drop an edge if doing so
would orphan an anchor (zero remaining incident edges).

Reads existing outputs:
  /data/way_city_graph.json     — chain edges with a, b, cost_m, geom
  /data/way_city_anchors.geojson — anchor refs

Writes:
  /data/way_city_graph.json     — filtered chain edges
  /data/way_city_graph.geojson  — same, as feature collection
  /data/way_city_anchors.geojson — updated in_graph flags
"""
from __future__ import annotations

import heapq
import json
import time
from pathlib import Path


CHAIN_GRAPH_IN_JSON     = Path("/data/way_city_graph.json")
ANCHORS_IN              = Path("/data/way_city_anchors.geojson")
CHAIN_GRAPH_OUT_JSON    = Path("/data/way_city_graph.json")
CHAIN_GRAPH_OUT_GEOJSON = Path("/data/way_city_graph.geojson")

# Ratio: drop A→B if multi-hop chain cost is ≤ RATIO × direct A→B cost.
# 1.3 means the multi-hop has to be no more than 30% longer than the
# direct edge — long-corridor edges (where the chain follows the same
# road) typically have alt cost ≈ 1.0 × direct cost.
RATIO = 1.3


def _shortest_excluding(adj: dict[int, list[tuple[int, float, int]]],
                        u_start: int, u_end: int,
                        exclude_edge_idx: int, max_cost: float
                        ) -> float:
    """Dijkstra from u_start to u_end avoiding the chain edge with
    index `exclude_edge_idx`. Returns shortest cost or inf if none
    ≤ max_cost is reachable."""
    dist: dict[int, float] = {u_start: 0.0}
    heap: list[tuple[float, int]] = [(0.0, u_start)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == u_end:
            return d
        if d > max_cost:
            return float("inf")
        if d > dist.get(u, float("inf")):
            continue
        for v, c, ei in adj.get(u, ()):
            if ei == exclude_edge_idx:
                continue
            nd = d + c
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return float("inf")


def main() -> None:
    t0 = time.time()
    chain_edges = json.loads(CHAIN_GRAPH_IN_JSON.read_text())
    anchors_fc = json.loads(ANCHORS_IN.read_text())
    print(f"[refilter] loaded {len(chain_edges):,} edges, "
          f"{len(anchors_fc['features']):,} anchors", flush=True)

    # Build index maps
    ref_to_i = {f["properties"]["ref"]: i
                for i, f in enumerate(anchors_fc["features"])}

    # Build adjacency list. edge_idx is the position in chain_edges.
    adj: dict[int, list[tuple[int, float, int]]] = {}
    for i, e in enumerate(chain_edges):
        u = ref_to_i[e["a"]]
        v = ref_to_i[e["b"]]
        c = float(e["cost_m"])
        adj.setdefault(u, []).append((v, c, i))
        adj.setdefault(v, []).append((u, c, i))

    # Sort edges longest first. Longer edges are more likely to be
    # redundant (have multi-hop alternatives) and processing them first
    # lets short corridor edges survive as the canonical chains.
    edges_by_cost = sorted(range(len(chain_edges)),
                           key=lambda i: -chain_edges[i]["cost_m"])

    # Per-anchor incident-edge counter so we never drop an edge that
    # would orphan an anchor.
    degree: dict[int, int] = {}
    for nbrs in adj.values():
        for v, _c, _ei in nbrs:
            degree[v] = degree.get(v, 0) + 1
    # adj has each edge listed twice (u→v and v→u), so degree[v] above
    # is correct (count of distinct edges incident to v).

    dropped: set[int] = set()
    drop_log: list[tuple[int, float, float]] = []  # (edge_idx, direct, alt)
    for ei in edges_by_cost:
        e = chain_edges[ei]
        u = ref_to_i[e["a"]]
        v = ref_to_i[e["b"]]
        direct_cost = float(e["cost_m"])
        # Skip if dropping would orphan either endpoint.
        if degree.get(u, 0) <= 1 or degree.get(v, 0) <= 1:
            continue
        # Find shortest u→v via remaining edges, excluding both this
        # edge and previously-dropped edges.
        alt_cost = _shortest_excluding_with_drops(
            adj, u, v, ei, direct_cost * RATIO, dropped,
        )
        if alt_cost <= direct_cost * RATIO:
            dropped.add(ei)
            drop_log.append((ei, direct_cost, alt_cost))
            # Decrement degree of both endpoints
            degree[u] -= 1
            degree[v] -= 1

    print(f"[refilter] dropped {len(dropped):,} edges  "
          f"({len(chain_edges) - len(dropped):,} remain)  "
          f"in {time.time()-t0:.1f}s", flush=True)
    if drop_log:
        # Stats on the dropped edges
        ratios = [alt / direct for _ei, direct, alt in drop_log]
        long_drops = sum(1 for _ei, d, _a in drop_log if d > 100_000)
        print(f"[refilter]   dropped edge ratios: "
              f"min={min(ratios):.2f} median={sorted(ratios)[len(ratios)//2]:.2f} "
              f"max={max(ratios):.2f}", flush=True)
        print(f"[refilter]   of dropped, {long_drops} had direct cost > 100km",
              flush=True)

    kept = [e for i, e in enumerate(chain_edges) if i not in dropped]
    kept.sort(key=lambda e: (e["a"], e["b"]))

    CHAIN_GRAPH_OUT_JSON.write_text(json.dumps(kept, ensure_ascii=False))
    fc_edges = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString",
                             "coordinates": [list(c) for c in e["geom"]]},
                "properties": {
                    "a": e["a"], "a_name": e["a_name"],
                    "b": e["b"], "b_name": e["b_name"],
                    "cost_km": round(e["cost_m"]/1000.0, 2),
                },
            } for e in kept
        ],
    }
    CHAIN_GRAPH_OUT_GEOJSON.write_text(json.dumps(fc_edges, ensure_ascii=False))
    print(f"[refilter] wrote {CHAIN_GRAPH_OUT_JSON} + "
          f"{CHAIN_GRAPH_OUT_GEOJSON.name}", flush=True)

    # Rewrite anchors with in_graph reflecting the new edge set.
    participants = {e["a"] for e in kept} | {e["b"] for e in kept}
    n_in = 0
    for feat in anchors_fc["features"]:
        in_graph = feat["properties"]["ref"] in participants
        feat["properties"]["in_graph"] = in_graph
        if in_graph:
            n_in += 1
    ANCHORS_IN.write_text(json.dumps(anchors_fc, ensure_ascii=False))
    print(f"[refilter] re-wrote way_city_anchors.geojson  "
          f"({n_in:,} in-graph / {len(anchors_fc['features']):,})", flush=True)
    print(f"[refilter] DONE in {time.time()-t0:.1f}s", flush=True)


def _shortest_excluding_with_drops(
    adj: dict[int, list[tuple[int, float, int]]],
    u_start: int, u_end: int,
    exclude_edge_idx: int, max_cost: float,
    dropped: set[int],
) -> float:
    """Same as _shortest_excluding but also skips edges in `dropped`."""
    dist: dict[int, float] = {u_start: 0.0}
    heap: list[tuple[float, int]] = [(0.0, u_start)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == u_end:
            return d
        if d > max_cost:
            return float("inf")
        if d > dist.get(u, float("inf")):
            continue
        for v, c, ei in adj.get(u, ()):
            if ei == exclude_edge_idx or ei in dropped:
                continue
            nd = d + c
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return float("inf")


if __name__ == "__main__":
    main()
