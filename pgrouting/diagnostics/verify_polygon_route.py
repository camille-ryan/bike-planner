"""Verify Graz→Cph routing over the polygon-derived city_graph.json.

Standalone script — does Dijkstra over data/spt/<profile>/city_graph.json
without needing the API or the paired-trunks.db. Validates that:
  1. The chain of polygon SPTs + adapter produces a connected
     city-graph that spans AT → DE → DK.
  2. The min-cost path between Graz and Copenhagen exists and has
     reasonable cost.

Usage:
  SPT_PROFILE=views python3 verify_polygon_route.py
  SPT_PROFILE=views python3 verify_polygon_route.py "Bregenz" "Tønder"
"""
from __future__ import annotations

import heapq
import json
import os
import re
import sys
import time
from pathlib import Path


PROFILE = os.environ.get("SPT_PROFILE", "views")
if not re.fullmatch(r"[a-z_][a-z0-9_]*", PROFILE):
    raise SystemExit(f"unsafe profile name: {PROFILE!r}")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
PAIRED   = DATA_DIR / "spt" / PROFILE


def _load_city_graph() -> tuple[dict[int, dict], dict[int, list[tuple[int, float]]]]:
    cities = json.load(open(PAIRED / "cities.json"))
    by_idx = {c["city_idx"]: c for c in cities}
    cg = json.load(open(PAIRED / "city_graph.json"))
    adj: dict[int, list[tuple[int, float]]] = {}
    for fa, tb, w in zip(cg["from_city"], cg["to_city"], cg["weight"]):
        adj.setdefault(fa, []).append((tb, w))
    print(f"[verify] {len(cities):,} cities, {len(cg['weight']):,} directed edges",
          flush=True)
    return by_idx, adj


def _find_city_by_name(by_idx: dict[int, dict], needles: list[str]) -> int | None:
    """Case-insensitive prefix match on name. Returns city_idx."""
    needles_l = [n.lower() for n in needles]
    candidates = []
    for ci, c in by_idx.items():
        name_l = (c.get("name") or "").lower()
        for n in needles_l:
            if n in name_l:
                candidates.append((len(name_l), ci, c["name"]))
                break
    candidates.sort()
    if not candidates:
        return None
    print(f"[verify] matched {len(candidates)} for {needles}: "
          f"{[c[2] for c in candidates[:5]]}", flush=True)
    return candidates[0][1]


def _dijkstra(adj: dict[int, list[tuple[int, float]]], src: int, dst: int):
    dist: dict[int, float] = {src: 0.0}
    prev: dict[int, int] = {}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dst not in dist:
        return None, None
    # Reconstruct path
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return dist[dst], path


def main() -> None:
    if len(sys.argv) >= 3:
        src_needles = [sys.argv[1]]
        dst_needles = [sys.argv[2]]
    else:
        src_needles = ["graz"]
        dst_needles = ["copenhagen", "kobenhavn", "københavn"]

    by_idx, adj = _load_city_graph()
    src_ci = _find_city_by_name(by_idx, src_needles)
    dst_ci = _find_city_by_name(by_idx, dst_needles)
    if src_ci is None or dst_ci is None:
        print(f"[verify] FAIL: no match for src={src_needles} or dst={dst_needles}",
              flush=True)
        sys.exit(1)
    src = by_idx[src_ci]
    dst = by_idx[dst_ci]
    print(f"[verify] src=ci{src_ci} {src['name']} ({src['country']}, "
          f"{src['lon']:.3f}/{src['lat']:.3f})", flush=True)
    print(f"[verify] dst=ci{dst_ci} {dst['name']} ({dst['country']}, "
          f"{dst['lon']:.3f}/{dst['lat']:.3f})", flush=True)

    t0 = time.time()
    cost, path = _dijkstra(adj, src_ci, dst_ci)
    print(f"[verify] dijkstra in {(time.time()-t0)*1000:.0f}ms", flush=True)
    if cost is None:
        print(f"[verify] FAIL: no path from {src['name']} to {dst['name']}",
              flush=True)
        sys.exit(2)

    print(f"[verify] SUCCESS — cost={cost:,.0f}  hops={len(path)-1}",
          flush=True)
    print(f"[verify] path:", flush=True)
    for i, ci in enumerate(path):
        c = by_idx[ci]
        print(f"  {i:>3}. ci{ci:>4} {c['name']} ({c['country']})", flush=True)


if __name__ == "__main__":
    main()
