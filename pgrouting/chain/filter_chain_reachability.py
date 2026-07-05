"""Stage 7.5: drop unreachable chain edges + reweight survivors.

Reads:
  - /data/way_city_graph.json         (candidate chain edges from stage 4/5)
  - /data/spt/views/cities.json       (per-anchor snap_vids)
  - /data/spt/views_polygon/*.npz     (per-anchor polygon SPT)

For each edge (A, B):
  1. Look up A's SPT NPZ.
  2. Check if ANY of B's snap_vids appears in A's node_global.
  3. If reachable — keep the edge, replace cost_m with the actual SPT
     distance A→B (the minimum cost across B's snap_vids that are in
     A's SPT).
  4. If not reachable — drop it.

This turns the road-oblivious crow-flies chain graph into a
road-verified one, without needing another postgres pass. Uses only
per-anchor NPZ files that stage 7 already produced.

Rerun stages 6 and 7 with the cleaned graph and the polygons will be
smaller, SPTs will not reach across water/mountains, and adapt (stage
8) will not manufacture spurious paired trunks.

Runs quickly — one NPZ open per unique anchor, then in-memory
searchsorted lookups per edge. 2,000 anchors × 10k edges → seconds.

Env vars:
  SPT_PROFILE       (default views)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np


PROFILE = os.environ.get("SPT_PROFILE", "views")
DATA_DIR    = Path(os.environ.get("DATA_DIR", "/data"))
IN_GRAPH    = DATA_DIR / "way_city_graph.json"
IN_CITIES   = DATA_DIR / "spt" / PROFILE / "cities.json"
SPT_DIR     = DATA_DIR / "spt" / f"{PROFILE}_polygon"

OUT_GRAPH   = DATA_DIR / "way_city_graph.json"
OUT_DROPPED = DATA_DIR / "way_city_graph_dropped.json"


def main() -> None:
    t0 = time.time()
    print(f"[filter-chain] profile: {PROFILE}", flush=True)
    print(f"[filter-chain] input:  {IN_GRAPH}", flush=True)
    print(f"[filter-chain] SPT dir: {SPT_DIR}", flush=True)

    edges: list[dict] = json.loads(IN_GRAPH.read_text())
    cities: list[dict] = json.loads(IN_CITIES.read_text())
    print(f"[filter-chain] {len(edges):,} candidate edges, "
          f"{len(cities):,} anchors", flush=True)

    # Index cities by ref → (city_idx, snap_vids array).
    ref_to_city: dict[str, dict] = {c["ref"]: c for c in cities}
    missing_refs = [
        e["a"] for e in edges if e["a"] not in ref_to_city
    ] + [e["b"] for e in edges if e["b"] not in ref_to_city]
    if missing_refs:
        print(f"[filter-chain] WARN: {len(set(missing_refs)):,} refs "
              f"in graph not in cities.json — dropping those edges",
              flush=True)

    # Cache SPT NPZ loads. Each edge needs A's SPT; we can hit the same
    # A many times. Only keep node_global (sorted) + cost arrays.
    spt_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _load_spt(city_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        hit = spt_cache.get(city_idx)
        if hit is not None:
            return hit
        path = SPT_DIR / f"{city_idx}.npz"
        if not path.exists():
            spt_cache[city_idx] = (np.empty(0, np.int64), np.empty(0, np.float32))
            return spt_cache[city_idx]
        with np.load(path, allow_pickle=False) as d:
            ng = d["node_global"]
            cost = d["cost"]
        # Sort by node_global for searchsorted lookups.
        order = np.argsort(ng)
        pair = (ng[order].astype(np.int64, copy=False),
                cost[order].astype(np.float32, copy=False))
        spt_cache[city_idx] = pair
        # Bound the cache — 2079 anchors × ~50MB = 100 GB worst case.
        # Evict LRU-ish by dropping the smallest entries when > 200.
        if len(spt_cache) > 200:
            # Drop the entries with fewest reached vertices (small SPTs).
            small = sorted(spt_cache.items(),
                           key=lambda kv: len(kv[1][0]))[:50]
            for k, _ in small:
                if k != city_idx:
                    del spt_cache[k]
        return pair

    kept: list[dict] = []
    dropped: list[dict] = []
    t_ref = time.time()
    for i, e in enumerate(edges):
        if e["a"] not in ref_to_city or e["b"] not in ref_to_city:
            dropped.append({**e, "_reason": "ref missing from cities.json"})
            continue
        a_city = ref_to_city[e["a"]]
        b_city = ref_to_city[e["b"]]
        a_spt = _load_spt(a_city["city_idx"])
        if a_spt is None or len(a_spt[0]) == 0:
            dropped.append({**e, "_reason": f"no SPT for {e['a']}"})
            continue
        a_ng, a_cost = a_spt

        # B's snap vertices (multi-component).
        b_snaps = np.asarray(b_city.get("snap_vids") or
                             [b_city["snap_vertex_id"]],
                             dtype=np.int64)
        # Find intersection of b_snaps with a_ng (sorted).
        idxs = np.searchsorted(a_ng, b_snaps)
        idxs = np.clip(idxs, 0, len(a_ng) - 1)
        hits = a_ng[idxs] == b_snaps
        if not hits.any():
            dropped.append({**e, "_reason": "unreachable"})
            continue

        # Cost is the minimum SPT cost across B's reachable snap vids.
        cost_m = float(a_cost[idxs[hits]].min())
        kept_edge = {**e, "cost_m": cost_m}
        kept.append(kept_edge)

        if (i + 1) % 2000 == 0:
            dt = time.time() - t_ref
            print(f"[filter-chain]   {i+1:,}/{len(edges):,} in {dt:.0f}s "
                  f"— kept {len(kept):,}, dropped {len(dropped):,}, "
                  f"cache={len(spt_cache)}", flush=True)

    print(f"[filter-chain] DONE: kept {len(kept):,} / "
          f"{len(edges):,} edges ({len(kept)*100//max(len(edges),1)}%), "
          f"dropped {len(dropped):,}", flush=True)

    # Overwrite the graph JSON with survivors + save dropped for audit.
    OUT_GRAPH.write_text(json.dumps(kept, ensure_ascii=False))
    OUT_DROPPED.write_text(json.dumps(dropped, ensure_ascii=False))
    print(f"[filter-chain] wrote {OUT_GRAPH} "
          f"({OUT_GRAPH.stat().st_size/(1<<20):.1f} MB)", flush=True)
    print(f"[filter-chain] wrote {OUT_DROPPED} for audit", flush=True)
    print(f"[filter-chain] elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
