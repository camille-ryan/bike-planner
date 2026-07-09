"""Task #49: classify ferry piers as sea vs river.

Reads `data/ferry_piers.geojsonseq` (411 piers as of 2026-07) and the
`ways WHERE is_ferry` subgraph from postgres. Groups piers by
connected component of the ferry-only subgraph, then classifies each
component on TWO signals:

  * The MAX single ferry-way length in the component. A real sea
    crossing shows up as one long way (Rødby-Puttgarden is 19 km on
    one way). A river cluster is many short (< 1 km) crossings even
    if they sum to > 20 km — so total-length alone misclassifies.
  * The component total length as a secondary heuristic (default not
    used; only kicks in if MAX_SEA_WAY_KM is set to 0 to disable).

Classification:
  * max single-way length ≥ MAX_SEA_WAY_KM (default 5 km) → **sea**
    (piers become chain anchors, get their own polygon SPT).
  * otherwise → **river**
    (piers dropped from the anchor set; the ferry way stays in
    `ways` as an is_ferry route the bike Dijkstra can still take,
    just not a chain hop).

Outputs:
  * `data/sea_piers.geojsonseq`   — subset of ferry_piers to keep
  * `data/river_piers.geojsonseq` — dropped subset (kept for audit /
    inspection; not consumed downstream).
  * `data/ferry_classification.json` — per-component summary:
      { "components": [{
          "id": 0,
          "kind": "sea"|"river",
          "total_len_m": 47000,
          "piers": ["ferry:12345", ...],
      }], ... }

Runs quickly (~5 s) — the ferry subgraph is small (<10k edges).

Env vars:
  * PGDATABASE
  * MAX_SEA_WAY_KM (default 5.0) — a component with any single ferry
    way at least this long is classified sea. Set to 0 to fall back
    on the old component-total heuristic.
  * FERRY_SEA_THRESHOLD_KM (default 20) — legacy component-total
    threshold, still used when MAX_SEA_WAY_KM is 0.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import psycopg

import config


DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
IN_PIERS      = DATA_DIR / "ferry_piers.geojsonseq"
OUT_SEA_PIERS   = DATA_DIR / "sea_piers.geojsonseq"
OUT_RIVER_PIERS = DATA_DIR / "river_piers.geojsonseq"
OUT_SUMMARY     = DATA_DIR / "ferry_classification.json"

THRESHOLD_KM = float(os.environ.get("FERRY_SEA_THRESHOLD_KM", "20"))
THRESHOLD_M  = THRESHOLD_KM * 1000.0
MAX_SEA_WAY_KM = float(os.environ.get("MAX_SEA_WAY_KM", "5.0"))
MAX_SEA_WAY_M  = MAX_SEA_WAY_KM * 1000.0


def _load_piers() -> list[dict]:
    if not IN_PIERS.exists():
        raise SystemExit(f"[classify_piers] missing {IN_PIERS}")
    piers: list[dict] = []
    with open(IN_PIERS) as f:
        for line in f:
            line = line.strip().lstrip("\x1e").strip()
            if not line:
                continue
            piers.append(json.loads(line))
    return piers


def _load_ferry_edges(conn: psycopg.Connection) -> list[tuple[int, int, float]]:
    """Return every (source, target, length_m) for ferry ways.

    Classification is topological — just the ferry subgraph structure —
    so we don't apply bike-routability filters here. All is_ferry ways
    with a real length participate."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT source, target, length_m
            FROM ways
            WHERE is_ferry
              AND length_m > 0.0
        """)
        return list(cur.fetchall())


def _components(
    edges: list[tuple[int, int, float]],
) -> tuple[dict[int, int], list[float]]:
    """Union-find grouping of vertices connected by ferry edges.

    Returns (vertex → component_id, list[component_total_len_m])."""
    parent: dict[int, int] = {}
    rank:   dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for s, t, _ in edges:
        if s not in parent:
            parent[s] = s; rank[s] = 0
        if t not in parent:
            parent[t] = t; rank[t] = 0
        union(s, t)

    # Consolidate roots to compact component IDs.
    root_to_id: dict[int, int] = {}
    vid_to_cid: dict[int, int] = {}
    for v in parent:
        r = find(v)
        if r not in root_to_id:
            root_to_id[r] = len(root_to_id)
        vid_to_cid[v] = root_to_id[r]

    n_components = len(root_to_id)
    total_len_by_cid: list[float] = [0.0] * n_components
    max_len_by_cid:   list[float] = [0.0] * n_components
    for s, _t, length_m in edges:
        cid = vid_to_cid[s]
        total_len_by_cid[cid] += float(length_m)
        if float(length_m) > max_len_by_cid[cid]:
            max_len_by_cid[cid] = float(length_m)
    return vid_to_cid, total_len_by_cid, max_len_by_cid


def _write_geojsonseq(path: Path, features: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for feat in features:
            f.write(json.dumps(feat) + "\n")


def main() -> None:
    t0 = time.time()
    print(f"[classify_piers] threshold: {THRESHOLD_KM} km", flush=True)

    piers = _load_piers()
    print(f"[classify_piers] loaded {len(piers)} piers from {IN_PIERS.name}",
          flush=True)

    with psycopg.connect(config.PG_DSN) as conn:
        edges = _load_ferry_edges(conn)
    print(f"[classify_piers] loaded {len(edges):,} ferry edges", flush=True)

    vid_to_cid, total_len_by_cid, max_len_by_cid = _components(edges)
    n_components = len(total_len_by_cid)
    print(f"[classify_piers] {n_components} connected ferry components  "
          f"(max_sea_way={MAX_SEA_WAY_KM} km)",
          flush=True)

    def _is_sea(cid: int) -> bool:
        # Primary rule: a single ferry way in the component ≥ 5 km.
        # Falls back to legacy component-total heuristic when
        # MAX_SEA_WAY_KM is 0.
        if MAX_SEA_WAY_M > 0:
            return max_len_by_cid[cid] >= MAX_SEA_WAY_M
        return total_len_by_cid[cid] > THRESHOLD_M

    # Bucket piers by component.
    sea_piers: list[dict] = []
    river_piers: list[dict] = []
    unclassified: list[dict] = []
    component_piers: list[list[str]] = [[] for _ in range(n_components)]
    for pier in piers:
        vid = int(pier["properties"].get("vid") or 0)
        cid = vid_to_cid.get(vid)
        if cid is None:
            # Pier not in ferry subgraph — its ways aren't bike-routable
            # (bike_excluded or missing cost). Not usable for routing.
            unclassified.append(pier)
            continue
        component_piers[cid].append(pier["id"])
        if _is_sea(cid):
            sea_piers.append(pier)
        else:
            river_piers.append(pier)

    _write_geojsonseq(OUT_SEA_PIERS, sea_piers)
    _write_geojsonseq(OUT_RIVER_PIERS, river_piers)

    # Compact summary of every component (sea + river).
    summary = {
        "threshold_km": THRESHOLD_KM,
        "n_piers_source": len(piers),
        "n_sea_piers":   len(sea_piers),
        "n_river_piers": len(river_piers),
        "n_unclassified_piers": len(unclassified),
        "n_ferry_edges": len(edges),
        "components": [
            {
                "id":          cid,
                "kind":        "sea" if _is_sea(cid) else "river",
                "total_len_m": round(total_len_by_cid[cid], 1),
                "max_len_m":   round(max_len_by_cid[cid], 1),
                "n_piers":     len(component_piers[cid]),
                "piers":       component_piers[cid],
            }
            for cid in range(n_components)
        ],
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

    # Human-readable overview.
    sea_components   = [c for c in summary["components"] if c["kind"] == "sea"]
    river_components = [c for c in summary["components"] if c["kind"] == "river"]
    total_sea_km   = sum(c["total_len_m"] for c in sea_components)   / 1000.0
    total_river_km = sum(c["total_len_m"] for c in river_components) / 1000.0
    print(f"[classify_piers] SEA:   {len(sea_piers):>3} piers in "
          f"{len(sea_components)} components, "
          f"{total_sea_km:.1f} km total", flush=True)
    print(f"[classify_piers] RIVER: {len(river_piers):>3} piers in "
          f"{len(river_components)} components, "
          f"{total_river_km:.1f} km total", flush=True)
    if unclassified:
        print(f"[classify_piers] unclassified (not in ferry subgraph): "
              f"{len(unclassified)}", flush=True)
    # Show the 5 longest sea + 5 longest river components as a sanity check.
    for label, comps in (("SEA",   sea_components),
                         ("RIVER", river_components)):
        biggest = sorted(comps, key=lambda c: -c["total_len_m"])[:5]
        for c in biggest:
            print(f"[classify_piers]   {label} c{c['id']}: "
                  f"{c['total_len_m']/1000:.1f} km, "
                  f"{c['n_piers']} piers", flush=True)

    print(f"[classify_piers] DONE in {time.time()-t0:.1f}s "
          f"— sea → {OUT_SEA_PIERS.name}, river → {OUT_RIVER_PIERS.name}",
          flush=True)


if __name__ == "__main__":
    main()
