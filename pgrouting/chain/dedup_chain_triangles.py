"""DEPRECATED — replaced by build_chain_graph_proximity.py (stage 6 as
of 2026-07-08). This file is kept for audit / A-B comparison. It does
not run in the current pipeline. The replacement folds triangle
removal into a single multi-source-Dijkstra pass whose proximity-
based intermediate-C test catches passes-through-anchor cases this
script's cost-only test could miss.

Drop redundant chain-graph triangles.

An edge (A, C) is redundant when the chain graph also contains
(A, B) and (B, C) for some B and cost(A,B) + cost(B,C) is within
DEDUP_TOL of the direct cost(A, C). Chain-Dijkstra will always
find the A → B → C multi-hop, so (A, C) adds no routing value —
but it does inflate A's and C's polygons (each gains a chain
neighbor to hull toward), which balloons SPT compute (stage 9)
and trunk builds (stage 11).

Empirically on the July 2026 chain graph, ~30 % of edges were
redundant, climbing to 71-74 % for edges longer than 80 km.

Reads / writes:
  /data/way_city_graph.json          (bidir output; overwritten)
  /data/way_city_graph.geojson       (regenerated for map viz)
  /data/way_city_graph.pre_dedup.json (backup, first-run only)

Env vars:
  DEDUP_TOL     (default 1.1)  two-hop cost tolerance vs direct
"""
from __future__ import annotations

import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path


DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
IN_GRAPH   = DATA_DIR / "way_city_graph.json"
OUT_GRAPH  = DATA_DIR / "way_city_graph.json"
OUT_GEOJSON = DATA_DIR / "way_city_graph.geojson"
BACKUP     = DATA_DIR / "way_city_graph.pre_dedup.json"

TOL = float(os.environ.get("DEDUP_TOL", "1.1"))


def main() -> None:
    t0 = time.time()
    edges = json.loads(IN_GRAPH.read_text())
    print(f"[dedup] read {len(edges):,} chain edges "
          f"(tolerance = {TOL:.2f}× direct)", flush=True)

    # Undirected cost adjacency for quick two-hop lookups.
    adj: dict[str, dict[str, float]] = defaultdict(dict)
    for e in edges:
        c = float(e.get("cost_m", 0) or 0)
        if c <= 0:
            continue
        adj[e["a"]][e["b"]] = c
        adj[e["b"]][e["a"]] = c

    # Mark redundant undirected pairs.
    redundant: set[tuple[str, str]] = set()
    for e in edges:
        a, c_ref = e["a"], e["b"]
        direct = adj[a].get(c_ref, 0.0)
        if direct <= 0:
            continue
        cap = TOL * direct
        # Scan A's other neighbors for a B with (A→B→C) ≤ cap.
        for b, ab in adj[a].items():
            if b == c_ref:
                continue
            bc = adj[b].get(c_ref)
            if bc is None:
                continue
            if ab + bc <= cap:
                redundant.add((a, c_ref))
                redundant.add((c_ref, a))
                break

    kept = [e for e in edges if (e["a"], e["b"]) not in redundant]
    n_dropped = len(edges) - len(kept)
    print(f"[dedup]   dropped {n_dropped:,} redundant edges "
          f"({100 * n_dropped / max(len(edges), 1):.1f}%)  "
          f"→ {len(kept):,} kept", flush=True)

    # Backup the pre-dedup graph on the first run so an audit is possible.
    if not BACKUP.exists():
        shutil.copy(IN_GRAPH, BACKUP)
        print(f"[dedup]   backed up pre-dedup graph → {BACKUP.name}",
              flush=True)

    OUT_GRAPH.write_text(json.dumps(kept, ensure_ascii=False))

    # Regenerate the geojson so the web viz reflects the deduped graph.
    features = [{
        "type": "Feature",
        "geometry": {"type": "LineString",
                     "coordinates": e.get("geom") or []},
        "properties": {
            "a": e["a"], "b": e["b"],
            "a_name": e.get("a_name"), "b_name": e.get("b_name"),
            "cost_m": e.get("cost_m"),
        },
    } for e in kept]
    OUT_GEOJSON.write_text(json.dumps(
        {"type": "FeatureCollection", "features": features},
        ensure_ascii=False))

    print(f"[dedup] wrote {OUT_GRAPH.name} + {OUT_GEOJSON.name} in "
          f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
