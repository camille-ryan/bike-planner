"""Walk an explicit chain of cities with dynamic gradient switching.

Doesn't load the global CSR. At each step, the walker picks the
farthest-in-chain city whose SPT contains the current vertex, and
follows its gradient one step. This means we naturally cross from
one city's SPT into the next without explicitly stopping at
intermediate polygons — same algorithm as the verify_chainless.py
tests on the AT smoke set.

Usage:
  python walk_chain.py START_CITY HOP1 HOP2 ... END_CITY
"""
import os
import sys
from pathlib import Path

import numpy as np
import psycopg
import config


SPT_DIR = Path(os.environ.get("DATA_DIR", "/data")) / "spt" / "lht"


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} START_CITY HOP1 HOP2 ... END_CITY")
    chain_names = sys.argv[1:]
    print(f"[walk] chain: {' -> '.join(chain_names)}")

    conn = psycopg.connect(config.PG_DSN)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, snap_vertex_id FROM anchors WHERE name = ANY(%s)",
        (chain_names,),
    )
    rows = {r[1]: (int(r[0]), int(r[2])) for r in cur.fetchall()}
    chain_aids = []
    for n in chain_names:
        if n not in rows:
            sys.exit(f"missing anchor: {n}")
        chain_aids.append(rows[n][0])
    end_aid = chain_aids[-1]

    # Load SPT for every chain city. Eager load (np.load) into RAM —
    # mmap reads are very slow on /mnt/e (WSL 9P filesystem) when we
    # do many searchsorted lookups across the file. Each chain SPT is
    # ~30 MB; for a 10-city chain that's 300 MB total.
    spts: dict[int, dict] = {}
    for aid in chain_aids:
        ci = aid - 1
        path = SPT_DIR / "spt" / f"{ci}.npz"
        if not path.exists():
            sys.exit(f"missing SPT: city_idx={ci} ({path})")
        with np.load(path) as data:
            spts[aid] = {
                "node_global":  np.array(data["node_global"]),
                "parent_local": np.array(data["parent_local"]),
                "cost":         np.array(data["cost"]),
            }
        print(f"[walk] loaded SPT for city {aid - 1}: "
              f"{len(spts[aid]['node_global']):,} vertices")

    # End polygon set for completion detection.
    cur.execute("""
        SELECT v.id FROM ways_vertices_pgr v
        JOIN anchors a ON a.id = %s
        WHERE a.geom_boundary IS NOT NULL
          AND ST_Contains(a.geom_boundary, v.the_geom)
    """, (end_aid,))
    end_polygon = set(int(r[0]) for r in cur.fetchall())
    if not end_polygon:
        end_polygon = {rows[chain_names[-1]][1]}
    conn.close()
    print(f"[walk] end polygon size: {len(end_polygon):,}")

    # Dynamic-switching walk. At each step, find the farthest chain city
    # whose SPT contains current vertex, follow that city's gradient one
    # step. Stop when current vertex is in end city's polygon.
    current_vid = rows[chain_names[0]][1]
    total_cost = 0.0
    total_edges = 0
    last_city_used = None
    switches = 0

    while current_vid not in end_polygon:
        # Find farthest chain city (excluding start) whose SPT contains current.
        best_aid = None
        best_local_idx = None
        for aid in reversed(chain_aids[1:]):
            ng = spts[aid]["node_global"]
            i = int(np.searchsorted(ng, current_vid))
            if i < len(ng) and int(ng[i]) == current_vid:
                best_aid = aid
                best_local_idx = i
                break
        if best_aid is None:
            print(f"[walk] FAIL after {total_edges} edges, cost {total_cost:.0f}: "
                  f"vid {current_vid} not in any remaining chain city's SPT")
            sys.exit(1)
        if best_aid != last_city_used:
            city_name = chain_names[chain_aids.index(best_aid)]
            print(f"[walk]   switch -> following {city_name}'s gradient "
                  f"(at edge {total_edges}, cost {total_cost:.0f})")
            last_city_used = best_aid
            switches += 1
        # Follow this city's gradient one step.
        npz = spts[best_aid]
        par = int(npz["parent_local"][best_local_idx])
        if par == -9999:
            # Reached this city's polygon. Drop it from chain and re-evaluate.
            chain_aids.remove(best_aid)
            if not chain_aids[1:]:
                break
            continue
        edge_cost = float(npz["cost"][best_local_idx]) - float(npz["cost"][par])
        total_cost += edge_cost
        total_edges += 1
        current_vid = int(npz["node_global"][par])

    print(f"[walk] CHAIN total = {total_cost:.0f}  ({total_edges} edges, "
          f"{switches} city switches)")
    print(f"[walk] ended at vid {current_vid} inside {chain_names[-1]}'s polygon")


if __name__ == "__main__":
    main()
