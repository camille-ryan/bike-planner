"""Print BFS reach from a starting city through done anchors.

Shows how far the chainless reach graph extends from a given start.
Useful for "we didn't make it to End — how close did we get?"
"""
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import psycopg
import config


SPT_DIR = Path(os.environ.get("DATA_DIR", "/data")) / "spt" / "lht"


def main():
    start_name = sys.argv[1] if len(sys.argv) > 1 else "Graz"
    spt_dir = SPT_DIR / "spt"
    done_ids = set(int(p.stem) for p in spt_dir.glob("*.npz"))
    print(f"[trace] {len(done_ids):,} done SPTs")

    with psycopg.connect(config.PG_DSN) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, name, country, snap_vertex_id FROM anchors
            WHERE snap_vertex_id IS NOT NULL ORDER BY id
        """)
        rows = cur.fetchall()

    name_by_id = {int(r[0]): r[1] for r in rows}
    country_by_id = {int(r[0]): r[2] for r in rows}
    by_name = {r[1]: int(r[0]) for r in rows}
    snap_vids = np.asarray([int(r[3]) for r in rows], dtype=np.int64)
    snap_to_aid = {int(r[3]): int(r[0]) for r in rows}

    if start_name not in by_name:
        sys.exit(f"missing: {start_name}")
    start_id = by_name[start_name]

    # Build reach graph.
    print("[trace] building reach graph...")
    adj: dict[int, set[int]] = defaultdict(set)
    for n, ci in enumerate(sorted(done_ids)):
        spt_path = spt_dir / f"{ci}.npz"
        spt = np.load(spt_path, mmap_mode="r")
        ng = spt["node_global"]
        idx = np.searchsorted(ng, snap_vids)
        in_range = idx < len(ng)
        matched = np.zeros_like(in_range)
        matched[in_range] = ng[idx[in_range]] == snap_vids[in_range]
        a_id = ci + 1
        for vid_idx in np.flatnonzero(matched):
            target_aid = snap_to_aid[int(snap_vids[vid_idx])]
            if target_aid != a_id:
                adj[a_id].add(target_aid)
        if (n + 1) % 200 == 0:
            print(f"[trace]   {n + 1:,}/{len(done_ids):,}")

    # BFS by depth.
    parent = {start_id: None}
    depth = {start_id: 0}
    queue = deque([start_id])
    while queue:
        cur_id = queue.popleft()
        for nxt in adj.get(cur_id, set()):
            if nxt not in parent:
                parent[nxt] = cur_id
                depth[nxt] = depth[cur_id] + 1
                queue.append(nxt)

    # Print reachable cities by hop count.
    by_depth: dict[int, list[int]] = defaultdict(list)
    for aid, d in depth.items():
        by_depth[d].append(aid)

    print(f"[trace] reachable from {start_name}: {len(depth):,} cities")
    for d in sorted(by_depth):
        cities = sorted(by_depth[d], key=lambda a: name_by_id[a])
        countries = sorted(set(country_by_id[a] for a in cities))
        print(f"  hop={d}  {len(cities):,} cities, countries={countries}")
        if d <= 3:
            sample = ", ".join(name_by_id[a] for a in cities[:8])
            if len(cities) > 8:
                sample += f", ... ({len(cities)} total)"
            print(f"          {sample}")

    # Find the deepest reach in each country.
    for country in ("austria", "czech-republic", "germany", "denmark"):
        in_country = [(aid, d) for aid, d in depth.items() if country_by_id[aid] == country]
        if in_country:
            farthest = max(in_country, key=lambda x: x[1])
            print(f"  farthest in {country}: {name_by_id[farthest[0]]} at hop {farthest[1]}")
        else:
            print(f"  ✗ no reach into {country}")


if __name__ == "__main__":
    main()
