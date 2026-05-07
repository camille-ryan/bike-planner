"""Incremental BFS chain finder.

Only loads each city's SPT once we visit it via BFS, instead of
preprocessing the full all-pairs reach graph. Massively faster when
we just want a Graz->København chain — we typically visit ~10-20
cities, not all 800.

Algorithm:
  1. Pre-fetch all polygon vertex sets in one query.
  2. BFS from start. For each visited city A, load A's SPT, find
     which other cities' polygons are inside A's SPT, enqueue those.
  3. Stop when end is reached (or queue empty).

For most chains this needs to load ~depth × branch ~= 30-100 SPTs,
not 800.
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
    end_name = sys.argv[2] if len(sys.argv) > 2 else "København"

    spt_dir = SPT_DIR / "spt"
    done_ids = set(int(p.stem) for p in spt_dir.glob("*.npz"))
    print(f"[find-chain-lazy] {len(done_ids):,} done SPTs")

    conn = psycopg.connect(config.PG_DSN)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, name, snap_vertex_id, geom_boundary IS NOT NULL
        FROM   anchors WHERE snap_vertex_id IS NOT NULL ORDER BY id
    """)
    all_anchors = [(int(r[0]), r[1], int(r[2]), bool(r[3]))
                   for r in cur.fetchall()]
    by_name = {n: a for a, n, _, _ in all_anchors}
    if start_name not in by_name or end_name not in by_name:
        sys.exit(f"missing anchor: have={list(by_name)[:5]}...")
    start_id = by_name[start_name]
    end_id = by_name[end_name]
    name_by_id = {a: n for a, n, _, _ in all_anchors}

    # Batched polygon-vertex fetch.
    print("[find-chain-lazy] fetching all polygon vertex sets in one join...")
    cur.execute("""
        SELECT a.id, v.id
        FROM   anchors a
        JOIN   ways_vertices_pgr v
          ON   ST_Contains(a.geom_boundary, v.the_geom)
        WHERE  a.geom_boundary IS NOT NULL
          AND  a.snap_vertex_id IS NOT NULL
    """)
    polygon_vids_raw: dict[int, list[int]] = defaultdict(list)
    for aid, vid in cur.fetchall():
        polygon_vids_raw[int(aid)].append(int(vid))
    polygon_vids: dict[int, np.ndarray] = {}
    for aid, _name, snap_vid, has_poly in all_anchors:
        if has_poly and aid in polygon_vids_raw:
            polygon_vids[aid] = np.asarray(
                sorted(polygon_vids_raw[aid]), dtype=np.int64,
            )
        else:
            polygon_vids[aid] = np.asarray([snap_vid], dtype=np.int64)
    conn.close()
    print(f"[find-chain-lazy]   {sum(len(v) for v in polygon_vids.values()):,} "
          f"polygon vertices over {len(polygon_vids):,} anchors")

    # BFS with lazy SPT loading.
    parent = {start_id: None}
    queue = deque([start_id])
    visited_count = 0

    if (start_id - 1) not in done_ids:
        sys.exit(f"start city {start_name}'s SPT not yet computed")

    while queue:
        cur_id = queue.popleft()
        if cur_id == end_id:
            break
        if (cur_id - 1) not in done_ids:
            continue   # can't outbound from a city whose SPT isn't computed
        spt = np.load(spt_dir / f"{cur_id - 1}.npz", mmap_mode="r")
        ng = spt["node_global"]
        visited_count += 1
        if visited_count % 25 == 0:
            print(f"[find-chain-lazy]   visited {visited_count} cities, "
                  f"queue={len(queue)}, found={len(parent)}")
        # Find which other anchors' polygons are in cur's SPT.
        for nxt_id, polys in polygon_vids.items():
            if nxt_id == cur_id or nxt_id in parent or len(polys) == 0:
                continue
            idx = np.searchsorted(ng, polys)
            in_range = idx < len(ng)
            if not in_range.any():
                continue
            matched = ng[np.minimum(idx, len(ng) - 1)] == polys
            matched &= in_range
            if matched.any():
                parent[nxt_id] = cur_id
                queue.append(nxt_id)
                if nxt_id == end_id:
                    break

    if end_id not in parent:
        print(f"[find-chain-lazy] NO CHAIN found from {start_name} to {end_name}")
        # Show the deepest reach
        visited = sorted(parent.keys(), key=lambda a: name_by_id.get(a, ""))
        print(f"[find-chain-lazy] reached {len(visited)} cities so far:")
        sample = [name_by_id[a] for a in visited[:30]]
        print("  " + ", ".join(sample) + (", ..." if len(visited) > 30 else ""))
        sys.exit(2)

    chain = []
    c = end_id
    while c is not None:
        chain.append(name_by_id[c])
        c = parent[c]
    chain.reverse()
    print(f"[find-chain-lazy] chain ({len(chain)} hops):")
    for n in chain:
        print(f"  {n}")
    quoted = " ".join(f'"{n}"' for n in chain[1:-1])
    print(f"\nTry: python verify_chain.py \"{chain[0]}\" \"{chain[-1]}\" {quoted}")


if __name__ == "__main__":
    main()
