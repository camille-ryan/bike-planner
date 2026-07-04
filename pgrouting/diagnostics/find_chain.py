"""Find a shortest chain of done cities from START to END.

Edge (A -> B) exists iff A's SPT reaches **any vertex of B's polygon**
(or B's snap_vertex_id, for polygonless anchors). This matches how
`compute_spts._build_city_graph` derives city_graph.json — looking for
a single snap vertex would be too brittle because snap can land on a
disconnected stub (excluded edges, OSM stitching gaps).

Useful when city_graph.json doesn't exist yet. Hopping between done
anchors gives us a chain we can hand to verify_chain.py.
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
    print(f"[find-chain] {len(done_ids):,} done SPTs")

    with psycopg.connect(config.PG_DSN) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, name, snap_vertex_id, geom_boundary IS NOT NULL
            FROM   anchors WHERE snap_vertex_id IS NOT NULL ORDER BY id
        """)
        all_anchors = [(int(r[0]), r[1], int(r[2]), bool(r[3]))
                       for r in cur.fetchall()]

        # Batched fetch of all polygon-interior vertices in one
        # spatial join. Iterating per-anchor would be 595 separate
        # queries — slow when postgres is busy with another workload.
        # Polygonless anchors fall back to the snap vertex.
        print(f"[find-chain] fetching all polygon vertex sets in one join...")
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
        print(f"[find-chain]   {sum(len(v) for v in polygon_vids.values()):,} "
              f"polygon vertices over {len(polygon_vids):,} anchors")

    by_name = {n: a for a, n, _, _ in all_anchors}
    if start_name not in by_name or end_name not in by_name:
        sys.exit(f"missing anchor: have={list(by_name)[:5]}...")
    start_id = by_name[start_name]
    end_id = by_name[end_name]
    name_by_id = {a: n for a, n, _, _ in all_anchors}

    # Build adjacency: A -> [B's polygon reached by A's SPT]
    print(f"[find-chain] building reach graph over {len(done_ids):,} done cities "
          f"(polygon membership)...")
    adj: dict[int, set[int]] = defaultdict(set)
    sorted_done_aids = sorted(done_ids)
    for n, ci_a in enumerate(sorted_done_aids):
        spt_path = spt_dir / f"{ci_a}.npz"
        if not spt_path.exists():
            continue
        spt = np.load(spt_path, mmap_mode="r")
        ng = spt["node_global"]
        a_id = ci_a + 1
        for b_id, b_polys in polygon_vids.items():
            if b_id == a_id or len(b_polys) == 0:
                continue
            idx = np.searchsorted(ng, b_polys)
            in_range = idx < len(ng)
            if not in_range.any():
                continue
            matched = ng[np.minimum(idx, len(ng) - 1)] == b_polys
            matched &= in_range
            if matched.any():
                adj[a_id].add(b_id)
        if (n + 1) % 100 == 0:
            print(f"[find-chain]   processed {n+1:,}/{len(sorted_done_aids):,}")
    print(f"[find-chain] reach graph: {sum(len(v) for v in adj.values()):,} directed edges "
          f"over {len(adj):,} source cities")

    # BFS for shortest hop chain start_id -> end_id.
    if start_id not in adj:
        sys.exit(f"start city {start_name} (id={start_id}) has no outgoing edges — its SPT "
                 f"reaches no other done anchor's snap")
    parent = {start_id: None}
    queue = deque([start_id])
    while queue:
        cur_id = queue.popleft()
        if cur_id == end_id:
            break
        for nxt in adj.get(cur_id, set()):
            if nxt not in parent:
                parent[nxt] = cur_id
                queue.append(nxt)

    if end_id not in parent:
        print(f"[find-chain] NO CHAIN found from {start_name} to {end_name} via done cities")
        # Print closest things in reach.
        print(f"[find-chain] cities reachable from {start_name}: "
              f"{sorted(name_by_id[a] for a in adj.get(start_id, set()))[:20]}")
        sys.exit(2)

    chain = []
    cur_id = end_id
    while cur_id is not None:
        chain.append(name_by_id[cur_id])
        cur_id = parent[cur_id]
    chain.reverse()
    print(f"[find-chain] chain ({len(chain)} hops):")
    for n in chain:
        print(f"  {n}")
    print()
    quoted = " ".join(f'"{n}"' for n in chain[1:-1])
    print(f"Try: python verify_chain.py \"{chain[0]}\" \"{chain[-1]}\" {quoted}")


if __name__ == "__main__":
    main()
