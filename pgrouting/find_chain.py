"""Find a shortest chain of done cities from START to END.

Edge (A -> B) exists iff B's snap_vertex_id is in A's SPT (i.e. cost
from A's polygon to B's snap is < SPT_MAX_COST). BFS on this graph.

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
            SELECT id, name, snap_vertex_id FROM anchors
            WHERE snap_vertex_id IS NOT NULL ORDER BY id
        """)
        all_anchors = [(int(r[0]), r[1], int(r[2])) for r in cur.fetchall()]

    by_name = {n: (a, v) for a, n, v in all_anchors}
    if start_name not in by_name or end_name not in by_name:
        sys.exit(f"missing anchor: have={list(by_name)[:5]}...")
    start_id = by_name[start_name][0]
    end_id = by_name[end_name][0]

    # Build adjacency: A -> [B's that are reachable from A's SPT]
    # Strategy: load A's SPT once, batch-check all anchor snap_vids.
    print(f"[find-chain] building reach graph over {len(done_ids):,} done cities...")
    snap_vids = np.asarray([v for (_, _, v) in all_anchors], dtype=np.int64)
    snap_to_aid = {v: a for a, _, v in all_anchors}
    name_by_id = {a: n for a, n, _ in all_anchors}

    adj: dict[int, set[int]] = defaultdict(set)
    sorted_done_aids = sorted(done_ids)
    for n, ci_a in enumerate(sorted_done_aids):
        spt_path = spt_dir / f"{ci_a}.npz"
        if not spt_path.exists():
            continue
        spt = np.load(spt_path, mmap_mode="r")
        ng = spt["node_global"]
        idx = np.searchsorted(ng, snap_vids)
        in_range = idx < len(ng)
        matched = np.zeros_like(in_range)
        matched[in_range] = ng[idx[in_range]] == snap_vids[in_range]
        a_id = ci_a + 1
        for vid_idx in np.flatnonzero(matched):
            target_aid = snap_to_aid[int(snap_vids[vid_idx])]
            if target_aid != a_id:
                adj[a_id].add(target_aid)
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
