"""Lower ferry cost factor in `ways` and rebuild only the SPTs that
touch a ferry endpoint.

Step-by-step:
  1. Find every vertex that is a source or target of a `is_ferry=true`
     edge — these are ferry-endpoint vertices.
  2. For each existing SPT npz, intersect `node_global` with the ferry
     endpoint set. If a hit, the SPT's reach touches a ferry — re-run.
  3. SQL `UPDATE ways` to halve ferry cost (8.0 → 4.0 effective factor).
  4. Delete the to-rerun npz files + city_graph.json.
  5. Caller re-runs `compute_spts spts`; resume mode regenerates only
     the deleted anchors. city_graph.json rebuilds at the end.

Usage:
    python3 apply_ferry_fix.py [--dry-run]
"""
import os
import sys
from pathlib import Path

import numpy as np
import psycopg

import config


SPT_DIR = Path(os.environ.get("DATA_DIR", "/data")) / "spt" / "lht"


def main():
    dry_run = "--dry-run" in sys.argv

    print("[ferry-fix] connecting to postgres")
    with psycopg.connect(config.PG_DSN) as conn:
        # Step 1 — collect ferry endpoint vertex ids.
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT vid FROM (
                    SELECT source AS vid FROM ways WHERE is_ferry AND cost >= 0
                    UNION
                    SELECT target AS vid FROM ways WHERE is_ferry AND cost >= 0
                    UNION
                    SELECT source AS vid FROM ways WHERE is_ferry AND reverse_cost >= 0
                    UNION
                    SELECT target AS vid FROM ways WHERE is_ferry AND reverse_cost >= 0
                ) f
                ORDER BY vid
            """)
            ferry_vids = np.asarray(
                [int(r[0]) for r in cur.fetchall()], dtype=np.int64,
            )
        print(f"[ferry-fix] {len(ferry_vids):,} ferry endpoint vertices")

        # Step 2 — scan every SPT npz for intersection with ferry vids.
        spt_dir = SPT_DIR / "spt"
        npz_files = sorted(spt_dir.glob("*.npz"))
        print(f"[ferry-fix] scanning {len(npz_files):,} npzs for ferry-touching SPTs")
        to_rerun: list[int] = []
        for path in npz_files:
            try:
                with np.load(path, mmap_mode="r") as data:
                    ng = np.asarray(data["node_global"], dtype=np.int64)
            except Exception as e:
                print(f"  skip {path.name}: {e}")
                continue
            # Sorted intersect via searchsorted.
            idx = np.searchsorted(ng, ferry_vids)
            in_range = idx < len(ng)
            matched = np.zeros_like(in_range, dtype=bool)
            matched[in_range] = ng[idx[in_range]] == ferry_vids[in_range]
            if matched.any():
                to_rerun.append(int(path.stem))
        print(f"[ferry-fix] {len(to_rerun):,} SPTs touch a ferry endpoint, "
              f"{len(npz_files) - len(to_rerun):,} unaffected")

        if dry_run:
            print("[ferry-fix] dry run — would update ways + delete affected npzs + city_graph.json")
            return

        # Step 3 — halve ferry edge costs in `ways`.
        # cost.py was factor 8.0; in-place divide gets us to factor 4.0
        # without re-ingesting.
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE ways
                SET cost = cost / 2.0,
                    reverse_cost = reverse_cost / 2.0
                WHERE is_ferry = true
                  AND (cost >= 0 OR reverse_cost >= 0)
            """)
            updated = cur.rowcount
        conn.commit()
        print(f"[ferry-fix] updated {updated:,} ferry rows (cost factor 8.0 → 4.0)")

    # Step 4 — delete affected npz files + city_graph.json.
    for ci in to_rerun:
        path = SPT_DIR / "spt" / f"{ci}.npz"
        if path.exists():
            path.unlink()
    cg_path = SPT_DIR / "city_graph.json"
    if cg_path.exists():
        cg_path.unlink()
        print(f"[ferry-fix] deleted city_graph.json")
    print(f"[ferry-fix] deleted {len(to_rerun):,} ferry-touching npzs")
    print(f"[ferry-fix] now run `pgrouting spts --profile lht` to regenerate")


if __name__ == "__main__":
    main()
