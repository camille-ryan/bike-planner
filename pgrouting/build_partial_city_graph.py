"""Build a city_graph.json from whatever npz files currently exist.

Useful for testing routing mid-preprocess: the full city_graph
derivation runs at the END of compute_spts.run() over all anchors.
This script does the same thing but only for the per-city SPTs that
have been written so far. Drops the partial result at
data/spt/lht/city_graph.partial.json.

The verify_chainless.py script can be pointed at this with:
    python verify_chainless.py Graz København  # uses city_graph.json
After this script runs, you can `mv city_graph.partial.json
city_graph.json` to test (and rename back, since the running
preprocess will overwrite it on completion).
"""
import json
import os
from pathlib import Path

import numpy as np
import psycopg

import compute_spts
import config


SPT_DIR = Path(os.environ.get("DATA_DIR", "/data")) / "spt" / "lht"


def main():
    spt_dir = SPT_DIR / "spt"

    # Find which anchor ids have npz files.
    npz_files = sorted(spt_dir.glob("*.npz"))
    done_ids = sorted(int(p.stem) for p in npz_files)
    print(f"[partial] {len(done_ids):,} per-city SPTs found in {spt_dir}")

    # Fetch all anchors (need polygon_sets for adjacency).
    with psycopg.connect(config.PG_DSN) as conn:
        anchors = compute_spts._fetch_anchors(conn)
        print(f"[partial] {len(anchors):,} anchors total in DB")

        polygon_sets: dict[int, np.ndarray] = {}
        for a in anchors:
            ci = a["anchor_id"] - 1
            if ci not in done_ids:
                continue
            if a["has_polygon"]:
                polygon_sets[ci] = compute_spts._polygon_vertices(conn, a["anchor_id"])
            else:
                polygon_sets[ci] = np.asarray([a["snap_vertex_id"]], dtype=np.int64)

    print(f"[partial] computed polygon vertex sets for {len(polygon_sets):,} done cities")

    # Cross-check: for each done city A, which other done city's polygon vertices
    # are reached by A's SPT?
    edges = []
    for ci_a in done_ids:
        spt_path = spt_dir / f"{ci_a}.npz"
        data = np.load(spt_path)
        a_node_global = data["node_global"]
        a_cost = data["cost"]
        for ci_b, b_polys in polygon_sets.items():
            if ci_b == ci_a or len(b_polys) == 0:
                continue
            idx = np.searchsorted(a_node_global, b_polys)
            in_range = idx < len(a_node_global)
            matched = np.zeros_like(in_range)
            matched[in_range] = a_node_global[idx[in_range]] == b_polys[in_range]
            if not matched.any():
                continue
            hit_costs = a_cost[idx[matched]]
            edges.append((ci_a, ci_b, float(hit_costs.min())))
    print(f"[partial] {len(edges):,} directed edges in partial city_graph")

    cg = {
        "from_city": [e[0] for e in edges],
        "to_city":   [e[1] for e in edges],
        "weight":    [e[2] for e in edges],
    }
    out = SPT_DIR / "city_graph.partial.json"
    with open(out, "w") as fh:
        json.dump(cg, fh)
    print(f"[partial] wrote {out}")


if __name__ == "__main__":
    main()
