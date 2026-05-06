"""Persist the SPT preprocess outputs.

Layout under `data/spt/<profile>/`:

  graph_nodes.npz      — lon, lat, osm_id arrays (read by API at startup)
  graph_edges.npz      — src, dst, cost, length arrays (debug + future use)
  cities.json          — list of {idx, osm_id, name, lon, lat, node_idx}
  spt_fwd.npz          — cost, parent, city_idx (city → everywhere)
  spt_rev.npz          — cost, parent, city_idx (everywhere → city)
  city_graph.json      — adjacency for upper-layer Dijkstra
  cells.geojson        — FeatureCollection of cell polygons

Numpy `npz` (uncompressed) is mmap-friendly via np.load(mmap_mode='r')
which is exactly what the API needs at query time. JSON for the small
metadata so it's human-readable during debugging.
"""
import json
from pathlib import Path

import numpy as np


def write_all(
    out_dir: Path,
    graph,
    fwd_spt,
    rev_spt,
    cities: list[dict],
    city_graph,
    polygons: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / "graph_nodes.npz",
             lon=graph.node_lon, lat=graph.node_lat, osm_id=graph.node_osm_id)
    np.savez(out_dir / "graph_edges.npz",
             src=graph.edge_src, dst=graph.edge_dst,
             cost=graph.edge_cost, length=graph.edge_length_m)

    # spt_fwd.npz / spt_rev.npz aren't read by any downstream code —
    # the API uses global_assignment.npz (city_idx only) and the
    # per-city SPTs. Writing the full forward SPT was costing ~1.4 GB
    # of disk per profile and forcing us to keep the parent array alive
    # in memory through the cells phase. Skip it.

    with open(out_dir / "cities.json", "w") as fh:
        json.dump(cities, fh, ensure_ascii=False, indent=1)

    cg = {
        "from_city": city_graph.from_city.tolist(),
        "to_city":   city_graph.to_city.tolist(),
        "weight":    [float(w) for w in city_graph.weight],
    }
    with open(out_dir / "city_graph.json", "w") as fh:
        json.dump(cg, fh)

    with open(out_dir / "cells.geojson", "w") as fh:
        json.dump(polygons, fh, ensure_ascii=False)

    print(f"[save] wrote outputs to {out_dir}")
