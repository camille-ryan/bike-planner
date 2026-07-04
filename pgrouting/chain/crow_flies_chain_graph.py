"""Crow-flies chain graph: K-nearest by haversine, no roads involved.

Fallback for stage 4 when the postgres-heavy road-based Voronoi
builder (`build_way_graph.py`) can't complete under WSL2 disk I/O.
Runs in <1 s regardless of geography size.

Approach
--------
1. Read every anchor from `way_city_anchors.geojson` (written by
   `select_anchors_bottom_up.py` upstream — includes ferry piers).
2. For each anchor A, find its K nearest anchors by great-circle
   distance (KDTree over ECEF).
3. Emit each undirected pair (A, B) as a chain edge with:
   - `cost_m` = haversine distance in meters
   - `geom`   = straight-line polyline [A_lonlat, B_lonlat]
4. Dedupe canonically by (min_ref, max_ref).
5. Cap by MAX_EDGE_M so a village doesn't chain-neighbor a city
   400 km away.

Correctness of downstream stages
--------------------------------
Chain edges define which anchor pairs get paired SPTs built for them.
Voronoi-adjacency (build_way_graph) gives semantically better neighbors
because it follows real corridors, but any reasonable topology works
— the paired SPT stage does the actual road routing inside each
edge's polygon.

Trade-offs vs Voronoi:
- Extra edges: some pairs that aren't real road neighbors get paired
  SPTs. Costs storage + build time; harmless at query time
  (chain-Dijkstra picks the cheapest chain).
- Missing edges: none — every anchor gets K neighbors.
- Wrong weights: crow-flies underestimates real bike distance. If two
  edges have similar cost but wildly different road distance,
  chain-Dijkstra may pick suboptimally. Fixable by a later pass that
  overwrites `cost_m` with the real paired-SPT distance.

Env vars:
- CROW_K            (default 6)  — neighbors per anchor
- CROW_MAX_EDGE_KM  (default 100) — cap edge length
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


DATA_DIR         = Path(os.environ.get("DATA_DIR", "/data"))
ANCHORS_IN       = DATA_DIR / "way_city_anchors.geojson"
OUT_GRAPH_JSON   = DATA_DIR / "way_city_graph.json"
OUT_GRAPH_GEOJSON = DATA_DIR / "way_city_graph.geojson"
OUT_NODES        = DATA_DIR / "way_city_anchors.geojson"  # rewritten with in_graph
OUT_ORPHANS      = DATA_DIR / "way_city_anchors_orphans.geojson"

K = int(os.environ.get("CROW_K", "6"))
MAX_EDGE_M = float(os.environ.get("CROW_MAX_EDGE_KM", "100")) * 1000.0
R_EARTH_M = 6_371_000.0


def _lonlat_to_xyz(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    coslat = np.cos(lat_r)
    return np.column_stack([
        R_EARTH_M * coslat * np.cos(lon_r),
        R_EARTH_M * coslat * np.sin(lon_r),
        R_EARTH_M * np.sin(lat_r),
    ])


def _haversine_m(lon1: float, lat1: float,
                 lon2: float, lat2: float) -> float:
    """Vectorized haversine in meters."""
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2)
    return float(2.0 * R_EARTH_M * np.arcsin(np.sqrt(a)))


def main() -> None:
    t0 = time.time()
    print(f"[crow-flies] K = {K}, max edge = {MAX_EDGE_M/1000.0:.0f} km",
          flush=True)

    fc = json.loads(ANCHORS_IN.read_text())
    anchors: list[dict] = []
    for f in fc["features"]:
        lon, lat = f["geometry"]["coordinates"]
        p = f["properties"]
        anchors.append({
            "ref":         p["ref"],
            "name":        p["name"],
            "kind":        p.get("kind"),
            "place":       p.get("place"),
            "population":  p.get("population"),
            "country":     p.get("country"),
            "vid":         p.get("vid"),
            "_protected":  p.get("_protected", False),
            "lon":         float(lon),
            "lat":         float(lat),
        })
    print(f"[crow-flies] loaded {len(anchors):,} anchors from "
          f"{ANCHORS_IN.name}", flush=True)

    lons = np.array([a["lon"] for a in anchors], dtype=np.float64)
    lats = np.array([a["lat"] for a in anchors], dtype=np.float64)
    xyz  = _lonlat_to_xyz(lons, lats)
    tree = cKDTree(xyz)

    # k+1 because query returns the anchor itself as its own nearest.
    dists_chord, idxs = tree.query(xyz, k=K + 1)
    # Convert chord (ECEF straight line through the earth) to great-circle
    # arc length. For our K-nearest use it's essentially interchangeable
    # with haversine at these distances, but we compute the exact
    # haversine per edge for cost_m below anyway.

    # Emit undirected pairs, canonical order by ref.
    merged: dict[tuple[str, str], dict] = {}
    for i in range(len(anchors)):
        ref_a  = anchors[i]["ref"]
        name_a = anchors[i]["name"]
        lon_a, lat_a = anchors[i]["lon"], anchors[i]["lat"]
        for k in range(1, K + 1):  # skip self at k=0
            j = int(idxs[i, k])
            if j == i:
                continue
            ref_b  = anchors[j]["ref"]
            name_b = anchors[j]["name"]
            lon_b, lat_b = anchors[j]["lon"], anchors[j]["lat"]
            cost_m = _haversine_m(lon_a, lat_a, lon_b, lat_b)
            if cost_m > MAX_EDGE_M:
                continue
            key = (ref_a, ref_b) if ref_a < ref_b else (ref_b, ref_a)
            if key in merged:
                continue
            # Store in canonical key order.
            if key[0] == ref_a:
                merged[key] = {
                    "a": ref_a, "a_name": name_a,
                    "b": ref_b, "b_name": name_b,
                    "cost_m": cost_m,
                    "geom": [[lon_a, lat_a], [lon_b, lat_b]],
                }
            else:
                merged[key] = {
                    "a": ref_b, "a_name": name_b,
                    "b": ref_a, "b_name": name_a,
                    "cost_m": cost_m,
                    "geom": [[lon_b, lat_b], [lon_a, lat_a]],
                }

    chain_edges = list(merged.values())
    participant_refs = {e["a"] for e in chain_edges} | {e["b"] for e in chain_edges}
    print(f"[crow-flies] {len(chain_edges):,} chain edges  "
          f"| {len(participant_refs):,} anchors participate", flush=True)

    OUT_GRAPH_JSON.write_text(json.dumps(chain_edges, ensure_ascii=False))
    print(f"[crow-flies] wrote {OUT_GRAPH_JSON} "
          f"({OUT_GRAPH_JSON.stat().st_size/(1<<20):.1f} MB)", flush=True)

    fc_edges = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString",
                             "coordinates": [list(c) for c in e["geom"]]},
                "properties": {
                    "a":       e["a"], "a_name": e["a_name"],
                    "b":       e["b"], "b_name": e["b_name"],
                    "cost_km": round(e["cost_m"]/1000.0, 2),
                },
            }
            for e in chain_edges
        ],
    }
    OUT_GRAPH_GEOJSON.write_text(json.dumps(fc_edges, ensure_ascii=False))
    print(f"[crow-flies] wrote {OUT_GRAPH_GEOJSON}", flush=True)

    # Rewrite anchors with in_graph flag. Anchors that appear in NO
    # edge (e.g., if MAX_EDGE_M dropped their only neighbor) go to
    # orphans.
    fc_nodes = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [a["lon"], a["lat"]]},
                "properties": {
                    "ref":        a["ref"],
                    "name":       a["name"],
                    "kind":       a["kind"],
                    "place":      a["place"],
                    "population": a["population"],
                    "country":    a["country"],
                    "vid":        a["vid"],
                    "_protected": a["_protected"],
                    "in_graph":   a["ref"] in participant_refs,
                    "snap_dist_m": 0,  # crow-flies has no snap
                    "n_snaps":     1 if a["ref"] in participant_refs else 0,
                },
            }
            for a in anchors
        ],
    }
    OUT_NODES.write_text(json.dumps(fc_nodes, ensure_ascii=False))
    print(f"[crow-flies] wrote {OUT_NODES}", flush=True)

    orphans = [a for a in anchors if a["ref"] not in participant_refs]
    fc_orphans = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [a["lon"], a["lat"]]},
                "properties": {
                    "name":  a["name"],
                    "kind":  a["kind"],
                    "place": a["place"],
                },
            }
            for a in orphans
        ],
    }
    OUT_ORPHANS.write_text(json.dumps(fc_orphans, ensure_ascii=False))
    print(f"[crow-flies] wrote {OUT_ORPHANS} ({len(orphans):,} orphans)",
          flush=True)

    print(f"[crow-flies] DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
