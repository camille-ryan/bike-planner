"""Crow-flies chain graph: closest-per-sector by haversine, no roads.

Fallback for stage 4 when the postgres-heavy road-based Voronoi
builder (`build_way_graph.py`) can't complete under WSL2 disk I/O.
Runs in <1 s regardless of geography size.

Approach
--------
For each anchor A:
1. Find every candidate anchor within CROW_MAX_EDGE_M via KDTree.
2. Compute the compass bearing A→candidate.
3. Bin candidates into overlapping sectors: CROW_N_SECTORS starts of
   `SECTOR_WIDTH_DEG` degrees each, with `SECTOR_STRIDE_DEG` degrees
   between starts. Default 12 starts of 60° every 30° — every
   candidate falls into 2 sectors, giving each anchor a "second
   chance" to be picked when a slightly closer alternative dominates
   the closer sector.
4. Keep the CLOSEST candidate per non-empty sector. Dedupe across
   sectors by taking each ref's minimum distance.

Sector-based selection gives directionally-balanced neighbors:
- A central anchor with ring-of-neighbors: up to N sectors' worth of
  distinct neighbors, one per direction.
- A coastal / edge anchor: fewer edges, only in directions where
  anchors actually exist. Won't waste chain-neighbors on sea.

Overlap (60°/30° stride vs the previous 45° non-overlapping) helps
edge cases:
- A coastal town whose closest neighbor sits right at a 45° boundary
  now gets picked in both adjacent overlapping sectors.
- Slightly further neighbors in a nearby direction are no longer
  masked by an unrelated near neighbor sitting exactly at the sector
  boundary.

Emits each undirected pair (A, B) with:
- `cost_m` = haversine distance in meters
- `geom`   = straight-line polyline [A_lonlat, B_lonlat]

Dedupes canonically by (min_ref, max_ref).

Correctness of downstream stages
--------------------------------
Chain edges define which anchor pairs get paired SPTs built for them.
Voronoi-adjacency (build_way_graph) gives semantically better neighbors
because it follows real corridors, but any reasonable topology works
— the paired SPT stage does the actual road routing inside each
edge's polygon.

Trade-offs vs Voronoi:
- Wrong weights: crow-flies underestimates real bike distance. If two
  edges have similar cost but wildly different road distance,
  chain-Dijkstra may pick suboptimally. Fixable by a later pass that
  overwrites `cost_m` with the real paired-SPT distance.

Env vars:
- CROW_N_SECTORS    (default 8)   — angular bins around each anchor
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

SECTOR_WIDTH_DEG  = float(os.environ.get("CROW_SECTOR_WIDTH_DEG", "60"))
SECTOR_STRIDE_DEG = float(os.environ.get("CROW_SECTOR_STRIDE_DEG", "30"))
# Derived: number of overlapping sector starts around the compass.
N_SECTORS = int(round(360.0 / SECTOR_STRIDE_DEG))
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


def _bearing_deg(lon1: float, lat1: float,
                 lons2: np.ndarray, lats2: np.ndarray) -> np.ndarray:
    """Initial compass bearing from (lon1, lat1) to each (lons2, lats2).

    Returns degrees in [0, 360). 0 = north, 90 = east, 180 = south,
    270 = west. Uses the standard great-circle initial-bearing
    formula so sectors are well-defined at any latitude.
    """
    lat1_r  = np.radians(lat1)
    lats2_r = np.radians(lats2)
    dlon    = np.radians(lons2 - lon1)
    y = np.sin(dlon) * np.cos(lats2_r)
    x = (np.cos(lat1_r) * np.sin(lats2_r)
         - np.sin(lat1_r) * np.cos(lats2_r) * np.cos(dlon))
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def _chord_for_arc(arc_m: float) -> float:
    return 2.0 * R_EARTH_M * np.sin(arc_m / (2.0 * R_EARTH_M))


def main() -> None:
    t0 = time.time()
    print(f"[crow-flies] sectors: {N_SECTORS} × {SECTOR_WIDTH_DEG:.0f}° wide "
          f"(stride {SECTOR_STRIDE_DEG:.0f}° — "
          f"{SECTOR_WIDTH_DEG - SECTOR_STRIDE_DEG:.0f}° overlap)  "
          f"max edge = {MAX_EDGE_M/1000.0:.0f} km",
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
    chord_cap = _chord_for_arc(MAX_EDGE_M)

    # For each anchor: find every candidate within MAX_EDGE_M, bin by
    # bearing sector (overlapping), keep the closest per non-empty
    # sector, dedupe by ref.
    merged: dict[tuple[str, str], dict] = {}
    per_anchor_neighbors = np.zeros(len(anchors), dtype=np.int32)
    sector_starts = [s * SECTOR_STRIDE_DEG for s in range(N_SECTORS)]

    def _in_sector(bearing, start, width):
        end = (start + width) % 360.0
        if start < end:
            return start <= bearing < end
        # wraps 360°
        return bearing >= start or bearing < end

    for i in range(len(anchors)):
        cand_idxs = tree.query_ball_point(xyz[i], r=chord_cap)
        cand_idxs = [j for j in cand_idxs if j != i]
        if not cand_idxs:
            continue
        cand_lons = lons[cand_idxs]
        cand_lats = lats[cand_idxs]
        bearings = _bearing_deg(lons[i], lats[i], cand_lons, cand_lats)
        dists = np.array([
            _haversine_m(lons[i], lats[i], cand_lons[k], cand_lats[k])
            for k in range(len(cand_idxs))
        ])
        # Per overlapping sector, keep the closest candidate.
        best_per_sector: dict[float, tuple[int, float]] = {}
        for k in range(len(cand_idxs)):
            if dists[k] > MAX_EDGE_M:
                continue
            for start in sector_starts:
                if _in_sector(bearings[k], start, SECTOR_WIDTH_DEG):
                    cur = best_per_sector.get(start)
                    if cur is None or dists[k] < cur[1]:
                        best_per_sector[start] = (cand_idxs[k], float(dists[k]))
        # Dedupe by candidate ref — same anchor may win multiple
        # sectors; we count it once and take its (already unique)
        # distance.
        seen: set[int] = set()
        for cand_idx, cand_dist in best_per_sector.values():
            seen.add(cand_idx)
        per_anchor_neighbors[i] = len(seen)

        ref_a  = anchors[i]["ref"]
        name_a = anchors[i]["name"]
        lon_a, lat_a = anchors[i]["lon"], anchors[i]["lat"]
        # Emit each unique candidate once (many sectors may pick the
        # same neighbor in the overlap band).
        emitted: set[int] = set()
        for j, cost_m in best_per_sector.values():
            if j in emitted:
                continue
            emitted.add(j)
            ref_b  = anchors[j]["ref"]
            name_b = anchors[j]["name"]
            lon_b, lat_b = anchors[j]["lon"], anchors[j]["lat"]
            key = (ref_a, ref_b) if ref_a < ref_b else (ref_b, ref_a)
            if key in merged:
                continue
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
    print(f"[crow-flies] per-anchor neighbor counts: "
          f"min {per_anchor_neighbors.min()}, "
          f"median {int(np.median(per_anchor_neighbors))}, "
          f"max {per_anchor_neighbors.max()}", flush=True)

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
