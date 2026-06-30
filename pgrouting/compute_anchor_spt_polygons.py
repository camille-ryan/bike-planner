"""Compute per-anchor SPT bounding polygons for the chain-graph experiment.

Replaces the uniform 30 km radius SPT cap (`compute_spts.py:SPT_RADIUS_M`)
with a dynamic per-anchor polygon shaped by the chain graph. The
polygon is:

    hull(
      chain-edge-geometry from A to each adjacent B,
      + for each B: chain-edge-geometry from B to each onward C,
                    up to the path-midpoint of (B, C)
    )
    ∪ 5 km circle around A

The "1.5-hop" hull captures the actual road envelope around A — long
and thin along valley corridors, wider in dense suburbs — instead of
the 30 km disc that wastes coverage on directions where no road goes
and clips early in long-thin valleys.

The 5 km circle is a hard floor so isolated villages and orphans (no
chain edges) still get a useful local SPT.

Inputs:
  /data/way_city_graph.json    -- list of chain edges with geom[]
  /data/way_city_anchors.geojson  -- FeatureCollection with ref, lon, lat

Output:
  /data/way_city_spt_polygons.json -- dict[ref -> [[lon, lat], ...]]
                                       (polygon vertex ring, closed)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


CHAIN_GRAPH_IN     = Path("/data/way_city_graph.json")
ANCHORS_IN         = Path("/data/way_city_anchors.geojson")
POLYGONS_OUT       = Path("/data/way_city_spt_polygons.json")
POLYGONS_OUT_GEOJSON = Path("/data/way_city_spt_polygons.geojson")

FLOOR_RADIUS_M = 5_000.0  # minimum disc each polygon must contain
CIRCLE_VERTICES = 32      # polygon vertices used to approximate the disc

R_EARTH_M = 6_371_000.0


def _haversine_segment_lengths(coords: list[list[float]]) -> list[float]:
    """Length of each segment in meters, for a polyline of (lon, lat) points."""
    out: list[float] = []
    for (lon1, lat1), (lon2, lat2) in zip(coords[:-1], coords[1:]):
        rl1 = math.radians(lat1); rl2 = math.radians(lat2)
        dl  = math.radians(lat2 - lat1)
        dn  = math.radians(lon2 - lon1)
        a = math.sin(dl/2)**2 + math.cos(rl1)*math.cos(rl2)*math.sin(dn/2)**2
        out.append(2 * R_EARTH_M * math.asin(math.sqrt(a)))
    return out


def _walk_to_path_midpoint(coords: list[list[float]]) -> list[list[float]]:
    """Return the prefix of `coords` up to the path-midpoint vertex
    (the first vertex past the halfway mark of the total path length).
    Includes both endpoints of the prefix."""
    if len(coords) < 2:
        return list(coords)
    seg_lens = _haversine_segment_lengths(coords)
    total = sum(seg_lens)
    half = total / 2.0
    accum = 0.0
    for i, sl in enumerate(seg_lens):
        accum += sl
        if accum >= half:
            return coords[:i + 2]  # include the segment that crosses half
    return list(coords)


def _circle_points(lon: float, lat: float, radius_m: float,
                   n: int) -> list[tuple[float, float]]:
    """`n`-vertex polygon approximating a great-circle disc of `radius_m`
    around (lon, lat). Latitude is treated as constant — accurate to
    ~0.1% within 5 km at temperate latitudes."""
    lat_per_m = 1.0 / 111_320.0           # m per degree latitude
    lon_per_m = lat_per_m / max(math.cos(math.radians(lat)), 1e-6)
    out: list[tuple[float, float]] = []
    for k in range(n):
        theta = 2.0 * math.pi * k / n
        dlat = math.cos(theta) * radius_m * lat_per_m
        dlon = math.sin(theta) * radius_m * lon_per_m
        out.append((lon + dlon, lat + dlat))
    return out


def _load_inputs() -> tuple[list[dict], list[dict]]:
    chain_edges = json.loads(CHAIN_GRAPH_IN.read_text())
    fc = json.loads(ANCHORS_IN.read_text())
    anchors: list[dict] = []
    for f in fc["features"]:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        anchors.append({
            "ref": p["ref"],
            "name": p["name"],
            "lon": float(lon),
            "lat": float(lat),
        })
    return chain_edges, anchors


def _edges_by_endpoint(chain_edges: list[dict]
                       ) -> dict[str, list[tuple[str, list[list[float]]]]]:
    """ref -> [(other_ref, geom_oriented_away_from_ref), ...]
    Geom is oriented so geom[0] is at `ref` and geom[-1] is at `other_ref`."""
    out: dict[str, list[tuple[str, list[list[float]]]]] = {}
    for e in chain_edges:
        a, b, geom = e["a"], e["b"], e["geom"]
        out.setdefault(a, []).append((b, list(geom)))
        out.setdefault(b, []).append((a, list(reversed(geom))))
    return out


def _build_polygon(anchor: dict,
                   chain_by_ref: dict[str, list[tuple[str, list[list[float]]]]]
                   ) -> list[tuple[float, float]]:
    """Convex hull of the 1.5-hop chain geometry, unioned with a
    FLOOR_RADIUS_M disc around the anchor. Returns the polygon ring
    as a closed list of (lon, lat) tuples."""
    a_ref = anchor["ref"]
    points: list[tuple[float, float]] = []

    # 5 km floor disc — guarantees SPT coverage for orphans and
    # ensures the hull never collapses to a degenerate line.
    points.extend(_circle_points(anchor["lon"], anchor["lat"],
                                 FLOOR_RADIUS_M, CIRCLE_VERTICES))

    # 1-hop: full A→B geometry for every adjacent B.
    a_edges = chain_by_ref.get(a_ref, [])
    for b_ref, geom_ab in a_edges:
        points.extend((float(x), float(y)) for x, y in geom_ab)

        # 0.5-hop further: from B, walk each onward B→C up to path-midpoint.
        for c_ref, geom_bc in chain_by_ref.get(b_ref, []):
            if c_ref == a_ref:
                continue
            prefix = _walk_to_path_midpoint(geom_bc)
            points.extend((float(x), float(y)) for x, y in prefix)

    if len(points) < 3:
        return list(points)
    arr = np.array(points, dtype=np.float64)
    hull = ConvexHull(arr)
    ring = [tuple(arr[i]) for i in hull.vertices]
    # Close the ring (first point == last point) so consumers don't need to.
    ring.append(ring[0])
    return ring


def main() -> None:
    t0 = time.time()
    chain_edges, anchors = _load_inputs()
    print(f"[polygons] {len(anchors):,} anchors, "
          f"{len(chain_edges):,} chain edges", flush=True)

    chain_by_ref = _edges_by_endpoint(chain_edges)
    n_orphan = sum(1 for a in anchors if a["ref"] not in chain_by_ref)
    print(f"[polygons] {n_orphan} orphans (no chain edges) — "
          f"will get the {FLOOR_RADIUS_M/1000:.0f} km floor disc only",
          flush=True)

    polygons: dict[str, list[tuple[float, float]]] = {}
    sizes: list[int] = []
    for a in anchors:
        ring = _build_polygon(a, chain_by_ref)
        polygons[a["ref"]] = ring
        sizes.append(len(ring))

    POLYGONS_OUT.write_text(json.dumps(polygons))
    print(f"[polygons] wrote {POLYGONS_OUT} "
          f"({len(polygons):,} polygons, "
          f"avg {sum(sizes)/len(sizes):.1f} verts, max {max(sizes)})",
          flush=True)

    # GeoJSON for visual inspection in the web app.
    features = []
    for a in anchors:
        ring = polygons[a["ref"]]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[float(x), float(y)] for x, y in ring]],
            },
            "properties": {
                "ref": a["ref"], "name": a["name"],
                "n_vertices": len(ring),
            },
        })
    POLYGONS_OUT_GEOJSON.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": features,
    }))
    print(f"[polygons] wrote {POLYGONS_OUT_GEOJSON} for visual inspection",
          flush=True)
    print(f"[polygons] DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
