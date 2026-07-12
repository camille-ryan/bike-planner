"""Fast augmentation: add ferry chain edges + pier-to-land-anchor edges
to the existing way_city_graph.json without re-running the massive
build_way_graph.py JOIN.

Assumes:
  * data/way_city_graph.json — pre-ferry land chain graph (any freshness).
  * data/way_city_anchors.geojson — anchors including 411 ferry piers
    (produced by select_anchors_bottom_up.py with the ferry-piers
    loader).

Outputs:
  * data/way_city_graph.json (overwritten) — appended with:
      - ferry chain edges (pier_A ↔ pier_B via each ferry way), one
        per ferry `ways` row, straight-line geom pier-to-pier.
      - pier↔land chain edges: for each ferry pier, KDTree lookup on
        land anchors within 30 km, add straight-line chain edge to
        the nearest 2 (avoids isolating piers on the chain graph).

Rationale: the massive `ways × way_tags × ways_vertices_pgr` JOIN in
build_way_graph.py is unnecessary for this delta. Ferry endpoints and
their coords come from a fast partial-index query; land anchors are
already in the anchors geojson.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import psycopg
from scipy.spatial import cKDTree

import config


ANCHORS_IN         = Path("/data/way_city_anchors.geojson")
CHAIN_GRAPH_INOUT  = Path("/data/way_city_graph.json")
CHAIN_GRAPH_GEOJSON = Path("/data/way_city_graph.geojson")
PIER_LAND_MAX_KM   = 30.0    # KDTree radius for pier↔land chain edges
PIER_LAND_K        = 2        # nearest-K land anchors per pier


def _hav_km(a_lon, a_lat, b_lon, b_lat):
    R = 6371.0
    from math import radians, sin, cos, asin, sqrt
    p1, p2 = radians(a_lat), radians(b_lat)
    dp = radians(b_lat - a_lat); dl = radians(b_lon - a_lon)
    s = sin(dp/2)**2 + cos(p1)*cos(p2)*sin(dl/2)**2
    return 2 * R * asin(sqrt(s))


def _load_ferry_ways(conn: psycopg.Connection) -> list[dict]:
    """Return one dict per ferry way: {src_vid, dst_vid, len_m, src_lon,
    src_lat, dst_lon, dst_lat}. Uses the partial index on ways.is_ferry."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT w.source, w.target, w.length_m,
                   ST_X(vs.the_geom), ST_Y(vs.the_geom),
                   ST_X(vt.the_geom), ST_Y(vt.the_geom)
            FROM ways w
            JOIN ways_vertices_pgr vs ON vs.id = w.source
            JOIN ways_vertices_pgr vt ON vt.id = w.target
            WHERE w.is_ferry AND NOT w.bike_excluded
              AND w.cost_views IS NOT NULL
              AND w.length_m >= 100
        """)
        return [{
            "src_vid": int(r[0]), "dst_vid": int(r[1]),
            "len_m":   float(r[2]),
            "src_lon": float(r[3]), "src_lat": float(r[4]),
            "dst_lon": float(r[5]), "dst_lat": float(r[6]),
        } for r in cur.fetchall()]


def _load_anchors_by_ref() -> tuple[dict[str, dict], list[dict], list[dict]]:
    """Load all anchors + split into (piers, land)."""
    fc = json.loads(ANCHORS_IN.read_text())
    anchors_by_ref: dict[str, dict] = {}
    piers: list[dict] = []
    land: list[dict] = []
    for f in fc["features"]:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        a = {
            "ref":  p["ref"],
            "name": p["name"],
            "kind": p.get("kind"),
            "lon":  float(lon),
            "lat":  float(lat),
        }
        anchors_by_ref[a["ref"]] = a
        if a["kind"] == "ferry_pier":
            piers.append(a)
        else:
            land.append(a)
    return anchors_by_ref, piers, land


def main() -> None:
    t0 = time.time()
    print(f"[augment] loading anchors + existing chain graph…", flush=True)
    anchors_by_ref, piers, land = _load_anchors_by_ref()
    print(f"[augment]   {len(piers):,} ferry piers, {len(land):,} land anchors",
          flush=True)

    chain_edges = json.loads(CHAIN_GRAPH_INOUT.read_text())
    n_before = len(chain_edges)
    existing_pairs: set[tuple[str, str]] = set()
    for e in chain_edges:
        existing_pairs.add((e["a"], e["b"]))
        existing_pairs.add((e["b"], e["a"]))
    print(f"[augment]   {n_before:,} existing chain edges "
          f"({len(existing_pairs):,} directed pair slots)", flush=True)

    # Query ferry ways from postgres (fast — partial index).
    print(f"[augment] querying ferry ways from postgres…", flush=True)
    t = time.time()
    with psycopg.connect(config.PG_DSN) as conn:
        ferry_rows = _load_ferry_ways(conn)
    print(f"[augment]   {len(ferry_rows):,} ferry `ways` rows in "
          f"{time.time()-t:.1f}s", flush=True)

    # Persist ferry (src_vid, dst_vid) pairs so the stage-6 proximity
    # Dijkstra can mask ferry edges out of its road graph. Design
    # intent: LAND anchors reach chain neighbors via road only; ferry
    # crossings are chain-level hops between piers (added below via the
    # pier↔pier BFS). Without this mask, the proximity Dijkstra rides a
    # heavily-weighted sea ferry way and manufactures bogus LAND↔PIER
    # chain edges like Rostock LAND → Gedser PIER.
    ferry_edges_out = Path("/data/ferry_edges.json")
    ferry_edges_out.write_text(json.dumps(
        [[int(r["src_vid"]), int(r["dst_vid"])] for r in ferry_rows]))
    print(f"[augment]   wrote {ferry_edges_out.name} "
          f"({ferry_edges_out.stat().st_size / 1024:.0f} KB)", flush=True)

    # 1) Ferry chain edges (pier ↔ pier via ferry-only subgraph).
    #
    # Ferry ways in OSM are usually split into a *chain* of short
    # segments between the two piers — the piers are the endpoints of
    # the chain, and every vertex in between has ONLY ferry edges. So
    # we can't just take rows where both endpoints are piers (that
    # missed all multi-segment crossings, including DE↔DK).
    #
    # Instead: build the ferry-only vertex graph, then BFS from each
    # pier and record every OTHER pier reached. Cost = sum of ferry
    # segment lengths on the BFS path; geom = concatenated segment
    # midpoints (a piecewise-linear approximation of the real ferry
    # route — good enough for polygon compute, since actual routing
    # uses cell files).
    from collections import defaultdict, deque
    ferry_adj: dict[int, list[tuple[int, float, tuple[float,float], tuple[float,float]]]] = defaultdict(list)
    vid_coord: dict[int, tuple[float, float]] = {}
    for r in ferry_rows:
        s, t = r["src_vid"], r["dst_vid"]
        sll = (r["src_lon"], r["src_lat"]); tll = (r["dst_lon"], r["dst_lat"])
        vid_coord[s] = sll; vid_coord[t] = tll
        ferry_adj[s].append((t, r["len_m"], sll, tll))
        ferry_adj[t].append((s, r["len_m"], tll, sll))
    pier_vids = {int(p["ref"].split(":", 1)[1]): p for p in piers
                 if p["ref"].startswith("ferry:")}
    print(f"[augment]   ferry subgraph: {len(ferry_adj):,} vids, "
          f"{len(pier_vids):,} pier vids", flush=True)

    n_ferry_added = 0
    for start_vid in pier_vids:
        if start_vid not in ferry_adj:
            continue
        # BFS with cumulative cost + accumulated coord path.
        # We only want the shortest path (by cost) to each other pier —
        # deque + revisit-if-improved is fine for the small subgraph.
        cost_to: dict[int, float] = {start_vid: 0.0}
        path_to: dict[int, list[tuple[float, float]]] = {
            start_vid: [vid_coord[start_vid]]
        }
        q = deque([start_vid])
        while q:
            u = q.popleft()
            for v, w, u_ll, v_ll in ferry_adj[u]:
                new_cost = cost_to[u] + w
                if v not in cost_to or new_cost < cost_to[v]:
                    cost_to[v] = new_cost
                    path_to[v] = path_to[u] + [v_ll]
                    q.append(v)
        # Emit chain edges to every OTHER pier we reached.
        a_ref = f"ferry:{start_vid}"
        for end_vid, total in cost_to.items():
            if end_vid == start_vid or end_vid not in pier_vids:
                continue
            # Emit each pair once (start_vid < end_vid).
            if start_vid > end_vid:
                continue
            b_ref = f"ferry:{end_vid}"
            if a_ref not in anchors_by_ref or b_ref not in anchors_by_ref:
                continue
            if (a_ref, b_ref) in existing_pairs or (b_ref, a_ref) in existing_pairs:
                continue
            chain_edges.append({
                "a": a_ref, "a_name": anchors_by_ref[a_ref]["name"],
                "b": b_ref, "b_name": anchors_by_ref[b_ref]["name"],
                "cost_m": total,
                "geom":   [list(pt) for pt in path_to[end_vid]],
                "_is_ferry": True,
            })
            existing_pairs.add((a_ref, b_ref))
            existing_pairs.add((b_ref, a_ref))
            n_ferry_added += 1
    print(f"[augment]   added {n_ferry_added:,} pier↔pier ferry chain edges "
          f"(via ferry-subgraph BFS)", flush=True)

    # 2) Pier ↔ nearest-K land anchor chain edges.
    # For each pier, KDTree-lookup the K nearest land anchors within
    # PIER_LAND_MAX_KM. Chain edges get straight-line geom (fine for
    # polygon-compute — SPT still uses the actual road network via
    # cell edges).
    if land and piers:
        land_lonlat = np.array([[a["lon"], a["lat"]] for a in land])
        tree = cKDTree(land_lonlat)
        # Convert PIER_LAND_MAX_KM to degrees (rough; latitude scaling).
        max_deg = PIER_LAND_MAX_KM / 111.0
        n_pl_added = 0
        for pier in piers:
            dists, idxs = tree.query(
                [pier["lon"], pier["lat"]], k=PIER_LAND_K,
                distance_upper_bound=max_deg,
            )
            if np.isscalar(dists):
                dists, idxs = [dists], [idxs]
            for d, i in zip(dists, idxs):
                if not np.isfinite(d) or i >= len(land):
                    continue
                la = land[i]
                a_ref, b_ref = pier["ref"], la["ref"]
                if (a_ref, b_ref) in existing_pairs or (b_ref, a_ref) in existing_pairs:
                    continue
                dist_km = _hav_km(pier["lon"], pier["lat"], la["lon"], la["lat"])
                chain_edges.append({
                    "a": a_ref, "a_name": pier["name"],
                    "b": b_ref, "b_name": la["name"],
                    "cost_m": dist_km * 1000.0,
                    "geom":   [[pier["lon"], pier["lat"]],
                               [la["lon"], la["lat"]]],
                    "_pier_land": True,
                })
                existing_pairs.add((a_ref, b_ref))
                existing_pairs.add((b_ref, a_ref))
                n_pl_added += 1
        print(f"[augment]   added {n_pl_added:,} pier↔land chain edges "
              f"(K={PIER_LAND_K}, max {PIER_LAND_MAX_KM:.0f} km)",
              flush=True)

    print(f"[augment] writing {CHAIN_GRAPH_INOUT} "
          f"({n_before:,} → {len(chain_edges):,} edges)", flush=True)
    CHAIN_GRAPH_INOUT.write_text(json.dumps(chain_edges))

    # GeoJSON for UI visualization.
    features = []
    for e in chain_edges:
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": e["geom"]},
            "properties": {
                "a": e["a"], "b": e["b"],
                "cost_m": e["cost_m"],
                "is_ferry": bool(e.get("_is_ferry")),
                "pier_land": bool(e.get("_pier_land")),
            },
        })
    CHAIN_GRAPH_GEOJSON.write_text(json.dumps({
        "type": "FeatureCollection", "features": features,
    }))
    print(f"[augment] DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
