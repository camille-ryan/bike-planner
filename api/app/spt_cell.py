"""Per-anchor SPT visualization + amenities for the chainless preprocess.

Endpoints (wired in main.py):
  GET /spt/cell/{city_idx}             — SPT as GeoJSON FeatureCollection of
                                         LineStrings (each non-seed vertex →
                                         its parent_local edge), colored by
                                         cost-from-anchor.
  GET /spt/cell/{city_idx}/amenities   — POIs grouped by category, scoped to
                                         this anchor's admin polygon (or 1 km
                                         bbox around its place node otherwise).

Both endpoints read the chainless preprocess artifacts directly:
  - SPT: <DATA_DIR>/spt/<profile>/spt/<ci>.npz
  - Anchor geometry: postgres `anchors` table
  - POIs: pois.sqlite (SpatiaLite)
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

from .settings import POIS_DB, SPT_DIR

# MapLibre comfortably renders ~50 K LineStrings on a laptop GPU. Above
# that, panning gets sluggish. This is the cap for the *output*: as long
# as filtering (e.g. cost-cap) brings the kept-edge count below it, we
# ship everything without spatial subsampling.
HARD_FEATURE_CAP = 50_000

# Spatial-grid budget when we DO have to subsample (i.e. the post-filter
# count would still exceed the hard cap). Lower than the cap so the
# subsampled view still feels snappy.
GRID_FEATURE_BUDGET = 25_000


def gradient_for(
    conn: psycopg.Connection, profile: str, city_idx: int,
    max_cost: float | None = None,
) -> dict | None:
    """Read <DATA_DIR>/spt/<profile>/spt/<city_idx>.npz and emit a
    GeoJSON FeatureCollection of edges (each non-seed vertex → its
    parent), colored by cost. Returns None if the npz doesn't exist
    so the caller can raise 404.

    `max_cost` filters edges to those with cost-from-anchor < the value
    (units: same as the SPT cost — meters of bike-equivalent). Useful
    for zooming in on the immediate vicinity of an anchor: cost < 30 000
    typically yields a few tens of thousands of edges (vs. millions
    unfiltered) which can be shipped without subsampling for a much
    sharper visualization.

    If after filtering the kept-edge count is below `HARD_FEATURE_CAP`,
    we ship every edge as-is. Otherwise we fall back to adaptive
    spatial-grid subsampling: each edge's midpoint maps to a grid cell,
    we keep one representative per cell, and the cell size is chosen
    so the output lands near `GRID_FEATURE_BUDGET`. Spatial grid
    spreads kept edges evenly across the reach instead of clustering;
    pure stride sampling at the edge-id level produces visible dots
    because the kept edges are micro-segments scattered randomly.
    """
    npz_path = SPT_DIR / profile / "spt" / f"{city_idx}.npz"
    if not npz_path.exists():
        return None

    with np.load(npz_path) as f:
        node_global = np.asarray(f["node_global"])
        parent_local = np.asarray(f["parent_local"])
        cost = np.asarray(f["cost"])

    valid_mask = parent_local >= 0
    if max_cost is not None:
        valid_mask = valid_mask & (cost < float(max_cost))
    valid_idx = np.where(valid_mask)[0]
    n_valid = int(len(valid_idx))

    if n_valid == 0:
        return {
            "type": "FeatureCollection", "features": [],
            "city_idx": city_idx,
            "total_visited": int(len(node_global)),
            "cost_min": 0.0, "cost_max": 0.0,
            "grid_deg": None,
            "subsampled": False,
            "filtered_max_cost": max_cost,
        }

    # Decide whether spatial subsampling is needed at all.
    # - If `max_cost` was specified, the caller has already chosen a
    #   bounded slice — trust them and ship every kept edge unsubsampled
    #   even if it's tens of thousands of features. Tighten the filter
    #   if rendering gets sluggish.
    # - If unfiltered, subsample anything over HARD_FEATURE_CAP so the
    #   browser doesn't try to render millions of LineStrings.
    will_subsample = (max_cost is None) and (n_valid > HARD_FEATURE_CAP)

    if will_subsample:
        # Pre-stride to a coarse pool BEFORE doing any postgres lookups —
        # otherwise we'd ANY() over 1.5 M+ vertex ids and the coord
        # fetch alone takes ~2 minutes. The pool is a multiple of the
        # spatial-grid budget so each cell still has multiple candidates.
        pool_target = GRID_FEATURE_BUDGET * 5
        if n_valid > pool_target:
            pool_step = n_valid // pool_target + 1
            pool_idx = valid_idx[::pool_step]
        else:
            pool_idx = valid_idx
    else:
        pool_idx = valid_idx
    n_pool = int(len(pool_idx))

    child_global = node_global[pool_idx].astype(np.int64)
    parent_global = node_global[parent_local[pool_idx]].astype(np.int64)
    edge_costs = cost[pool_idx].astype(np.float32)

    all_vids = np.unique(np.concatenate([child_global, parent_global])).tolist()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT v.id, ST_X(v.the_geom), ST_Y(v.the_geom)
            FROM ways_vertices_pgr v
            WHERE v.id = ANY(%s)
        """, (all_vids,))
        coord_map = {
            int(r[0]): (float(r[1]), float(r[2])) for r in cur.fetchall()
        }

    # Build endpoint coordinate arrays for the pool. Edges that lost an
    # endpoint (shouldn't happen in a clean ingest) get filtered.
    cx = np.empty(n_pool, dtype=np.float64)
    cy = np.empty(n_pool, dtype=np.float64)
    px = np.empty(n_pool, dtype=np.float64)
    py = np.empty(n_pool, dtype=np.float64)
    keep = np.zeros(n_pool, dtype=bool)
    for i, (c, p) in enumerate(zip(child_global.tolist(), parent_global.tolist())):
        c_xy = coord_map.get(c)
        p_xy = coord_map.get(p)
        if c_xy is None or p_xy is None:
            continue
        cx[i], cy[i] = c_xy
        px[i], py[i] = p_xy
        keep[i] = True

    if not keep.any():
        return {
            "type": "FeatureCollection", "features": [],
            "city_idx": city_idx,
            "total_visited": int(len(node_global)),
            "cost_min": 0.0, "cost_max": 0.0,
            "grid_deg": None,
        }

    cx, cy, px, py = cx[keep], cy[keep], px[keep], py[keep]
    edge_costs = edge_costs[keep]

    # Adaptive spatial grid over the (already pre-strided) pool. Only
    # apply when we're in the subsample path; otherwise ship every edge.
    grid_deg = None
    if will_subsample and len(cx) > GRID_FEATURE_BUDGET:
        mid_x = (cx + px) * 0.5
        mid_y = (cy + py) * 0.5
        bbox_area = max(
            1e-9,
            (mid_x.max() - mid_x.min()) * (mid_y.max() - mid_y.min()),
        )
        grid_deg = float(np.clip(np.sqrt(bbox_area / GRID_FEATURE_BUDGET),
                                  0.001, 0.05))
        cell_x = np.floor(mid_x / grid_deg).astype(np.int64)
        cell_y = np.floor(mid_y / grid_deg).astype(np.int64)
        cell_key = (cell_x.astype(np.int64) << 32) | (cell_y.astype(np.int64) & 0xFFFFFFFF)
        _, first_per_cell = np.unique(cell_key, return_index=True)
        sel = np.sort(first_per_cell)
        cx, cy = cx[sel], cy[sel]
        px, py = px[sel], py[sel]
        edge_costs = edge_costs[sel]

    features: list[dict] = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(cx_i), float(cy_i)],
                    [float(px_i), float(py_i)],
                ],
            },
            "properties": {"cost": float(k)},
        }
        for cx_i, cy_i, px_i, py_i, k in zip(cx, cy, px, py, edge_costs)
    ]

    return {
        "type": "FeatureCollection",
        "features": features,
        "city_idx": city_idx,
        "total_visited": int(len(node_global)),
        # Cost stats for the FULL SPT (regardless of filter / subsample).
        "cost_min": float(cost[valid_mask].min()),
        "cost_max": float(cost.max()),
        # Cost stats for the SHIPPED features only — use these for the
        # color ramp so the gradient stretches across what's visible.
        "shown_cost_min": float(edge_costs.min()) if len(edge_costs) > 0 else 0.0,
        "shown_cost_max": float(edge_costs.max()) if len(edge_costs) > 0 else 0.0,
        "grid_deg": grid_deg,
        "subsampled": bool(will_subsample),
        "filtered_max_cost": float(max_cost) if max_cost is not None else None,
        "kept_count": len(features),
    }


def amenities_for(
    conn: psycopg.Connection, city_idx: int,
) -> dict | None:
    """Group POIs by category for the anchor's spatial footprint.

    Footprint = admin polygon when present (`anchors.geom_boundary`),
    otherwise a 1 km bbox around the place=city|town node. Reads
    pois.sqlite via SpatiaLite for the actual POI lookup.
    """
    anchor_id = city_idx + 1
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.name,
                   ST_AsText(a.geom_boundary),
                   ST_X(a.geom), ST_Y(a.geom),
                   a.geom_boundary IS NOT NULL
            FROM anchors a
            WHERE a.id = %s
        """, (anchor_id,))
        row = cur.fetchone()
    if not row:
        return None
    name, poly_wkt, lon, lat, has_polygon = row

    sqlite_conn = sqlite3.connect(str(POIS_DB))
    try:
        sqlite_conn.enable_load_extension(True)
        sqlite_conn.load_extension("mod_spatialite")
        sqlite_conn.enable_load_extension(False)

        if has_polygon and poly_wkt:
            # SpatialIndex prefilter via the polygon's envelope, then
            # exact Within() check. The two GeomFromText calls share a
            # query plan; SpatiaLite parses the WKT each time but for
            # admin-level polygons (a few hundred vertices) this is ms.
            rows = sqlite_conn.execute("""
                SELECT category, subtype, name
                FROM pois
                WHERE ROWID IN (
                    SELECT ROWID FROM SpatialIndex
                    WHERE f_table_name='pois'
                      AND search_frame=Envelope(GeomFromText(?, 4326))
                )
                AND Within(geom, GeomFromText(?, 4326))
            """, (poly_wkt, poly_wkt)).fetchall()
        else:
            # ~1 km bbox in degrees — generous enough at 56°N
            # (1 km ≈ 0.009° lat, ~0.016° lon at that latitude). Going
            # 0.015° each way overscans a touch in lat but keeps
            # behavior stable across the corridor.
            d = 0.015
            rows = sqlite_conn.execute("""
                SELECT category, subtype, name
                FROM pois
                WHERE ROWID IN (
                    SELECT ROWID FROM SpatialIndex
                    WHERE f_table_name='pois'
                      AND search_frame=BuildMbr(?,?,?,?,4326)
                )
            """, (lon - d, lat - d, lon + d, lat + d)).fetchall()
    finally:
        sqlite_conn.close()

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for cat, sub, poi_name in rows:
        by_cat[cat].append({
            "subtype": sub,
            "name": poi_name or "(unnamed)",
        })

    return {
        "city_idx": city_idx,
        "name": name,
        "has_polygon": bool(has_polygon),
        "footprint": "polygon" if has_polygon else "1km_bbox",
        "by_category": {
            cat: {
                "count": len(items),
                "samples": items[:8],
            }
            for cat, items in sorted(by_cat.items())
        },
    }
