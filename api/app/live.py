"""Live read-through to the pgRouting tables.

These endpoints work *during* the multi-source wave loop — they query
Postgres directly rather than reading the legacy npz artifacts the
older API code expected. As the Bellman-Ford waves converge, the
gradient endpoint shows cells expanding outward in real time.

  GET /live/cities                 — anchor list (id, name, lon, lat, snap_vertex_id)
  GET /live/cell/{city_idx}/gradient
                                   — cost-from-anchor across the cell's
                                     road network as one LineString per
                                     OSM way, colored by cost-at-target.
                                     Returns the full network when the
                                     cell is small; subsamples at the
                                     way-id level for huge cells.
"""
from collections import defaultdict

import psycopg

# Soft cap on the number of OSM ways included in the response. Below
# this threshold a cell ships its complete road network as connected
# LineStrings; above it we subsample by `osm_way_id % step`. ~10k
# LineStrings × ~10 coords each is comfortable for MapLibre on a
# laptop GPU.
MAX_WAYS = 10_000


def list_cities(
    conn: psycopg.Connection, profile: str | None = None,
) -> list[dict]:
    """Anchors with snap targets, ordered for stable city_idx mapping.
    `city_idx` matches the index used by the per-city SPT files
    (anchor.id - 1, since anchors.id is 1-based bigserial).

    If `profile` is given, filter to only the anchors that have a
    finished `<ci>.npz` on disk for that profile. Useful for the web UI
    so users don't see anchor pins they can't usefully click while the
    chainless preprocess is mid-flight.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, name, place, country,
                   ST_X(geom), ST_Y(geom),
                   snap_vertex_id
            FROM anchors
            WHERE snap_vertex_id IS NOT NULL
            ORDER BY id
        """)
        rows = cur.fetchall()

    if profile is None:
        ready_set: set[int] | None = None
    else:
        from .settings import SPT_DIR
        spt_dir = SPT_DIR / profile / "spt"
        ready_set = (
            {int(p.stem) for p in spt_dir.glob("*.npz")}
            if spt_dir.exists() else set()
        )

    out: list[dict] = []
    for r in rows:
        ci = int(r[0]) - 1
        if ready_set is not None and ci not in ready_set:
            continue
        out.append({
            "city_idx": ci,
            "name": r[1], "place": r[2], "country": r[3],
            "lon": float(r[4]), "lat": float(r[5]),
            "snap_vertex_id": int(r[6]),
        })
    return out


def gradient_for(conn: psycopg.Connection, city_idx: int) -> dict:
    """Return cost-gradient as a GeoJSON FeatureCollection of LineStrings.

    One LineString per OSM way (not per edge): we fetch all edges
    inside the cell, group by `osm_way_id` in Python, and emit one
    polyline that traces the way's geometry. Color is the maximum
    cost-from-anchor along the way, so the visualization fades from
    green (close) to red (far edge of the cell).

    For very large cells (say 50k+ ways) we sample by `osm_way_id %
    step`. Below that threshold we ship the full network so the road
    graph appears connected.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*), MIN(cost), MAX(cost) FROM visited WHERE city_id = %s",
            (city_idx,),
        )
        total, min_cost, max_cost = cur.fetchone()
        total = int(total or 0)
        if total == 0:
            return {
                "type": "FeatureCollection", "features": [],
                "city_idx": city_idx, "total_visited": 0,
            }

        # Approx way count from visited node count (~3 nodes per way
        # on average inside a cell). Subsample only when we'd otherwise
        # exceed MAX_WAYS LineStrings.
        approx_ways = max(1, total // 3)
        step = max(1, approx_ways // MAX_WAYS)

        # Order matters: rows for the same way must be consecutive and
        # ordered along the way's polyline so we can chain endpoint
        # coordinates into a single LineString. `gid` is monotonic per
        # way (pyosmium emits edges in node-order during ingest).
        cur.execute("""
            SELECT w.osm_way_id, w.gid,
                   ws.lon, ws.lat, wt.lon, wt.lat, vt.cost
            FROM ways w
            JOIN visited vs ON vs.vid = w.source AND vs.city_id = %s
            JOIN visited vt ON vt.vid = w.target AND vt.city_id = %s
            JOIN ways_vertices_pgr ws ON ws.id = w.source
            JOIN ways_vertices_pgr wt ON wt.id = w.target
            WHERE w.osm_way_id %% %s = 0
            ORDER BY w.osm_way_id, w.gid
        """, (city_idx, city_idx, step))
        rows = cur.fetchall()

    # One LineString per edge — *not* per OSM way. We tried merging
    # edges into per-way polylines but the source -> target chain
    # depends on gid ordering, and gid doesn't actually follow the
    # way's node order (the JOIN in _resolve_into_final reorders
    # rows). Drawing each edge as its own 2-point LineString avoids
    # the chain-reconstruction problem entirely; adjacent edges share
    # endpoints, so on the map they tile into continuous roads.
    features = [{
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [float(slon), float(slat)],
                [float(tlon), float(tlat)],
            ],
        },
        "properties": {"cost": float(cost)},
    } for (_osm_way_id, _gid, slon, slat, tlon, tlat, cost) in rows]

    return {
        "type": "FeatureCollection",
        "features": features,
        "city_idx": city_idx,
        "total_visited": total,
        "subsample_step": step,
        "cost_min": float(min_cost) if min_cost is not None else None,
        "cost_max": float(max_cost) if max_cost is not None else None,
    }
