"""All-pairs routing between 6 anchor cities, all 5 cost profiles.

Uses pgr_bdAstar (bidirectional A*) with a line-buffer corridor (a
geography-buffer of the great-circle line A↔B), not axis-aligned
bbox + pgr_dijkstra. Both swaps are deliberate:

  - **Bidirectional A*** explores a corridor around the optimal path
    instead of expanding outward in all directions; it touches roughly
    √2 fewer nodes than plain Dijkstra and uses straight-line distance
    as a heuristic so it walks toward the goal.

  - **Line-buffer corridor** filters edges by ST_DWithin to a 30 km
    band along the A↔B line, not a (typically huge) rectangular bbox
    that would also include unrelated terrain north and south. For
    Innsbruck↔Vienna this drops the loaded edge count from ~30M to
    ~8M and pgr_bdAstar's heap stays well under postgres's 10 GB cap.

The single ways.cost column is the cost the routing uses; the caller
runs `recompute-cost --profile <name>` between profiles in the outer
pipeline. This avoids needing per-profile cost columns for v1.
"""
from __future__ import annotations
import time
from dataclasses import dataclass

import psycopg


# Six official Austrian cities (place=city in OSM). Lon, lat.
CITIES: dict[str, tuple[float, float]] = {
    "INN": (11.3927685, 47.2654296),   # Innsbruck
    "SAL": (13.0464806, 47.7981346),   # Salzburg
    "LIN": (14.2861980, 48.3059078),   # Linz
    "WIE": (16.3725042, 48.2083537),   # Wien
    "GRA": (15.4382786, 47.0708678),   # Graz
    "KLA": (14.3075976, 46.6239430),   # Klagenfurt
}

# Half-width of the corridor (meters) around the great-circle line
# between two cities.
#
# 30 km was the original default — small enough to keep the loaded
# edge count under ~10M for INN-WIE — but it produced "no path found"
# for INN-SAL because the natural cyclable route runs via the Pinzgau
# valley (~47.3°N), which sits ~35 km south of the line INN→SAL at its
# midpoint (the line trends NE toward Salzburg while the cyclable
# route detours south to avoid the Bavarian Alps and Inn-valley
# cross-border edges we don't have in our Austria-only ways table).
#
# 50 km is wide enough to include the southern detour routes for every
# 6-city pair. INN-WIE corridor grows from ~7M to ~11M edges, still
# under the 10 GB postgres cap.
CORRIDOR_M_DEFAULT = 50_000.0

# All 15 unordered city pairs.
def all_pairs() -> list[tuple[str, str]]:
    keys = list(CITIES.keys())
    return [(a, b) for i, a in enumerate(keys) for b in keys[i+1:]]


@dataclass
class RouteResult:
    a: str
    b: str
    profile: str
    length_km: float
    cost: float
    n_edges: int
    geom_geojson: str   # full LineString geometry as a GeoJSON dict string


def _snap_vertex(conn: psycopg.Connection, lon: float, lat: float) -> int:
    """Nearest ways_vertices_pgr.id to the given lon/lat that lies on
    the **bike-routable** road network.

    Plain nearest-neighbor snapped Salzburg's city coord to a 5 m
    pedestrian pebblestone deadend (vid 21205025) — an orphan piece
    of the graph not connected to the cyclable network. pgr_bdAstar
    then returned "no path found" for every pair involving SAL.

    Fix: require the snapped vertex to touch at least one edge whose
    highway tag is bike-routable. The exclusion list covers the
    common foot/transit-only types; `path` and `track` are KEPT
    because they're legitimately bike-routable in Austria. Order by
    nearest within that set.
    """
    EXCLUDED = ("pedestrian", "footway", "steps", "platform", "corridor",
                "elevator", "escalator")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT v.id FROM ways_vertices_pgr v "
            "WHERE EXISTS ("
            "  SELECT 1 FROM ways w "
            "  WHERE (w.source = v.id OR w.target = v.id) "
            "    AND w.highway <> ALL(%s::text[])"
            ") "
            "ORDER BY v.the_geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326) "
            "LIMIT 1",
            (list(EXCLUDED), lon, lat),
        )
        return int(cur.fetchone()[0])


def route_one_pair(conn: psycopg.Connection,
                   a: str, b: str, profile: str,
                   corridor_m: float = CORRIDOR_M_DEFAULT,
                   ) -> RouteResult | None:
    """Run a single pgr_bdAstar pair on the current ways.cost column."""
    a_lon, a_lat = CITIES[a]
    b_lon, b_lat = CITIES[b]
    t0 = time.time()

    # Pre-flight: edge count in corridor, for the log.
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        # Keep work_mem moderate: pgr_bdAstar's bounded heap is the
        # real memory consumer; we don't want a wide sort spill on top.
        cur.execute("SET work_mem = '256MB'")
        cur.execute(
            "SELECT COUNT(*) FROM ways w "
            "JOIN ways_vertices_pgr vs ON vs.id = w.source "
            "WHERE ST_DWithin("
            "  vs.the_geom::geography, "
            "  ST_SetSRID(ST_MakeLine("
            "    ST_MakePoint(%s, %s), ST_MakePoint(%s, %s)"
            "  ), 4326)::geography, %s)",
            (a_lon, a_lat, b_lon, b_lat, corridor_m),
        )
        n_corridor = int(cur.fetchone()[0])
        print(f"[city-route] {a}->{b} profile={profile}: corridor has "
              f"{n_corridor:,} edges", flush=True)

        start_vid = _snap_vertex(conn, a_lon, a_lat)
        end_vid   = _snap_vertex(conn, b_lon, b_lat)

        # Edges sql: filter by source vertex inside the line-buffer
        # corridor; include the target vertex's lon/lat too so
        # pgr_bdAstar has x1/y1/x2/y2 for the heuristic. Edges leaving
        # the corridor are still included (because we only filter by
        # source vertex), so a path can reach the corridor boundary
        # smoothly without an artificial cliff.
        #
        # We SELECT the profile-specific cost columns (cost_<profile>,
        # reverse_cost_<profile>) and alias them as the standard `cost`
        # / `reverse_cost` names pgr_bdAstar expects. This is how we
        # support all 5 profiles in one ways table without overwriting
        # each profile on every recompute. The COALESCE guards against
        # NULL (edges not yet recomputed for this profile) by using a
        # very high cost — pgRouting needs non-NULL, and an unrouted
        # edge should be effectively unreachable rather than free.
        cost_col = f"cost_{profile}"
        rev_col  = f"reverse_cost_{profile}"
        edges_sql = (
            f"SELECT w.gid AS id, w.source, w.target, "
            f"       COALESCE(w.{cost_col}, 1e15)::double precision AS cost, "
            f"       COALESCE(w.{rev_col},  1e15)::double precision AS reverse_cost, "
            f"       vs.lon AS x1, vs.lat AS y1, "
            f"       vt.lon AS x2, vt.lat AS y2 "
            f"FROM ways w "
            f"JOIN ways_vertices_pgr vs ON vs.id = w.source "
            f"JOIN ways_vertices_pgr vt ON vt.id = w.target "
            f"WHERE ST_DWithin("
            f"  vs.the_geom::geography, "
            f"  ST_SetSRID(ST_MakeLine("
            f"    ST_MakePoint({a_lon}, {a_lat}), "
            f"    ST_MakePoint({b_lon}, {b_lat})"
            f"  ), 4326)::geography, {corridor_m})"
        )
        sql_quoted = edges_sql.replace("'", "''")

        cur.execute(
            f"WITH rt AS ("
            f"  SELECT seq, node, edge, cost "
            f"  FROM pgr_bdAstar('{sql_quoted}', "
            f"                    {int(start_vid)}, {int(end_vid)}, "
            f"                    directed := true) "
            f"  WHERE edge != -1 "
            f"), edge_geoms AS ("
            f"  SELECT rt.seq, "
            f"         CASE WHEN w.source = rt.node "
            f"              THEN ST_MakeLine(vs.the_geom, vt.the_geom) "
            f"              ELSE ST_MakeLine(vt.the_geom, vs.the_geom) "
            f"         END AS g, "
            f"         w.length_m, rt.cost "
            f"  FROM rt "
            f"  JOIN ways w ON w.gid = rt.edge "
            f"  JOIN ways_vertices_pgr vs ON vs.id = w.source "
            f"  JOIN ways_vertices_pgr vt ON vt.id = w.target "
            f") "
            f"SELECT ST_AsGeoJSON(ST_MakeLine(g ORDER BY seq)), "
            f"       SUM(length_m), SUM(cost), COUNT(*) "
            f"FROM edge_geoms"
        )
        row = cur.fetchone()
        if row is None or row[0] is None:
            print(f"[city-route] {a}->{b}: no path found", flush=True)
            return None
        geom_json, length_m, total_cost, n_edges = row
        wall = time.time() - t0
        print(f"[city-route] {a}->{b}: {length_m/1000:.1f} km, "
              f"{int(n_edges):,} edges, cost={float(total_cost):.0f}, "
              f"in {wall:.1f}s", flush=True)
        return RouteResult(
            a=a, b=b, profile=profile,
            length_km=float(length_m) / 1000.0,
            cost=float(total_cost),
            n_edges=int(n_edges),
            geom_geojson=str(geom_json),
        )


def route_all_pairs(conn: psycopg.Connection,
                    profile: str,
                    corridor_m: float = CORRIDOR_M_DEFAULT,
                    pairs: list[tuple[str, str]] | None = None,
                    ) -> list[RouteResult]:
    """Run all 15 pairs (or a subset) on the current ways.cost column."""
    if pairs is None:
        pairs = all_pairs()
    results: list[RouteResult] = []
    for a, b in pairs:
        r = route_one_pair(conn, a, b, profile, corridor_m=corridor_m)
        if r is not None:
            results.append(r)
    return results
