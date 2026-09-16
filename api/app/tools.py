"""Bike-planner tool implementations — shared by the FastAPI chat
endpoint (`api.app.chat`) and the stdio MCP server
(`mcp_server.mcp_bike_planner`).

Every tool is:
  * Pure — takes an `input` dict, returns an `output` dict.
  * Side-effect-free from the caller's perspective (may populate the
    in-process route cache, but that's a within-process optimisation).
  * JSON-safe — inputs and outputs are plain dicts/lists/primitives.

Layout:
  TOOLS       — JSON-schema tool descriptors for the Anthropic
                messages.stream API (also picked up by the MCP server).
  TOOL_IMPLS  — dict mapping tool name → Python callable.
  POI_CATEGORIES — enum-like tuple of allowed POI category values.

The shared route cache (`_ROUTE_CACHE`) is process-local. A route
computed via a chat tool call and then queried via `pois_along_route`
in the same process reuses the polyline; across processes, the
`pois_along_route` call recomputes the route lazily.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import pois, trunk_router
from .settings import DEFAULT_PROFILE, DATA_DIR


POI_CATEGORIES = ("food", "viewpoint", "lodging", "water", "bike_service")
STATIONS_PATH = Path(DATA_DIR) / "rail_stations.geojson"
R_EARTH_M = 6_371_000.0


# ---------------------------------------------------------------------------
# Tool schemas — the shape both Claude (via the messages API) and MCP
# clients see. Kept as plain Python data so it's trivially serialisable.

TOOLS = [
    {
        "name": "search_anchors",
        "description": (
            "Find bike-tour anchor cities/towns by name (case-insensitive "
            "substring). Returns ref, name, country, population, lon, lat. "
            "Use this to resolve place names before calling `route`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Substring of the anchor's name"},
                "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 30},
            },
            "required": ["query"],
        },
    },
    {
        "name": "route",
        "description": (
            "Compute a bike route between two anchors or lon,lat coords. "
            "Returns the polyline, total distance in km, and per-anchor "
            "waypoints. Prefer passing anchor `ref` values (from "
            "`search_anchors`) over raw coords when possible."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "from_ref": {"type": "string", "description": "Anchor ref, e.g. 'db:2604'"},
                "to_ref":   {"type": "string", "description": "Anchor ref"},
                "from_lonlat": {"type": "string", "description": "Fallback: 'lon,lat'"},
                "to_lonlat":   {"type": "string", "description": "Fallback: 'lon,lat'"},
                "via_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Intermediate stops in order (each is an anchor ref).",
                },
            },
        },
    },
    {
        "name": "stations_near",
        "description": (
            "List rail stations within `radius_km` of (lon, lat). Sorted "
            "by distance ascending. Each result includes name, "
            "n_routes_rail (rail-classified routes serving the stop), "
            "n_routes_bus (bus-classified routes at the same stop — some "
            "national feeds mis-label a subset of bus routes as rail, "
            "so a stop with n_routes_rail >> n_routes_bus is a much "
            "more confident 'real train station' signal than raw "
            "n_routes alone), n_routes (total, kept for backwards "
            "compat), and distance_km. Use to find where a partner could "
            "arrive/depart by train near a day's overnight. Coverage: "
            "Austria (ÖBB), Germany (DB + regional), Denmark "
            "(Rejseplanen), Czech Republic (tangero national aggregate). "
            "PREFER `n_routes_rail >= 2` as the rail-accessible threshold."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lon": {"type": "number"},
                "lat": {"type": "number"},
                "radius_km": {"type": "number", "default": 15, "minimum": 1, "maximum": 100},
                "limit": {"type": "integer", "default": 8, "minimum": 1, "maximum": 30},
            },
            "required": ["lon", "lat"],
        },
    },
    {
        "name": "pois_near_anchor",
        "description": (
            "List points-of-interest (OSM) within `radius_km` of an anchor "
            "city, filtered by category. Categories: food, viewpoint, "
            "lodging, water, bike_service. Use this to enrich an overnight "
            "stop with nearby lodging or attractions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Anchor ref, e.g. 'db:224'"},
                "category": {
                    "type": "string",
                    "enum": list(POI_CATEGORIES),
                },
                "radius_km": {"type": "number", "default": 5, "minimum": 0.2, "maximum": 30},
                "limit": {"type": "integer", "default": 15, "minimum": 1, "maximum": 100},
            },
            "required": ["ref", "category"],
        },
    },
    {
        "name": "pois_along_route",
        "description": (
            "List points-of-interest (OSM) within `buffer_km` of the LAST "
            "computed route's polyline, filtered by category. Reads the "
            "polyline from the cached route (same one `split_into_stages` "
            "uses) so you don't need to pass it. Categories: food, viewpoint, "
            "lodging, water, bike_service. Use this to enrich a leg with "
            "scenic detours or refreshment stops."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "from_ref": {"type": "string", "description": "Same as `route`'s from_ref (used to look up the cached polyline)"},
                "to_ref":   {"type": "string", "description": "Same as `route`'s to_ref"},
                "via_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Same `via_refs` you used with `route` (for cache match).",
                },
                "category": {
                    "type": "string",
                    "enum": list(POI_CATEGORIES),
                },
                "buffer_km": {"type": "number", "default": 3, "minimum": 0.2, "maximum": 20},
                "limit": {"type": "integer", "default": 30, "minimum": 1, "maximum": 200},
            },
            "required": ["from_ref", "to_ref", "category"],
        },
    },
    {
        "name": "stations_along_route",
        "description": (
            "Rail-accessible ANCHOR CITIES along the last-computed "
            "route's polyline. Each result: anchor `ref`, `name`, "
            "`lon`, `lat`, `km_along_route` (cumulative km from route "
            "start to the anchor's closest polyline vertex), and "
            "`stations` (top rail stations within `station_radius_km` "
            "of that anchor, each with n_routes_rail / n_routes_bus / "
            "distance_km). Only anchors that have at least one station "
            "with n_routes_rail >= `min_routes_rail` (default 2) are "
            "returned. Reads the cached polyline from the last matching "
            "`route(from_ref, to_ref, via_refs)` call. Use this INSTEAD "
            "of calling `stations_near` per candidate town when you "
            "need to pick rail-accessible overnights across a "
            "multi-day plan — one call replaces N * stations_near "
            "round-trips."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "from_ref": {"type": "string"},
                "to_ref":   {"type": "string"},
                "via_refs": {
                    "type": "array", "items": {"type": "string"},
                },
                "station_radius_km": {
                    "type": "number", "default": 5, "minimum": 0.5, "maximum": 25,
                },
                "min_routes_rail": {
                    "type": "integer", "default": 2, "minimum": 1, "maximum": 50,
                    "description": "Minimum n_routes_rail on the best-served station near each anchor. 2 is a solid 'has actual train service' threshold given the per-mode split from task #78.",
                },
                "corridor_km": {
                    "type": "number", "default": 8, "minimum": 1, "maximum": 30,
                    "description": "Only consider anchors within this many km of the polyline (avoids picking cities that share the corridor but aren't actually near it).",
                },
                "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
            },
            "required": ["from_ref", "to_ref"],
        },
    },
    {
        "name": "direct_rail_service",
        "description": (
            "USE THIS whenever the user asks about a DIRECT, "
            "NON-TRANSFER, ONE-SEAT, or SINGLE-CHANGE train "
            "connection between two anchors. `stations_near` and "
            "`n_routes_rail` only tell you a station EXISTS at a "
            "given place — they do NOT prove a direct train runs from "
            "there to any specific corridor hub. If the user's prompt "
            "asks for rail-accessible overnights reachable by direct "
            "train, call this tool for EACH candidate overnight paired "
            "with the relevant hub (Graz, Copenhagen, or an "
            "intermediate corridor hub like Wien, Praha, Berlin, "
            "Hamburg) — do not infer direct service from route counts.\n\n"
            "Behavior: finds the closest rail station to each anchor's "
            "center (within `max_station_dist_km`) that has at least "
            "`min_routes_rail` GTFS rail routes, and intersects their "
            "route_id sets. Returns `direct_service` (bool), "
            "`n_shared_routes`, the two matched stations, and up to 20 "
            "shared route_ids. Coverage: same 4 national GTFS feeds as "
            "`stations_near` (Austria, Germany, Denmark, Czech "
            "Republic).\n\n"
            "Caveat: only detects direct service that both operates in "
            "the same GTFS feed AND uses the same route_id. Cross-feed "
            "pairs (Wien→Praha, Berlin→København) may legitimately "
            "return False even when a real Railjet/EuroCity crosses "
            "them, because the operators file the service under "
            "different route_ids in each national feed. Prefer running "
            "the check inside a single country when possible; treat a "
            "False on a cross-feed pair as 'probably needs a transfer' "
            "rather than a hard proof."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "from_ref": {"type": "string", "description": "First anchor ref"},
                "to_ref":   {"type": "string", "description": "Second anchor ref"},
                "max_station_dist_km": {
                    "type": "number", "default": 5, "minimum": 0.5, "maximum": 25,
                    "description": "Only consider a station reachable from the anchor if it's within this many km.",
                },
                "min_routes_rail": {
                    "type": "integer", "default": 1, "minimum": 1, "maximum": 50,
                    "description": "Skip stops with fewer than this many rail routes. 1 keeps the check permissive (any rail service); raise to 2+ to require a busier station.",
                },
            },
            "required": ["from_ref", "to_ref"],
        },
    },
    {
        "name": "split_into_stages",
        "description": (
            "Split a route into daily stages of roughly `target_km_per_day`. "
            "Each stage ends at the anchor nearest the km-target on the "
            "polyline (so the tour terminates at real towns, not arbitrary "
            "coordinates). Returns a list of stages: "
            "[{day, from_ref, to_ref, km, from_lonlat, to_lonlat}]. "
            "Pass the same from_ref/to_ref you used with `route`; this tool "
            "reads the last-computed polyline internally so you never need "
            "to carry it around."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "from_ref": {"type": "string", "description": "Route start anchor ref"},
                "to_ref":   {"type": "string", "description": "Route end anchor ref"},
                "via_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The SAME via_refs you passed to `route`, in the same order. Required for the cache lookup to match.",
                },
                "target_km_per_day": {"type": "number", "minimum": 20, "maximum": 300},
            },
            "required": ["from_ref", "to_ref", "target_km_per_day"],
        },
    },
]


# ---------------------------------------------------------------------------
# Anchor name aliasing — English → local-language variants so a user
# query in either language resolves. `search_anchors` auto-tries these.

CITY_ALIASES: dict[str, list[str]] = {
    "copenhagen": ["københavn", "koebenhavn"],
    "københavn":  ["copenhagen"],
    "vienna":     ["wien"],
    "wien":       ["vienna"],
    "prague":     ["praha"],
    "praha":      ["prague"],
    "munich":     ["münchen", "muenchen"],
    "münchen":    ["munich"],
    "cologne":    ["köln", "koeln"],
    "köln":       ["cologne"],
    "nuremberg":  ["nürnberg", "nuernberg"],
    "nürnberg":   ["nuremberg"],
    "aarhus":     ["århus"],
    "århus":      ["aarhus"],
    "gothenburg": ["göteborg"],
    "göteborg":   ["gothenburg"],
    "warsaw":     ["warszawa"],
    "warszawa":   ["warsaw"],
    "brno":       [],
    "salzburg":   [],
    "graz":       [],
    "linz":       [],
    "berlin":     [],
    "hamburg":    [],
    "dresden":    [],
    "leipzig":    [],
    "hannover":   [],
    "bremen":     [],
    "flensburg":  [],
    "odense":     [],
}


def _query_variants(q: str) -> list[str]:
    """Expand a place-name query with any known local-language variants."""
    qn = q.strip().lower()
    if not qn:
        return []
    variants = [qn]
    for extra in CITY_ALIASES.get(qn, []):
        if extra not in variants:
            variants.append(extra)
    return variants


def _hav_m(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R_EARTH_M * math.asin(math.sqrt(a))


# Per-process cache of computed routes so tools like split_into_stages,
# pois_along_route, and stations_along_route can read a polyline
# without the LLM having to carry it. Keyed on the canonical
# (from_ref, to_ref, via_refs_tuple) triple.
_ROUTE_CACHE: dict[tuple, list[list[float]]] = {}


def _route_cache_key(inp: dict) -> tuple:
    return (
        inp.get("from_ref") or inp.get("from_lonlat"),
        inp.get("to_ref")   or inp.get("to_lonlat"),
        tuple(inp.get("via_refs") or []),
    )


# ---------------------------------------------------------------------------
# Tool implementations.

def _tool_search_anchors(inp: dict) -> dict:
    prof = trunk_router._load_profile(DEFAULT_PROFILE)
    q = (inp.get("query") or "").strip()
    limit = int(inp.get("limit", 10))
    variants = _query_variants(q)
    if not variants:
        ranked = sorted(prof.cities, key=lambda c: -(c.get("population") or 0))[:limit]
    else:
        prefix, contains = [], []
        seen: set[str] = set()
        for c in prof.cities:
            nm = (c.get("name") or "").lower()
            if not nm or c.get("ref") in seen:
                continue
            for v in variants:
                if nm.startswith(v):
                    prefix.append(c); seen.add(c["ref"]); break
                elif v in nm:
                    contains.append(c); seen.add(c["ref"]); break
        prefix.sort(key=lambda c: -(c.get("population") or 0))
        contains.sort(key=lambda c: -(c.get("population") or 0))
        ranked = (prefix + contains)[:limit]
    return {
        "results": [
            {
                "ref": c.get("ref"),
                "name": c.get("name"),
                "country": c.get("country"),
                "population": c.get("population"),
                "lon": c.get("lon"),
                "lat": c.get("lat"),
            }
            for c in ranked
        ],
        "query_variants_tried": variants,
    }


def _tool_route(inp: dict) -> dict:
    """Call trunk_router.route() and flatten the response for the LLM."""
    kwargs: dict[str, Any] = {"profile": DEFAULT_PROFILE,
                              "start": None, "end": None}
    if inp.get("from_ref"):
        kwargs["start_ref"] = inp["from_ref"]
    elif inp.get("from_lonlat"):
        lon, lat = (float(x) for x in inp["from_lonlat"].split(","))
        kwargs["start"] = (lon, lat)
    else:
        return {"error": "Provide from_ref or from_lonlat"}
    if inp.get("to_ref"):
        kwargs["end_ref"] = inp["to_ref"]
    elif inp.get("to_lonlat"):
        lon, lat = (float(x) for x in inp["to_lonlat"].split(","))
        kwargs["end"] = (lon, lat)
    else:
        return {"error": "Provide to_ref or to_lonlat"}
    if inp.get("via_refs"):
        kwargs["via"] = [(r, None) for r in inp["via_refs"]]
    try:
        result = trunk_router.route(**kwargs)
    except Exception as e:
        return {"error": f"route failed: {e}"}

    # trunk_router.route() returns a GeoJSON Feature. Distill it into
    # something small enough for the LLM to reason about AND rich enough
    # for the frontend to redraw.
    props = result.get("properties", {}) if isinstance(result, dict) else {}
    geom = result.get("geometry", {}) if isinstance(result, dict) else {}
    coords = geom.get("coordinates", []) if geom.get("type") == "LineString" else []
    total_km = (props.get("gross_length_m") or 0) / 1000.0
    _ROUTE_CACHE[_route_cache_key(inp)] = coords
    # Enriched chain metadata: each hop's name + population + country + kind
    # so the model can pick "major cities" (population filter) or reason
    # about country transitions without extra search_anchors calls.
    prof = trunk_router._load_profile(DEFAULT_PROFILE)
    chain_stops = []
    for name, ci in zip(props.get("chain_names", []),
                        props.get("chain_city_idx", [])):
        c = prof.cities[int(ci)] if 0 <= int(ci) < len(prof.cities) else {}
        chain_stops.append({
            "name": name,
            "ref": c.get("ref"),
            "population": c.get("population"),
            "country": c.get("country"),
            "kind": c.get("place"),
        })
    return {
        "total_km": round(total_km, 1),
        "polyline": coords,
        "chain_stops": chain_stops,
        "chain_names":    props.get("chain_names", []),
        "chain_city_idx": list(props.get("chain_city_idx", [])),
        "n_bridges": len(props.get("bridges", [])),
    }


def _stations_kdtree():
    """Cached scipy.spatial.cKDTree over station lon/lat, plus the
    parallel list of station dicts. Both are None if no stations
    are loaded yet or if scipy isn't available.
    Returns (tree, stations_list) — both indexed the same way."""
    if not hasattr(_stations_kdtree, "_cache"):
        stations = _load_stations_cache()
        if not stations:
            _stations_kdtree._cache = (None, [])
        else:
            try:
                import numpy as np
                from scipy.spatial import cKDTree
                arr = np.array([(s["lon"], s["lat"]) for s in stations],
                               dtype=np.float64)
                _stations_kdtree._cache = (cKDTree(arr), stations)
            except ImportError:
                _stations_kdtree._cache = (None, stations)
    return _stations_kdtree._cache


def _load_stations_cache() -> list[dict]:
    if not STATIONS_PATH.exists():
        return []
    if not hasattr(_load_stations_cache, "_cache"):
        fc = json.loads(STATIONS_PATH.read_text())
        _load_stations_cache._cache = [
            {
                "name": f["properties"].get("name"),
                # `n_routes_rail` is the reliable "has real train service"
                # signal after task #78; prefer it over raw `n_routes` in
                # any filtering.
                "n_routes":      f["properties"].get("n_routes", 0),
                "n_routes_rail": f["properties"].get("n_routes_rail",
                                                    f["properties"].get("n_routes", 0)),
                "n_routes_bus":  f["properties"].get("n_routes_bus", 0),
                "n_lodging":     f["properties"].get("n_lodging", 0),
                # country + gtfs_id are the join keys into the route-list
                # sidecar (`rail_station_routes.json`) used by the
                # `direct_rail_service` tool.
                "country":       f["properties"].get("country"),
                "gtfs_id":       f["properties"].get("gtfs_id"),
                "lon": f["geometry"]["coordinates"][0],
                "lat": f["geometry"]["coordinates"][1],
            }
            for f in fc.get("features", [])
        ]
    return _load_stations_cache._cache


STATION_ROUTES_PATH = Path(DATA_DIR) / "rail_station_routes.json"


def _load_station_routes_cache() -> dict[str, set[str]]:
    """Load the per-station rail-route-id sidecar emitted by
    `pgrouting/main.py export-rail-routes`. Keys are
    `<country>:<gtfs_id>`; values are sets of GTFS route_ids serving
    that station. Empty dict if the sidecar hasn't been produced yet."""
    if not hasattr(_load_station_routes_cache, "_cache"):
        if not STATION_ROUTES_PATH.exists():
            _load_station_routes_cache._cache = {}
        else:
            data = json.loads(STATION_ROUTES_PATH.read_text())
            _load_station_routes_cache._cache = {
                k: set(v) for k, v in data.items()
            }
    return _load_station_routes_cache._cache


def _stations_for_anchor(alon: float, alat: float,
                          max_dist_m: float,
                          min_routes_rail: int) -> list[tuple[float, dict]]:
    """All rail stations near (alon, alat) meeting `min_routes_rail`,
    within `max_dist_m`, sorted by distance. Empty list if none.

    We return ALL nearby stations (not just the closest) so callers
    like `direct_rail_service` can union the route_id sets — GTFS feed
    dedup often misses sibling platforms and multi-terminal cities
    (Wien Hbf vs Wien Mitte vs Praterstern) that a passenger would
    consider interchangeable for "is this reachable by direct train"
    purposes."""
    stree, stations = _stations_kdtree()
    hits: list[tuple[float, dict]] = []
    if stree is None:
        for s in stations:
            if s.get("n_routes_rail", 0) < min_routes_rail:
                continue
            d = _hav_m(alon, alat, s["lon"], s["lat"])
            if d <= max_dist_m:
                hits.append((d, s))
    else:
        _MDEGLAT = 111_320.0
        _MDEGLON = 111_320.0 * math.cos(math.radians(alat))
        _min_mdeg = min(_MDEGLAT, _MDEGLON)
        radius_deg = max_dist_m / _min_mdeg
        for i in stree.query_ball_point([alon, alat], r=radius_deg):
            s = stations[i]
            if s.get("n_routes_rail", 0) < min_routes_rail:
                continue
            d = _hav_m(alon, alat, s["lon"], s["lat"])
            if d <= max_dist_m:
                hits.append((d, s))
    hits.sort(key=lambda h: h[0])
    return hits


def _tool_stations_near(inp: dict) -> dict:
    lon, lat = float(inp["lon"]), float(inp["lat"])
    radius_m = float(inp.get("radius_km", 15)) * 1000.0
    limit = int(inp.get("limit", 8))
    hits = []
    for s in _load_stations_cache():
        d = _hav_m(lon, lat, s["lon"], s["lat"])
        if d <= radius_m:
            hits.append({**s, "distance_km": round(d / 1000.0, 1)})
    hits.sort(key=lambda h: h["distance_km"])
    return {"stations": hits[:limit]}


def _tool_split_into_stages(inp: dict) -> dict:
    """Walk the polyline, cut it into ~target_km chunks, snap each cut
    point to the nearest anchor. Returns stage boundaries. Reads the
    polyline from the last matching `route` call; the LLM only passes
    the anchor refs it already knows."""
    from_ref = inp.get("from_ref")
    to_ref = inp.get("to_ref")
    via_refs = tuple(inp.get("via_refs") or [])
    target_km = float(inp.get("target_km_per_day", 80))
    poly = _ROUTE_CACHE.get((from_ref, to_ref, via_refs))
    if not poly:
        for (fr, tr, _via), coords in _ROUTE_CACHE.items():
            if fr == from_ref and tr == to_ref:
                poly = coords
                break
    if not poly:
        route_result = _tool_route({
            "from_ref": from_ref,
            "to_ref": to_ref,
            "via_refs": list(via_refs) if via_refs else None,
        })
        poly = route_result.get("polyline") or []
    if len(poly) < 2:
        return {"error": "Could not compute a route for that (from_ref, to_ref)."}
    cum = [0.0]
    for i in range(1, len(poly)):
        cum.append(cum[-1] + _hav_m(poly[i-1][0], poly[i-1][1],
                                     poly[i][0], poly[i][1]))
    total_km = cum[-1] / 1000.0
    n_days = max(1, round(total_km / target_km))
    cut_km = [total_km * (i + 1) / n_days for i in range(n_days - 1)]

    prof = trunk_router._load_profile(DEFAULT_PROFILE)
    anchors = [(c.get("lon"), c.get("lat"), c.get("name"), c.get("ref"))
               for c in prof.cities
               if c.get("lon") is not None and c.get("lat") is not None]

    def _nearest_anchor(lon, lat):
        best = None; best_d = float("inf")
        for alon, alat, aname, aref in anchors:
            d = _hav_m(lon, lat, alon, alat)
            if d < best_d:
                best_d = d; best = (aname, aref, alon, alat, d)
        return best

    def _point_at_km(km_target):
        target_m = km_target * 1000.0
        for i in range(1, len(cum)):
            if cum[i] >= target_m:
                seg = cum[i] - cum[i-1] or 1e-9
                t = (target_m - cum[i-1]) / seg
                lon = poly[i-1][0] + t * (poly[i][0] - poly[i-1][0])
                lat = poly[i-1][1] + t * (poly[i][1] - poly[i-1][1])
                return lon, lat
        return poly[-1][0], poly[-1][1]

    def _km_at_nearest_polyline_vertex(alon, alat) -> float:
        best_i = 0; best_d = float("inf")
        for i, (px, py) in enumerate(poly):
            d = (px - alon) ** 2 + (py - alat) ** 2
            if d < best_d:
                best_d = d; best_i = i
        return cum[best_i] / 1000.0

    stages = []
    prev_anchor = _nearest_anchor(poly[0][0], poly[0][1])
    prev_km_actual = _km_at_nearest_polyline_vertex(prev_anchor[2], prev_anchor[3]) \
                     if prev_anchor else 0.0
    for day, km_at in enumerate(cut_km + [total_km], start=1):
        target_pt = _point_at_km(km_at)
        cur_anchor = _nearest_anchor(*target_pt)
        cur_km_actual = _km_at_nearest_polyline_vertex(cur_anchor[2], cur_anchor[3]) \
                        if cur_anchor else km_at
        leg_km = max(0.0, cur_km_actual - prev_km_actual)
        stages.append({
            "day": day,
            "from_ref": prev_anchor[1] if prev_anchor else None,
            "from_name": prev_anchor[0] if prev_anchor else None,
            "from_lonlat": [prev_anchor[2], prev_anchor[3]] if prev_anchor else None,
            "to_ref": cur_anchor[1] if cur_anchor else None,
            "to_name": cur_anchor[0] if cur_anchor else None,
            "to_lonlat": [cur_anchor[2], cur_anchor[3]] if cur_anchor else list(target_pt),
            "km": round(leg_km, 1),
        })
        prev_anchor = cur_anchor
        prev_km_actual = cur_km_actual
    return {"total_km": round(total_km, 1), "n_days": n_days, "stages": stages}


def _tool_pois_near_anchor(inp: dict) -> dict:
    ref = inp.get("ref")
    category = inp.get("category")
    radius_km = float(inp.get("radius_km", 5))
    limit = int(inp.get("limit", 15))
    if not ref or not category:
        return {"error": "ref and category are required"}
    prof = trunk_router._load_profile(DEFAULT_PROFILE)
    ci = prof.city_idx_by_ref.get(ref)
    if ci is None:
        return {"error": f"unknown anchor ref: {ref}"}
    c = prof.cities[int(ci)]
    lon, lat = float(c["lon"]), float(c["lat"])
    radius_m = radius_km * 1000.0
    r_deg_lat = radius_km / 111.0
    r_deg_lon = r_deg_lat / max(math.cos(math.radians(lat)), 0.1)
    bbox = (lon - r_deg_lon, lat - r_deg_lat, lon + r_deg_lon, lat + r_deg_lat)
    raw = pois.query_bbox(bbox, [category], limit * 3)
    hits = []
    for p in raw:
        d = _hav_m(lon, lat, float(p["lon"]), float(p["lat"]))
        if d <= radius_m:
            hits.append({
                "name": p.get("name") or f"{p.get('category')}:{p.get('subtype') or '?'}",
                "subtype": p.get("subtype"),
                "lon": float(p["lon"]),
                "lat": float(p["lat"]),
                "distance_km": round(d / 1000.0, 2),
            })
    hits.sort(key=lambda h: h["distance_km"])
    return {"pois": hits[:limit], "anchor": {"ref": ref, "name": c.get("name"),
                                              "lon": lon, "lat": lat}}


def _tool_pois_along_route(inp: dict) -> dict:
    from_ref = inp.get("from_ref")
    to_ref = inp.get("to_ref")
    via_refs = tuple(inp.get("via_refs") or [])
    category = inp.get("category")
    buffer_km = float(inp.get("buffer_km", 3))
    limit = int(inp.get("limit", 30))
    if not category:
        return {"error": "category is required"}
    poly = _ROUTE_CACHE.get((from_ref, to_ref, via_refs))
    if not poly:
        for (fr, tr, _via), coords in _ROUTE_CACHE.items():
            if fr == from_ref and tr == to_ref:
                poly = coords
                break
    if not poly:
        route_result = _tool_route({
            "from_ref": from_ref, "to_ref": to_ref,
            "via_refs": list(via_refs) if via_refs else None,
        })
        poly = route_result.get("polyline") or []
    if len(poly) < 2:
        return {"error": "no cached route for that endpoint pair; call `route` first"}
    lons = [p[0] for p in poly]
    lats = [p[1] for p in poly]
    lat_mid = (min(lats) + max(lats)) / 2
    r_deg_lat = buffer_km / 111.0
    r_deg_lon = r_deg_lat / max(math.cos(math.radians(lat_mid)), 0.1)
    bbox = (min(lons) - r_deg_lon, min(lats) - r_deg_lat,
            max(lons) + r_deg_lon, max(lats) + r_deg_lat)
    raw = pois.query_bbox(bbox, [category], min(limit * 20, 2000))
    buffer_m = buffer_km * 1000.0

    def _nearest_seg_dist_m(plon, plat):
        best = float("inf")
        for i in range(len(poly)):
            d = _hav_m(plon, plat, poly[i][0], poly[i][1])
            if d < best:
                best = d
        return best

    hits = []
    for p in raw:
        d = _nearest_seg_dist_m(float(p["lon"]), float(p["lat"]))
        if d <= buffer_m:
            hits.append({
                "name": p.get("name") or f"{p.get('category')}:{p.get('subtype') or '?'}",
                "subtype": p.get("subtype"),
                "lon": float(p["lon"]),
                "lat": float(p["lat"]),
                "distance_km": round(d / 1000.0, 2),
            })
    hits.sort(key=lambda h: h["distance_km"])
    return {"pois": hits[:limit], "n_matched_full_buffer": len(hits)}


def _tool_stations_along_route(inp: dict) -> dict:
    """Batched 'which chain anchors along this route are rail-served?'
    lookup. Replaces N × stations_near round-trips with one call."""
    from_ref = inp.get("from_ref")
    to_ref = inp.get("to_ref")
    via_refs = tuple(inp.get("via_refs") or [])
    station_radius_m = float(inp.get("station_radius_km", 5)) * 1000.0
    min_routes_rail = int(inp.get("min_routes_rail", 2))
    corridor_m = float(inp.get("corridor_km", 8)) * 1000.0
    limit = int(inp.get("limit", 20))

    poly = _ROUTE_CACHE.get((from_ref, to_ref, via_refs))
    if not poly:
        for (fr, tr, _via), coords in _ROUTE_CACHE.items():
            if fr == from_ref and tr == to_ref:
                poly = coords
                break
    if not poly:
        route_result = _tool_route({
            "from_ref": from_ref, "to_ref": to_ref,
            "via_refs": list(via_refs) if via_refs else None,
        })
        poly = route_result.get("polyline") or []
    if len(poly) < 2:
        return {"error": "no cached route for that endpoint pair; call `route` first"}

    cum_m = [0.0]
    for i in range(1, len(poly)):
        cum_m.append(cum_m[-1] + _hav_m(
            poly[i-1][0], poly[i-1][1], poly[i][0], poly[i][1]))

    prof = trunk_router._load_profile(DEFAULT_PROFILE)
    stree, stations = _stations_kdtree()

    try:
        import numpy as np
        from scipy.spatial import cKDTree as _CKD
        poly_arr = np.array(poly, dtype=np.float64)
        poly_tree = _CKD(poly_arr)
    except ImportError:
        poly_tree = None

    lat_mid = poly[len(poly)//2][1] if poly else 50.0
    _MDEGLAT = 111_320.0
    _MDEGLON = 111_320.0 * math.cos(math.radians(lat_mid))
    _min_mdeg = min(_MDEGLAT, _MDEGLON)

    def _closest_poly_km(alon, alat):
        if poly_tree is not None:
            d_deg, idx = poly_tree.query([alon, alat], k=1)
            return d_deg * _min_mdeg, int(idx)
        best_d = float("inf"); best_i = 0
        for i, (px, py) in enumerate(poly):
            d = _hav_m(alon, alat, px, py)
            if d < best_d: best_d = d; best_i = i
        return best_d, best_i

    def _stations_near_anchor(alon, alat):
        if stree is None:
            hits = []
            for s in stations:
                d = _hav_m(alon, alat, s["lon"], s["lat"])
                if d <= station_radius_m:
                    hits.append((d, s))
            hits.sort(key=lambda h: h[0])
            return hits
        radius_deg = station_radius_m / _min_mdeg
        idxs = stree.query_ball_point([alon, alat], r=radius_deg)
        out = []
        for i in idxs:
            s = stations[i]
            d = _hav_m(alon, alat, s["lon"], s["lat"])
            if d <= station_radius_m:
                out.append((d, s))
        out.sort(key=lambda h: h[0])
        return out

    results = []
    for c in prof.cities:
        alon, alat = float(c.get("lon", 0)), float(c.get("lat", 0))
        if alon == 0 and alat == 0:
            continue
        d_to_poly, best_i = _closest_poly_km(alon, alat)
        if d_to_poly > corridor_m:
            continue
        st_hits = _stations_near_anchor(alon, alat)
        if not st_hits:
            continue
        best_rail = max((s.get("n_routes_rail", 0) for _, s in st_hits), default=0)
        if best_rail < min_routes_rail:
            continue
        results.append({
            "ref": c.get("ref"),
            "name": c.get("name"),
            "population": c.get("population"),
            "country": c.get("country"),
            "lon": alon, "lat": alat,
            "km_along_route": round(cum_m[best_i] / 1000.0, 1),
            "dist_from_route_km": round(d_to_poly / 1000.0, 2),
            "stations": [
                {
                    "name": s["name"],
                    "n_routes_rail": s.get("n_routes_rail", 0),
                    "n_routes_bus":  s.get("n_routes_bus", 0),
                    "distance_km": round(d / 1000.0, 2),
                }
                for d, s in st_hits[:3]
            ],
        })
    results.sort(key=lambda r: r["km_along_route"])
    return {
        "total_km": round(cum_m[-1] / 1000.0, 1),
        "n_matched_anchors": len(results),
        "anchors": results[:limit],
    }


def _tool_direct_rail_service(inp: dict) -> dict:
    """Intersect the route_id sets of the stations closest to two
    anchors. Returns `direct_service` = whether the intersection is
    non-empty, plus the matched stations and (trimmed) shared route
    list."""
    from_ref = inp.get("from_ref")
    to_ref = inp.get("to_ref")
    if not from_ref or not to_ref:
        return {"error": "from_ref and to_ref are required"}
    max_dist_m = float(inp.get("max_station_dist_km", 5)) * 1000.0
    min_routes_rail = int(inp.get("min_routes_rail", 1))

    prof = trunk_router._load_profile(DEFAULT_PROFILE)
    ci_a = prof.city_idx_by_ref.get(from_ref)
    ci_b = prof.city_idx_by_ref.get(to_ref)
    if ci_a is None:
        return {"error": f"unknown anchor ref: {from_ref}"}
    if ci_b is None:
        return {"error": f"unknown anchor ref: {to_ref}"}
    a = prof.cities[int(ci_a)]
    b = prof.cities[int(ci_b)]
    alon, alat = float(a["lon"]), float(a["lat"])
    blon, blat = float(b["lon"]), float(b["lat"])

    a_hits = _stations_for_anchor(alon, alat, max_dist_m, min_routes_rail)
    b_hits = _stations_for_anchor(blon, blat, max_dist_m, min_routes_rail)
    if not a_hits:
        return {
            "direct_service": False,
            "reason": (f"no station with >= {min_routes_rail} rail routes "
                       f"within {max_dist_m/1000:.1f} km of {a.get('name')}"),
            "from_anchor": {"ref": from_ref, "name": a.get("name")},
            "to_anchor":   {"ref": to_ref,   "name": b.get("name")},
        }
    if not b_hits:
        return {
            "direct_service": False,
            "reason": (f"no station with >= {min_routes_rail} rail routes "
                       f"within {max_dist_m/1000:.1f} km of {b.get('name')}"),
            "from_anchor": {"ref": from_ref, "name": a.get("name")},
            "to_anchor":   {"ref": to_ref,   "name": b.get("name")},
        }

    routes = _load_station_routes_cache()
    if not routes:
        return {
            "error": (
                "route-id sidecar not present. Run "
                "`pgrouting/main.py export-rail-routes` and copy the "
                "output to data/rail_station_routes.json."
            ),
        }

    def _union_routes(hits: list[tuple[float, dict]]) -> set[str]:
        acc: set[str] = set()
        for _, s in hits:
            key = f"{s.get('country')}:{s.get('gtfs_id')}"
            acc |= routes.get(key, set())
        return acc

    a_routes = _union_routes(a_hits)
    b_routes = _union_routes(b_hits)
    shared = sorted(a_routes & b_routes)
    # Report the highest-n_routes_rail hit at each end as the
    # representative (best proxy for the "main station").
    a_repr = max(a_hits, key=lambda h: h[1].get("n_routes_rail", 0))
    b_repr = max(b_hits, key=lambda h: h[1].get("n_routes_rail", 0))
    return {
        "direct_service": bool(shared),
        "n_shared_routes": len(shared),
        "from_station": {
            "name": a_repr[1].get("name"),
            "country": a_repr[1].get("country"),
            "distance_km": round(a_repr[0] / 1000.0, 2),
            "n_routes_rail": a_repr[1].get("n_routes_rail", 0),
        },
        "to_station": {
            "name": b_repr[1].get("name"),
            "country": b_repr[1].get("country"),
            "distance_km": round(b_repr[0] / 1000.0, 2),
            "n_routes_rail": b_repr[1].get("n_routes_rail", 0),
        },
        "n_from_stations_considered": len(a_hits),
        "n_to_stations_considered": len(b_hits),
        "shared_routes": shared[:20],
    }


TOOL_IMPLS = {
    "search_anchors":       _tool_search_anchors,
    "route":                _tool_route,
    "stations_near":        _tool_stations_near,
    "stations_along_route": _tool_stations_along_route,
    "direct_rail_service":  _tool_direct_rail_service,
    "split_into_stages":    _tool_split_into_stages,
    "pois_near_anchor":     _tool_pois_near_anchor,
    "pois_along_route":     _tool_pois_along_route,
}


def call_tool(name: str, args: dict) -> dict:
    """Convenience entry — dispatch a tool call by name. Returns
    `{"error": ...}` if the tool name is unknown or the impl raises."""
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return impl(args or {})
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
