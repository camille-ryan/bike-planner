"""Chat-driven planner: /chat SSE endpoint backed by Claude + tool use.

The backend is stateless — the browser sends the full message history on
every turn. That keeps the API simple to reason about; it also means
switching sessions / editing history is a purely client-side concern.

Tools:
  - route(from, to, profile)             → calls trunk_router.route()
  - search_anchors(query, near?, limit)  → wraps /anchors/search
  - stations_near(lon, lat, radius_km)   → reads rail_stations.geojson
  - split_into_stages(coords, target_km) → splits a polyline into
                                            day-length chunks anchored
                                            at cities
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterator

import anthropic
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from . import pois, trunk_router
from .settings import DEFAULT_PROFILE, DATA_DIR

POI_CATEGORIES = ("food", "viewpoint", "lodging", "water", "bike_service")


CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-sonnet-5")
CHAT_MAX_TOOL_ROUNDS = int(os.environ.get("CHAT_MAX_TOOL_ROUNDS", "30"))
STATIONS_PATH = Path(DATA_DIR) / "rail_stations.geojson"

router = APIRouter()

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise HTTPException(500, "ANTHROPIC_API_KEY not set in the API container's env")
        # Per-request timeout so a stalled upstream can't hang the SSE
        # loop indefinitely. 120s covers even long extended-thinking
        # rounds; anything longer than that is a genuine failure the
        # client should see quickly.
        _client = anthropic.Anthropic(api_key=key, timeout=120.0)
    return _client


# ---------------------------------------------------------------------------
# Tool schemas — passed to Claude as `tools=[...]`.

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
# Tool implementations.

R_EARTH_M = 6_371_000.0


# English → native name aliases along the Graz → Copenhagen corridor.
# Anchor names come from OSM's `name` tag, which is the local language.
# So Copenhagen won't match anything until we also try "København".
# Bidirectional so a user query in either language works.
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


# Per-tool-loop cache of computed routes so tools like split_into_stages
# can read a polyline without the LLM having to carry it. Keyed on the
# canonical `(from_ref, to_ref, via_refs_tuple)` triple.
_ROUTE_CACHE: dict[tuple, list[list[float]]] = {}


def _route_cache_key(inp: dict) -> tuple:
    return (
        inp.get("from_ref") or inp.get("from_lonlat"),
        inp.get("to_ref")   or inp.get("to_lonlat"),
        tuple(inp.get("via_refs") or []),
    )


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
        # Plain-list variants for the frontend's paired-SPT viz layer,
        # which reuses the sidebar's route-shape (chain_names +
        # chain_city_idx). Kept alongside `chain_stops` so the LLM
        # keeps its rich per-stop metadata.
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
                # Keep n_routes for backwards compat, but the model
                # should prefer n_routes_rail (rail-only) for the
                # rail-accessible overnight decision — task #78
                # exposed this split after the tangero feed was found
                # to conflate suburban bus routes into rail counts.
                "n_routes":      f["properties"].get("n_routes", 0),
                "n_routes_rail": f["properties"].get("n_routes_rail",
                                                    f["properties"].get("n_routes", 0)),
                "n_routes_bus":  f["properties"].get("n_routes_bus", 0),
                "n_lodging":     f["properties"].get("n_lodging", 0),
                "lon": f["geometry"]["coordinates"][0],
                "lat": f["geometry"]["coordinates"][1],
            }
            for f in fc.get("features", [])
        ]
    return _load_stations_cache._cache


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
    # Try exact key first (matches the route call), then fall back to
    # any cached route for the same endpoints regardless of via_refs.
    # The LLM sometimes forgets to echo via_refs — the fallback rescues
    # that case without needing another route call.
    poly = _ROUTE_CACHE.get((from_ref, to_ref, via_refs))
    if not poly:
        for (fr, tr, _via), coords in _ROUTE_CACHE.items():
            if fr == from_ref and tr == to_ref:
                poly = coords
                break
    if not poly:
        # No cache hit — compute the route inline so split_into_stages
        # always works even when the LLM tries to split before routing.
        # Cheaper than making the model retry.
        route_result = _tool_route({
            "from_ref": from_ref,
            "to_ref": to_ref,
            "via_refs": list(via_refs) if via_refs else None,
        })
        poly = route_result.get("polyline") or []
    if len(poly) < 2:
        return {"error": "Could not compute a route for that (from_ref, to_ref)."}
    # Cumulative distance along the polyline.
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
                # linear interp within the segment
                seg = cum[i] - cum[i-1] or 1e-9
                t = (target_m - cum[i-1]) / seg
                lon = poly[i-1][0] + t * (poly[i][0] - poly[i-1][0])
                lat = poly[i-1][1] + t * (poly[i][1] - poly[i-1][1])
                return lon, lat
        return poly[-1][0], poly[-1][1]

    # For each anchor we care about, find the polyline vertex nearest
    # its centroid. That vertex's cumulative-km value is the anchor's
    # "position along the route", which is what we need to report the
    # ACTUAL km per stage (not the evenly-split target km, which is
    # what a prior version reported and caused every stage to display
    # an identical distance).
    def _km_at_nearest_polyline_vertex(alon, alat) -> float:
        best_i = 0; best_d = float("inf")
        for i, (px, py) in enumerate(poly):
            d = (px - alon) ** 2 + (py - alat) ** 2   # squared deg — comparison only
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
        # Actual km from prev anchor to this anchor, measured along the
        # polyline (approximated by the km-of-nearest-vertex per anchor).
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
    # Coarse bbox around the anchor, then exact haversine filter.
    r_deg_lat = radius_km / 111.0
    r_deg_lon = r_deg_lat / max(math.cos(math.radians(lat)), 0.1)
    bbox = (lon - r_deg_lon, lat - r_deg_lat, lon + r_deg_lon, lat + r_deg_lat)
    # Query 3× the limit so exact-radius filtering has candidates.
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
    # Cache lookup mirrors split_into_stages: try exact key, fall back
    # to any cached route for the same endpoints, else compute inline.
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
    # Polyline bbox padded by buffer (rough — bbox will over-include,
    # exact haversine filter culls after).
    lons = [p[0] for p in poly]
    lats = [p[1] for p in poly]
    lat_mid = (min(lats) + max(lats)) / 2
    r_deg_lat = buffer_km / 111.0
    r_deg_lon = r_deg_lat / max(math.cos(math.radians(lat_mid)), 0.1)
    bbox = (min(lons) - r_deg_lon, min(lats) - r_deg_lat,
            max(lons) + r_deg_lon, max(lats) + r_deg_lat)
    # Pull more than `limit` because bbox is coarse; we'll filter to true
    # buffer and keep the closest N.
    raw = pois.query_bbox(bbox, [category], min(limit * 20, 2000))
    buffer_m = buffer_km * 1000.0

    # Nearest polyline vertex per POI. O(len(poly) × len(raw)) — fine
    # for a few hundred POIs × ~1000-vertex polyline; keeps it simple.
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
    lookup. Replaces the runaway N × stations_near loop where the model
    interrogates each candidate overnight one at a time — one call
    returns the full ranked corridor.

    Algorithm:
      1. Pull the cached polyline for (from_ref, to_ref, via_refs).
      2. Precompute cumulative km along the polyline.
      3. For each ANCHOR in the profile: skip if > corridor_km from
         the nearest polyline vertex.
      4. For each surviving anchor: find the top rail stations within
         station_radius_km using the cached stations list.
      5. Drop anchors whose best station has n_routes_rail < min_routes_rail.
      6. Sort remaining anchors by km_along_route ascending.
    """
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

    # Cumulative km along the polyline (in meters, converted at the end).
    cum_m = [0.0]
    for i in range(1, len(poly)):
        cum_m.append(cum_m[-1] + _hav_m(
            poly[i-1][0], poly[i-1][1], poly[i][0], poly[i][1]))

    # For each anchor, find the closest polyline vertex.
    prof = trunk_router._load_profile(DEFAULT_PROFILE)
    stree, stations = _stations_kdtree()

    # Build a KDTree over polyline vertices for fast per-anchor lookup.
    # Without this the per-anchor closest-vertex search is
    # O(N_anchors × N_poly_verts) — ~2M ops per 1000-vert route.
    try:
        import numpy as np
        from scipy.spatial import cKDTree as _CKD
        poly_arr = np.array(poly, dtype=np.float64)
        poly_tree = _CKD(poly_arr)
    except ImportError:
        poly_tree = None

    # Rough deg→m at the route midpoint for KDTree radius params.
    import math as _math
    lat_mid = poly[len(poly)//2][1] if poly else 50.0
    _MDEGLAT = 111_320.0
    _MDEGLON = 111_320.0 * _math.cos(_math.radians(lat_mid))
    _min_mdeg = min(_MDEGLAT, _MDEGLON)

    def _closest_poly_km(alon, alat):
        if poly_tree is not None:
            d_deg, idx = poly_tree.query([alon, alat], k=1)
            # deg → m upper bound (use smaller deg-per-m so a bounded
            # radius in deg is safely ≤ meters at any orientation).
            return d_deg * _min_mdeg, int(idx)
        # Fallback: linear scan
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
        # KDTree returns indices within a deg-radius upper bound.
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
        # Find rail stations near this anchor.
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


TOOL_IMPLS = {
    "search_anchors":       _tool_search_anchors,
    "route":                _tool_route,
    "stations_near":        _tool_stations_near,
    "stations_along_route": _tool_stations_along_route,
    "split_into_stages":    _tool_split_into_stages,
    "pois_near_anchor":     _tool_pois_near_anchor,
    "pois_along_route":     _tool_pois_along_route,
}


def _strip_bulk_for_llm(name: str, result: dict) -> dict:
    """Return a compact form of the tool result for the LLM's context.
    The frontend still receives the full result via the SSE frame."""
    if not isinstance(result, dict) or "error" in result:
        return result
    if name == "route":
        return {
            "total_km":    result.get("total_km"),
            "chain_stops": result.get("chain_stops", []),
            "n_bridges":   result.get("n_bridges"),
            "polyline_verts": len(result.get("polyline") or []),
        }
    if name == "split_into_stages":
        # Stages themselves are small (a dozen items × 6 fields) — keep.
        return result
    return result


# ---------------------------------------------------------------------------
# SSE endpoint.

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: Any  # string or list[block]


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


SYSTEM_PROMPT = """You are a bike-touring co-planner for a route spanning Austria, Czechia, Germany, and Denmark. The user is planning a Graz → Copenhagen tour (they can also plan sub-trips within that corridor).

Facts about the routing profile:
- Anchor references look like `db:1234` (city from population db) or `ferry:12345` (ferry pier). Prefer resolving names via `search_anchors` before calling `route`.
- The routing engine is bike-optimized (avoids highways, prefers bike lanes / dedicated paths).
- Ferry crossings are handled implicitly by the router — you don't need to plan them explicitly.
- Rail stations along the corridor matter because the user's partner meets them daily by train.
- Anchor names come from OSM's local-language `name` tag, so Copenhagen appears as "København", Vienna as "Wien", Prague as "Praha", Munich as "München", Cologne as "Köln", etc. `search_anchors` auto-tries a handful of English↔local pairs; if a search returns zero results, try the local-language name explicitly.

Workflow — do exactly what the user asked, no more. DO NOT stop mid-workflow for confirmation between steps that ARE needed:

- Always resolve every named place with `search_anchors` (English + local variants are auto-tried) before any other tool.
- Always call `route` between the resolved anchors if the user asked for a route.
- Call `split_into_stages` ONLY if the user asked for a multi-day plan, daily stages, km/day, or overnights — cues like "plan a X-day tour", "80 km/day", "break into stages". Do NOT split just because a route is long.
- Call `stations_near` ONLY if the user asked about rail, train, meeting the partner, or station-accessible overnights.
- Then write the final summary. Keep it proportional to what was asked — a single route gets one bullet with total km and chain waypoints, not a day-by-day breakdown.

Reformat / recall requests — DO NOT re-run tools:
- When the user asks to rephrase, reformat, summarize, tabulate, "make it prettier", "give me markdown", "show as a list", "just the overnights", "recap", or any variant that references content ALREADY produced in this conversation, work directly from the prior tool_result blocks and assistant messages in your context.
- Only call tools again if the user CHANGED a parameter (different endpoints, different km/day, added a via, "route via X" that wasn't there before, "add a stop", "shorten the days"). Anything that could produce a genuinely different route requires re-routing; anything that's pure presentation must NOT.
- If unsure whether a request is presentation-only vs. parameter-change, err on the side of NOT re-calling tools — reformat from context and add a one-sentence "let me know if you want me to re-route with different parameters."

Style:
- Be concise. No hedging preamble like "I'll help you plan…" — just start doing the work.
- If a tool errors, note briefly and try one alternative (e.g. local-language spelling) before giving up.
- Never invent anchors or coordinates. Every place-name is verified via `search_anchors` first."""


def _sse(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


def _msg_to_api(m: ChatMessage) -> dict:
    return {"role": m.role, "content": m.content}


def _friendly_error(exc: Exception) -> str:
    """Translate SDK / network / tool errors into a one-line message the
    user can read without seeing tracebacks or provider internals."""
    name = type(exc).__name__
    if isinstance(exc, anthropic.RateLimitError):
        return "I'm hitting the API rate limit — wait a moment and try again."
    if isinstance(exc, anthropic.AuthenticationError):
        return "The Anthropic API key isn't accepted. Check api/.env."
    if isinstance(exc, anthropic.BadRequestError):
        # Strip the 400 body (which is JSON with provider internals) and
        # just note the class. Log the full detail for the operator.
        print(f"[chat] BadRequestError: {exc}", flush=True)
        return "The model rejected the request — I've logged it. Try rephrasing, or hit Reset."
    if isinstance(exc, anthropic.APIStatusError):
        print(f"[chat] APIStatusError {exc.status_code}: {exc}", flush=True)
        return f"The model API returned {exc.status_code}. Try again in a moment."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Couldn't reach the model API — network issue. Try again in a moment."
    print(f"[chat] {name}: {exc}", flush=True)
    return f"Something broke internally ({name}). Try again, or hit Reset if it keeps failing."


def _run_chat(req: ChatRequest, client: anthropic.Anthropic) -> Iterator[bytes]:
    messages: list[dict] = [_msg_to_api(m) for m in req.messages]

    try:
        yield from _run_chat_inner(client, messages)
    except Exception as exc:
        # Never let a mid-stream exception crash the SSE with no signal
        # to the client — emit a clean `event: error` and close cleanly.
        yield _sse("error", {"message": _friendly_error(exc)})


def _run_chat_inner(client: anthropic.Anthropic, messages: list[dict]) -> Iterator[bytes]:
    for round_i in range(CHAT_MAX_TOOL_ROUNDS):
        with client.messages.stream(
            model=CHAT_MODEL,
            # 16k so a full multi-tool round (search_anchors × 2 +
            # route + split_into_stages + stations_near × N + a summary)
            # fits without stopping mid-plan for a token cap.
            max_tokens=16384,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            for event in stream:
                # Text deltas — forward each chunk immediately.
                if event.type == "text":
                    yield _sse("text", {"delta": event.text})
                elif event.type == "input_json":
                    # streaming tool-input arguments; ignore, we take the final
                    pass
            final = stream.get_final_message()

        # Log per-round outcome so we can see when a plan halts because
        # of max_tokens, max_rounds, or a clean end_turn. Without this
        # the frontend sees "silence" and there's no way to diagnose.
        n_text = sum(1 for b in final.content if b.type == "text")
        n_tool = sum(1 for b in final.content if b.type == "tool_use")
        print(f"[chat] round {round_i+1}/{CHAT_MAX_TOOL_ROUNDS}  "
              f"stop_reason={final.stop_reason}  "
              f"text_blocks={n_text}  tool_uses={n_tool}",
              flush=True)

        # Turn's assistant content — save to history for the next round
        # of the tool loop. Skip `thinking` blocks: Sonnet 5's extended
        # reasoning emits them, but their model_dump() shape isn't valid
        # as message-input and re-serializing them via the SDK's own
        # schema requires an exact match we don't get for free. Losing
        # them costs some cross-turn reasoning coherence but not
        # correctness — Claude re-reasons on the next round if needed.
        assistant_content = [
            block.model_dump() for block in final.content
            if block.type != "thinking"
        ]
        messages.append({"role": "assistant", "content": assistant_content})

        if final.stop_reason != "tool_use":
            # If the model stopped without ever emitting text on any
            # round, surface that as a friendly error so the UI doesn't
            # just look "done" with no answer.
            if final.stop_reason == "max_tokens" and n_text == 0:
                yield _sse("error", {"message": (
                    "The model hit its per-turn max_tokens without "
                    "finishing the answer. Raise max_tokens in chat.py "
                    "or ask for a smaller scope."
                )})
            yield _sse("done", {"stop_reason": final.stop_reason})
            return

        # Execute every tool_use block; append tool_result blocks as a
        # single user message (Claude API expects this shape).
        tool_uses = [b for b in final.content if b.type == "tool_use"]
        tool_results = []
        for tu in tool_uses:
            impl = TOOL_IMPLS.get(tu.name)
            if impl is None:
                result = {"error": f"unknown tool: {tu.name}"}
            else:
                try:
                    result = impl(tu.input)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}
            # Log every tool_use so a `docker logs bike-api` shows what
            # the model actually did with each turn. Trim polylines out
            # so the log stays readable.
            summary_result = _strip_bulk_for_llm(tu.name, result)
            print(f"[chat.tool] {tu.name}({tu.input}) → {summary_result}",
                  flush=True)
            # Stream the FULL tool call + result to the client — the
            # frontend needs the polyline / stages arrays to render the
            # map, and the client's context is cheap.
            yield _sse("tool_call", {
                "id": tu.id, "name": tu.name, "input": tu.input, "output": result,
            })
            # LLM only needs the reasoning-relevant fields. Stripping
            # the polyline (~1000+ vertices for a Graz→Cph route) keeps
            # each turn's context under control and prevents a single
            # tool_result from blowing past max_tokens.
            llm_result = _strip_bulk_for_llm(tu.name, result)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(llm_result),
            })
        messages.append({"role": "user", "content": tool_results})

    # Hit the round cap without a clean end. Tell the client so it can
    # render an "incomplete" note instead of just going silent.
    print(f"[chat] hit CHAT_MAX_TOOL_ROUNDS={CHAT_MAX_TOOL_ROUNDS} "
          f"without a final text response", flush=True)
    yield _sse("error", {"message": (
        f"Ran out of tool-loop rounds ({CHAT_MAX_TOOL_ROUNDS}) before "
        f"finishing the plan. Try a narrower prompt, or raise "
        f"CHAT_MAX_TOOL_ROUNDS in api/.env and restart the api."
    )})
    yield _sse("done", {"stop_reason": "max_rounds"})


@router.post("/chat")
def chat(req: ChatRequest):
    # Resolve the API key BEFORE starting the streaming response — an
    # HTTPException raised mid-stream would just close the socket with no
    # visible error to the client. Fail cleanly here instead.
    client = _get_client()
    # Sync generator; Starlette runs it in a threadpool so the anthropic
    # client's blocking IO doesn't stall the event loop.
    return StreamingResponse(_run_chat(req, client), media_type="text/event-stream")
