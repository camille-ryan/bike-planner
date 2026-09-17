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
            "`search_anchors`) over raw coords when possible.\n\n"
            "**Default is FAST mode** — uses the pre-computed city-graph "
            "chain (rail-line-following, ~50 ms) and returns a polyline "
            "of anchor coords + straight-line segments between them. "
            "Total km comes from the city-graph edge weights and is a "
            "good approximation (~5-10% of the true bike distance). "
            "Perfect for exploration — cheap enough to try many "
            "candidate corridors. Pass `precise: true` ONLY for the "
            "final route you want to hand to the user (adds ~1-3 s "
            "for full pathfinding + first/last-mile stitching)."
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
                "precise": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Full pathfinding (slow, precise polyline). "
                        "Only use for the final route the user will "
                        "see; defaults to false for planning."
                    ),
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
        "name": "rail_path",
        "description": (
            "Shortest chain of RAIL-connected anchors from `from_ref` to "
            "`to_ref` where every consecutive pair has direct-train "
            "service (they share at least one GTFS route_id). Uses the "
            "SAME shape as the bike chain graph but derived from GTFS "
            "rail data instead of bike-route trunks.\n\n"
            "**Call this FIRST for any rail-constrained tour.** The bike "
            "corridor should FOLLOW the rail spine, not the shortest-"
            "distance bike route: pass this tool's chain as `via_refs` "
            "to `route` and `split_into_stages` so every base overnight "
            "lands on a rail line and the consecutive-pair direct-train "
            "constraint is satisfied by construction. Doing bike routing "
            "first and checking rail after leaves overnights stranded "
            "when the direct rail line diverges from the bike route "
            "(Wien↔Praha rail runs via Brno; direct is bus-only).\n\n"
            "Returns `{reachable, n_hops, total_km, chain: [{ref, name, "
            "population, country, km_from_start}, ...]}`. Chain is "
            "ordered from `from_ref` to `to_ref`; `total_km` is the "
            "sum of geodesic hop distances (a lower bound on the bike "
            "distance that will follow it). If unreachable (isolated on "
            "the rail graph or missing stations), returns "
            "`reachable: false`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "from_ref": {"type": "string", "description": "Start anchor ref"},
                "to_ref":   {"type": "string", "description": "End anchor ref"},
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
        "name": "direct_rail_service_batch",
        "description": (
            "BATCHED version of `direct_rail_service`. Check up to 30 "
            "`(from_ref, to_ref)` pairs in ONE tool call. Prefer this "
            "over individual `direct_rail_service` calls whenever you "
            "need to verify rail connectivity for more than 2 pairs — "
            "e.g. a multi-overnight itinerary where you're checking "
            "every overnight against Graz + Copenhagen + a corridor "
            "hub. One call replaces N rounds; saves your tool-loop "
            "budget for the actual day-by-day narrative.\n\n"
            "Returns `{'results': [...]}` where each entry has the "
            "same shape as a single `direct_rail_service` response, "
            "in the SAME order as the input pairs, tagged with the "
            "input `from_ref`/`to_ref` for identification.\n\n"
            "Global `max_station_dist_km` / `min_routes_rail` apply "
            "to every pair; a pair can't override them individually."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pairs": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 30,
                    "description": "List of {from_ref, to_ref} pairs.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from_ref": {"type": "string"},
                            "to_ref":   {"type": "string"},
                        },
                        "required": ["from_ref", "to_ref"],
                    },
                },
                "max_station_dist_km": {
                    "type": "number", "default": 5, "minimum": 0.5, "maximum": 25,
                },
                "min_routes_rail": {
                    "type": "integer", "default": 1, "minimum": 1, "maximum": 50,
                },
            },
            "required": ["pairs"],
        },
    },
    {
        "name": "split_into_stages",
        "description": (
            "Split a tour into daily stages of roughly "
            "`target_km_per_day`. Uses the last-computed whole-trip "
            "polyline to pick which anchors to overnight at (nearest "
            "anchor to each km-cut point), then routes each daily leg "
            "SEPARATELY from previous overnight's center to today's "
            "overnight's center. Each daily route naturally arrives at "
            "and departs from actual city centers — different from just "
            "slicing the whole-trip polyline, which would leave the "
            "rider on the bike-corridor bypass rather than downtown.\n\n"
            "Returns `[{day, from_ref, from_name, from_lonlat, to_ref, "
            "to_name, to_lonlat, km}]`. `km` is the daily leg's total "
            "distance including the walk from previous overnight's "
            "center and into today's overnight's center. Pass the same "
            "from_ref/to_ref/via_refs you used with `route` so this "
            "tool can find that route's polyline in cache."
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

# Parallel cache of the total_km value the route call reported.
# Split_into_stages and friends measure haversine along the polyline
# — that's fine for a precise polyline (matches the router's own
# gross_length_m), but for a FAST polyline (straight-lines between
# chain anchors) the haversine total underestimates the real bike
# distance. When a km hint is present, day-count math uses THAT
# instead of the haversine total.
_ROUTE_KM_CACHE: dict[tuple, float] = {}


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


def _tool_route_fast(inp: dict) -> dict:
    """Chain-graph-only route: pairwise city-graph Dijkstra through
    the anchor chain (start → via[0] → via[1] → … → end), polyline
    is straight-line segments between the picked anchors, total_km
    is the sum of city-graph edge weights. ~50 ms per call, no
    trunk polylines loaded, no local Dijkstra.

    Approximate distance error is 5-10% vs the precise trunk-walk
    (which follows real roads); good enough for planning. The
    frontend can hydrate to precise later.
    """
    prof = trunk_router._load_profile(DEFAULT_PROFILE)

    def _resolve(ref: str | None) -> int | None:
        if not ref:
            return None
        return prof.city_idx_by_ref.get(ref)

    from_ref = inp.get("from_ref")
    to_ref   = inp.get("to_ref")
    if not from_ref or not to_ref:
        return {"error": "fast mode requires from_ref and to_ref (anchor refs)"}
    src = _resolve(from_ref)
    dst = _resolve(to_ref)
    if src is None:
        return {"error": f"unknown from_ref: {from_ref}"}
    if dst is None:
        return {"error": f"unknown to_ref: {to_ref}"}

    via_refs = list(inp.get("via_refs") or [])
    waypoints = [src]
    for vr in via_refs:
        vi = _resolve(vr)
        if vi is None:
            return {"error": f"unknown via_ref: {vr}"}
        waypoints.append(vi)
    waypoints.append(dst)

    chain: list[int] = []
    total_km = 0.0
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        seg = trunk_router._city_graph_dijkstra(prof.chain_adj, a, b)
        if seg is None:
            return {"error": (f"no chain-graph path from "
                              f"{prof.cities[a].get('name','?')} to "
                              f"{prof.cities[b].get('name','?')}")}
        # Sum edge weights for this segment. `chain_adj` weights are
        # in METERS (from city_graph.json's `weight` column, matching
        # the paired-SPT preprocess) — convert to km for the response.
        for u, v in zip(seg[:-1], seg[1:]):
            for nbr, w in prof.chain_adj.get(u, ()):
                if nbr == v:
                    total_km += float(w) / 1000.0
                    break
        # Concat, dedup at the shared waypoint.
        chain.extend(seg if not chain else seg[1:])
    # Empirical fudge: chain_adj weights are the paired-SPT preprocess's
    # DISC-to-DISC shortest bike paths, not city-center-to-city-center.
    # Summing them underreads the precise trunk-walked total by ~15-20%
    # because a real trip's first/last-mile stitches (city center → disc
    # boundary, disc boundary → city center) aren't in the edge weights.
    # A proper fix is additive per hop (~disc radius × 2), TBD when we
    # can read the preprocess disc radius; 1.20× lands close enough
    # today (Graz→Wien 178 → 214 vs 215; full trip 1296 → 1556 vs 1583).
    total_km *= 1.20

    coords: list[list[float]] = []
    chain_stops: list[dict] = []
    for ci in chain:
        c = prof.cities[ci]
        lon = float(c.get("lon", 0.0))
        lat = float(c.get("lat", 0.0))
        coords.append([lon, lat])
        chain_stops.append({
            "name":       c.get("name"),
            "ref":        c.get("ref"),
            "population": c.get("population"),
            "country":    c.get("country"),
            "kind":       c.get("place"),
        })

    key = _route_cache_key(inp)
    _ROUTE_CACHE[key]    = coords
    _ROUTE_KM_CACHE[key] = total_km
    return {
        "total_km":       round(total_km, 1),
        "polyline":       coords,
        "chain_stops":    chain_stops,
        "chain_names":    [s["name"] for s in chain_stops],
        "chain_city_idx": chain,
        "n_bridges":      0,
        "mode":           "fast",
    }


def _tool_route(inp: dict) -> dict:
    """Fast (default) chain-graph route, or precise trunk-walked route.

    Fast mode skips both trunk polyline walking and first/last-mile
    local Dijkstra — the polyline is straight lines between the
    chain anchors picked by city-graph Dijkstra, and total_km is
    the sum of city-graph edge weights. Cheap enough (~50 ms) that
    the model can call it many times during exploration.

    `precise: true` falls through to `trunk_router.route()` for
    the real pathfinding — use for the FINAL polyline the user
    sees. All the pathfinding cost (multi-leg local Dijkstra,
    trunk walk, decimation) lives here.
    """
    precise = bool(inp.get("precise", False))
    if not precise:
        return _tool_route_fast(inp)

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
    haversine_total_km = cum[-1] / 1000.0
    # A fast-mode polyline is straight-lines between chain anchors,
    # so `haversine_total_km` under-reads real bike distance by ~15-20%.
    # When the route call cached a km hint (fudged toward the precise
    # value), use that for day-count math and scale the cumulative
    # array so per-stage `km` values inherit the correction too.
    hint = _ROUTE_KM_CACHE.get((from_ref, to_ref, via_refs))
    if hint and haversine_total_km > 1e-9:
        _scale = (hint * 1000.0) / cum[-1]
        cum = [c * _scale for c in cum]
        total_km = hint
    else:
        total_km = haversine_total_km

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

    def _idx_at_km(km_target):
        """Polyline vertex index closest to a given cumulative km."""
        target_m = km_target * 1000.0
        for i in range(1, len(cum)):
            if cum[i] >= target_m:
                return i - 1 if (target_m - cum[i-1]) < (cum[i] - target_m) else i
        return len(cum) - 1

    def _anchor_from_ref(ref: str):
        """Look up an anchor by ref, return the (name, ref, lon, lat, 0) tuple
        `_nearest_anchor` returns."""
        ci = prof.city_idx_by_ref.get(ref)
        if ci is None:
            return None
        c = prof.cities[int(ci)]
        return (c.get("name"), c.get("ref"),
                float(c.get("lon", 0.0)), float(c.get("lat", 0.0)), 0.0)

    # STEP 1: fix the mandatory boundaries — from_ref, every via_ref
    # (in order), and to_ref MUST be overnights. Find each's index
    # along the polyline (nearest vertex to the anchor coord). This
    # gives us the km-along-route where each boundary sits, and lets
    # us split the trip into inter-boundary SEGMENTS to size day
    # counts within each. Fixes the "split stops in Hennigsdorf
    # instead of Berlin" bug: Berlin was in via_refs but the km-based
    # cut fell nearer a suburb, so it never became an overnight.
    boundary_refs = [from_ref, *via_refs, to_ref]
    boundary_anchors = []
    boundary_idxs = []
    for ref in boundary_refs:
        a = _anchor_from_ref(ref)
        if a is None:
            return {"error": f"unknown ref: {ref}"}
        # Nearest vertex on the polyline to this anchor.
        best_i, best_d = 0, float("inf")
        for i, (lon, lat) in enumerate(poly):
            d = _hav_m(a[2], a[3], lon, lat)
            if d < best_d:
                best_d, best_i = d, i
        boundary_anchors.append(a)
        boundary_idxs.append(best_i)

    # STEP 2: pick day boundaries within each inter-hub segment.
    # `target_km_per_day` sizes the day-count per segment; we cut
    # evenly and snap each cut to the nearest anchor.
    overnights = [boundary_anchors[0]]
    overnight_idxs = [boundary_idxs[0]]
    for seg_i in range(len(boundary_idxs) - 1):
        a_idx, b_idx = boundary_idxs[seg_i], boundary_idxs[seg_i + 1]
        if b_idx <= a_idx:
            # Degenerate segment (anchors collide on the polyline);
            # nothing to insert between them.
            overnights.append(boundary_anchors[seg_i + 1])
            overnight_idxs.append(b_idx)
            continue
        seg_km = (cum[b_idx] - cum[a_idx]) / 1000.0
        seg_days = max(1, round(seg_km / target_km))
        for i in range(1, seg_days):
            cut_km_abs = cum[a_idx] / 1000.0 + seg_km * i / seg_days
            cut_idx = _idx_at_km(cut_km_abs)
            # Only add if it moves us forward past the previous overnight.
            if cut_idx <= overnight_idxs[-1]:
                continue
            near = _nearest_anchor(poly[cut_idx][0], poly[cut_idx][1])
            if near is None or near[1] == overnight_idxs[-1]:
                continue
            overnights.append(near)
            overnight_idxs.append(cut_idx)
        overnights.append(boundary_anchors[seg_i + 1])
        overnight_idxs.append(b_idx)

    # STEP 3: build stage dicts by SLICING the already-computed
    # polyline instead of a nested `_tool_route` call per day. The
    # old per-day nested route was the 272-s stall in the flagship
    # trace (issue #9) — the polyline is already precise (or fast,
    # matching the caller's choice), so a slice gives the exact same
    # daily route for zero extra work.
    stages = []
    for day_i in range(1, len(overnights)):
        prev, cur = overnights[day_i - 1], overnights[day_i]
        a_idx, b_idx = overnight_idxs[day_i - 1], overnight_idxs[day_i]
        day_poly = poly[a_idx:b_idx + 1]
        day_km = round((cum[b_idx] - cum[a_idx]) / 1000.0, 1)
        stages.append({
            "day": day_i,
            "from_ref":    prev[1],
            "from_name":   prev[0],
            "from_lonlat": [prev[2], prev[3]],
            "to_ref":      cur[1],
            "to_name":     cur[0],
            "to_lonlat":   [cur[2], cur[3]],
            "km":          day_km,
            "polyline":    day_poly,
        })
    return {
        "total_km": round(sum(s["km"] for s in stages), 1),
        "n_days":   len(stages),
        "stages":   stages,
    }


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


def _tool_direct_rail_service_batch(inp: dict) -> dict:
    """Run `_tool_direct_rail_service` for each `(from_ref, to_ref)`
    pair in `pairs` and return an aligned results list. Applies the
    same global `max_station_dist_km` / `min_routes_rail` to every
    pair — a per-pair override would multiply the schema without
    much practical use.

    Cheap: all the data (stations kdtree, route-id sidecar, anchor
    lookups) is already loaded and cached. This is O(pairs) with a
    tiny constant. The value it adds isn't compute — it's tool-loop
    rounds. 20 pairs go from 20 model rounds to 1.
    """
    pairs = inp.get("pairs") or []
    if not isinstance(pairs, list) or not pairs:
        return {"error": "pairs must be a non-empty list"}
    if len(pairs) > 30:
        return {"error": f"at most 30 pairs per call (got {len(pairs)})"}
    max_km = inp.get("max_station_dist_km")
    min_r  = inp.get("min_routes_rail")
    results = []
    for p in pairs:
        if not isinstance(p, dict):
            results.append({"error": "pair must be an object"})
            continue
        one_inp = {"from_ref": p.get("from_ref"),
                   "to_ref":   p.get("to_ref")}
        if max_km is not None:
            one_inp["max_station_dist_km"] = max_km
        if min_r is not None:
            one_inp["min_routes_rail"] = min_r
        one_out = _tool_direct_rail_service(one_inp)
        # Echo the input refs so the caller can align results with
        # its own list without positional counting.
        one_out["from_ref"] = p.get("from_ref")
        one_out["to_ref"]   = p.get("to_ref")
        results.append(one_out)
    return {"results": results, "n_pairs": len(results)}


def _tool_rail_path(inp: dict) -> dict:
    """Rail-anchor-graph Dijkstra between two anchors. Returns the
    chain of intermediate rail-served anchors — the "rail spine" the
    bike corridor should follow so every consecutive overnight pair
    has direct-train service."""
    from . import rail_router

    from_ref = inp.get("from_ref")
    to_ref = inp.get("to_ref")
    if not from_ref or not to_ref:
        return {"error": "provide from_ref and to_ref"}
    prof = trunk_router._load_profile(DEFAULT_PROFILE)
    src_ci = prof.city_idx_by_ref.get(from_ref)
    dst_ci = prof.city_idx_by_ref.get(to_ref)
    if src_ci is None:
        return {"error": f"unknown from_ref: {from_ref}"}
    if dst_ci is None:
        return {"error": f"unknown to_ref: {to_ref}"}

    chain = rail_router.rail_shortest_path(DEFAULT_PROFILE, src_ci, dst_ci)
    if chain is None:
        return {
            "reachable": False,
            "chain": [],
            "note": (
                f"No direct-rail chain from {from_ref} to {to_ref}. "
                "Either one of them isn't within 5 km of a rail-served "
                "station, or they're in disconnected components on the "
                "rail graph. Falling back to bike-shortest routing is "
                "your only option."
            ),
        }

    # Cumulative km along the chain (geodesic per hop).
    entries: list[dict] = []
    prev_lon: float | None = None
    prev_lat: float | None = None
    cum_km = 0.0
    for ci in chain:
        c = prof.cities[ci]
        lon, lat = float(c["lon"]), float(c["lat"])
        if prev_lon is not None:
            cum_km += rail_router._hav_km(prev_lon, prev_lat, lon, lat)
        entries.append({
            "ref":           c.get("ref"),
            "name":          c.get("name"),
            "population":    c.get("population"),
            "country":       c.get("country"),
            "km_from_start": round(cum_km, 1),
        })
        prev_lon, prev_lat = lon, lat

    return {
        "reachable": True,
        "n_hops":    len(entries) - 1,
        "total_km":  round(cum_km, 1),
        "chain":     entries,
    }


TOOL_IMPLS = {
    "search_anchors":       _tool_search_anchors,
    "route":                _tool_route,
    "stations_near":        _tool_stations_near,
    "stations_along_route": _tool_stations_along_route,
    "rail_path":            _tool_rail_path,
    "direct_rail_service":  _tool_direct_rail_service,
    "direct_rail_service_batch": _tool_direct_rail_service_batch,
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
