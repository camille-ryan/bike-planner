"""HTTP client for the BRouter routing server.

BRouter sometimes returns 200 with a plain-text body containing an error
message (e.g. profile parse errors, or "no route found"). We treat any
non-JSON-looking body as an error and surface it clearly.

`fetch_split_route` is the cached path used for primary routes: it
splits an N-point request into N-1 single-leg BRouter calls, cacheing
each leg by `(profile, from, to)`. Alternative routes still go via
`fetch_alternatives` which calls BRouter end-to-end uncached (alts only
make sense as variations on the *full* search — caching alt-idx > 0
wouldn't compose).
"""
import httpx

from . import leg_cache
from .settings import BROUTER_URL


def _format_lonlats(points: list[tuple[float, float]]) -> str:
    return "|".join(f"{lon},{lat}" for lon, lat in points)


async def fetch_route(
    points: list[tuple[float, float]],
    profile: str,
    alternativeidx: int = 0,
) -> dict:
    """Fetch a single route from BRouter; raises RuntimeError on failure."""
    params = {
        "lonlats": _format_lonlats(points),
        "profile": profile,
        "alternativeidx": str(alternativeidx),
        "format": "geojson",
    }
    # Long multi-leg routes (point-to-point ~hundreds of km) can take BRouter
    # several minutes; the engine's own maxRunningTime cap is 1800s.
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=1800.0, write=10.0, pool=10.0)) as client:
        r = await client.get(f"{BROUTER_URL}/brouter", params=params)
    body = r.text
    if not body or not body.lstrip().startswith("{"):
        raise RuntimeError(f"brouter[{alternativeidx}]: {body[:300] or '(empty)'}")
    gj = r.json()
    feats = gj.get("features") or []
    if not feats:
        raise RuntimeError(f"brouter[{alternativeidx}]: no features")
    feat = feats[0]
    feat["properties"]["alternativeidx"] = alternativeidx
    return feat


async def fetch_alternatives(
    points: list[tuple[float, float]],
    profile: str,
    extra: int,
) -> list[dict]:
    """Fetch the primary route + up to `extra` alternatives.

    Stops at the first failure on alt index >= 1 (alts may not exist).
    Re-raises if the primary route itself fails.
    """
    out: list[dict] = []
    for idx in range(extra + 1):
        try:
            out.append(await fetch_route(points, profile, idx))
        except RuntimeError:
            if idx == 0:
                raise
            break
    return out


async def _fetch_or_cache_leg(
    a: tuple[float, float],
    b: tuple[float, float],
    profile: str,
) -> tuple[dict, bool]:
    """Return (feature, was_cache_hit) for a single A->B leg."""
    cached = leg_cache.get(profile, a, b)
    if cached is not None:
        return cached, True
    feature = await fetch_route([a, b], profile, alternativeidx=0)
    leg_cache.put(profile, a, b, feature)
    return feature, False


def _concat_legs(legs: list[dict], cache_hits: list[bool]) -> dict:
    """Concatenate adjacent BRouter LineString features into one.

    Boundary points are de-duplicated: leg N's coordinates start at
    leg N-1's last coordinate, so we drop the first sample of every
    leg after the first. `times` are cumulative within a leg, so we
    shift them by the running total-time before extending. `messages`
    have a header row (column names) that we keep once and skip on
    subsequent legs.
    """
    primary = legs[0]
    coords = list(primary["geometry"]["coordinates"])
    props = dict(primary["properties"])
    times = list(props.get("times", []))
    messages = list(props.get("messages", []))

    track_len  = float(props.get("track-length", 0) or 0)
    f_ascend   = float(props.get("filtered ascend", 0) or 0)
    p_ascend   = float(props.get("plain-ascend", 0) or 0)
    total_time = float(props.get("total-time", 0) or 0)
    total_eng  = float(props.get("total-energy", 0) or 0)
    cost       = float(props.get("cost", 0) or 0)

    for leg in legs[1:]:
        lp = leg["properties"]
        coords.extend(leg["geometry"]["coordinates"][1:])
        offset = total_time
        for t in (lp.get("times") or [])[1:]:
            times.append(t + offset)
        messages.extend((lp.get("messages") or [])[1:])
        track_len  += float(lp.get("track-length", 0) or 0)
        f_ascend   += float(lp.get("filtered ascend", 0) or 0)
        p_ascend   += float(lp.get("plain-ascend", 0) or 0)
        total_time += float(lp.get("total-time", 0) or 0)
        total_eng  += float(lp.get("total-energy", 0) or 0)
        cost       += float(lp.get("cost", 0) or 0)

    props["track-length"]    = int(track_len)
    props["filtered ascend"] = int(f_ascend)
    props["plain-ascend"]    = int(p_ascend)
    props["total-time"]      = total_time
    props["total-energy"]    = total_eng
    props["cost"]            = int(cost)
    props["times"]           = times
    props["messages"]        = messages
    props["cache-hits"]      = cache_hits
    props["alternativeidx"]  = 0

    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": props,
    }


async def fetch_split_route(
    points: list[tuple[float, float]],
    profile: str,
) -> dict:
    """Cached, leg-by-leg version of fetch_route for the primary alternative.

    Returns a single concatenated feature equivalent to one BRouter call
    with the full multi-point lonlats, but built by stitching cached
    per-leg features when available.
    """
    if len(points) < 2:
        raise ValueError("fetch_split_route requires at least two points")
    legs: list[dict] = []
    cache_hits: list[bool] = []
    for a, b in zip(points[:-1], points[1:]):
        feature, hit = await _fetch_or_cache_leg(a, b, profile)
        legs.append(feature)
        cache_hits.append(hit)
    if len(legs) == 1:
        only = dict(legs[0])
        only.setdefault("properties", {})["cache-hits"] = cache_hits
        return only
    return _concat_legs(legs, cache_hits)
