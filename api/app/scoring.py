"""Re-rank pass for BRouter alternative routes.

Each alternative gets a composite score combining:
  - BRouter's own cost (the routing engine's primary signal)
  - A curvature × |grade| penalty over descents (BRouter's DSL can't
    express smooth way curvature; we add it back here)
  - A scenic bonus from viewpoints near the route polyline

Lower composite score wins. Routes are returned sorted ascending; each
gets a `properties.scoring` block describing its contribution.
"""
from .geo import bearing_diff, haversine_m, initial_bearing
from .pois import query_bbox

# Tunable weights — these are first-pass guesses and can be calibrated
# once you have a few real routes to compare against.
W_CURVY_DESCENT = 5.0      # multiplier on (deg × downhill_pct × km)
W_VIEWPOINT_BONUS = 50.0   # cost units subtracted per viewpoint near route
VIEWPOINT_BUFFER_M = 500   # how close to the route a viewpoint must be


def _coords(feature: dict) -> list[list[float]]:
    return feature.get("geometry", {}).get("coordinates", []) or []


def _num(props: dict, key: str, default: float = 0.0) -> float:
    v = props.get(key, default)
    try:
        return float(v)
    except Exception:
        return default


def curvy_descent_penalty(coords: list[list[float]]) -> float:
    """Sum of (curvature_deg × downhill_% × segment_km) over the polyline.

    Only counts positive descents — flat or uphill bends contribute zero.
    Each coordinate is `[lon, lat, ele_m]` from BRouter.
    """
    if len(coords) < 3:
        return 0.0
    total = 0.0
    for i in range(1, len(coords) - 1):
        p_prev, p_cur, p_next = coords[i - 1], coords[i], coords[i + 1]
        d1 = haversine_m(p_prev, p_cur)
        d2 = haversine_m(p_cur, p_next)
        if d1 < 1.0 or d2 < 1.0:
            continue
        if len(p_prev) < 3 or len(p_next) < 3:
            continue
        b1 = initial_bearing(p_prev, p_cur)
        b2 = initial_bearing(p_cur, p_next)
        curve_deg = bearing_diff(b1, b2)
        dz = p_next[2] - p_prev[2]
        dxy = d1 + d2
        downhill_pct = -dz / dxy * 100.0  # >0 when descending
        if downhill_pct <= 0:
            continue
        total += curve_deg * downhill_pct * (dxy / 1000.0)
    return total


def viewpoints_near_route(coords: list[list[float]], buffer_m: int = VIEWPOINT_BUFFER_M) -> int:
    """Count distinct viewpoint POIs within buffer_m of the route polyline."""
    if not coords:
        return 0
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    deg = buffer_m / 110_000.0  # rough — fine for ~few-hundred-meter buffers
    bbox = (min(lons) - deg, min(lats) - deg, max(lons) + deg, max(lats) + deg)
    candidates = query_bbox(bbox, ["viewpoint"], 5000)
    if not candidates:
        return 0
    # Subsample the route — checking every coord against every viewpoint is
    # O(coords × viewpoints). On a long tour the route polyline can have
    # tens of thousands of points; 500 samples keeps this snappy.
    step = max(1, len(coords) // 500)
    sample = [(c[0], c[1]) for c in coords[::step]]
    n = 0
    for vp in candidates:
        vlonlat = (vp["lon"], vp["lat"])
        for s in sample:
            if haversine_m(s, vlonlat) <= buffer_m:
                n += 1
                break
    return n


def score_route(feature: dict) -> dict:
    coords = _coords(feature)
    props = feature.get("properties", {})
    track_len_km = _num(props, "track-length") / 1000.0
    # BRouter spells these with a space, not a dash. The frontend used to use
    # the dashed form too; both keys are now read so older clients keep working.
    ascend_m = _num(props, "filtered ascend") or _num(props, "filtered-ascend")
    raw_cost = _num(props, "cost")
    curvy = curvy_descent_penalty(coords)
    viewpoints = viewpoints_near_route(coords)
    composite = raw_cost + curvy * W_CURVY_DESCENT - viewpoints * W_VIEWPOINT_BONUS
    feature.setdefault("properties", {})["scoring"] = {
        "track_length_km": round(track_len_km, 2),
        "ascend_m": ascend_m,
        "raw_cost": raw_cost,
        "curvy_descent_penalty": round(curvy, 1),
        "viewpoints_near_route": viewpoints,
        "composite_score": round(composite, 1),
    }
    return feature


def rerank(routes: list[dict]) -> list[dict]:
    for r in routes:
        score_route(r)
    routes.sort(key=lambda r: r["properties"]["scoring"]["composite_score"])
    return routes
