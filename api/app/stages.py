"""Split a route into ~target_km legs and find lodging clusters near each split."""
from .brouter import fetch_route
from .geo import haversine_m
from .pois import query_bbox


async def plan(
    from_lonlat: tuple[float, float],
    to_lonlat: tuple[float, float],
    profile: str,
    target_km: float,
    lodging_radius_m: int,
) -> dict:
    feature = await fetch_route([from_lonlat, to_lonlat], profile, 0)
    coords = feature["geometry"]["coordinates"]
    target_m = target_km * 1000.0

    legs: list[dict] = []
    accumulated = 0.0
    leg_start_idx = 0
    for i in range(1, len(coords)):
        accumulated += haversine_m(
            (coords[i - 1][0], coords[i - 1][1]),
            (coords[i][0], coords[i][1]),
        )
        if accumulated >= target_m:
            legs.append({
                "start": [coords[leg_start_idx][0], coords[leg_start_idx][1]],
                "end":   [coords[i][0], coords[i][1]],
                "length_m": int(accumulated),
                "ascend_m": _ascend_between(coords, leg_start_idx, i),
            })
            leg_start_idx = i
            accumulated = 0.0
    if accumulated > 0 and leg_start_idx < len(coords) - 1:
        last_idx = len(coords) - 1
        legs.append({
            "start": [coords[leg_start_idx][0], coords[leg_start_idx][1]],
            "end":   [coords[last_idx][0], coords[last_idx][1]],
            "length_m": int(accumulated),
            "ascend_m": _ascend_between(coords, leg_start_idx, last_idx),
        })

    for leg in legs:
        leg["lodging"] = _lodging_near(leg["end"], lodging_radius_m)

    return {
        "from": list(from_lonlat),
        "to": list(to_lonlat),
        "profile": profile,
        "target_km": target_km,
        "total_legs": len(legs),
        "total_length_m": sum(l["length_m"] for l in legs),
        "legs": legs,
    }


def _ascend_between(coords: list[list[float]], i_start: int, i_end: int) -> int:
    """Sum of positive elevation deltas between two indices in the polyline."""
    total = 0.0
    for j in range(i_start + 1, i_end + 1):
        if len(coords[j]) >= 3 and len(coords[j - 1]) >= 3:
            dz = coords[j][2] - coords[j - 1][2]
            if dz > 0:
                total += dz
    return int(total)


def _lodging_near(point: list[float], radius_m: int) -> list[dict]:
    deg = radius_m / 110_000.0
    bbox = (point[0] - deg, point[1] - deg, point[0] + deg, point[1] + deg)
    candidates = query_bbox(bbox, ["lodging"], 200)
    p = (point[0], point[1])
    in_range = [
        {**c, "distance_m": int(haversine_m(p, (c["lon"], c["lat"])))}
        for c in candidates
        if haversine_m(p, (c["lon"], c["lat"])) <= radius_m
    ]
    in_range.sort(key=lambda c: c["distance_m"])
    return in_range[:15]
