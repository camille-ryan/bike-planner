"""Pure-Python geo utilities: haversine distance and forward bearing.

Kept dependency-free (no shapely) — we only need scalar math on (lon, lat)
points for the route-scoring pass.
"""
from math import acos, asin, atan2, cos, degrees, radians, sin, sqrt

EARTH_R = 6371000.0  # meters


def haversine_m(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    rl1, rl2 = radians(p1[1]), radians(p2[1])
    dlat = rl2 - rl1
    dlon = radians(p2[0] - p1[0])
    a = sin(dlat / 2) ** 2 + cos(rl1) * cos(rl2) * sin(dlon / 2) ** 2
    return 2 * EARTH_R * asin(sqrt(a))


def initial_bearing(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Forward azimuth (degrees, [0, 360)) from p1 to p2."""
    lat1, lat2 = radians(p1[1]), radians(p2[1])
    dlon = radians(p2[0] - p1[0])
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    return (degrees(atan2(y, x)) + 360) % 360


def bearing_diff(b1: float, b2: float) -> float:
    """Smallest unsigned angular difference between two bearings, in [0, 180]."""
    d = (b2 - b1 + 540) % 360 - 180
    return abs(d)


def along_cross_track_m(
    start: tuple[float, float],
    end: tuple[float, float],
    point: tuple[float, float],
) -> tuple[float, float]:
    """Project `point` onto the great-circle from `start` to `end`.

    Returns `(along_m, cross_m)`:
      - along_m is signed distance along the great-circle from start
        (positive = toward end, negative = behind start, > total = past end)
      - cross_m is unsigned perpendicular distance from the line

    Useful for "is this anchor near the corridor and how far along?"
    Standard great-circle formulas; sufficient for distances up to a few
    thousand km without flat-Earth artifacts.
    """
    d13 = haversine_m(start, point) / EARTH_R          # angular distance
    if d13 == 0.0:
        return (0.0, 0.0)
    theta12 = radians(initial_bearing(start, end))
    theta13 = radians(initial_bearing(start, point))
    cross_ang = asin(sin(d13) * sin(theta13 - theta12))
    cross_m = abs(cross_ang) * EARTH_R
    # Guard against acos domain edge cases when point is essentially on the line.
    cos_along = cos(d13) / cos(cross_ang) if abs(cross_ang) < 1.5 else 1.0
    cos_along = max(-1.0, min(1.0, cos_along))
    along_m = acos(cos_along) * EARTH_R
    # Sign: if the bearing-from-start to the point is > 90° away from the
    # bearing-to-end, the point is behind us along the great circle.
    db = bearing_diff(degrees(theta12), degrees(theta13))
    if db > 90.0:
        along_m = -along_m
    return (along_m, cross_m)
