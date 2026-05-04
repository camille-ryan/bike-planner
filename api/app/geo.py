"""Pure-Python geo utilities: haversine distance and forward bearing.

Kept dependency-free (no shapely) — we only need scalar math on (lon, lat)
points for the route-scoring pass.
"""
from math import asin, atan2, cos, degrees, radians, sin, sqrt

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
