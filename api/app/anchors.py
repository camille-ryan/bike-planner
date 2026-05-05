"""Auto-waypoint selection for long-distance routes.

BRouter's A* search cost grows roughly exponentially with start-end distance,
so a Graz->Copenhagen point-to-point query takes ~8 minutes on this stack
(see brouter/Dockerfile note). Adding intermediate via-points along the
corridor turns it into a sequence of short legs, each cheap.

This module:
  1. Pulls candidate `place=city|town` anchors near the start->end great circle
     from the SpatiaLite `anchors` table.
  2. Greedily walks from start toward end, picking the next anchor in a
     target spacing band, preferring those closest to the corridor line.

Empirically (BRouter direct, lht profile): direct 489s, 7 waypoints 209s,
10 waypoints (~120 km spacing) 140s. ~120 km is the sweet spot — denser
spacing wins more time but adds detour cost via off-corridor anchors.
"""
import sqlite3

from .geo import along_cross_track_m, haversine_m
from .pois import _conn  # reuse the spatialite connection helper

# Tunables — see module docstring for empirical justification.
TARGET_STEP_M = 120_000          # preferred spacing between auto-waypoints
MIN_STEP_M    =  60_000          # don't pick anchors closer than this
MAX_STEP_M    = 200_000          # if no anchor is closer than this, give up gracefully
MAX_CROSS_M   =  60_000          # ignore anchors farther than this from the corridor line
DISTANCE_TRIGGER_M = 250_000     # routes shorter than this don't need auto-waypointing


def _candidates_along_corridor(
    start: tuple[float, float],
    end: tuple[float, float],
) -> list[tuple[float, float, float, float, str]]:
    """Return (along_m, cross_m, lon, lat, name) for every anchor whose
    perpendicular distance to the start->end great circle is <= MAX_CROSS_M
    and which projects between start and end.

    A bbox prefilter via SpatialIndex narrows the SQL scan to a rectangle
    around the corridor; the precise along/cross filtering is done in Python
    because spatial extensions don't expose great-circle projection.
    """
    # Buffer the corridor bbox by MAX_CROSS_M (~60 km ≈ 0.55° at our latitudes).
    buffer_deg = MAX_CROSS_M / 100_000.0
    minlon = min(start[0], end[0]) - buffer_deg
    maxlon = max(start[0], end[0]) + buffer_deg
    minlat = min(start[1], end[1]) - buffer_deg
    maxlat = max(start[1], end[1]) + buffer_deg
    sql = (
        "SELECT name, X(geom) AS lon, Y(geom) AS lat "
        "FROM anchors "
        "WHERE ROWID IN (SELECT ROWID FROM SpatialIndex "
        "                 WHERE f_table_name='anchors' "
        "                   AND search_frame=BuildMbr(?,?,?,?,4326))"
    )
    try:
        with _conn() as conn:
            rows = conn.execute(sql, (minlon, minlat, maxlon, maxlat)).fetchall()
    except sqlite3.OperationalError:
        # anchors table missing — older DB. Caller will fall back to direct routing.
        return []
    total_m = haversine_m(start, end)
    out = []
    for name, lon, lat in rows:
        along_m, cross_m = along_cross_track_m(start, end, (lon, lat))
        if cross_m > MAX_CROSS_M:
            continue
        if along_m <= MIN_STEP_M or along_m >= total_m - MIN_STEP_M:
            continue  # too close to the endpoints to be a useful waypoint
        out.append((along_m, cross_m, lon, lat, name or ""))
    out.sort(key=lambda x: x[0])
    return out


def _greedy_pick(
    candidates: list[tuple[float, float, float, float, str]],
    total_m: float,
) -> list[tuple[float, float, str]]:
    """Walk from along=0 to along=total_m, greedily picking the next anchor
    that's roughly TARGET_STEP_M ahead and closest to the corridor line.

    Returns ordered list of (lon, lat, name) waypoints (excluding endpoints).
    """
    chosen: list[tuple[float, float, str]] = []
    cur = 0.0
    while True:
        remaining = total_m - cur
        if remaining <= TARGET_STEP_M * 1.3:
            break  # last leg is short enough — head straight to end
        # Preferred band: anchors in [target*0.7, target*1.3] ahead.
        lo, hi = cur + TARGET_STEP_M * 0.7, cur + TARGET_STEP_M * 1.3
        band = [c for c in candidates if lo <= c[0] <= hi]
        if not band:
            # Loosen: anything in [MIN_STEP, MAX_STEP] ahead.
            lo, hi = cur + MIN_STEP_M, cur + MAX_STEP_M
            band = [c for c in candidates if lo <= c[0] <= hi]
            if not band:
                break  # gap too large — accept the long leg rather than recurse
        # Among band candidates, prefer the one closest to the corridor line.
        best = min(band, key=lambda c: c[1])
        chosen.append((best[2], best[3], best[4]))
        cur = best[0]
    return chosen


def auto_waypoints(
    start: tuple[float, float],
    end: tuple[float, float],
) -> list[tuple[float, float, str]]:
    """Pick intermediate `place=city|town` waypoints between start and end.

    Returns [] if the route is short, or if no anchors table / no anchors
    near the corridor — caller should fall back to direct routing.
    """
    total_m = haversine_m(start, end)
    if total_m < DISTANCE_TRIGGER_M:
        return []
    candidates = _candidates_along_corridor(start, end)
    if not candidates:
        return []
    return _greedy_pick(candidates, total_m)
