"""SpatiaLite POI queries."""
import json
import sqlite3
from contextlib import contextmanager

from .settings import POIS_DB


@contextmanager
def _conn():
    conn = sqlite3.connect(POIS_DB)
    try:
        conn.enable_load_extension(True)
        conn.load_extension("mod_spatialite")
        conn.enable_load_extension(False)
        yield conn
    finally:
        conn.close()


def _row_to_dict(row, cols) -> dict:
    d = dict(zip(cols, row))
    if "tags" in d and d["tags"]:
        try:
            d["tags"] = json.loads(d["tags"])
        except Exception:
            pass
    return d


def query_bbox(
    bbox: tuple[float, float, float, float],
    categories: list[str] | None,
    limit: int,
) -> list[dict]:
    """Spatial-index lookup of POIs within `bbox` (minlon, minlat, maxlon, maxlat)."""
    minlon, minlat, maxlon, maxlat = bbox
    sql = (
        "SELECT id, osm_type, osm_id, category, subtype, name, country, tags, "
        "       X(geom) AS lon, Y(geom) AS lat "
        "FROM pois "
        "WHERE ROWID IN (SELECT ROWID FROM SpatialIndex "
        "                 WHERE f_table_name='pois' "
        "                   AND search_frame=BuildMbr(?,?,?,?,4326))"
    )
    args: list = [minlon, minlat, maxlon, maxlat]
    if categories:
        placeholders = ",".join("?" for _ in categories)
        sql += f" AND category IN ({placeholders})"
        args.extend(categories)
    sql += " LIMIT ?"
    args.append(limit)
    cols = [
        "id", "osm_type", "osm_id", "category", "subtype",
        "name", "country", "tags", "lon", "lat",
    ]
    with _conn() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [_row_to_dict(r, cols) for r in rows]
