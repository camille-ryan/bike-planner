"""Materialize lodging POIs from pois.sqlite into postgres.

The source `pois` table in sqlite stores SpatiaLite POINT blobs;
extracting lon/lat requires loading mod_spatialite (same pattern as
scenicness/rasters.py::rasterize_points).

Per-country, idempotent: DELETE-then-INSERT rows for the given country.
Pass `countries=None` to ingest everything in pois.sqlite at once.
"""
from __future__ import annotations
import sqlite3
import time
from pathlib import Path
from typing import Iterable

import psycopg

import config


# OSM tourism subtypes we consider "lodging" — keeps the table focused.
# Filtering by the user-facing subset (e.g. drop camp_site / wilderness_hut)
# happens at query time, not ingest time.
LODGING_SUBTYPES = (
    "hotel", "guest_house", "hostel", "motel",
    "camp_site", "wilderness_hut",
)


def _read_sqlite(sqlite_path: Path,
                 countries: Iterable[str] | None) -> list[tuple]:
    """Read (osm_id, name, subtype, country, lon, lat) rows for lodging
    POIs from the sqlite `pois` table."""
    con = sqlite3.connect(str(sqlite_path))
    con.enable_load_extension(True)
    con.execute("SELECT load_extension('mod_spatialite')")
    sql = (
        "SELECT osm_id, name, subtype, country, X(geom), Y(geom) "
        "FROM pois WHERE category = 'lodging' "
        "AND subtype IN (" + ",".join("?" * len(LODGING_SUBTYPES)) + ")"
    )
    params: list = list(LODGING_SUBTYPES)
    if countries:
        countries = list(countries)
        sql += " AND country IN (" + ",".join("?" * len(countries)) + ")"
        params.extend(countries)
    rows = con.execute(sql, params).fetchall()
    con.close()
    return rows


def ingest(conn: psycopg.Connection,
           countries: Iterable[str] | None = None,
           sqlite_path: Path | None = None) -> None:
    """Replace lodging rows for the given countries (or all countries)."""
    if sqlite_path is None:
        sqlite_path = config.POIS_DB
    if not sqlite_path.exists():
        raise SystemExit(f"missing pois.sqlite: {sqlite_path}")
    t0 = time.time()
    rows = _read_sqlite(sqlite_path, countries)
    print(f"[lodging] read {len(rows):,} rows from {sqlite_path} "
          f"in {time.time()-t0:.1f}s")
    if not rows:
        print("[lodging] nothing to ingest")
        return
    with conn.cursor() as cur:
        if countries:
            cur.execute(
                "DELETE FROM lodging WHERE country = ANY(%s)",
                (list(countries),),
            )
        else:
            cur.execute("TRUNCATE lodging")
        cur.executemany(
            "INSERT INTO lodging "
            "(osm_id, name, subtype, country, geom) VALUES "
            "(%s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))",
            rows,
        )
        cur.execute("ANALYZE lodging")
    conn.commit()
    print(f"[lodging] inserted {len(rows):,} rows in "
          f"{time.time()-t0:.1f}s total")

    # Per-subtype summary
    with conn.cursor() as cur:
        cur.execute(
            "SELECT subtype, COUNT(*) FROM lodging "
            + ("WHERE country = ANY(%s) " if countries else "")
            + "GROUP BY subtype ORDER BY 2 DESC",
            (list(countries),) if countries else (),
        )
        for subtype, n in cur:
            print(f"[lodging]   {subtype:18s} {n:,}")
