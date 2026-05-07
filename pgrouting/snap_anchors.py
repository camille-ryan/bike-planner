"""Load anchors from `pois.sqlite`, snap to nearest graph vertex.

The POI ingest (Phase 1) produced an `anchors` table in SpatiaLite —
all `place=city|town` nodes from the corridor PBFs. We mirror those
rows into a Postgres `anchors` table here, with one extra column:
`snap_vertex_id` is the nearest `ways_vertices_pgr.id`. That snapped
vertex is the source we'll feed `pgr_drivingDistance` from.

Snapping uses PostGIS's `<->` KNN distance operator with the spatial
index on `ways_vertices_pgr`, so this is bounded by the size of the
anchor list (a few thousand) regardless of graph size.
"""
import sqlite3

import psycopg


def load_anchors_from_sqlite(pois_db: str, countries: list[str]) -> list[dict]:
    """Pull `place=city|town` rows for the requested countries."""
    placeholders = ",".join("?" for _ in countries)
    sql = (
        "SELECT name, place, population, country, "
        "       X(geom) AS lon, Y(geom) AS lat, osm_id "
        f"FROM anchors WHERE country IN ({placeholders}) "
        "AND name IS NOT NULL "
        "ORDER BY country, name"
    )
    conn = sqlite3.connect(pois_db)
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")
    rows = conn.execute(sql, countries).fetchall()
    conn.close()
    return [
        {"name": n, "place": p, "population": pop, "country": c,
         "lon": float(lo), "lat": float(la), "osm_id": int(oi or 0)}
        for n, p, pop, c, lo, la, oi in rows
    ]


def populate_anchors_table(conn: psycopg.Connection, anchors: list[dict]) -> None:
    """Truncate + repopulate the Postgres `anchors` table from the
    list, then snap each row to its nearest graph vertex."""
    print(f"[snap] populating anchors table with {len(anchors):,} rows")
    with conn.cursor() as cur:
        cur.execute("TRUNCATE anchors RESTART IDENTITY")
        with cur.copy(
            "COPY anchors (osm_id, name, place, population, country, geom) "
            "FROM STDIN"
        ) as cp:
            for a in anchors:
                cp.write_row((
                    a["osm_id"], a["name"], a["place"],
                    a["population"], a["country"],
                    f"SRID=4326;POINT({a['lon']} {a['lat']})",
                ))
    conn.commit()

    print("[snap] snapping each anchor to nearest graph vertex (KNN)...")
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE anchors a
            SET snap_vertex_id = nearest.id
            FROM (
                SELECT a2.id AS anchor_id,
                       (SELECT v.id
                          FROM ways_vertices_pgr v
                         ORDER BY v.the_geom <-> a2.geom
                         LIMIT 1) AS id
                FROM anchors a2
            ) AS nearest
            WHERE a.id = nearest.anchor_id
        """)
        print(f"[snap]   updated {cur.rowcount:,} rows")
    conn.commit()
