"""Export a Graz->Wien route from the connected Postgres DB as one
feature of `web/public/data/graz_wien_compare.geojson`. Run before
and after a cost recompute to populate both variants in the same
file.

Typical use against the parallel V2 test DB:

  # 1) Current state (elev+curv on, no canopy yet):
  PGHOST=localhost PGDATABASE=bike_v2_test \\
    python3 export_route_compare.py --variant no_canopy

  # 2) ... landcover-ingest + canopy-compute + recompute-cost ...

  # 3) After canopy is folded in:
  PGHOST=localhost PGDATABASE=bike_v2_test \\
    python3 export_route_compare.py --variant with_canopy

Each run reads the existing GeoJSON (if any), drops the feature
matching `--variant`, runs pgr_dijkstra Graz->Wien against the
current `ways.cost` / `reverse_cost`, computes route properties,
and writes the file back with the new feature appended. Other
variants are preserved.
"""
import argparse
import json
from pathlib import Path

import psycopg

import config


GRAZ_LON, GRAZ_LAT = 15.4395, 47.0707
WIEN_LON, WIEN_LAT = 16.3725, 48.2082

# Bbox the pgr_dijkstra inner edges_sql to the Graz<->Wien corridor.
# Without this, pgr_dijkstra loads all ~22 M edges into Postgres
# memory and the backend OOMs at the 6 GB container cap. The bbox
# matches what canopy-compute / recompute-cost use.
CORRIDOR_BBOX = (14.5, 46.8, 17.0, 48.5)  # min_lon, min_lat, max_lon, max_lat

# Default sits outside DATA_DIR — convenient when running from the
# host, where the repo and the web tree are visible. Override with
# --out for other layouts.
DEFAULT_OUT = Path("/mnt/e/proj/bike/web/public/data/graz_wien_compare.geojson")

# Variant -> human-readable display name. Stored in feature.properties.name
# so the web app can render it without hard-coded mappings.
VARIANT_NAMES = {
    "no_canopy":   "V2 (elev+curv, no canopy)",
    "with_canopy": "V2 + canopy bonus",
}


def snap_vertex(cur, lon: float, lat: float) -> int:
    cur.execute(
        "SELECT id FROM ways_vertices_pgr "
        "ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326) "
        "LIMIT 1",
        (lon, lat),
    )
    row = cur.fetchone()
    if not row:
        raise SystemExit("ways_vertices_pgr is empty — bad DB?")
    return int(row[0])


def build_feature(conn, variant: str) -> dict:
    """pgr_dijkstra Graz->Wien, return the GeoJSON feature."""
    with conn.cursor() as cur:
        cur.execute("SET work_mem = '2GB'")
        start_vid = snap_vertex(cur, GRAZ_LON, GRAZ_LAT)
        end_vid   = snap_vertex(cur, WIEN_LON, WIEN_LAT)
        print(f"[export]   start_vid={start_vid} end_vid={end_vid}")

        # Bbox the inner edges_sql — loading the full 22 M-edge graph
        # OOMs the 6 GB Postgres container. Floats are safe to inline
        # into the SQL string.
        b = CORRIDOR_BBOX
        edges_sql = (
            f"SELECT w.gid AS id, w.source, w.target, "
            f"       w.cost, w.reverse_cost "
            f"FROM ways w "
            f"JOIN ways_vertices_pgr vs ON vs.id = w.source "
            f"JOIN ways_vertices_pgr vt ON vt.id = w.target "
            f"WHERE vs.lon BETWEEN {b[0]} AND {b[2]} "
            f"  AND vs.lat BETWEEN {b[1]} AND {b[3]} "
            f"  AND vt.lon BETWEEN {b[0]} AND {b[2]} "
            f"  AND vt.lat BETWEEN {b[1]} AND {b[3]}"
        )

        cur.execute(
            "SELECT p.path_seq, p.node, p.edge, "
            "       w.source AS w_source, w.length_m, w.canopy_frac, "
            "       vs.elev_m AS elev_src, vt.elev_m AS elev_dst, "
            "       ST_X(vs.the_geom) AS slon, ST_Y(vs.the_geom) AS slat, "
            "       ST_X(vt.the_geom) AS tlon, ST_Y(vt.the_geom) AS tlat "
            "FROM pgr_dijkstra(%s, %s, %s, true) p "
            "JOIN ways w ON w.gid = p.edge "
            "JOIN ways_vertices_pgr vs ON vs.id = w.source "
            "JOIN ways_vertices_pgr vt ON vt.id = w.target "
            "WHERE p.edge != -1 "
            "ORDER BY p.path_seq",
            (edges_sql, start_vid, end_vid),
        )
        rows = cur.fetchall()

    if not rows:
        raise SystemExit("pgr_dijkstra returned empty path")

    coords: list[list[list[float]]] = []  # MultiLineString of 2-point segments
    cumul_km = 0.0
    climb_m = 0.0
    canopy_m = 0.0
    profile: list[list[float | None]] = []
    n_edges = 0

    for (path_seq, node, edge, w_source,
         length_m, canopy_frac,
         e_src, e_dst, slon, slat, tlon, tlat) in rows:
        # pgr_dijkstra's `node` is the vertex at the start of `edge`
        # in the traversal direction. node == w_source means we're
        # going source->target; otherwise we're going target->source.
        forward = (int(node) == int(w_source))

        if forward:
            coords.append([[float(slon), float(slat)], [float(tlon), float(tlat)]])
            d_elev = (float(e_dst) - float(e_src)
                      if e_src is not None and e_dst is not None else None)
        else:
            coords.append([[float(tlon), float(tlat)], [float(slon), float(slat)]])
            d_elev = (float(e_src) - float(e_dst)
                      if e_src is not None and e_dst is not None else None)

        seg_km = float(length_m) / 1000.0
        cumul_km += seg_km
        if d_elev is not None and d_elev > 0:
            climb_m += d_elev
        mid_elev = (((float(e_src) + float(e_dst)) / 2.0)
                    if e_src is not None and e_dst is not None else None)
        profile.append([round(cumul_km, 3),
                        round(mid_elev, 1) if mid_elev is not None else None])
        canopy_m += float(canopy_frac or 0.0) * float(length_m)
        n_edges += 1

    return {
        "type": "Feature",
        "geometry": {"type": "MultiLineString", "coordinates": coords},
        "properties": {
            "name": VARIANT_NAMES.get(variant, variant),
            "variant": variant,
            "edges": n_edges,
            "length_km": round(cumul_km, 2),
            "climb_m": round(climb_m),
            "canopy_m": round(canopy_m),
            "canopy_km": round(canopy_m / 1000.0, 2),
            "profile": profile,
            "profile_name": config.PG_DSN.split("dbname=")[-1].strip(),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True,
                   help="Variant tag (e.g. no_canopy, with_canopy). Used as "
                        "feature.properties.variant and to dedupe against the "
                        "existing file.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output GeoJSON path (default: {DEFAULT_OUT})")
    args = p.parse_args()

    with psycopg.connect(config.PG_DSN) as conn:
        feat = build_feature(conn, args.variant)

    out: Path = args.out
    fc: dict
    if out.exists():
        with out.open() as h:
            fc = json.load(h)
        # Drop any existing feature with this variant — re-runs replace.
        fc["features"] = [f for f in fc.get("features", [])
                          if f.get("properties", {}).get("variant") != args.variant]
    else:
        fc = {"type": "FeatureCollection", "features": []}
    fc["features"].append(feat)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as h:
        json.dump(fc, h)

    props = feat["properties"]
    print(f"[export] variant={args.variant} -> {out}")
    print(f"[export]   edges={props['edges']} length_km={props['length_km']} "
          f"climb_m={props['climb_m']} canopy_km={props['canopy_km']}")
    print(f"[export]   features in file now: "
          f"{[f['properties']['variant'] for f in fc['features']]}")


if __name__ == "__main__":
    main()
