"""Re-annotate `canopy_km` on each feature in graz_wien_compare.geojson
by walking the feature's existing line geometry against the live
`ways.canopy_frac` column.

Why this exists: the `no_canopy` variant was exported BEFORE
`canopy_frac` was populated, so its `canopy_km` is incorrectly 0.
This script doesn't re-route — it just looks up the right number
for the route that's already in the GeoJSON.

Approach:
  - For each variant feature, iterate the MultiLineString segments
    (each segment is one 2-point edge line).
  - COPY all segment endpoints into a temp `_segs` table.
  - JOIN to `ways_vertices_pgr` and `ways` to map back to gid +
    canopy_frac + length_m. Handle both directions of traversal
    (segment endpoints can come in either source->target or
    target->source order depending on dijkstra's direction).
  - SUM(canopy_frac * length_m) for matched edges.
  - Write `canopy_m` and `canopy_km` back onto the feature.

Runs in-place on the GeoJSON. Other property fields are preserved.
"""
import argparse
import json
from pathlib import Path

import psycopg

import config


DEFAULT_PATH = Path("/data/graz_wien_compare.geojson")


def _annotate_feature(conn, feat: dict) -> None:
    geom = feat.get("geometry") or {}
    if geom.get("type") != "MultiLineString":
        print(f"  skip: not MultiLineString ({geom.get('type')!r})")
        return
    segs: list[tuple[float, float, float, float]] = []
    for ring in geom["coordinates"]:
        if len(ring) < 2:
            continue
        # Treat each adjacent pair as an edge segment.
        for i in range(len(ring) - 1):
            (slon, slat) = ring[i]
            (tlon, tlat) = ring[i + 1]
            segs.append((float(slon), float(slat),
                         float(tlon), float(tlat)))

    print(f"  segments: {len(segs):,}")
    if not segs:
        return

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS _segs")
        cur.execute(
            "CREATE TEMP TABLE _segs ("
            "  ord int, slon double precision, slat double precision, "
            "  tlon double precision, tlat double precision)"
        )
        with cur.copy(
            "COPY _segs (ord, slon, slat, tlon, tlat) FROM STDIN"
        ) as cp:
            for i, (slon, slat, tlon, tlat) in enumerate(segs):
                cp.write_row((i, slon, slat, tlon, tlat))

        cur.execute("CREATE INDEX ON _segs (slon, slat, tlon, tlat)")
        cur.execute("ANALYZE _segs")

        # Match either traversal direction. Vertex lon/lat are stored
        # as double precision and originated from the same source as
        # the GeoJSON (export queried ST_X/ST_Y on the same vertex),
        # so equality comparison is safe.
        cur.execute(
            "WITH matched AS ("
            "  SELECT s.ord, w.gid, w.length_m, w.canopy_frac_polygon AS canopy_frac "
            "  FROM _segs s "
            "  JOIN ways_vertices_pgr vs ON vs.lon = s.slon AND vs.lat = s.slat "
            "  JOIN ways_vertices_pgr vt ON vt.lon = s.tlon AND vt.lat = s.tlat "
            "  JOIN ways w ON w.source = vs.id AND w.target = vt.id "
            "  UNION ALL "
            "  SELECT s.ord, w.gid, w.length_m, w.canopy_frac_polygon AS canopy_frac "
            "  FROM _segs s "
            "  JOIN ways_vertices_pgr vs ON vs.lon = s.tlon AND vs.lat = s.tlat "
            "  JOIN ways_vertices_pgr vt ON vt.lon = s.slon AND vt.lat = s.slat "
            "  JOIN ways w ON w.source = vs.id AND w.target = vt.id "
            ") "
            "SELECT COUNT(*), "
            "       COALESCE(SUM(canopy_frac * length_m), 0), "
            "       COALESCE(SUM(length_m), 0) "
            "FROM ("
            "  SELECT DISTINCT ON (ord) ord, length_m, canopy_frac "
            "  FROM matched ORDER BY ord, gid"
            ") m"
        )
        n_matched, canopy_m, total_m = cur.fetchone()
    canopy_m = float(canopy_m)
    total_m = float(total_m)

    print(f"  matched {n_matched:,} of {len(segs):,} segments; "
          f"covered length {total_m/1000:.2f} km")
    canopy_km = canopy_m / 1000.0
    print(f"  canopy_m = {canopy_m:.1f}  →  canopy_km = {canopy_km:.2f}")

    feat["properties"]["canopy_m"] = round(canopy_m)
    feat["properties"]["canopy_km"] = round(canopy_km, 2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, default=DEFAULT_PATH,
                   help="Path to compare GeoJSON (modified in-place).")
    args = p.parse_args()
    inp: Path = args.inp
    if not inp.exists():
        raise SystemExit(f"missing GeoJSON: {inp}")
    with inp.open() as h:
        fc = json.load(h)

    with psycopg.connect(config.PG_DSN) as conn:
        for feat in fc.get("features", []):
            v = feat.get("properties", {}).get("variant", "?")
            print(f"[reannotate] variant={v}")
            _annotate_feature(conn, feat)

    with inp.open("w") as h:
        json.dump(fc, h)
    print(f"[reannotate] wrote {inp}")
    for feat in fc["features"]:
        p = feat["properties"]
        print(f"  variant={p.get('variant')}: "
              f"length_km={p.get('length_km')} "
              f"climb_m={p.get('climb_m')} "
              f"canopy_km={p.get('canopy_km')}")


if __name__ == "__main__":
    main()
