"""Dump the road edges that fall outside a 15 km euclidean buffer
around every anchor. The "gap" set is the routing dead zone where
paired-SPT b_frontier logic degenerates and routes detour around.

Output: /data/web_overlays/coverage_gap.geojson — toggleable layer
in the web UI for spotting where corridor anchors are most needed.

To keep the GeoJSON small enough for MapLibre to render smoothly, we
drop very short edges (< 100 m) — they're connector links inside
intersections and don't help diagnose corridor gaps.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import psycopg

import config


COVERAGE_RADIUS_M = 15_000.0
MIN_EDGE_LEN_M    = 100.0
OUT_PATH = Path("/data/web_overlays/coverage_gap.geojson")


def _covered_vids(conn: psycopg.Connection, radius_m: float) -> np.ndarray:
    """Union of ways_vertices_pgr.id within `radius_m` euclidean of any
    anchor. Per-anchor query is fast (GIST index + small bbox) so we
    avoid the global cross-join."""
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, ST_X(geom), ST_Y(geom) FROM anchors "
            "WHERE snap_vertex_id IS NOT NULL"
        )
        anchors = cur.fetchall()
    print(f"[gap] {len(anchors):,} anchors", flush=True)

    expand_deg = radius_m / 50_000.0 + 0.05   # ~0.35 for 15 km, generous prefilter
    covered: set[int] = set()
    for i, (anchor_id, lon, lat) in enumerate(anchors):
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM ways_vertices_pgr
                WHERE the_geom && ST_Expand(
                        ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s
                      )
                  AND ST_DWithin(
                        the_geom::geography,
                        ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                        %s
                      )
                """,
                (lon, lat, expand_deg, lon, lat, radius_m),
            )
            for r in cur:
                covered.add(int(r[0]))
        if (i + 1) % 25 == 0 or i == len(anchors) - 1:
            print(f"[gap]   processed {i+1:,}/{len(anchors):,} anchors  "
                  f"covered={len(covered):,}  ({time.time()-t0:.1f}s)",
                  flush=True)
    arr = np.fromiter(covered, dtype=np.int64)
    arr.sort()
    return arr


def _dump_gap_edges(
    conn: psycopg.Connection, covered: np.ndarray,
    min_len_m: float, out_path: Path,
) -> None:
    """Stream the ways table and write a GeoJSON FeatureCollection of
    edges where (a) both endpoints fall OUTSIDE `covered`, and (b) the
    edge is at least `min_len_m` long. Geometry comes from postgres
    ST_AsGeoJSON to avoid format mismatches."""
    print(f"[gap] streaming ways to find gap edges (length >= {min_len_m:.0f} m)…",
          flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_total = 0
    t0 = time.time()
    with open(out_path, "w") as fh:
        fh.write('{"type":"FeatureCollection","features":[\n')
        first = True
        with conn.cursor(name="gap_edge_cur") as cur:
            cur.itersize = 200_000
            # `ways` is the pgRouting logical graph — no geometry column.
            # Synthesize a straight-line LineString between source and
            # target vertex points; visually identical to the real curved
            # road at coverage-overlay zoom levels (≤14).
            cur.execute(
                """
                SELECT w.source, w.target, w.length_m,
                       ST_AsGeoJSON(ST_MakeLine(vs.the_geom, vt.the_geom), 6) AS geom
                FROM ways w
                JOIN ways_vertices_pgr vs ON vs.id = w.source
                JOIN ways_vertices_pgr vt ON vt.id = w.target
                WHERE w.length_m >= %s
                """,
                (min_len_m,),
            )
            batch_s: list[int] = []
            batch_t: list[int] = []
            batch_g: list[str] = []
            batch_l: list[float] = []
            for row in cur:
                n_total += 1
                batch_s.append(int(row[0]))
                batch_t.append(int(row[1]))
                batch_l.append(float(row[2]))
                batch_g.append(row[3])
                if len(batch_s) >= 200_000:
                    n_written += _flush_batch(
                        fh, covered, batch_s, batch_t, batch_l, batch_g,
                        is_first=first,
                    )
                    first = False
                    batch_s.clear(); batch_t.clear(); batch_l.clear(); batch_g.clear()
                    print(f"[gap]   scanned {n_total:,}  written {n_written:,}  "
                          f"({time.time()-t0:.1f}s)", flush=True)
            if batch_s:
                n_written += _flush_batch(
                    fh, covered, batch_s, batch_t, batch_l, batch_g,
                    is_first=first,
                )
        fh.write('\n]}')
    sz_mb = out_path.stat().st_size / 1e6
    print(f"[gap] wrote {n_written:,} gap edges (of {n_total:,} scanned, "
          f">= {min_len_m:.0f} m) to {out_path} ({sz_mb:.1f} MB) "
          f"in {time.time()-t0:.1f}s", flush=True)


def _flush_batch(
    fh, covered: np.ndarray,
    src: list[int], tgt: list[int], lens: list[float], geoms: list[str],
    is_first: bool,
) -> int:
    arr_s = np.fromiter(src, dtype=np.int64)
    arr_t = np.fromiter(tgt, dtype=np.int64)
    sp = np.searchsorted(covered, arr_s)
    tp = np.searchsorted(covered, arr_t)
    s_in = (sp < len(covered)) & (covered[np.clip(sp, 0, len(covered) - 1)] == arr_s)
    t_in = (tp < len(covered)) & (covered[np.clip(tp, 0, len(covered) - 1)] == arr_t)
    gap = ~s_in & ~t_in
    written = 0
    for i in np.flatnonzero(gap):
        if not is_first or written > 0:
            fh.write(",\n")
        # Tag the length so the UI can size/color by it if desired.
        fh.write('{"type":"Feature","properties":{"length_m":')
        fh.write(f"{lens[int(i)]:.0f}")
        fh.write('},"geometry":')
        fh.write(geoms[int(i)])
        fh.write('}')
        written += 1
    return written


def main() -> None:
    with psycopg.connect(config.PG_DSN) as conn:
        covered = _covered_vids(conn, COVERAGE_RADIUS_M)
        _dump_gap_edges(conn, covered, MIN_EDGE_LEN_M, OUT_PATH)


if __name__ == "__main__":
    main()
