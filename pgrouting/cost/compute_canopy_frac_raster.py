"""Raster-based canopy_frac compute — experimental alternative to the
polygon-based `compute_canopy_frac.py`.

Approach:
  1. Pull all forest polygons (landcover.class='forest') intersecting
     the bbox into memory as shapely geometries.
  2. Rasterize them into a binary uint8 forest mask at fixed
     pixel resolution (default ~20 m).
  3. Stream every bike edge whose endpoints fall in the bbox; sample
     the mask at N points along each 2-point line; canopy_frac is the
     fraction of samples that land in a forest pixel.
  4. COPY (gid, canopy_frac) back into ways via a temp table.

Why this is faster than the polygon path:
  - O(N_edges × samples_per_edge) instead of
    O(N_edges × avg_forests_per_edge × avg_vertices_per_forest).
  - All per-edge work is numpy array lookup — no PostGIS per-pair
    ST_Intersection.
  - The same forest raster can be reused (with a convolution) for
    `nearby_forest_frac`, `low_traffic_feel`, etc. — see
    feature_profiles_and_scenicness memory for the broader plan.

Trade-off: pixel resolution becomes the canopy precision floor. At
20 m, an edge passing within ~10 m of a forest edge may be classified
either way. For canopy this is well below the precision we care
about; the polygon path was already losing ~5 m to
ST_SimplifyPreserveTopology, so the gap is small.

Validation: run against the same DB after a polygon-based bake +
snapshot of `ways.canopy_frac` into `ways.canopy_frac_polygon` so the
two columns can be diffed.

Run:
  PGDATABASE=bike_v2_test PYTHONUNBUFFERED=1 \\
    python3 compute_canopy_frac_raster.py --bbox 14.5,46.8,17.0,48.5
"""
import argparse
import time

import numpy as np
import psycopg
import rasterio.features
import rasterio.transform
import shapely.wkb

import config


# 20 m gives a ~89 MB raster over the Austria bbox (2.5°×1.7°) and is
# small enough that the convolutions needed for nearby_forest_frac
# stay cheap on the same grid.
_RES_M_DEFAULT = 20.0

# Per-edge sample spacing in meters. Sample count = clamp(length_m /
# spacing, 3, 50). For a typical 50 m edge, 5 samples; for a 500 m
# edge, 50.
_SAMPLE_SPACING_M = 10.0
_SAMPLES_MIN = 3
_SAMPLES_MAX = 50

# Edges streamed per DB cursor batch. Tunable; the per-edge work is
# cheap numpy, so larger batches reduce round-trip overhead.
_EDGE_BATCH = 100_000


def _deg_per_meter_at(lat: float) -> tuple[float, float]:
    """Return (deg_per_meter_lon, deg_per_meter_lat) at this latitude.

    Earth ≈ 111 km per degree of latitude everywhere. Longitude scales
    by cos(lat). Single-anchor approximation is fine for an Austria-
    sized bbox — for a full-Europe bake we'd want per-row scaling.
    """
    deg_per_m_lat = 1.0 / 111_000.0
    deg_per_m_lon = 1.0 / (111_000.0 * np.cos(np.radians(lat)))
    return deg_per_m_lon, deg_per_m_lat


def build_forest_raster(conn,
                        bbox: tuple[float, float, float, float],
                        res_m: float
                        ) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Load forest polygons inside bbox and rasterize to a binary mask.

    Returns (mask, transform). `mask` is uint8 shape (H, W); 1 = forest.
    `transform` maps pixel (col, row) → (lon, lat) per rasterio.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = (min_lat + max_lat) / 2.0
    deg_lon_per_m, deg_lat_per_m = _deg_per_meter_at(mid_lat)
    px_lon = res_m * deg_lon_per_m
    px_lat = res_m * deg_lat_per_m
    width = int(np.ceil((max_lon - min_lon) / px_lon))
    height = int(np.ceil((max_lat - min_lat) / px_lat))
    print(f"[raster] grid: {width}x{height} ({width*height/1e6:.1f}M px) "
          f"@ {res_m:.0f}m resolution", flush=True)

    transform = rasterio.transform.from_bounds(
        min_lon, min_lat, max_lon, max_lat, width, height,
    )

    print(f"[raster] loading forest polygons in bbox...", flush=True)
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ST_AsBinary(geom) FROM landcover "
            "WHERE class = 'forest' "
            "  AND ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 4326))",
            bbox,
        )
        polys = [shapely.wkb.loads(bytes(row[0])) for row in cur]
    print(f"[raster] loaded {len(polys):,} forest polygons in "
          f"{time.time()-t0:.1f}s", flush=True)

    print(f"[raster] rasterizing...", flush=True)
    t0 = time.time()
    mask = rasterio.features.rasterize(
        ((g, 1) for g in polys),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    forest_frac = mask.sum() / mask.size
    print(f"[raster] rasterized in {time.time()-t0:.1f}s "
          f"({forest_frac*100:.1f}% of bbox is forest)", flush=True)
    return mask, transform


def _sample_edges(mask: np.ndarray,
                  transform: "rasterio.Affine",
                  rows: list[tuple]
                  ) -> list[tuple[int, float]]:
    """Vectorized: for each edge row, sample mask at N points along the
    2-point line and return (gid, canopy_frac).

    `rows` is a list of (gid, slon, slat, tlon, tlat, length_m).
    """
    h, w = mask.shape
    # rasterio's transform maps (col, row) -> (lon, lat). The inverse
    # gives (col, row) from (lon, lat).
    inv = ~transform
    results: list[tuple[int, float]] = []
    for (gid, slon, slat, tlon, tlat, length_m) in rows:
        n = int(round(max(_SAMPLES_MIN,
                          min(_SAMPLES_MAX, length_m / _SAMPLE_SPACING_M))))
        ts = np.linspace(0.0, 1.0, n)
        lons = slon + (tlon - slon) * ts
        lats = slat + (tlat - slat) * ts
        # Vectorized affine inverse via rasterio.transform.rowcol
        # (it's a thin wrapper around `~transform`). Using inv * pair
        # would be per-point; numpy approach is faster.
        cols = ((lons - transform.c) / transform.a).astype(np.int64)
        rs   = ((lats - transform.f) / transform.e).astype(np.int64)
        valid = (cols >= 0) & (cols < w) & (rs >= 0) & (rs < h)
        if not valid.any():
            results.append((gid, 0.0))
            continue
        hits = mask[rs[valid], cols[valid]].sum()
        frac = float(hits) / int(valid.sum())
        results.append((gid, frac))
    return results


def compute(conn: psycopg.Connection,
            bbox: tuple[float, float, float, float],
            res_m: float = _RES_M_DEFAULT) -> None:
    """Populate ways.canopy_frac inside bbox via raster sampling."""
    t_total = time.time()
    mask, transform = build_forest_raster(conn, bbox, res_m)

    # Stream edges + write back via temp table for batched COPY.
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _canopy_raster ("
            "  gid bigint PRIMARY KEY, frac real NOT NULL)"
        )
        cur.execute("TRUNCATE _canopy_raster")
    conn.commit()

    print(f"[raster] streaming edges + sampling...", flush=True)
    t0 = time.time()
    total_edges = 0
    nonzero = 0

    # Stream-process via a server-side cursor: postgres yields one
    # row at a time (or in `itersize` chunks), we sample and COPY
    # straight into _canopy_raster within the same uncommitted
    # transaction. Memory stays bounded to the raster (88 MB) + one
    # batch of edge tuples (a few MB). No commit inside the loop —
    # the server-side cursor would die on commit (psycopg
    # InvalidCursorName).
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute("SET work_mem = '512MB'")
    print(f"[raster] streaming edges (server cursor, "
          f"{_EDGE_BATCH:,}/fetch)...", flush=True)
    t_sample = time.time()
    last_print = time.time()

    with conn.cursor(name="raster_edge_scan") as scan_cur:
        scan_cur.itersize = _EDGE_BATCH
        scan_cur.execute(
            "SELECT w.gid, ST_X(vs.the_geom), ST_Y(vs.the_geom), "
            "       ST_X(vt.the_geom), ST_Y(vt.the_geom), w.length_m "
            "FROM ways w "
            "JOIN ways_vertices_pgr vs ON vs.id = w.source "
            "JOIN ways_vertices_pgr vt ON vt.id = w.target "
            "WHERE vs.lon BETWEEN %s AND %s AND vs.lat BETWEEN %s AND %s "
            "  AND vt.lon BETWEEN %s AND %s AND vt.lat BETWEEN %s AND %s",
            (bbox[0], bbox[2], bbox[1], bbox[3]) * 2,
        )

        batch: list[tuple] = []
        for row in scan_cur:
            batch.append(row)
            if len(batch) < _EDGE_BATCH:
                continue
            results = _sample_edges(mask, transform, batch)
            with conn.cursor() as wcur:
                with wcur.copy(
                    "COPY _canopy_raster (gid, frac) FROM STDIN"
                ) as cp:
                    for gid, frac in results:
                        cp.write_row((gid, frac))
                        if frac > 0:
                            nonzero += 1
            total_edges += len(batch)
            del batch, results
            batch = []
            if time.time() - last_print > 5:
                rate = total_edges / max(time.time() - t_sample, 1e-3)
                print(f"[raster]   {total_edges:,} edges sampled "
                      f"({rate:.0f}/s, nonzero={nonzero:,})",
                      flush=True)
                last_print = time.time()
        if batch:
            results = _sample_edges(mask, transform, batch)
            with conn.cursor() as wcur:
                with wcur.copy(
                    "COPY _canopy_raster (gid, frac) FROM STDIN"
                ) as cp:
                    for gid, frac in results:
                        cp.write_row((gid, frac))
                        if frac > 0:
                            nonzero += 1
            total_edges += len(batch)

    conn.commit()
    print(f"[raster] sampling+COPY done: {total_edges:,} edges in "
          f"{(time.time()-t_sample)/60:.1f} min; nonzero={nonzero:,}",
          flush=True)

    sample_dt = time.time() - t0
    print(f"[raster] sampling done: {total_edges:,} edges in "
          f"{sample_dt/60:.1f} min ({total_edges/sample_dt:.0f}/s); "
          f"nonzero={nonzero:,}", flush=True)

    # Bulk UPDATE from temp table.
    print(f"[raster] applying to ways.canopy_frac...", flush=True)
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute(
            "UPDATE ways w SET canopy_frac = "
            "  LEAST(1.0::real, GREATEST(0.0::real, r.frac::real)) "
            "FROM _canopy_raster r WHERE w.gid = r.gid"
        )
        updated = cur.rowcount
    conn.commit()
    print(f"[raster] applied to {updated:,} edges in "
          f"{time.time()-t0:.1f}s", flush=True)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT AVG(canopy_frac), MAX(canopy_frac), "
            "  COUNT(*) FILTER (WHERE canopy_frac > 0), "
            "  COUNT(*) FILTER (WHERE canopy_frac > 0.5), "
            "  COUNT(*) FILTER (WHERE canopy_frac >= 0.99) "
            "FROM ways"
        )
        avg, mx, gt0, gt50, full = cur.fetchone()
    print(f"[raster] totals: avg={avg:.4f} max={mx:.4f} "
          f"nonzero={gt0:,} >50%={gt50:,} ~full={full:,}", flush=True)
    print(f"[raster] done in {(time.time()-t_total)/60:.1f} min total",
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bbox", required=True,
                   help="min_lon,min_lat,max_lon,max_lat")
    p.add_argument("--res-m", type=float, default=_RES_M_DEFAULT,
                   help=f"Raster resolution in meters (default "
                        f"{_RES_M_DEFAULT})")
    args = p.parse_args()
    parts = tuple(float(x) for x in args.bbox.split(","))
    if len(parts) != 4:
        raise SystemExit("--bbox must be 4 comma-separated floats")
    with psycopg.connect(config.PG_DSN) as conn:
        compute(conn, parts, res_m=args.res_m)
