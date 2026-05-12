"""Sample Copernicus DEM GLO-30 elevations at every graph vertex.

Run order: download_dem → ingest_pbf → ingest_dem.

Strategy:
  - Enumerate DEM tiles already present in `data/dem/`.
  - For each tile, query vertices whose lon/lat fall inside the
    tile's 1°×1° box AND whose `elev_m` is still NULL (so the step
    is incremental — re-running over a partially-populated table
    only touches the gaps).
  - Bilinear-sample the tile at every vertex's (lon, lat) in one
    vectorized pass.
  - COPY (id, elev_m) into a temp table and UPDATE ways_vertices_pgr
    in one statement per tile.

Bilinear (vs nearest-neighbor) matters here: at 30 m DEM resolution,
nearest-neighbor errors easily add several meters of sampling noise,
which translates to spurious grade on a 50 m edge. Bilinear smooths
that out.
"""
import argparse
from pathlib import Path

import numpy as np
import psycopg
import rasterio

import config
import download_dem


def _enumerate_tiles(dem_dir: Path) -> dict[tuple[int, int], Path]:
    """Map (sw_lat, sw_lon) → tile path for all GLO-30 tiles on disk."""
    out: dict[tuple[int, int], Path] = {}
    for p in sorted(dem_dir.glob("Copernicus_DSM_COG_10_*_DEM.tif")):
        # Parse e.g. Copernicus_DSM_COG_10_N47_00_E015_00_DEM
        stem = p.stem
        parts = stem.split("_")
        # parts = ["Copernicus","DSM","COG","10","N47","00","E015","00","DEM"]
        try:
            ns, ew = parts[4], parts[6]
            lat = int(ns[1:]) * (1 if ns[0] == "N" else -1)
            lon = int(ew[1:]) * (1 if ew[0] == "E" else -1)
            out[(lat, lon)] = p
        except (ValueError, IndexError):
            print(f"[dem-ingest] skipping unparseable filename: {p.name}")
    return out


def _sample_bilinear(tile_path: Path,
                     lons: np.ndarray,
                     lats: np.ndarray) -> np.ndarray:
    """Vectorized bilinear sample of a single DEM tile.

    Returns a float32 array of elevations; NaN where the sample falls
    in a no-data cell or hits the tile edge.
    """
    with rasterio.open(tile_path) as ds:
        arr = ds.read(1).astype(np.float32)
        # Convert lon/lat → fractional pixel coordinates via affine inverse.
        # rasterio's `ds.transform` maps (col, row) → (x, y).
        inv = ~ds.transform
        cols = np.empty_like(lons)
        rows = np.empty_like(lats)
        for i in range(len(lons)):
            c, r = inv * (lons[i], lats[i])
            cols[i] = c
            rows[i] = r
        c0 = np.floor(cols).astype(np.int64)
        r0 = np.floor(rows).astype(np.int64)
        fc = cols - c0
        fr = rows - r0

        h, w = arr.shape
        in_bounds = (r0 >= 0) & (r0 < h - 1) & (c0 >= 0) & (c0 < w - 1)
        out = np.full(len(lons), np.nan, dtype=np.float32)
        # Clamp for safe indexing on the in_bounds subset.
        r0c = np.clip(r0, 0, h - 2)
        c0c = np.clip(c0, 0, w - 2)
        v00 = arr[r0c, c0c]
        v01 = arr[r0c, c0c + 1]
        v10 = arr[r0c + 1, c0c]
        v11 = arr[r0c + 1, c0c + 1]
        # Mask no-data (rasterio surfaces it as ds.nodata, typically int16).
        nodata = ds.nodata
        if nodata is not None:
            bad = (v00 == nodata) | (v01 == nodata) | (v10 == nodata) | (v11 == nodata)
            in_bounds &= ~bad

        z = (v00 * (1 - fr) * (1 - fc)
             + v01 * (1 - fr) * fc
             + v10 * fr * (1 - fc)
             + v11 * fr * fc)
        out[in_bounds] = z[in_bounds].astype(np.float32)
        return out


def ingest(conn: psycopg.Connection, dem_dir: Path | None = None) -> None:
    dem_dir = dem_dir or config.DEM_DIR
    if not dem_dir.exists():
        raise SystemExit(f"DEM dir does not exist: {dem_dir} — run download_dem first")
    tiles = _enumerate_tiles(dem_dir)
    if not tiles:
        raise SystemExit(f"no DEM tiles found in {dem_dir}")
    print(f"[dem-ingest] {len(tiles)} tiles on disk, sampling vertices...")

    total_vertices = 0
    total_updated  = 0
    total_skipped_no_data = 0

    with conn.cursor() as cur:
        # NB: no ON COMMIT DROP — we commit per-tile (so a long run can
        # resume by re-running with the remaining `elev_m IS NULL`
        # vertices). Temp tables are session-scoped by default; the
        # connection closing at end of `ingest()` cleans up.
        cur.execute("""
            CREATE TEMP TABLE IF NOT EXISTS _vertex_elev (
                id     bigint PRIMARY KEY,
                elev_m real NOT NULL
            )
        """)
        # Each tile is independent; commit per-tile so a long run can
        # be resumed by re-running ingest (incremental: WHERE elev_m IS NULL).
        for i, ((lat, lon), tile_path) in enumerate(sorted(tiles.items()), 1):
            cur.execute(
                "SELECT id, lon, lat FROM ways_vertices_pgr "
                "WHERE elev_m IS NULL AND lat >= %s AND lat < %s "
                "AND lon >= %s AND lon < %s",
                (lat, lat + 1, lon, lon + 1),
            )
            rows = cur.fetchall()
            if not rows:
                print(f"[dem-ingest]   {i}/{len(tiles)} {tile_path.name}: "
                      f"no vertices in box, skip")
                continue

            ids  = np.array([r[0] for r in rows], dtype=np.int64)
            lons = np.array([r[1] for r in rows], dtype=np.float64)
            lats = np.array([r[2] for r in rows], dtype=np.float64)
            elevs = _sample_bilinear(tile_path, lons, lats)

            valid = np.isfinite(elevs)
            n_valid = int(valid.sum())
            n_total = len(rows)
            total_vertices += n_total
            total_skipped_no_data += (n_total - n_valid)

            if n_valid:
                cur.execute("TRUNCATE _vertex_elev")
                with cur.copy("COPY _vertex_elev (id, elev_m) FROM STDIN") as cp:
                    for vid, ve in zip(ids[valid], elevs[valid]):
                        cp.write_row((int(vid), float(ve)))
                cur.execute(
                    "UPDATE ways_vertices_pgr v "
                    "SET elev_m = e.elev_m "
                    "FROM _vertex_elev e WHERE v.id = e.id"
                )
                total_updated += cur.rowcount
            conn.commit()
            print(f"[dem-ingest]   {i}/{len(tiles)} {tile_path.name}: "
                  f"{n_valid}/{n_total} vertices updated")

    print(f"[dem-ingest] done. vertices_seen={total_vertices:,} "
          f"updated={total_updated:,} skipped_no_data={total_skipped_no_data:,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dem-dir", type=Path, default=None,
                   help="Override default DEM tile directory.")
    args = p.parse_args()
    with psycopg.connect(config.PG_DSN) as conn:
        ingest(conn, args.dem_dir)
