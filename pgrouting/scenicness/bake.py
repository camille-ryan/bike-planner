"""Top-level scenicness orchestrator (tiled).

Bakes signals over the ways-table extent (or --bbox override), tiling
internally to keep per-tile memory bounded. Lets the bake scale to
arbitrary country sizes without rebuilding the full raster in RAM.

Pipeline:
  1. Determine bake extent (from ways_vertices_pgr, or --bbox override).
  2. Tile the extent at `tile_size_deg` granularity.
  3. For each non-empty tile:
     a. Build the unique source rasters expanded by per-kernel halo
        (one builder call per unique source key).
     b. Apply each unique kernel to its source, then crop back to the
        tile bbox for PNG output. The halo'd version is kept for edge
        sampling so edges near tile boundaries see correct context.
     c. Stream ways edges whose SOURCE vertex falls in this tile, sample
        every signal at edge midpoint / along edge, COPY into temp.
     d. If exporting rasters: write per-tile per-signal PNG slices to
        `{tiles_dir}/{column}_{ti}.png` (cropped to tile bbox).
  4. Stitch per-tile PNGs into one global per-signal PNG; apply the
     lat-to-mercator row warp once on the assembled image.
  5. One UPDATE applies temp _scenicness back to ways.

Each edge is sampled exactly once (it's owned by the tile containing
its source vertex). Halo width per kernel ensures samples near tile
boundaries see correct kernel output.
"""
from __future__ import annotations
import time
import json
from pathlib import Path

import numpy as np
import psycopg
import rasterio

import config
from . import rasters, signals as signals_mod


# Edges per server-cursor fetch chunk. Per-row work is tiny but Python
# loop overhead dominates, so big chunks help. Memory cost per chunk
# is ~6 * 8 bytes * N = a few MB at 100k.
_EDGE_BATCH = 100_000


# ---------------------------------------------------------------------
# Halo / tiling helpers
# ---------------------------------------------------------------------

def _kernel_halo_m(kernel) -> float:
    """Conservative halo (meters) needed around a tile so this kernel's
    output is correct within the tile after halo-cropping."""
    if kernel is None or kernel.kind is None:
        return 0.0
    if kernel.kind == "uniform_blur":
        return float(kernel.param_m)
    if kernel.kind == "stddev_filter":
        return float(kernel.param_m)
    if kernel.kind == "subtract_gaussian_blur":
        return 3.0 * float(kernel.param_m)   # 3σ covers >99% of weight
    if kernel.kind == "distance_to_high_relief":
        # Drama distance: halo bounds how far away drama can register.
        # 30 km comfortably exceeds the rider's perception range and
        # the signal's intended useful range (~15 km).
        return 30_000.0
    raise ValueError(f"unknown kernel kind: {kernel.kind}")


def _signal_halo_m(sig) -> float:
    return _kernel_halo_m(sig.kernel)


def _source_halo_m(sigs, source_key) -> float:
    """Max halo across all signals using this source — the source needs
    to be built that big so every signal's kernel has enough context."""
    return max(
        _signal_halo_m(s)
        for s in sigs
        if (s.source.kind, s.source.landcover_class, s.source.poi_category) == source_key
    )


def _bake_extent(conn) -> tuple[float, float, float, float]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MIN(lon), MIN(lat), MAX(lon), MAX(lat) "
            "FROM ways_vertices_pgr"
        )
        row = cur.fetchone()
    return tuple(float(v) for v in row)


def _iter_tiles(extent, tile_size_deg):
    """Tile-grid covering extent. Tiles snap to integer multiples of
    tile_size_deg so grids are deterministic across runs."""
    xmin, ymin, xmax, ymax = extent
    x_lo = np.floor(xmin / tile_size_deg) * tile_size_deg
    y_lo = np.floor(ymin / tile_size_deg) * tile_size_deg
    nx = int(np.ceil((xmax - x_lo) / tile_size_deg))
    ny = int(np.ceil((ymax - y_lo) / tile_size_deg))
    tiles = []
    for j in range(ny):
        for i in range(nx):
            x0 = x_lo + i * tile_size_deg
            y0 = y_lo + j * tile_size_deg
            tiles.append((round(x0, 6), round(y0, 6),
                          round(x0 + tile_size_deg, 6),
                          round(y0 + tile_size_deg, 6)))
    return tiles


def _expand_bbox(bbox, halo_m):
    if halo_m <= 0:
        return bbox
    xmin, ymin, xmax, ymax = bbox
    mid_lat = (ymin + ymax) / 2.0
    deg_lat = halo_m / 111_000.0
    deg_lon = halo_m / (111_000.0 * np.cos(np.radians(mid_lat)))
    return (xmin - deg_lon, ymin - deg_lat, xmax + deg_lon, ymax + deg_lat)


def _crop_to(raster, transform, target_bbox):
    """Crop a halo'd raster back to target_bbox. Returns the cropped
    array and a fresh affine transform spanning target_bbox."""
    xmin, ymin, xmax, ymax = target_bbox
    col_lo = int(round((xmin - transform.c) / transform.a))
    col_hi = int(round((xmax - transform.c) / transform.a))
    # transform.e is negative (row 0 at top = max lat of source bbox)
    row_lo = int(round((ymax - transform.f) / transform.e))
    row_hi = int(round((ymin - transform.f) / transform.e))
    h, w = raster.shape
    col_lo = max(0, col_lo); col_hi = min(w, col_hi)
    row_lo = max(0, row_lo); row_hi = min(h, row_hi)
    cropped = raster[row_lo:row_hi, col_lo:col_hi]
    new_xform = rasterio.transform.from_bounds(
        xmin, ymin, xmax, ymax, col_hi - col_lo, row_hi - row_lo,
    )
    return cropped, new_xform


def _tile_edge_count(conn, tile_bbox) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM ways_vertices_pgr "
            "WHERE lon BETWEEN %s AND %s AND lat BETWEEN %s AND %s",
            (tile_bbox[0], tile_bbox[2], tile_bbox[1], tile_bbox[3]),
        )
        return int(cur.fetchone()[0])


# ---------------------------------------------------------------------
# Per-edge sampler (unchanged from the pre-tiling version)
# ---------------------------------------------------------------------

def _sample_one_edge(signal_rasters, gid, slon, slat, tlon, tlat, length_m):
    values: list[float] = []
    for raster, transform, sample_mode, sample_spacing_m in signal_rasters:
        h, w = raster.shape
        if sample_mode == "at_midpoint":
            lon = 0.5 * (slon + tlon)
            lat = 0.5 * (slat + tlat)
            col = int((lon - transform.c) / transform.a)
            row = int((lat - transform.f) / transform.e)
            if 0 <= col < w and 0 <= row < h:
                v = float(raster[row, col])
                if v != v:
                    v = 0.0
            else:
                v = 0.0
        elif sample_mode == "along_edge":
            n = max(3, min(50, int(length_m / sample_spacing_m)))
            ts = np.linspace(0.0, 1.0, n)
            lons = slon + (tlon - slon) * ts
            lats = slat + (tlat - slat) * ts
            cols = ((lons - transform.c) / transform.a).astype(np.int64)
            rs = ((lats - transform.f) / transform.e).astype(np.int64)
            valid = (cols >= 0) & (cols < w) & (rs >= 0) & (rs < h)
            if not valid.any():
                v = 0.0
            else:
                vals = raster[rs[valid], cols[valid]].astype(np.float32)
                finite = np.isfinite(vals)
                v = float(vals[finite].mean()) if finite.any() else 0.0
        else:
            raise ValueError(f"unknown sample_mode: {sample_mode}")
        values.append(v)
    return (gid, *values)


# ---------------------------------------------------------------------
# Per-tile work
# ---------------------------------------------------------------------

def _apply_kernel(src_raster, transform, kernel):
    if kernel is None or kernel.kind is None:
        return src_raster
    if kernel.kind == "uniform_blur":
        return rasters.uniform_blur(src_raster, transform, kernel.param_m)
    if kernel.kind == "subtract_gaussian_blur":
        return rasters.subtract_gaussian_blur(src_raster, transform, kernel.param_m)
    if kernel.kind == "stddev_filter":
        return rasters.stddev_filter(src_raster, transform, kernel.param_m)
    if kernel.kind == "distance_to_high_relief":
        return rasters.distance_to_high_relief(src_raster, transform, kernel.param_m)
    raise ValueError(f"unknown kernel kind: {kernel.kind}")


def _build_tile_rasters(conn, sigs, tile_bbox, res_m, dem_dir, cos_lat):
    """For one tile, build (source halo'd) + (kernel applied) per signal.
    Returns a list of (raster, transform, sample_mode, sample_spacing_m)
    aligned with `sigs`, and a parallel list of (cropped_raster,
    cropped_transform) for PNG export (or None if export disabled).

    `cos_lat` is the global latitude-correction factor — passing the
    same value across all tiles keeps pixel sizes consistent so PNG
    slices align without drift in the stitch step."""
    source_cache: dict[tuple, tuple[np.ndarray, object, tuple]] = {}

    def _ensure_source(sig):
        key = (sig.source.kind, sig.source.landcover_class, sig.source.poi_category)
        if key in source_cache:
            return source_cache[key]
        halo = _source_halo_m(sigs, key)
        src_bbox = _expand_bbox(tile_bbox, halo)
        if sig.source.kind == "polygons":
            raster, transform = rasters.rasterize_polygons(
                conn, sig.source.landcover_class, src_bbox, res_m,
                cos_lat=cos_lat,
            )
        elif sig.source.kind == "dem":
            if dem_dir is None:
                raise SystemExit("DEM signal requested but dem_dir not provided")
            raster, transform = rasters.stitch_dem(
                dem_dir, src_bbox, res_m, cos_lat=cos_lat,
            )
        elif sig.source.kind == "points":
            raster, transform = rasters.rasterize_points(
                config.POIS_DB, sig.source.poi_category, src_bbox, res_m,
                cos_lat=cos_lat,
            )
        else:
            raise ValueError(f"unknown source kind: {sig.source.kind}")
        source_cache[key] = (raster, transform, src_bbox)
        return source_cache[key]

    # Per-signal kernel application, deduped by (source_key, kernel).
    final_cache: dict[tuple, tuple[np.ndarray, object]] = {}

    def _ensure_final(sig):
        src_key = (sig.source.kind, sig.source.landcover_class, sig.source.poi_category)
        kernel_key = (sig.kernel.kind, sig.kernel.param_m) if sig.kernel else None
        cache_key = (src_key, kernel_key)
        if cache_key in final_cache:
            return final_cache[cache_key]
        src_raster, transform, _ = _ensure_source(sig)
        final = _apply_kernel(src_raster, transform, sig.kernel)
        final_cache[cache_key] = (final, transform)
        return final, transform

    signal_rasters: list[tuple] = []
    cropped_for_png: list[tuple] = []
    for sig in sigs:
        raster, transform = _ensure_final(sig)
        signal_rasters.append(
            (raster, transform, sig.sample_mode, sig.sample_spacing_m),
        )
        # Crop now for PNG output; the un-cropped halo'd raster stays in
        # signal_rasters for edge sampling so boundary edges are correct.
        cr, ct = _crop_to(raster, transform, tile_bbox)
        cropped_for_png.append((cr, ct))
    return signal_rasters, cropped_for_png


def _sample_tile_edges(conn, sigs, signal_rasters, tile_bbox, col_list,
                       sample_stats):
    """Stream ways edges whose source vertex is in tile_bbox; sample
    every signal; COPY into _scenicness. Returns total edges sampled
    in this tile (also updates sample_stats counters in place)."""
    t0 = time.time()
    last_print = time.time()
    total = 0
    with conn.cursor(name=f"scenicness_tile_scan_{int(time.time()*1000)}") as scan_cur:
        scan_cur.itersize = _EDGE_BATCH
        scan_cur.execute(
            "SELECT w.gid, ST_X(vs.the_geom), ST_Y(vs.the_geom), "
            "       ST_X(vt.the_geom), ST_Y(vt.the_geom), w.length_m "
            "FROM ways w "
            "JOIN ways_vertices_pgr vs ON vs.id = w.source "
            "JOIN ways_vertices_pgr vt ON vt.id = w.target "
            "WHERE vs.lon >= %s AND vs.lon < %s "
            "  AND vs.lat >= %s AND vs.lat < %s",
            (tile_bbox[0], tile_bbox[2], tile_bbox[1], tile_bbox[3]),
        )
        batch: list[tuple] = []

        def _flush(rows):
            results = [_sample_one_edge(signal_rasters, *r) for r in rows]
            with conn.cursor() as wcur:
                with wcur.copy(
                    f"COPY _scenicness ({col_list}) FROM STDIN"
                ) as cp:
                    for r in results:
                        cp.write_row(r)

        for row in scan_cur:
            batch.append(row)
            if len(batch) >= _EDGE_BATCH:
                _flush(batch)
                total += len(batch)
                batch = []
                if time.time() - last_print > 5:
                    rate = total / max(time.time() - t0, 1e-3)
                    print(f"[scenicness]   tile sample {total:,} ({rate:.0f}/s)",
                          flush=True)
                    last_print = time.time()
        if batch:
            _flush(batch)
            total += len(batch)
    conn.commit()
    sample_stats["edges"] += total
    sample_stats["sec"] += time.time() - t0
    return total


def _write_tile_pngs(sigs, cropped_for_png, tile_bbox, ti, tiles_dir):
    """Write one PNG per signal for this tile to {tiles_dir}/{column}_{ti}.png.
    NO mercator warp — that's applied once on the stitched global PNG."""
    tiles_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for sig, (raster, _xform) in zip(sigs, cropped_for_png):
        png_path = tiles_dir / f"{sig.column}_t{ti:03d}.png"
        # Pass bbox=None so write_signal_png skips the mercator warp.
        rasters.write_signal_png(raster, png_path, sig.column, bbox=None)
        out.append((sig.column, png_path, raster.shape, tile_bbox))
    return out


# ---------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------

def _stitch_signal(column, tile_entries, extent, res_m, out_path, cos_lat):
    """Read every per-tile PNG for this signal, paste it into a global
    uint8 RGBA buffer at the tile's pixel location, mercator-warp once,
    save as a single PNG.

    Memory peak: one global buffer (H*W*4 bytes). For Austria at 20 m
    that's ~1.75 GB; safe on a 32 GB box.
    """
    width, height, transform = rasters.grid_dims(extent, res_m, cos_lat=cos_lat)
    canvas = np.zeros((height, width, 4), dtype=np.uint8)

    for entry in tile_entries:
        _col, png_path, _shape, tile_bbox = entry
        tx_min, _ty_min, _tx_max, ty_max = tile_bbox
        col_off = int(round((tx_min - transform.c) / transform.a))
        # transform.e < 0, top row = transform.f (= ymax of canvas).
        row_off = int(round((ty_max - transform.f) / transform.e))
        # Read all 4 bands; rasterio gave us a (bands, h, w) array.
        with rasterio.open(png_path) as ds:
            tile_arr = ds.read()
        arr = np.transpose(tile_arr, (1, 2, 0))  # (h, w, bands)
        if arr.shape[2] == 3:
            # opaque RGB → add full-alpha channel
            alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
            arr = np.concatenate([arr, alpha], axis=2)
        ah, aw = arr.shape[:2]
        c0 = max(0, col_off); r0 = max(0, row_off)
        c1 = min(width, col_off + aw); r1 = min(height, row_off + ah)
        if c1 <= c0 or r1 <= r0:
            continue
        src_c0 = c0 - col_off; src_r0 = r0 - row_off
        src_c1 = src_c0 + (c1 - c0); src_r1 = src_r0 + (r1 - r0)
        canvas[r0:r1, c0:c1, :] = arr[src_r0:src_r1, src_c0:src_c1, :]

    # Mercator row warp on the assembled image (one-shot, full extent).
    canvas = rasters.warp_lat_to_mercator_rows(canvas, extent)

    # Downsample for browser display. WebGL textures cap at 4096-16384
    # px on typical hardware; the full-res canvas (≈30k×15k for Austria)
    # exceeds that and MapLibre would fail to upload it.
    #
    # Max-pool (per factor×factor block, take the brightest pixel)
    # rather than nearest-neighbor stride: thin features like 10m-wide
    # waterway polygons are 1 source pixel wide, and stride sampling
    # would drop most of them, leaving "dotted line" artifacts. Max
    # over the block keeps any non-zero pixel visible in the output.
    # The per-edge values in `ways` columns sample the native-resolution
    # raster directly (not the PNG), so this is purely a visual fix.
    max_dim = 4096
    fh, fw = canvas.shape[:2]
    if max(fh, fw) > max_dim:
        factor = int(np.ceil(max(fh, fw) / max_dim))
        new_h = fh // factor
        new_w = fw // factor
        # Crop to a clean multiple of factor before reshape.
        canvas_crop = canvas[:new_h * factor, :new_w * factor]
        # Reshape (H, W, 4) -> (new_h, factor, new_w, factor, 4) is a
        # view (no copy). max over axes 1 and 3 reduces each block.
        canvas = canvas_crop.reshape(
            new_h, factor, new_w, factor, 4,
        ).max(axis=(1, 3))
        del canvas_crop
        print(f"[scenicness]   max-pool {factor}x downsample: "
              f"{fw}x{fh} -> {canvas.shape[1]}x{canvas.shape[0]}",
              flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="PNG",
        width=canvas.shape[1], height=canvas.shape[0],
        count=4, dtype="uint8",
    ) as dst:
        for b in range(4):
            dst.write(canvas[..., b], b + 1)
    print(f"[scenicness] wrote stitched PNG {out_path} "
          f"({canvas.shape[1]}x{canvas.shape[0]})", flush=True)
    # Free the canvas explicitly; numpy arrays of this size can linger
    # in Python's heap for a while before GC otherwise.
    del canvas
    import gc; gc.collect()


# ---------------------------------------------------------------------
# Top-level bake
# ---------------------------------------------------------------------

def bake(conn: psycopg.Connection,
         signal_names: list[str],
         bbox: tuple[float, float, float, float] | None = None,
         res_m: float = 20.0,
         dem_dir: Path | None = None,
         export_rasters_dir: Path | None = None,
         tile_size_deg: float = 1.0,
         ) -> None:
    """Compute every named signal over the ways-table extent (or bbox
    override) and write to ways. Internal tiling keeps memory bounded.
    """
    sigs = [signals_mod.SIGNALS[name] for name in signal_names]
    if not sigs:
        raise SystemExit("no signals requested")
    print(f"[scenicness] requested: {[s.name for s in sigs]}", flush=True)

    if bbox is None:
        extent = _bake_extent(conn)
        print(f"[scenicness] bake extent from ways: {extent}", flush=True)
    else:
        extent = bbox
        print(f"[scenicness] bake extent from --bbox: {extent}", flush=True)

    tiles = _iter_tiles(extent, tile_size_deg)
    print(f"[scenicness] tile_size_deg={tile_size_deg} → "
          f"{len(tiles)} tiles before filter", flush=True)

    # Filter: only run tiles that actually contain vertices.
    nonempty: list[tuple[int, tuple[float, float, float, float], int]] = []
    for ti, tb in enumerate(tiles):
        n = _tile_edge_count(conn, tb)
        if n > 0:
            nonempty.append((ti, tb, n))
    print(f"[scenicness] {len(nonempty)} non-empty tiles "
          f"(skipping {len(tiles) - len(nonempty)} empty)", flush=True)
    if not nonempty:
        raise SystemExit("no ways vertices in any tile — nothing to bake")

    # Temp table
    cols_def = ", ".join(f"{s.column} real" for s in sigs)
    col_list = "gid, " + ", ".join(s.column for s in sigs)
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute("SET work_mem = '512MB'")
        cur.execute("DROP TABLE IF EXISTS _scenicness")
        cur.execute(f"CREATE TEMP TABLE _scenicness "
                    f"(gid bigint PRIMARY KEY, {cols_def})")

    # Use a single cos(mid_lat) for every tile so pixel sizes stay
    # consistent and tile slices paste together without drift.
    global_mid_lat = (extent[1] + extent[3]) / 2.0
    cos_lat = float(np.cos(np.radians(global_mid_lat)))
    print(f"[scenicness] global mid_lat={global_mid_lat:.3f} "
          f"cos_lat={cos_lat:.4f}", flush=True)

    tiles_dir = None
    if export_rasters_dir is not None:
        tiles_dir = export_rasters_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)

    # Per-tile loop
    all_tile_entries: dict[str, list] = {s.column: [] for s in sigs}
    sample_stats = {"edges": 0, "sec": 0.0}
    t_total = time.time()
    for idx, (ti, tile_bbox, n_verts) in enumerate(nonempty):
        print(f"\n[scenicness] === tile {idx+1}/{len(nonempty)} "
              f"(index {ti}) {tile_bbox} verts={n_verts:,} ===", flush=True)
        t_tile = time.time()
        signal_rasters, cropped_for_png = _build_tile_rasters(
            conn, sigs, tile_bbox, res_m, dem_dir, cos_lat,
        )
        _sample_tile_edges(conn, sigs, signal_rasters, tile_bbox, col_list,
                           sample_stats)
        if tiles_dir is not None:
            entries = _write_tile_pngs(sigs, cropped_for_png, tile_bbox, ti,
                                       tiles_dir)
            for col, png_path, shape, tb in entries:
                all_tile_entries[col].append((col, png_path, shape, tb))
        # Free per-tile rasters before next tile
        del signal_rasters, cropped_for_png
        print(f"[scenicness]   tile done in {time.time()-t_tile:.1f}s "
              f"(total sampled: {sample_stats['edges']:,})", flush=True)

    print(f"\n[scenicness] all tiles done: {sample_stats['edges']:,} edges in "
          f"{(time.time()-t_total)/60:.1f} min", flush=True)

    # Stitch PNGs (one signal at a time, ~1.75 GB peak per signal)
    if export_rasters_dir is not None:
        export_rasters_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "bbox": list(extent),
            "res_m": res_m,
            "tile_size_deg": tile_size_deg,
            "signals": {},
        }
        for sig in sigs:
            entries = all_tile_entries[sig.column]
            if not entries:
                continue
            out_path = export_rasters_dir / f"{sig.column}.png"
            print(f"\n[scenicness] stitching {sig.column} "
                  f"from {len(entries)} tiles → {out_path}", flush=True)
            _stitch_signal(sig.column, entries, extent, res_m, out_path, cos_lat)
            manifest["signals"][sig.column] = {
                "name": sig.name,
                "description": sig.description,
                "png": out_path.name,
                "colormap": rasters.COLORMAPS.get(sig.column, {"kind": "green"}),
            }
        manifest_path = export_rasters_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[scenicness] wrote manifest {manifest_path} "
              f"({len(manifest['signals'])} signals)", flush=True)

    # Final UPDATE
    print(f"\n[scenicness] applying to ways: {[s.column for s in sigs]}...",
          flush=True)
    t_update = time.time()
    set_clause = ", ".join(f"{s.column} = r.{s.column}" for s in sigs)
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute(
            f"UPDATE ways w SET {set_clause} "
            "FROM _scenicness r WHERE w.gid = r.gid"
        )
        updated = cur.rowcount
        cur.execute("DROP TABLE IF EXISTS _scenicness")
    conn.commit()
    print(f"[scenicness] applied to {updated:,} edges in "
          f"{time.time()-t_update:.1f}s", flush=True)

    # Summary
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        select_parts = []
        for s in sigs:
            select_parts.append(f"AVG({s.column}) AS avg_{s.column}")
            select_parts.append(f"MIN({s.column}) AS min_{s.column}")
            select_parts.append(f"MAX({s.column}) AS max_{s.column}")
            select_parts.append(
                f"COUNT(*) FILTER (WHERE {s.column} <> 0) AS nz_{s.column}"
            )
        cur.execute("SELECT " + ", ".join(select_parts) + " FROM ways")
        row = cur.fetchone()
    print("[scenicness] summary:", flush=True)
    for i, s in enumerate(sigs):
        avg, mn, mx, nz = row[i*4:i*4 + 4]
        print(f"[scenicness]   {s.column:20s} avg={avg:.4f} "
              f"min={mn:.4f} max={mx:.4f} nonzero={nz:,}", flush=True)
