"""Top-level scenicness orchestrator.

Given a list of registered signal names + a bbox, this:
  1. Builds the unique set of source rasters (deduped — multiple
     signals can share the same forest mask, the same DEM, etc.).
  2. Applies each signal's kernel to produce a final raster.
  3. Streams every corridor edge via a memory-safe server cursor.
  4. For each edge, samples each signal's raster (at_midpoint or
     along_edge per signal config) → one (gid, val1, val2, …) row.
  5. COPYs into `_scenicness` temp table.
  6. Single UPDATE writes all per-signal columns back to `ways` in
     one statement.

The expensive parts — the corridor SELECT and the per-row sampling
loop — happen ONCE per bake, regardless of how many signals are
included. Adding a fifth signal to an existing four-signal bake is
roughly free.
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import psycopg

import config
from . import rasters, signals as signals_mod


# Edges per server-cursor fetch chunk. Per-row work is tiny but Python
# loop overhead dominates, so big chunks help. Memory cost per chunk
# is ~6 * 8 bytes * N = a few MB at 100k.
_EDGE_BATCH = 100_000


def _sample_one_edge(signal_rasters, gid, slon, slat, tlon, tlat, length_m):
    """Sample every signal's raster for one edge. Returns a tuple
    (gid, val1, val2, …) in the order of `signal_rasters`.
    """
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
                if v != v:  # NaN (DEM no-data); treat as 0
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
                # NaN-safe mean (DEM no-data → ignore)
                finite = np.isfinite(vals)
                v = float(vals[finite].mean()) if finite.any() else 0.0
        else:
            raise ValueError(f"unknown sample_mode: {sample_mode}")
        values.append(v)
    return (gid, *values)


def bake(conn: psycopg.Connection,
         signal_names: list[str],
         bbox: tuple[float, float, float, float],
         res_m: float = 20.0,
         dem_dir: Path | None = None,
         export_rasters_dir: Path | None = None,
         ) -> None:
    """Compute every named signal over the bbox and write to ways.

    When `export_rasters_dir` is given, also write one PNG per signal
    plus a `manifest.json` mapping {column → {png, bbox}}. The web
    app can fetch the manifest and overlay the PNGs as MapLibre image
    sources without any GIS round-tripping.
    """
    import json
    sigs = [signals_mod.SIGNALS[name] for name in signal_names]
    if not sigs:
        raise SystemExit("no signals requested")
    print(f"[scenicness] requested: {[s.name for s in sigs]}", flush=True)

    # ----------------------------------------------------------------
    # 1) Build source rasters (deduped).
    # ----------------------------------------------------------------
    source_cache: dict[tuple, tuple[np.ndarray, object]] = {}

    def _ensure_source(sig: signals_mod.Signal):
        key = (sig.source.kind, sig.source.landcover_class)
        if key in source_cache:
            return source_cache[key]
        if sig.source.kind == "polygons":
            raster, transform = rasters.rasterize_polygons(
                conn, sig.source.landcover_class, bbox, res_m,
            )
        elif sig.source.kind == "dem":
            if dem_dir is None:
                raise SystemExit("DEM signal requested but dem_dir not provided")
            raster, transform = rasters.stitch_dem(dem_dir, bbox, res_m)
        else:
            raise ValueError(f"unknown source kind: {sig.source.kind}")
        source_cache[key] = (raster, transform)
        return raster, transform

    # ----------------------------------------------------------------
    # 2) Apply kernels (deduped by (source, kernel) pair).
    # ----------------------------------------------------------------
    final_cache: dict[tuple, tuple[np.ndarray, object]] = {}

    def _ensure_final(sig: signals_mod.Signal):
        src_key = (sig.source.kind, sig.source.landcover_class)
        kernel_key = (sig.kernel.kind, sig.kernel.param_m) if sig.kernel else None
        cache_key = (src_key, kernel_key)
        if cache_key in final_cache:
            return final_cache[cache_key]
        src_raster, transform = _ensure_source(sig)
        if sig.kernel is None or sig.kernel.kind is None:
            final = src_raster
        elif sig.kernel.kind == "uniform_blur":
            final = rasters.uniform_blur(src_raster, transform, sig.kernel.param_m)
        elif sig.kernel.kind == "subtract_gaussian_blur":
            final = rasters.subtract_gaussian_blur(
                src_raster, transform, sig.kernel.param_m,
            )
        elif sig.kernel.kind == "stddev_filter":
            final = rasters.stddev_filter(
                src_raster, transform, sig.kernel.param_m,
            )
        elif sig.kernel.kind == "distance_to_high_relief":
            final = rasters.distance_to_high_relief(
                src_raster, transform, sig.kernel.param_m,
            )
        else:
            raise ValueError(f"unknown kernel kind: {sig.kernel.kind}")
        final_cache[cache_key] = (final, transform)
        return final, transform

    # Materialize the per-signal raster list in declared order.
    signal_rasters: list[tuple] = []
    for sig in sigs:
        raster, transform = _ensure_final(sig)
        signal_rasters.append(
            (raster, transform, sig.sample_mode, sig.sample_spacing_m),
        )

    # Optional: write each signal's final raster as a PNG overlay +
    # manifest for the web app.
    if export_rasters_dir is not None:
        export_rasters_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = export_rasters_dir / "manifest.json"
        # Merge with any prior manifest in the same directory so a multi-
        # pass bake (split for memory safety) accumulates signals rather
        # than overwriting them. Bbox/res_m must match across passes —
        # if they differ we drop the prior entries with a warning rather
        # than emitting an inconsistent manifest.
        manifest: dict
        if manifest_path.exists():
            try:
                prior = json.loads(manifest_path.read_text())
                if (prior.get("bbox") == list(bbox)
                        and prior.get("res_m") == res_m):
                    manifest = prior
                    manifest.setdefault("signals", {})
                    print(f"[scenicness] merging into existing manifest "
                          f"({len(manifest['signals'])} prior signals)",
                          flush=True)
                else:
                    print(f"[scenicness] WARNING: prior manifest bbox/res "
                          f"mismatch — replacing", flush=True)
                    manifest = {"bbox": list(bbox), "res_m": res_m, "signals": {}}
            except (OSError, ValueError) as e:
                print(f"[scenicness] could not read prior manifest "
                      f"({e}); starting fresh", flush=True)
                manifest = {"bbox": list(bbox), "res_m": res_m, "signals": {}}
        else:
            manifest = {"bbox": list(bbox), "res_m": res_m, "signals": {}}
        for sig in sigs:
            raster, _transform = _ensure_final(sig)
            png_path = export_rasters_dir / f"{sig.column}.png"
            # Pass bbox so write_signal_png warps the rows from linear-in-
            # lat to linear-in-mercator — required for MapLibre's image
            # source to display each pixel at its intended latitude.
            rasters.write_signal_png(raster, png_path, sig.column, bbox=bbox)
            manifest["signals"][sig.column] = {
                "name": sig.name,
                "description": sig.description,
                "png": png_path.name,
                "shape": list(raster.shape),
                "colormap": rasters.COLORMAPS.get(sig.column, {"kind": "green"}),
            }
            print(f"[scenicness] wrote PNG {png_path} "
                  f"({raster.shape[1]}x{raster.shape[0]})", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[scenicness] wrote manifest {manifest_path} "
              f"({len(manifest['signals'])} signals total)", flush=True)

    # ----------------------------------------------------------------
    # 3) Stream edges + 4) sample + 5) COPY into temp table.
    # ----------------------------------------------------------------
    cols_def = ", ".join(f"{s.column} real" for s in sigs)
    col_list = "gid, " + ", ".join(s.column for s in sigs)

    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute("SET work_mem = '512MB'")
        cur.execute("DROP TABLE IF EXISTS _scenicness")
        cur.execute(f"CREATE TEMP TABLE _scenicness (gid bigint PRIMARY KEY, {cols_def})")

    print(f"[scenicness] streaming edges (server cursor, "
          f"{_EDGE_BATCH:,}/fetch)...", flush=True)
    t_sample = time.time()
    last_print = time.time()
    total_edges = 0

    with conn.cursor(name="scenicness_edge_scan") as scan_cur:
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

        def _flush(rows: list[tuple]) -> None:
            results = [
                _sample_one_edge(signal_rasters, *r) for r in rows
            ]
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
                total_edges += len(batch)
                batch = []
                if time.time() - last_print > 5:
                    rate = total_edges / max(time.time() - t_sample, 1e-3)
                    print(f"[scenicness]   {total_edges:,} edges sampled "
                          f"({rate:.0f}/s)", flush=True)
                    last_print = time.time()
        if batch:
            _flush(batch)
            total_edges += len(batch)

    conn.commit()
    print(f"[scenicness] sampling+COPY done: {total_edges:,} edges in "
          f"{(time.time()-t_sample)/60:.1f} min", flush=True)

    # ----------------------------------------------------------------
    # 6) One UPDATE writes every signal column at once.
    # ----------------------------------------------------------------
    print(f"[scenicness] applying to ways: {[s.column for s in sigs]}...",
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

    # Per-column summary.
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
