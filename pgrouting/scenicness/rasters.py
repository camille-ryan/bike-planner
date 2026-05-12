"""Raster builders + kernels for scenicness signals.

Source types:
  - polygons: rasterize landcover polygons (forest / water / urban / ...)
  - dem:      stitch Copernicus DEM GLO-30 tiles into a single grid

Kernels (post-processing applied to the source raster):
  - uniform_blur:    scipy.ndimage.uniform_filter with size = 2·radius/res
  - gaussian_blur:   scipy.ndimage.gaussian_filter with sigma = sigma_m/res
  - subtract_blur:   raster - gaussian_blur(raster) — for view_dominance
  - distance_edt:    distance_transform_edt — meters from nearest True cell

All rasters are returned as `(ndarray, rasterio.Affine)`. The transform
maps pixel (col, row) → (lon, lat) per rasterio convention.
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
import scipy.ndimage
import shapely.wkb


# Approximate meters per degree of latitude. Longitude scales by cos(lat),
# computed at the bbox midpoint — accurate to ~0.5% over an Austria-sized
# area; for a Europe-wide bake we'd want a more careful projection.
_M_PER_DEG_LAT = 111_000.0


def grid_dims(bbox: tuple[float, float, float, float],
              res_m: float
              ) -> tuple[int, int, "rasterio.Affine"]:
    """Compute (width, height, transform) for a meter-resolution grid
    over the given lon/lat bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = (min_lat + max_lat) / 2.0
    deg_lon_per_m = 1.0 / (_M_PER_DEG_LAT * np.cos(np.radians(mid_lat)))
    deg_lat_per_m = 1.0 / _M_PER_DEG_LAT
    px_lon = res_m * deg_lon_per_m
    px_lat = res_m * deg_lat_per_m
    width = int(np.ceil((max_lon - min_lon) / px_lon))
    height = int(np.ceil((max_lat - min_lat) / px_lat))
    transform = rasterio.transform.from_bounds(
        min_lon, min_lat, max_lon, max_lat, width, height,
    )
    return width, height, transform


def res_to_pixels(transform: "rasterio.Affine", meters: float) -> float:
    """Convert a length in meters to a (rough) pixel count for this
    transform. Uses the latitudinal pixel size, which is locally
    invariant; longitudinal pixels are wider near the poles but the
    framework treats kernels as isotropic for simplicity.
    """
    px_size_deg = abs(transform.e)
    px_size_m = px_size_deg * _M_PER_DEG_LAT
    return meters / px_size_m


# ---------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------

def rasterize_polygons(conn,
                       landcover_class: str,
                       bbox: tuple[float, float, float, float],
                       res_m: float,
                       ) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Rasterize every `landcover` polygon of the given class within
    bbox into a binary uint8 mask. Returns (mask, transform).
    """
    width, height, transform = grid_dims(bbox, res_m)
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ST_AsBinary(geom) FROM landcover "
            "WHERE class = %s "
            "  AND ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 4326))",
            (landcover_class, *bbox),
        )
        polys = [shapely.wkb.loads(bytes(r[0])) for r in cur]
    t_load = time.time() - t0
    print(f"[rasters] polygons/{landcover_class}: loaded {len(polys):,} "
          f"polys in {t_load:.1f}s", flush=True)
    if not polys:
        return np.zeros((height, width), dtype=np.uint8), transform
    t0 = time.time()
    mask = rasterio.features.rasterize(
        ((g, 1) for g in polys),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    cover = mask.sum() / mask.size
    print(f"[rasters] polygons/{landcover_class}: rasterized "
          f"{width}x{height} ({cover*100:.1f}% coverage) in "
          f"{time.time()-t0:.1f}s", flush=True)
    return mask, transform


def stitch_dem(dem_dir: Path,
               bbox: tuple[float, float, float, float],
               res_m: float,
               ) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Stitch Copernicus DEM GLO-30 tiles intersecting bbox into a
    single float32 raster at the target resolution.

    Tiles are 1°×1° at ~30m native; we resample with nearest neighbor
    onto the target grid (cheap, adequate for the smooth elevation
    signals we derive). No-data cells become NaN.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    width, height, transform = grid_dims(bbox, res_m)
    out = np.full((height, width), np.nan, dtype=np.float32)
    inv = ~transform
    # Tile filenames look like Copernicus_DSM_COG_10_N47_00_E015_00_DEM.tif,
    # one per integer lat/lon SW corner.
    lat_lo = int(np.floor(min_lat))
    lat_hi = int(np.floor(max_lat))
    lon_lo = int(np.floor(min_lon))
    lon_hi = int(np.floor(max_lon))
    n_tiles_seen = 0
    t0 = time.time()
    for lat in range(lat_lo, lat_hi + 1):
        for lon in range(lon_lo, lon_hi + 1):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tile_name = (f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_"
                         f"{ew}{abs(lon):03d}_00_DEM.tif")
            tile_path = dem_dir / tile_name
            if not tile_path.exists():
                continue
            n_tiles_seen += 1
            with rasterio.open(tile_path) as ds:
                arr = ds.read(1).astype(np.float32)
                nodata = ds.nodata
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                tile_h, tile_w = arr.shape
                tile_xform = ds.transform
                # For each output pixel inside the tile's bbox, look up
                # the source pixel via the source transform's inverse.
                # Vectorized: build a grid of (lon, lat) for the output
                # subrange, transform to source pixels, gather.
                tile_min_lon = tile_xform.c
                tile_max_lat = tile_xform.f
                tile_max_lon = tile_min_lon + tile_w * tile_xform.a
                tile_min_lat = tile_max_lat + tile_h * tile_xform.e  # tile_xform.e < 0
                # Determine the output sub-window covered by this tile
                col_lo = max(0, int(np.floor(
                    (max(tile_min_lon, min_lon) - transform.c) / transform.a)))
                col_hi = min(width, int(np.ceil(
                    (min(tile_max_lon, max_lon) - transform.c) / transform.a)))
                row_lo = max(0, int(np.floor(
                    (transform.f - min(tile_max_lat, max_lat)) / -transform.e)))
                row_hi = min(height, int(np.ceil(
                    (transform.f - max(tile_min_lat, min_lat)) / -transform.e)))
                if col_hi <= col_lo or row_hi <= row_lo:
                    continue
                cols = np.arange(col_lo, col_hi)
                rows = np.arange(row_lo, row_hi)
                # Output pixel centers in lon/lat
                lons = transform.c + (cols + 0.5) * transform.a
                lats = transform.f + (rows + 0.5) * transform.e
                # Source pixel coords (nearest-neighbor lookup)
                src_cols = ((lons - tile_xform.c) / tile_xform.a).astype(np.int64)
                src_rows = ((lats - tile_xform.f) / tile_xform.e).astype(np.int64)
                # Clip to tile bounds (guard against off-by-one)
                src_cols = np.clip(src_cols, 0, tile_w - 1)
                src_rows = np.clip(src_rows, 0, tile_h - 1)
                # Broadcast to a (rows × cols) 2D grid of source samples
                rr, cc = np.meshgrid(src_rows, src_cols, indexing="ij")
                out[row_lo:row_hi, col_lo:col_hi] = arr[rr, cc]
    print(f"[rasters] dem: stitched {n_tiles_seen} tiles into "
          f"{width}x{height} grid in {time.time()-t0:.1f}s", flush=True)
    return out, transform


# ---------------------------------------------------------------------
# Kernels (in-place transforms on a raster)
# ---------------------------------------------------------------------

def uniform_blur(raster: np.ndarray,
                 transform: "rasterio.Affine",
                 radius_m: float
                 ) -> np.ndarray:
    """Mean-filter with a square window of side = 2·radius/pixel_size.
    Cheaper than a true disc, and the difference is invisible at
    blurs of more than a few pixels.
    """
    radius_px = max(1, int(round(res_to_pixels(transform, radius_m))))
    size = 2 * radius_px + 1
    t0 = time.time()
    # Convert to float32 for the filter (uniform_filter on uint8 wraps);
    # output is mean fraction in [0, 1] for a binary input.
    out = scipy.ndimage.uniform_filter(
        raster.astype(np.float32), size=size, mode="constant", cval=0.0,
    )
    print(f"[kernels] uniform_blur r={radius_m:.0f}m ({radius_px}px window) "
          f"in {time.time()-t0:.1f}s", flush=True)
    return out


# ---------------------------------------------------------------------
# PNG export for web overlays
# ---------------------------------------------------------------------

def _colormap_green(values: np.ndarray) -> np.ndarray:
    """Green gradient colormap, alpha = value.

    Input: float32 ∈ [0, 1]. Output: (H, W, 4) uint8 RGBA.
    """
    v = np.clip(values, 0.0, 1.0).astype(np.float32)
    h, w = v.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Stay slightly transparent even at max to keep map readable.
    rgba[..., 0] = (20 + 30 * v).astype(np.uint8)      # R
    rgba[..., 1] = (80 + 120 * v).astype(np.uint8)     # G
    rgba[..., 2] = (20 + 20 * v).astype(np.uint8)      # B
    rgba[..., 3] = (160 * v).astype(np.uint8)          # alpha
    return rgba


def _colormap_blue(values: np.ndarray) -> np.ndarray:
    """Blue gradient colormap for water-family signals (water, sea,
    waterway). Same shape as _colormap_green but in the blue end of
    the spectrum so water + forest overlays are visually
    distinguishable when toggled together.
    """
    v = np.clip(values, 0.0, 1.0).astype(np.float32)
    h, w = v.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = (10 + 20 * v).astype(np.uint8)      # R: stays low
    rgba[..., 1] = (50 + 80 * v).astype(np.uint8)      # G: mid for cyan tint
    rgba[..., 2] = (120 + 130 * v).astype(np.uint8)    # B: strong
    rgba[..., 3] = (170 * v).astype(np.uint8)          # alpha
    return rgba


def _colormap_teal(values: np.ndarray) -> np.ndarray:
    """Teal gradient for wetlands. Sits between green (forest) and
    blue (water) — wetlands are conceptually that mix.
    """
    v = np.clip(values, 0.0, 1.0).astype(np.float32)
    h, w = v.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = (10 + 20 * v).astype(np.uint8)
    rgba[..., 1] = (100 + 120 * v).astype(np.uint8)    # G: stronger
    rgba[..., 2] = (90 + 80 * v).astype(np.uint8)      # B: moderate
    rgba[..., 3] = (170 * v).astype(np.uint8)
    return rgba


def _colormap_intensity(values: np.ndarray,
                        vmax: float
                        ) -> np.ndarray:
    """Single-hue intensity colormap: transparent → amber → red.
    Used for unsigned magnitude signals like local_relief.
    """
    v = np.clip(values / max(vmax, 1e-6), 0.0, 1.0).astype(np.float32)
    h, w = v.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = (200 + 55 * v).astype(np.uint8)        # R: bright at high
    rgba[..., 1] = (160 - 80 * v).astype(np.uint8)        # G: pulls warm
    rgba[..., 2] = (40 + 20 * (1 - v)).astype(np.uint8)   # B: small
    rgba[..., 3] = (180 * v).astype(np.uint8)             # alpha by magnitude
    return rgba


def _colormap_inverse_intensity(values: np.ndarray,
                                vmax: float
                                ) -> np.ndarray:
    """Inverse intensity: bright purple at low values, transparent
    at high. For distance-style signals where SMALL = interesting
    (e.g. distance_to_drama — closer to mountains is more visible).
    """
    # Treat values >= vmax as fully faded.
    v = 1.0 - np.clip(values / max(vmax, 1e-6), 0.0, 1.0).astype(np.float32)
    h, w = v.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = (140 + 70 * v).astype(np.uint8)        # R: purple
    rgba[..., 1] = (40 + 30 * (1 - v)).astype(np.uint8)   # G: low
    rgba[..., 2] = (180 + 60 * v).astype(np.uint8)        # B: purple
    rgba[..., 3] = (180 * v).astype(np.uint8)             # alpha by closeness
    return rgba


def _colormap_diverging(values: np.ndarray,
                        vmax: float
                        ) -> np.ndarray:
    """Diverging blue-(transparent)-red colormap.

    Input: float32 (signed). Output: (H, W, 4) uint8 RGBA.
    `vmax` is the absolute value at full saturation; symmetric.
    """
    v = values.astype(np.float32)
    v = np.clip(v / max(vmax, 1e-6), -1.0, 1.0)
    h, w = v.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    neg = v < 0
    pos = v > 0
    a = np.abs(v)
    # Negative (valleys) → cool blue
    rgba[neg, 0] = (40 + 40 * (1 - a[neg])).astype(np.uint8)
    rgba[neg, 1] = (80 + 60 * (1 - a[neg])).astype(np.uint8)
    rgba[neg, 2] = (140 + 100 * a[neg]).astype(np.uint8)
    # Positive (ridges) → warm brown / red
    rgba[pos, 0] = (140 + 100 * a[pos]).astype(np.uint8)
    rgba[pos, 1] = (90 + 30 * (1 - a[pos])).astype(np.uint8)
    rgba[pos, 2] = (40 + 30 * (1 - a[pos])).astype(np.uint8)
    # Alpha: more visible near extremes, transparent at zero
    rgba[..., 3] = (180 * a).astype(np.uint8)
    return rgba


# Maps `column` → which colormap and what vmax to use. Add an entry
# per signal as it's added to the registry.
COLORMAPS: dict[str, dict] = {
    "forest_local":   {"kind": "green"},
    "forest_wide":    {"kind": "green"},
    "view_dominance": {"kind": "diverging", "vmax": 200.0},  # ±200m saturates
    "local_relief":   {"kind": "intensity", "vmax": 120.0},  # ~120m σ saturates
    "regional_relief":  {"kind": "intensity", "vmax": 250.0},  # broader saturation
    "distance_to_drama": {"kind": "inverse_intensity", "vmax": 15000.0},  # 15 km fade
    "canopy_frac":    {"kind": "green"},
    # Water family: blue gradient. Sea slightly more saturated than
    # water_local downstream by virtue of larger contiguous polygons
    # (don't need a separate colormap).
    "water_local":          {"kind": "blue"},
    "water_wide":           {"kind": "blue"},
    "sea_local":            {"kind": "blue"},
    "sea_wide":             {"kind": "blue"},
    "waterway_along_edge":  {"kind": "blue"},
    "waterway_local":       {"kind": "blue"},
    # Wetlands sit between forest and water visually + conceptually.
    "wetland_local":        {"kind": "teal"},
}


def write_signal_png(raster: np.ndarray,
                     out_path: Path,
                     column: str
                     ) -> None:
    """Render a signal raster to a web-ready PNG with the column's
    colormap. Geographic registration is carried in the bake manifest
    alongside the PNG, not embedded in the file."""
    cm = COLORMAPS.get(column, {"kind": "green"})
    if cm["kind"] == "green":
        rgba = _colormap_green(raster)
    elif cm["kind"] == "blue":
        rgba = _colormap_blue(raster)
    elif cm["kind"] == "teal":
        rgba = _colormap_teal(raster)
    elif cm["kind"] == "diverging":
        rgba = _colormap_diverging(raster, vmax=cm["vmax"])
    elif cm["kind"] == "intensity":
        rgba = _colormap_intensity(raster, vmax=cm["vmax"])
    elif cm["kind"] == "inverse_intensity":
        rgba = _colormap_inverse_intensity(raster, vmax=cm["vmax"])
    else:
        raise ValueError(f"unknown colormap kind: {cm['kind']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write via rasterio with the PNG driver. RGBA → 4 bands.
    h, w = rgba.shape[:2]
    with rasterio.open(
        out_path, "w", driver="PNG",
        width=w, height=h, count=4, dtype="uint8",
    ) as dst:
        for b in range(4):
            dst.write(rgba[..., b], b + 1)


def stddev_filter(raster: np.ndarray,
                  transform: "rasterio.Affine",
                  radius_m: float
                  ) -> np.ndarray:
    """Per-pixel standard deviation within a (2·radius+1)² window.

    Uses the identity σ = √(E[X²] − E[X]²) with two separable
    uniform_filter calls, so cost is O(W·H) regardless of radius.
    NaN cells (DEM no-data) are zero-filled before filtering — fine
    in regions with continuous DEM coverage; small bias on tile
    boundaries.

    For the elevation signal `local_relief`, the result is "meters
    of terrain variation within radius_m of this pixel" — high on
    gorge walls / mountainsides, ~0 on flat plains.
    """
    radius_px = max(1, int(round(res_to_pixels(transform, radius_m))))
    size = 2 * radius_px + 1
    t0 = time.time()
    arr = raster.astype(np.float32)
    arr = np.where(np.isnan(arr), 0.0, arr)
    mean = scipy.ndimage.uniform_filter(arr, size=size, mode="nearest")
    mean_sq = scipy.ndimage.uniform_filter(arr * arr, size=size, mode="nearest")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    out = np.sqrt(var).astype(np.float32)
    print(f"[kernels] stddev_filter r={radius_m:.0f}m "
          f"({size}px window) in {time.time()-t0:.1f}s", flush=True)
    return out


def distance_to_high_relief(raster: np.ndarray,
                            transform: "rasterio.Affine",
                            threshold_m: float,
                            relief_radius_m: float = 500.0,
                            ) -> np.ndarray:
    """Meters from each cell to the nearest cell with `local_relief >=
    threshold_m`. "Drama proximity" — captures "you can see distant
    mountains" without doing real viewshed math.

    Internally: stddev_filter(raster, relief_radius_m) → threshold →
    distance_transform_edt. The intermediate stddev pass is roughly
    the same as the `local_relief` signal would compute on its own,
    so re-running it here costs a few extra seconds vs full
    chained-kernel plumbing.
    """
    t0 = time.time()
    relief = stddev_filter(raster, transform, relief_radius_m)
    mask = relief > threshold_m
    # distance_transform_edt returns Euclidean distance (in pixel
    # units) from each cell to the nearest zero cell. We want each
    # cell's distance to the nearest "drama" cell, so the input mask
    # is inverted: drama cells = 0, others = 1.
    dist_px = scipy.ndimage.distance_transform_edt(~mask)
    # Pixel size in meters. The raster grid is constructed to be
    # locally square in meters (see grid_dims), so latitudinal pixel
    # size suffices.
    res_m = abs(transform.e) * _M_PER_DEG_LAT
    dist_m = (dist_px * res_m).astype(np.float32)
    print(f"[kernels] distance_to_high_relief threshold={threshold_m:.0f}m "
          f"(relief r={relief_radius_m:.0f}m) in {time.time()-t0:.1f}s",
          flush=True)
    return dist_m


def subtract_gaussian_blur(raster: np.ndarray,
                           transform: "rasterio.Affine",
                           sigma_m: float
                           ) -> np.ndarray:
    """Return `raster - gaussian_blur(raster, sigma=sigma_m)`.

    For elevation rasters, this yields a "local prominence" signal:
    positive where the cell is higher than its neighborhood average
    (ridges, summits), negative where lower (valleys). The Gaussian
    is separable so the cost is O(width × height) regardless of
    sigma.
    """
    sigma_px = max(1.0, res_to_pixels(transform, sigma_m))
    t0 = time.time()
    src = raster.astype(np.float32)
    # NaN-safe Gaussian: replace NaN with neighborhood mean via a
    # weighted blur. For simplicity here we just fill NaN with 0
    # before blurring; the subtract step preserves NaN in the
    # output via the source-side NaN.
    src_filled = np.where(np.isnan(src), 0.0, src)
    blur = scipy.ndimage.gaussian_filter(src_filled, sigma=sigma_px,
                                          mode="nearest")
    out = src - blur
    print(f"[kernels] subtract_gaussian_blur σ={sigma_m:.0f}m "
          f"({sigma_px:.1f}px) in {time.time()-t0:.1f}s", flush=True)
    return out
