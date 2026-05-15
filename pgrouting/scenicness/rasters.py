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
              res_m: float,
              cos_lat: float | None = None,
              ) -> tuple[int, int, "rasterio.Affine"]:
    """Compute (width, height, transform) for a meter-resolution grid
    over the given lon/lat bbox.

    `cos_lat` overrides the latitude-correction factor used for the
    longitudinal pixel size. Passing a fixed cos_lat across all tiles
    in a tiled bake keeps pixel sizes consistent so the tile slices
    paste together without drift; default (None) computes cos(mid_lat)
    from the bbox.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    if cos_lat is None:
        cos_lat = float(np.cos(np.radians((min_lat + max_lat) / 2.0)))
    deg_lon_per_m = 1.0 / (_M_PER_DEG_LAT * cos_lat)
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

def rasterize_points(sqlite_path,
                     category: str,
                     bbox: tuple[float, float, float, float],
                     res_m: float,
                     cos_lat: float | None = None,
                     ) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Rasterize POIs (single-pixel hits) of the given category onto a
    binary uint8 mask. Returns (mask, transform).

    The pois.sqlite DB uses SpatiaLite, but for *reading* X/Y of points
    we don't need the extension — store schema includes geom but a
    plain X(geom)/Y(geom) requires SpatiaLite functions. Load the
    mod_spatialite extension to access them."""
    import sqlite3
    width, height, transform = grid_dims(bbox, res_m, cos_lat=cos_lat)
    out = np.zeros((height, width), dtype=np.uint8)
    con = sqlite3.connect(str(sqlite_path))
    con.enable_load_extension(True)
    con.execute("SELECT load_extension('mod_spatialite')")
    rows = con.execute("""
        SELECT X(geom), Y(geom) FROM pois
        WHERE category = ?
          AND X(geom) BETWEEN ? AND ?
          AND Y(geom) BETWEEN ? AND ?
    """, (category, bbox[0], bbox[2], bbox[1], bbox[3])).fetchall()
    con.close()
    # Inverse-transform each (lon, lat) → (col, row) once. Affine inverse
    # gives a callable that handles the +0.5 corner→center semantics.
    inv = ~transform
    n_in = 0
    for lon, lat in rows:
        col, row = inv * (lon, lat)
        c, r = int(col), int(row)
        if 0 <= c < width and 0 <= r < height:
            out[r, c] = 1
            n_in += 1
    print(f"[rasters] points/{category}: {len(rows):,} POIs, "
          f"{n_in:,} inside grid ({width}x{height})", flush=True)
    return out, transform


def rasterize_polygons(conn,
                       landcover_class: str,
                       bbox: tuple[float, float, float, float],
                       res_m: float,
                       cos_lat: float | None = None,
                       ) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Rasterize every `landcover` polygon of the given class within
    bbox into a binary uint8 mask. Returns (mask, transform).
    """
    width, height, transform = grid_dims(bbox, res_m, cos_lat=cos_lat)
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
               cos_lat: float | None = None,
               ) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Stitch Copernicus DEM GLO-30 tiles intersecting bbox into a
    single float32 raster at the target resolution.

    Tiles are 1°×1° at ~30m native; we resample with nearest neighbor
    onto the target grid (cheap, adequate for the smooth elevation
    signals we derive). No-data cells become NaN.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    width, height, transform = grid_dims(bbox, res_m, cos_lat=cos_lat)
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


def _colormap_purple(values: np.ndarray) -> np.ndarray:
    """Wine-purple gradient for vineyards. Distinct from the diverging
    (red/blue) and intensity (amber/red) maps used by terrain signals."""
    v = np.clip(values, 0.0, 1.0).astype(np.float32)
    h, w = v.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = (90  + 90  * v).astype(np.uint8)
    rgba[..., 1] = (30  + 30  * v).astype(np.uint8)
    rgba[..., 2] = (110 + 100 * v).astype(np.uint8)
    rgba[..., 3] = (170 * v).astype(np.uint8)
    return rgba


def _colormap_intensity(values: np.ndarray,
                        vmax: float
                        ) -> np.ndarray:
    """Single-hue intensity colormap: transparent → amber → red.
    Used for unsigned magnitude signals like local_relief.

    NaN cells become fully transparent (no-data regions).
    """
    invalid = np.isnan(values)
    v = np.where(invalid, 0.0,
                 np.clip(values / max(vmax, 1e-6), 0.0, 1.0)).astype(np.float32)
    h, w = v.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = (200 + 55 * v).astype(np.uint8)
    rgba[..., 1] = (160 - 80 * v).astype(np.uint8)
    rgba[..., 2] = (40 + 20 * (1 - v)).astype(np.uint8)
    alpha = (180 * v).astype(np.uint8)
    rgba[..., 3] = np.where(invalid, 0, alpha).astype(np.uint8)
    return rgba


def _colormap_inverse_intensity(values: np.ndarray,
                                vmax: float
                                ) -> np.ndarray:
    """Inverse intensity: bright purple at low values, transparent
    at high. For distance-style signals where SMALL = interesting
    (e.g. distance_to_drama — closer to mountains is more visible).

    NaN cells become fully transparent (no-data regions).
    """
    invalid = np.isnan(values)
    v = np.where(invalid, 1.0,
                 1.0 - np.clip(values / max(vmax, 1e-6), 0.0, 1.0)
                 ).astype(np.float32)
    h, w = v.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = (140 + 70 * v).astype(np.uint8)
    rgba[..., 1] = (40 + 30 * (1 - v)).astype(np.uint8)
    rgba[..., 2] = (180 + 60 * v).astype(np.uint8)
    alpha = (180 * v).astype(np.uint8)
    rgba[..., 3] = np.where(invalid, 0, alpha).astype(np.uint8)
    return rgba


def _colormap_diverging(values: np.ndarray,
                        vmax: float
                        ) -> np.ndarray:
    """Diverging blue-(transparent)-red colormap.

    Input: float32 (signed). Output: (H, W, 4) uint8 RGBA.
    `vmax` is the absolute value at full saturation; symmetric.

    NaN cells become fully transparent (no-data regions).
    """
    invalid = np.isnan(values)
    v = np.where(invalid, 0.0, values).astype(np.float32)
    v = np.clip(v / max(vmax, 1e-6), -1.0, 1.0)
    h, w = v.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    neg = (v < 0) & ~invalid
    pos = (v > 0) & ~invalid
    a = np.abs(v)
    rgba[neg, 0] = (40 + 40 * (1 - a[neg])).astype(np.uint8)
    rgba[neg, 1] = (80 + 60 * (1 - a[neg])).astype(np.uint8)
    rgba[neg, 2] = (140 + 100 * a[neg]).astype(np.uint8)
    rgba[pos, 0] = (140 + 100 * a[pos]).astype(np.uint8)
    rgba[pos, 1] = (90 + 30 * (1 - a[pos])).astype(np.uint8)
    rgba[pos, 2] = (40 + 30 * (1 - a[pos])).astype(np.uint8)
    alpha = (180 * a).astype(np.uint8)
    rgba[..., 3] = np.where(invalid, 0, alpha).astype(np.uint8)
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
    # Vineyards: distinct purple — cultivated land but positively scored.
    "vineyard_local":       {"kind": "purple"},
    # Viewpoints: POI-density values are tiny (binary mask × uniform
    # blur), so use intensity with a vmax matched to observed maxima.
    # Bake summary (Austria-wide): vp_local max=0.074, vp_regional max=0.0013.
    "viewpoint_local":      {"kind": "intensity", "vmax": 0.03},
    "viewpoint_regional":   {"kind": "intensity", "vmax": 0.0008},
}


def warp_lat_to_mercator_rows(rgba: np.ndarray,
                              bbox: tuple[float, float, float, float],
                              ) -> np.ndarray:
    """Resample a raster's rows from linear-in-latitude spacing to
    linear-in-mercator-Y spacing.

    Necessary because MapLibre's image source renders the image quad
    by bilinear interpolation in mercator (display) space, while our
    rasterio.transform.from_bounds raster is generated with uniform
    latitude per row. At Austrian latitudes that mismatch puts each
    row's data ~1 km too far north when displayed; the fix is to
    pre-warp the PNG so MapLibre's mercator interpolation sees the
    intended geographic alignment.

    Columns are NOT warped: longitude is linear in mercator-X (and at
    our small bbox sizes the cos-correction is well under a pixel),
    so column-wise alignment was never the problem.

    Nearest-neighbor row resampling. Linear interpolation would be
    smoother but the worst case is sub-pixel at our 20 m resolution.
    """
    h, w = rgba.shape[:2]
    min_lat, max_lat = bbox[1], bbox[3]
    # Mercator Y in dimensionless units (R_earth scaling cancels in the
    # ratio below).
    def lat_to_y(lat_deg):
        return np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))
    def y_to_lat(y):
        return np.degrees(2 * (np.arctan(np.exp(y)) - np.pi / 4))
    y_top = lat_to_y(max_lat)
    y_bot = lat_to_y(min_lat)
    # For each OUTPUT row R_out, MapLibre will display its content at
    # the lat that mercator-Y-interpolates to (linear in mercator).
    # We want that displayed lat to equal the lat our SOURCE row R_src
    # was generated for (linear in lat). Solve for R_src(R_out):
    R_out = np.arange(h)
    y_at_out = y_top - (R_out / h) * (y_top - y_bot)
    lat_at_out = y_to_lat(y_at_out)
    R_src = (max_lat - lat_at_out) / (max_lat - min_lat) * h
    R_src_int = np.clip(R_src.astype(np.int64), 0, h - 1)
    return rgba[R_src_int]


def write_signal_png(raster: np.ndarray,
                     out_path: Path,
                     column: str,
                     bbox: tuple[float, float, float, float] | None = None,
                     ) -> None:
    """Render a signal raster to a web-ready PNG with the column's
    colormap. Geographic registration is carried in the bake manifest
    alongside the PNG, not embedded in the file.

    If `bbox` is supplied the output rows are warped to linear-in-
    mercator-Y spacing so MapLibre's image-source renderer (which
    interpolates the quad linearly in mercator) places each row at
    its intended geographic latitude. Omit `bbox` only for synthetic
    test rasters or non-geographic use."""
    cm = COLORMAPS.get(column, {"kind": "green"})
    if cm["kind"] == "green":
        rgba = _colormap_green(raster)
    elif cm["kind"] == "blue":
        rgba = _colormap_blue(raster)
    elif cm["kind"] == "teal":
        rgba = _colormap_teal(raster)
    elif cm["kind"] == "purple":
        rgba = _colormap_purple(raster)
    elif cm["kind"] == "diverging":
        rgba = _colormap_diverging(raster, vmax=cm["vmax"])
    elif cm["kind"] == "intensity":
        rgba = _colormap_intensity(raster, vmax=cm["vmax"])
    elif cm["kind"] == "inverse_intensity":
        rgba = _colormap_inverse_intensity(raster, vmax=cm["vmax"])
    else:
        raise ValueError(f"unknown colormap kind: {cm['kind']}")
    if bbox is not None:
        rgba = warp_lat_to_mercator_rows(rgba, bbox)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write via rasterio with the PNG driver. RGBA → 4 bands.
    h, w = rgba.shape[:2]
    with rasterio.open(
        out_path, "w", driver="PNG",
        width=w, height=h, count=4, dtype="uint8",
    ) as dst:
        for b in range(4):
            dst.write(rgba[..., b], b + 1)


def _nan_uniform_filter(arr: np.ndarray, size: int) -> np.ndarray:
    """Uniform-window mean over valid (non-NaN) cells. Output is NaN
    where no valid cells are in the window. Used so DEM no-data regions
    don't bias kernel outputs toward 0 at their boundaries.
    """
    valid = ~np.isnan(arr)
    arr_zero = np.where(valid, arr, 0.0).astype(np.float32)
    valid_f = valid.astype(np.float32)
    sum_avg = scipy.ndimage.uniform_filter(arr_zero, size=size, mode="nearest")
    valid_avg = scipy.ndimage.uniform_filter(valid_f, size=size, mode="nearest")
    out = np.where(valid_avg > 1e-6, sum_avg / valid_avg, np.nan)
    return out.astype(np.float32)


def stddev_filter(raster: np.ndarray,
                  transform: "rasterio.Affine",
                  radius_m: float
                  ) -> np.ndarray:
    """Per-pixel standard deviation within a (2·radius+1)² window,
    computed only over valid (non-NaN) cells.

    NaN-aware: DEM no-data regions don't bias the filter. Cells whose
    entire window is NaN come out as NaN (preserved through colormap
    → fully transparent in the PNG).

    Uses σ = √(E[X²] − E[X]²) with two NaN-aware uniform filters, so
    cost is O(W·H) regardless of radius.
    """
    radius_px = max(1, int(round(res_to_pixels(transform, radius_m))))
    size = 2 * radius_px + 1
    t0 = time.time()
    arr = raster.astype(np.float32)
    mean = _nan_uniform_filter(arr, size)
    mean_sq = _nan_uniform_filter(arr * arr, size)
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

    Internally: stddev_filter (NaN-aware) → threshold → distance
    transform. Cells whose source DEM was NaN come out as NaN (the
    distance from missing data is meaningless and would otherwise
    show as "very far from drama" everywhere outside coverage).
    """
    t0 = time.time()
    relief = stddev_filter(raster, transform, relief_radius_m)
    # NaN-aware threshold: NaN > threshold → False, so NaN cells are
    # treated as "not drama" (correct — they're unknown, not dramatic).
    mask = np.where(np.isnan(relief), False, relief > threshold_m)
    dist_px = scipy.ndimage.distance_transform_edt(~mask)
    res_m = abs(transform.e) * _M_PER_DEG_LAT
    dist_m = (dist_px * res_m).astype(np.float32)
    # Wherever DEM was missing, the "distance to drama" is undefined.
    dist_m[np.isnan(raster) | np.isnan(relief)] = np.nan
    print(f"[kernels] distance_to_high_relief threshold={threshold_m:.0f}m "
          f"(relief r={relief_radius_m:.0f}m) in {time.time()-t0:.1f}s",
          flush=True)
    return dist_m


def subtract_gaussian_blur(raster: np.ndarray,
                           transform: "rasterio.Affine",
                           sigma_m: float
                           ) -> np.ndarray:
    """Return `raster - gaussian_blur(raster, sigma=sigma_m)`,
    NaN-aware: the blur is the Gaussian-weighted mean over valid
    cells only, so DEM no-data regions don't drag down the blur of
    nearby valid cells. NaN propagates through the subtraction so
    output pixels with no source data remain NaN.
    """
    sigma_px = max(1.0, res_to_pixels(transform, sigma_m))
    t0 = time.time()
    src = raster.astype(np.float32)
    valid = ~np.isnan(src)
    src_zero = np.where(valid, src, 0.0).astype(np.float32)
    valid_f = valid.astype(np.float32)
    sum_blur = scipy.ndimage.gaussian_filter(
        src_zero, sigma=sigma_px, mode="nearest")
    valid_blur = scipy.ndimage.gaussian_filter(
        valid_f, sigma=sigma_px, mode="nearest")
    blur = np.where(valid_blur > 1e-6,
                    sum_blur / valid_blur,
                    np.nan).astype(np.float32)
    out = src - blur
    print(f"[kernels] subtract_gaussian_blur σ={sigma_m:.0f}m "
          f"({sigma_px:.1f}px) in {time.time()-t0:.1f}s", flush=True)
    return out
