"""Web-Mercator XYZ raster tile pyramid from the scenicness bake's per-tile
PNG fragments. Pure rasterio — the container has no GDAL CLI or osgeo bindings.

Why: a single full-extent overlay PNG does not scale. The 4-country grid is
~104k x 87k px (~34 GB) at 20 m, far past what fits in RAM or uploads to a
browser as one texture. Instead we emit standard {z}/{x}/{y}.png tiles that
MapLibre lazy-loads per viewport — full resolution when zoomed in, nothing
loaded off-screen.

Approach (memory-safe, fast):
  - BASE zoom (zoom_max): reproject each 1-degree fragment ONCE into a
    tile-aligned EPSG:3857 grid, then slice that into 256x256 tiles. One
    reproject per fragment (not per output tile) keeps it ~O(fragments).
    Fragment-boundary tiles are written by two neighbours, so we max-merge
    against any tile already on disk.
  - OVERVIEWS (zoom_max-1 .. zoom_min): build each parent by max-pooling its
    up-to-four children from the zoom above. Never touches the source again,
    so low zooms can't trigger a full-mosaic read.

Max-pool (not average) on the overview downsample keeps thin features (e.g.
10 m waterways, 1 source px wide) visible instead of fading them out.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds as _tf_from_bounds
from rasterio.warp import reproject, Resampling

from . import rasters

_R = 6378137.0
_ORIGIN = math.pi * _R  # 20037508.342789244
TILE_PX = 256
_LAT_LIMIT = 85.05112878


def _tile_bounds_3857(z: int, x: int, y: int):
    n = 2 ** z
    span = 2.0 * _ORIGIN / n
    minx = -_ORIGIN + x * span
    maxy = _ORIGIN - y * span
    return minx, maxy - span, minx + span, maxy


def _lonlat_to_tile(lon: float, lat: float, z: int):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat = max(min(lat, _LAT_LIMIT), -_LAT_LIMIT)
    yr = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(yr)) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def _save_tile(arr, xyz_root: Path, col: str, z: int, x: int, y: int) -> bool:
    """arr: (4, 256, 256) uint8. Skips fully-transparent tiles. Returns written."""
    if int(arr[3].max()) == 0:
        return False
    out = xyz_root / col / str(z) / str(x)
    out.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(arr)
    with rasterio.open(out / f"{y}.png", "w", driver="PNG",
                       width=TILE_PX, height=TILE_PX, count=4, dtype="uint8") as dst:
        dst.write(arr)
    return True


def _load_tile(xyz_root: Path, col: str, z: int, x: int, y: int):
    p = xyz_root / col / str(z) / str(x) / f"{y}.png"
    if not p.exists():
        return None
    with rasterio.open(p) as ds:
        return ds.read()  # (4, 256, 256)


def write_tile_xyz(rgba, tile_bbox, xyz_root, col, zoom_max) -> int:
    """Write base-zoom (`zoom_max`) XYZ tiles for one bake fragment, taking
    the colorized RGBA array directly — no fragment PNG intermediate. Each
    fragment is reprojected ONCE from EPSG:4326 (with the bbox-derived
    src_transform) into a tile-aligned EPSG:3857 grid, then sliced into
    256×256 chunks. Adjacent fragments' 1-degree seams write the same z12
    tile twice; we max-merge against any on-disk tile so both halves appear.

    `rgba` may be (h, w, 4) or (4, h, w). Returns the number of tiles
    written. The colormap should already have set alpha=0 in no-data
    pixels — those tiles get skipped."""
    if rgba.ndim != 3:
        raise ValueError(f"expected 3-d RGBA, got shape {rgba.shape}")
    # Normalize to band-first (4, h, w) for reproject.
    if rgba.shape[-1] == 4:
        src = np.transpose(rgba, (2, 0, 1))
    elif rgba.shape[0] == 4:
        src = rgba
    else:
        raise ValueError(f"no 4-channel axis in shape {rgba.shape}")
    if int(src[3].max()) == 0:
        return 0
    src = np.ascontiguousarray(src)
    _, h, w = src.shape
    src_tf = _tf_from_bounds(tile_bbox[0], tile_bbox[1], tile_bbox[2], tile_bbox[3], w, h)
    # z_max tile range covering this fragment
    x_nw, y_nw = _lonlat_to_tile(tile_bbox[0], tile_bbox[3], zoom_max)
    x_se, y_se = _lonlat_to_tile(tile_bbox[2], tile_bbox[1], zoom_max)
    tx0, tx1 = min(x_nw, x_se), max(x_nw, x_se)
    ty0, ty1 = min(y_nw, y_se), max(y_nw, y_se)
    # tile-aligned destination grid in 3857
    left, _, _, top = _tile_bounds_3857(zoom_max, tx0, ty0)
    _, bottom, right, _ = _tile_bounds_3857(zoom_max, tx1, ty1)
    dw = (tx1 - tx0 + 1) * TILE_PX
    dh = (ty1 - ty0 + 1) * TILE_PX
    dst_tf = _tf_from_bounds(left, bottom, right, top, dw, dh)
    dst = np.zeros((4, dh, dw), dtype=np.uint8)
    for bi in range(4):
        reproject(
            source=src[bi], destination=dst[bi],
            src_transform=src_tf, src_crs="EPSG:4326",
            dst_transform=dst_tf, dst_crs="EPSG:3857",
            resampling=Resampling.average,
        )
    n = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            sub = dst[:, (ty - ty0) * TILE_PX:(ty - ty0 + 1) * TILE_PX,
                      (tx - tx0) * TILE_PX:(tx - tx0 + 1) * TILE_PX]
            if int(sub[3].max()) == 0:
                continue
            existing = _load_tile(xyz_root, col, zoom_max, tx, ty)
            out = np.maximum(sub, existing) if existing is not None else sub
            if _save_tile(out, xyz_root, col, zoom_max, tx, ty):
                n += 1
    return n


def _build_base(col, frag_paths, all_tiles, xyz_root, zoom_max):
    """Legacy fragment-migration path: for each on-disk fragment PNG, hand
    it to write_tile_xyz. Used by `build-tiles` to backfill XYZ tiles from
    existing 4326 fragments (pre-integrated-bake)."""
    for p in frag_paths:
        ti = int(p.stem.rsplit("_t", 1)[1])
        tb = all_tiles[ti]
        with rasterio.open(p) as ds:
            src = ds.read()                      # (4, h, w)
        write_tile_xyz(src, tb, xyz_root, col, zoom_max)


def _build_overviews(col, xyz_root, zoom_min, zoom_max):
    for z in range(zoom_max - 1, zoom_min - 1, -1):
        child_dir = xyz_root / col / str(z + 1)
        if not child_dir.is_dir():
            break
        parents: set[tuple[int, int]] = set()
        for xd in child_dir.iterdir():
            if not xd.is_dir():
                continue
            cx = int(xd.name)
            for yp in xd.glob("*.png"):
                parents.add((cx // 2, int(yp.stem) // 2))
        for (px, py) in parents:
            big = np.zeros((4, 2 * TILE_PX, 2 * TILE_PX), dtype=np.uint8)
            got = False
            for dx in (0, 1):
                for dy in (0, 1):
                    child = _load_tile(xyz_root, col, z + 1, 2 * px + dx, 2 * py + dy)
                    if child is not None:
                        big[:, dy * TILE_PX:(dy + 1) * TILE_PX,
                            dx * TILE_PX:(dx + 1) * TILE_PX] = child
                        got = True
            if not got:
                continue
            small = big.reshape(4, TILE_PX, 2, TILE_PX, 2).max(axis=(2, 4))
            _save_tile(small.astype(np.uint8), xyz_root, col, z, px, py)


def build_overviews(xyz_root, signals, zoom_min: int = 4, zoom_max: int = 12) -> dict:
    """Build z<zoom_min>..z<zoom_max-1> overviews from existing z<zoom_max>
    children. Used by the integrated-bake end-of-run after every fragment has
    written its base tiles in-line. Returns {col: tile_count}."""
    xyz_root = Path(xyz_root)
    summary: dict[str, int] = {}
    for col in signals:
        if not (xyz_root / col / str(zoom_max)).is_dir():
            continue
        _build_overviews(col, xyz_root, zoom_min, zoom_max)
        n = sum(1 for _ in (xyz_root / col).rglob("*.png"))
        summary[col] = n
        print(f"[tiles] overviews {col}: {n} tiles total (z{zoom_min}-{zoom_max})",
              flush=True)
    return summary


def build_pyramid(export_dir, signals, extent, res_m, tile_size_deg, cos_lat,
                  zoom_min: int = 4, zoom_max: int = 12) -> dict:
    """Build {export_dir}/xyz/<col>/{z}/{x}/{y}.png for each signal that has
    fragments. Returns {col: tile_count}. `res_m`/`cos_lat`/`extent` are kept
    in the signature for parity with the bake's grid params (the per-fragment
    reproject derives geometry from each fragment's own tile bbox)."""
    from .bake import _iter_tiles  # lazy: avoid import cycle

    export_dir = Path(export_dir)
    tiles_dir = export_dir / "tiles"
    xyz_root = export_dir / "xyz"
    all_tiles = _iter_tiles(extent, tile_size_deg)
    summary: dict[str, int] = {}

    for col in signals:
        frag_paths = sorted(tiles_dir.glob(f"{col}_t*.png"))
        if not frag_paths:
            continue
        _build_base(col, frag_paths, all_tiles, xyz_root, zoom_max)
        _build_overviews(col, xyz_root, zoom_min, zoom_max)
        n = sum(1 for _ in (xyz_root / col).rglob("*.png")) if (xyz_root / col).is_dir() else 0
        summary[col] = n
        print(f"[tiles] {col}: {n} tiles (z{zoom_min}-{zoom_max})", flush=True)
    return summary
