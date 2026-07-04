"""Download Copernicus DEM GLO-30 GeoTIFF tiles for a bounding box.

The Copernicus DEM is a public dataset available from AWS without
authentication. Each tile covers a 1°×1° area, named by its
south-west integer corner. We download only the tiles whose extent
intersects the requested bbox.

Tile naming convention:
    Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM
where lat/lon are the south-west corner integer degrees (N for
northern hemisphere, S for southern; E for east of prime meridian,
W for west).

URL pattern (HTTPS, no auth):
    https://copernicus-dem-30m.s3.amazonaws.com/{tile}/{tile}.tif

Idempotent: existing files on disk are skipped. 404s (ocean tiles
or missing tiles in oblique parts of the bucket) are reported but
not fatal.
"""
import argparse
from pathlib import Path

import requests

import config


_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"


def _tile_name(lat: int, lon: int) -> str:
    """Tile basename without extension. lat/lon are SW-corner integers."""
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def _bbox_tiles(min_lat: float, min_lon: float, max_lat: float, max_lon: float):
    """Yield (lat, lon) SW-corner integers for every tile intersecting the bbox."""
    import math
    lat0 = math.floor(min_lat)
    lat1 = math.floor(max_lat - 1e-9)   # inclusive of integer-aligned upper
    lon0 = math.floor(min_lon)
    lon1 = math.floor(max_lon - 1e-9)
    for lat in range(lat0, lat1 + 1):
        for lon in range(lon0, lon1 + 1):
            yield lat, lon


def _download_one(tile: str, out_path: Path) -> str:
    """Download one tile. Returns 'ok' / 'cached' / '404' / 'err: ...'."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return "cached"
    url = f"{_BASE_URL}/{tile}/{tile}.tif"
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code == 404:
                return "404"
            r.raise_for_status()
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            tmp.rename(out_path)
            return "ok"
    except requests.RequestException as e:
        # Clean up partial file
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        if tmp.exists():
            tmp.unlink()
        return f"err: {e}"


def download(bbox: tuple[float, float, float, float],
             out_dir: Path | None = None) -> dict:
    """Download all DEM tiles intersecting `bbox = (min_lat, min_lon, max_lat, max_lon)`.

    Returns a dict summary: {'ok': N, 'cached': N, '404': N, 'err': [...]}
    """
    out_dir = out_dir or config.DEM_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = list(_bbox_tiles(*bbox))
    print(f"[dem] {len(tiles)} tiles intersect bbox {bbox}, out={out_dir}")

    stats = {"ok": 0, "cached": 0, "404": 0, "err": []}
    for i, (lat, lon) in enumerate(tiles, 1):
        tile = _tile_name(lat, lon)
        path = out_dir / f"{tile}.tif"
        status = _download_one(tile, path)
        if status in ("ok", "cached", "404"):
            stats[status] += 1
        else:
            stats["err"].append((tile, status))
        if i % 10 == 0 or i == len(tiles):
            print(f"[dem]   {i}/{len(tiles)}  ok={stats['ok']} cached={stats['cached']} "
                  f"404={stats['404']} err={len(stats['err'])}")
    if stats["err"]:
        print(f"[dem] errors:")
        for tile, msg in stats["err"]:
            print(f"  {tile}: {msg}")
    return stats


_COUNTRY_BBOX = {
    # Coarse bboxes (min_lat, min_lon, max_lat, max_lon) for the corridor.
    "austria":         (46.0,  9.0, 49.0, 17.0),
    "czech-republic":  (48.0, 12.0, 51.0, 19.0),
    "germany":         (47.0,  5.0, 55.0, 15.0),
    "denmark":         (54.0,  8.0, 58.0, 13.0),
}


def bbox_for_countries(countries: list[str]) -> tuple[float, float, float, float]:
    """Union of pre-computed country bboxes; useful before any ingest exists."""
    boxes = [_COUNTRY_BBOX[c] for c in countries if c in _COUNTRY_BBOX]
    if not boxes:
        raise ValueError(f"unknown countries: {countries}")
    min_lat = min(b[0] for b in boxes)
    min_lon = min(b[1] for b in boxes)
    max_lat = max(b[2] for b in boxes)
    max_lon = max(b[3] for b in boxes)
    return (min_lat, min_lon, max_lat, max_lon)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--bbox", type=str,
                   help="min_lat,min_lon,max_lat,max_lon (e.g. 47,8,56,17)")
    g.add_argument("--countries", type=str,
                   help="comma-separated country names from _COUNTRY_BBOX")
    args = p.parse_args()
    if args.bbox:
        bbox = tuple(float(x) for x in args.bbox.split(","))
        if len(bbox) != 4:
            raise SystemExit("--bbox must be 4 comma-separated floats")
    else:
        countries = [c.strip() for c in args.countries.split(",") if c.strip()]
        bbox = bbox_for_countries(countries)
    download(bbox)
