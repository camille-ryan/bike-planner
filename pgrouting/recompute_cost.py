"""Recompute per-edge cost for ONE cost profile.

Writes into the profile-specific column pair:

    cost_<profile>          real
    reverse_cost_<profile>  real

Each of the 5 production profiles (direct, vineyard_lover, forest_lover,
views, water) has its own column pair so they don't overwrite each
other. The legacy `ways.cost` / `ways.reverse_cost` columns are left
untouched (existing callers can continue to use them; new callers
should select `cost_<profile> AS cost` in their edges_sql).

Tile-mode UPDATE
----------------
The OOM-historic single-UPDATE on 22M edges is replaced with N
per-tile UPDATEs of ~1M edges each, with commits between tiles.
Each tile:
  1. opens a server-side cursor over edges whose SOURCE vertex falls
     in the tile bbox (each edge owned by exactly one tile → no
     duplicate work),
  2. computes new costs in Python and COPYs them into a small temp
     table,
  3. UPDATEs the per-profile columns from the temp table,
  4. COMMITs and moves on.

Per-tile postgres backend memory peak: ~500 MB on a 1M-row UPDATE,
well under the 10 GB container cap.

Run via the CLI: `python3 main.py recompute-cost --profile <name>`.
"""
import argparse
import time
import re

import numpy as np
import psycopg

import config
from cost import bike_edge_cost, _PROFILES


_BATCH = 100_000      # rows per server-cursor fetch chunk
_TILE_SIZE_DEG = 1.0  # 1° tiles cover Austria in ~22 chunks


# -----------------------------------------------------------------------
# Per-edge cost computation (unchanged from the pre-tiling version).
# -----------------------------------------------------------------------

def _recompute_one(row: tuple, profile: str) -> tuple[int, float, float]:
    """Return (gid, new_cost, new_reverse_cost) for a single edge row."""
    (gid, length_m, is_ferry, reverse_cost_in,
     highway, surface, tracktype, oneway, bicycle, cycleway,
     bicycle_road, access, curv_fwd, curv_rev, canopy_frac,
     forest_local, forest_wide, vineyard_local,
     water_local, water_wide, sea_local, sea_wide,
     waterway_along_edge, waterway_local, wetland_local,
     view_dominance, local_relief, regional_relief, distance_to_drama,
     viewpoint_local, viewpoint_regional,
     elev_src, elev_dst) = row

    if elev_src is None or elev_dst is None or length_m <= 0:
        grade_pct_fwd = 0.0
    else:
        grade_pct_fwd = (float(elev_dst) - float(elev_src)) / float(length_m) * 100.0

    scenic_kwargs = dict(
        profile=profile,
        forest_local=float(forest_local), forest_wide=float(forest_wide),
        vineyard_local=float(vineyard_local),
        water_local=float(water_local), water_wide=float(water_wide),
        sea_local=float(sea_local), sea_wide=float(sea_wide),
        waterway_along_edge=float(waterway_along_edge),
        waterway_local=float(waterway_local),
        wetland_local=float(wetland_local),
        view_dominance=float(view_dominance),
        local_relief=float(local_relief),
        regional_relief=float(regional_relief),
        distance_to_drama=float(distance_to_drama),
        viewpoint_local=float(viewpoint_local),
        viewpoint_regional=float(viewpoint_regional),
    )

    fwd_factor = bike_edge_cost(
        highway=highway, surface=surface, tracktype=tracktype,
        bicycle=bicycle, cycleway=cycleway, access=access,
        bicycle_road=bicycle_road, is_ferry=is_ferry,
        grade_pct=grade_pct_fwd, curv=float(curv_fwd),
        canopy_frac=float(canopy_frac),
        **scenic_kwargs,
    )
    if fwd_factor is None:
        return (gid, None, None)
    new_cost = float(fwd_factor) * float(length_m)

    if reverse_cost_in < 0:
        new_reverse_cost = -1.0
    else:
        rev_factor = bike_edge_cost(
            highway=highway, surface=surface, tracktype=tracktype,
            bicycle=bicycle, cycleway=cycleway, access=access,
            bicycle_road=bicycle_road, is_ferry=is_ferry,
            grade_pct=-grade_pct_fwd, curv=float(curv_rev),
            canopy_frac=float(canopy_frac),
            **scenic_kwargs,
        )
        new_reverse_cost = (float(rev_factor) * float(length_m)
                            if rev_factor is not None
                            else -1.0)
    return (gid, new_cost, new_reverse_cost)


# -----------------------------------------------------------------------
# Tile iteration over the ways extent.
# -----------------------------------------------------------------------

def _profile_column_names(profile: str) -> tuple[str, str]:
    """(cost_col, reverse_cost_col) for the given profile.

    Profile names come from cost._PROFILES which is curated, but
    defensively check for SQL-injection-safe identifier shape so this
    function can be reused without revalidation upstream.
    """
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", profile):
        raise ValueError(f"profile name not SQL-safe: {profile!r}")
    return (f"cost_{profile}", f"reverse_cost_{profile}")


def _bake_extent(conn: psycopg.Connection,
                 bbox: tuple[float, float, float, float] | None,
                 ) -> tuple[float, float, float, float]:
    """Extent to tile over — caller's bbox if given, else the full
    ways_vertices_pgr min/max."""
    if bbox is not None:
        return bbox
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MIN(lon), MIN(lat), MAX(lon), MAX(lat) "
            "FROM ways_vertices_pgr"
        )
        row = cur.fetchone()
    return tuple(float(v) for v in row)


def _iter_tiles(extent: tuple[float, float, float, float],
                tile_size_deg: float
                ) -> list[tuple[float, float, float, float]]:
    """Tile grid covering extent, snapped to integer multiples of
    tile_size_deg so re-runs hit identical tile boundaries."""
    xmin, ymin, xmax, ymax = extent
    x_lo = np.floor(xmin / tile_size_deg) * tile_size_deg
    y_lo = np.floor(ymin / tile_size_deg) * tile_size_deg
    nx = int(np.ceil((xmax - x_lo) / tile_size_deg))
    ny = int(np.ceil((ymax - y_lo) / tile_size_deg))
    out: list[tuple[float, float, float, float]] = []
    for j in range(ny):
        for i in range(nx):
            x0 = x_lo + i * tile_size_deg
            y0 = y_lo + j * tile_size_deg
            out.append((round(x0, 6), round(y0, 6),
                        round(x0 + tile_size_deg, 6),
                        round(y0 + tile_size_deg, 6)))
    return out


def _tile_edge_count(conn: psycopg.Connection,
                     tile_bbox: tuple[float, float, float, float]) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM ways w "
            "JOIN ways_vertices_pgr vs ON vs.id = w.source "
            "WHERE vs.lon >= %s AND vs.lon < %s "
            "  AND vs.lat >= %s AND vs.lat < %s",
            (tile_bbox[0], tile_bbox[2], tile_bbox[1], tile_bbox[3]),
        )
        return int(cur.fetchone()[0])


# -----------------------------------------------------------------------
# Per-tile recompute.
# -----------------------------------------------------------------------

def _recompute_tile(conn: psycopg.Connection,
                    tile_bbox: tuple[float, float, float, float],
                    profile: str,
                    cost_col: str, rev_col: str,
                    ) -> tuple[int, int]:
    """Recompute one tile's edges and apply to the per-profile columns.
    Returns (n_costed, n_skipped). Commits at the end of the tile so
    the next tile starts with locks released."""
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute("SET work_mem = '256MB'")
        # Per-tile temp table. ON COMMIT DROP would invalidate our
        # commit-between-tiles pattern; PRESERVE ROWS is the default so
        # we explicitly DROP at the bottom.
        cur.execute("DROP TABLE IF EXISTS _new_cost_tile")
        cur.execute(
            "CREATE TEMP TABLE _new_cost_tile ("
            "  gid          bigint PRIMARY KEY, "
            "  cost         real NOT NULL, "
            "  reverse_cost real NOT NULL"
            ")"
        )

    # Scan tile's edges via server cursor. Edge owned by tile containing
    # its source vertex — each edge processed in exactly one tile.
    n_costed = 0
    n_skipped = 0
    buf: list[tuple] = []
    with conn.cursor(name="recompute_tile_scan") as scan_cur:
        scan_cur.itersize = _BATCH
        scan_cur.execute(
            "SELECT w.gid, w.length_m, w.is_ferry, w.reverse_cost, "
            "w.highway, w.surface, w.tracktype, w.oneway, "
            "w.bicycle, w.cycleway, w.bicycle_road, w.access, "
            "w.curv_fwd, w.curv_rev, w.canopy_frac, "
            "w.forest_local, w.forest_wide, w.vineyard_local, "
            "w.water_local, w.water_wide, "
            "w.sea_local, w.sea_wide, "
            "w.waterway_along_edge, w.waterway_local, w.wetland_local, "
            "w.view_dominance, w.local_relief, "
            "w.regional_relief, w.distance_to_drama, "
            "w.viewpoint_local, w.viewpoint_regional, "
            "vs.elev_m AS elev_src, vt.elev_m AS elev_dst "
            "FROM ways w "
            "JOIN ways_vertices_pgr vs ON vs.id = w.source "
            "JOIN ways_vertices_pgr vt ON vt.id = w.target "
            "WHERE vs.lon >= %s AND vs.lon < %s "
            "  AND vs.lat >= %s AND vs.lat < %s",
            (tile_bbox[0], tile_bbox[2], tile_bbox[1], tile_bbox[3]),
        )

        def _flush(rows: list[tuple]) -> None:
            with conn.cursor() as wcur:
                with wcur.copy(
                    "COPY _new_cost_tile (gid, cost, reverse_cost) FROM STDIN"
                ) as cp:
                    for t in rows:
                        cp.write_row(t)

        for row in scan_cur:
            gid, new_cost, new_rev = _recompute_one(row, profile)
            if new_cost is None:
                n_skipped += 1
                continue
            buf.append((gid, new_cost, new_rev))
            if len(buf) >= _BATCH:
                _flush(buf)
                n_costed += len(buf)
                buf.clear()
        if buf:
            _flush(buf)
            n_costed += len(buf)
    t_scan = time.time() - t0

    # Per-tile UPDATE: bounded to the tile's row count, never the full
    # 22M edges, so the OOM-historic giant UPDATE never happens.
    t_upd = time.time()
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute(
            f"UPDATE ways w SET "
            f"  {cost_col} = n.cost, "
            f"  {rev_col} = n.reverse_cost "
            f"FROM _new_cost_tile n WHERE w.gid = n.gid"
        )
        cur.execute("DROP TABLE _new_cost_tile")
    conn.commit()
    print(f"[recompute]   tile {tile_bbox}: "
          f"{n_costed:,} edges, scan {t_scan:.1f}s + UPDATE {time.time()-t_upd:.1f}s",
          flush=True)
    return n_costed, n_skipped


# -----------------------------------------------------------------------
# Top-level
# -----------------------------------------------------------------------

def recompute(conn: psycopg.Connection,
              bbox: tuple[float, float, float, float] | None = None,
              profile: str = "direct",
              tile_size_deg: float = _TILE_SIZE_DEG,
              ) -> None:
    """Recompute edges for one profile, tile-by-tile, writing into
    the profile-specific column pair."""
    if profile not in _PROFILES:
        raise SystemExit(
            f"unknown profile {profile!r}; "
            f"expected one of {sorted(_PROFILES.keys())}"
        )
    cost_col, rev_col = _profile_column_names(profile)

    # Ensure the columns exist (idempotent — schema.sql defines them
    # but we also tolerate fresh deployments without the migration).
    with conn.cursor() as cur:
        cur.execute(f"ALTER TABLE ways ADD COLUMN IF NOT EXISTS {cost_col} real")
        cur.execute(f"ALTER TABLE ways ADD COLUMN IF NOT EXISTS {rev_col} real")
    conn.commit()

    extent = _bake_extent(conn, bbox)
    all_tiles = _iter_tiles(extent, tile_size_deg)
    # Filter empty tiles (corners of the bbox covering no ways data).
    tiles = [t for t in all_tiles if _tile_edge_count(conn, t) > 0]

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM ways_vertices_pgr WHERE elev_m IS NULL")
        n_null_elev = int(cur.fetchone()[0])

    print(f"[recompute] profile={profile} "
          f"writing to ({cost_col}, {rev_col}) "
          f"over {len(tiles)} non-empty tiles "
          f"(of {len(all_tiles)} candidate tiles, tile_size={tile_size_deg}°). "
          f"{n_null_elev:,} vertices have NULL elev (grade=0 there).")
    t_total = time.time()
    n_total_costed = 0
    n_total_skipped = 0
    for ti, tile_bbox in enumerate(tiles):
        print(f"[recompute] === tile {ti+1}/{len(tiles)} {tile_bbox} ===",
              flush=True)
        c, s = _recompute_tile(conn, tile_bbox, profile, cost_col, rev_col)
        n_total_costed += c
        n_total_skipped += s
    elapsed = (time.time() - t_total) / 60
    print(f"[recompute] DONE profile={profile}: "
          f"{n_total_costed:,} edges costed, {n_total_skipped:,} skipped "
          f"in {elapsed:.1f} min across {len(tiles)} tiles.")

    # Sanity: any NULL remaining in the profile's columns?
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM ways WHERE {cost_col} IS NULL")
        n_null = int(cur.fetchone()[0])
    if n_null > 0:
        print(f"[recompute] WARNING: {n_null:,} edges have NULL {cost_col} "
              "after recompute (outside any tile? bbox filter?)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bbox", type=str, default=None,
                   help="min_lon,min_lat,max_lon,max_lat — restrict to this bbox.")
    p.add_argument("--profile", default="direct",
                   help="cost profile name (must exist in cost._PROFILES).")
    p.add_argument("--tile-size-deg", type=float, default=_TILE_SIZE_DEG)
    args = p.parse_args()
    bbox: tuple[float, float, float, float] | None = None
    if args.bbox:
        parts = tuple(float(x) for x in args.bbox.split(","))
        if len(parts) != 4:
            raise SystemExit("--bbox must be 4 comma-separated floats")
        bbox = parts
    with psycopg.connect(config.PG_DSN) as conn:
        recompute(conn, bbox=bbox, profile=args.profile,
                  tile_size_deg=args.tile_size_deg)
