"""Recompute edge cost using elevation + directional curvature.

Run after ingest_pbf + ingest_dem. Re-applies `bike_edge_cost` to
every edge in `ways`, this time supplying `grade_pct` (derived from
the vertex elevations now sitting in `ways_vertices_pgr`) and the
per-edge directional curvature (`curv_fwd` / `curv_rev`) that
ingest already stamped onto the edge.

Direction handling:
  - Forward grade  = (elev_target - elev_source) / length_m × 100
  - Reverse grade  = -forward grade (same edge, opposite direction)
  - Forward cost   uses `curv_fwd` (bends ahead when traveling source→target)
  - Reverse cost   uses `curv_rev` (bends ahead when traveling target→source)
  - If the edge is oneway (reverse_cost == -1 in V1/V2 convention),
    reverse_cost stays -1.

When either endpoint elevation is NULL (DEM didn't cover it),
grade_pct is treated as 0.0 — same as the V2 default no-op.

Performance: streams through `ways` in batches via a server-side
cursor so the working set stays bounded. Recomputed (gid, cost,
reverse_cost) triples land in a temp table, then a single
UPDATE…FROM applies them in bulk per batch.
"""
import argparse

import psycopg

import config
from cost import bike_edge_cost


_BATCH = 100_000   # rows per fetch from the server-side cursor


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
        # An edge that previously priced now refuses to. Should not
        # happen with V2 (no surface excludes), but be defensive:
        # keep the original cost so the graph stays connected.
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


def recompute(conn: psycopg.Connection,
              bbox: tuple[float, float, float, float] | None = None,
              profile: str = "direct") -> None:
    """Recompute edge costs over the whole graph or a bbox subset.

    bbox = (min_lon, min_lat, max_lon, max_lat) — when supplied, both
    endpoints of an edge must fall inside the box for the edge to be
    recomputed. Useful for fast validation runs over a specific
    corridor without re-pricing the whole graph.

    profile selects the cost model. 'direct' is the no-scenic-discount
    baseline; 'balanced' / 'scenic' apply the multiplicative scenic
    discount from cost._scenic_factor. The persistent ways.cost /
    ways.reverse_cost columns are overwritten each run, so to A/B
    profiles in series, recompute one and export its routes before
    switching to the next.
    """
    with conn.cursor() as cur:
        # Parallel hash join over ways+vertices twice can blow past the
        # postgres container's /dev/shm. We don't need parallelism here.
        cur.execute("SET max_parallel_workers_per_gather = 0")
        if bbox:
            cur.execute(
                "SELECT COUNT(*) FROM ways w "
                "JOIN ways_vertices_pgr vs ON vs.id = w.source "
                "JOIN ways_vertices_pgr vt ON vt.id = w.target "
                "WHERE vs.lon BETWEEN %s AND %s AND vs.lat BETWEEN %s AND %s "
                "AND vt.lon BETWEEN %s AND %s AND vt.lat BETWEEN %s AND %s",
                (bbox[0], bbox[2], bbox[1], bbox[3]) * 2,
            )
        else:
            cur.execute("SELECT COUNT(*) FROM ways")
        n_edges = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM ways_vertices_pgr WHERE elev_m IS NULL")
        n_null_elev = int(cur.fetchone()[0])
    print(f"[recompute] {n_edges:,} edges{' (bbox-limited)' if bbox else ''}; "
          f"{n_null_elev:,} vertices have NULL elevation (will use grade=0 there); "
          f"profile={profile}")

    # Single server-side cursor over the bbox-limited corridor. Postgres
    # plans the bbox JOIN once and streams rows; we accumulate triples
    # in Python and COPY them into _new_cost in chunks. NO commits
    # during the scan — they kill the server cursor. One final UPDATE
    # joins _new_cost to ways at the end. Matches the pattern in
    # scenicness/bake.py.
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute("DROP TABLE IF EXISTS _new_cost")
        cur.execute("""
            CREATE TEMP TABLE _new_cost (
                gid          bigint PRIMARY KEY,
                cost         double precision NOT NULL,
                reverse_cost double precision NOT NULL
            )
        """)

    bbox_filter = ""
    bbox_params: tuple = ()
    if bbox:
        bbox_filter = (
            " WHERE vs.lon BETWEEN %s AND %s AND vs.lat BETWEEN %s AND %s "
            " AND vt.lon BETWEEN %s AND %s AND vt.lat BETWEEN %s AND %s "
        )
        bbox_params = (bbox[0], bbox[2], bbox[1], bbox[3]) * 2

    import time
    t_scan = time.time()
    last_print = t_scan
    processed = 0
    skipped_no_cost = 0

    with conn.cursor(name="recompute_edge_scan") as scan_cur:
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
            + bbox_filter,
            bbox_params,
        )

        buf: list[tuple] = []

        def _flush(rows: list[tuple]) -> None:
            with conn.cursor() as wcur:
                with wcur.copy(
                    "COPY _new_cost (gid, cost, reverse_cost) FROM STDIN"
                ) as cp:
                    for t in rows:
                        cp.write_row(t)

        for row in scan_cur:
            gid, new_cost, new_rev = _recompute_one(row, profile)
            if new_cost is None:
                skipped_no_cost += 1
                continue
            buf.append((gid, new_cost, new_rev))
            if len(buf) >= _BATCH:
                _flush(buf)
                processed += len(buf)
                buf.clear()
                if time.time() - last_print > 5:
                    rate = processed / max(time.time() - t_scan, 1e-3)
                    print(f"[recompute]   {processed:,} edges costed "
                          f"({rate:.0f}/s, skipped {skipped_no_cost:,})",
                          flush=True)
                    last_print = time.time()
        if buf:
            _flush(buf)
            processed += len(buf)

    print(f"[recompute] scan+COPY done: {processed:,} costed edges in "
          f"{(time.time()-t_scan)/60:.1f} min", flush=True)

    # Single UPDATE applies all new costs.
    t_upd = time.time()
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute(
            "UPDATE ways w "
            "SET cost = n.cost, reverse_cost = n.reverse_cost "
            "FROM _new_cost n WHERE w.gid = n.gid"
        )
        updated = cur.rowcount
        cur.execute("DROP TABLE _new_cost")
    conn.commit()
    print(f"[recompute] applied to {updated:,} edges in "
          f"{time.time()-t_upd:.1f}s "
          f"(skipped {skipped_no_cost:,} no-cost edges)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bbox", type=str, default=None,
                   help="min_lon,min_lat,max_lon,max_lat — restrict to edges "
                        "whose both endpoints fall inside this box.")
    p.add_argument("--profile", default="direct",
                   choices=("direct", "balanced", "scenic",
                            "direct_thresh5", "direct_minus3",
                            "direct_scenic_2x", "direct_scenic_5x", "direct_scenic_10x"),
                   help="cost model: direct (no scenic discount), balanced "
                        "(~20%% detour budget), scenic (~50%% detour budget), "
                        "or one of the experimental profiles.")
    args = p.parse_args()
    bbox: tuple[float, float, float, float] | None = None
    if args.bbox:
        parts = tuple(float(x) for x in args.bbox.split(","))
        if len(parts) != 4:
            raise SystemExit("--bbox must be 4 comma-separated floats")
        bbox = parts
    with psycopg.connect(config.PG_DSN) as conn:
        recompute(conn, bbox=bbox, profile=args.profile)
