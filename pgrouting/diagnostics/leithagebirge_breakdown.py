"""One-shot diagnostic: compare the free-routing Graz→Hainburg path
against a forced detour through the Leithagebirge, breaking down where
each route's cost actually goes.

Assumes ways.cost currently reflects the SCENIC profile (run
`recompute-cost --profile scenic` first). Uses pgr_dijkstraVia with a
fully-inlined edges_sql to avoid the parameterized-plan OOM that hit us
earlier (see compare_corridors.py for the original write-up).
"""
import psycopg
import config

GRAZ      = (15.4395, 47.0707)
HAINBURG  = (16.9407, 48.1428)
LEITHA_MID = (16.6,   47.95)    # center of Leithagebirge (gentle forested ridge SE of Wien)

BBOX = (14.5, 46.8, 17.0, 48.5)


def _snap(cur, lon: float, lat: float) -> int:
    cur.execute(
        "SELECT id FROM ways_vertices_pgr "
        "ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326) "
        "LIMIT 1", (lon, lat),
    )
    return int(cur.fetchone()[0])


def _edges_sql() -> str:
    b = BBOX
    return (
        "SELECT w.gid AS id, w.source, w.target, w.cost, w.reverse_cost "
        "FROM ways w "
        "JOIN ways_vertices_pgr vs ON vs.id = w.source "
        "JOIN ways_vertices_pgr vt ON vt.id = w.target "
        f"WHERE vs.lon BETWEEN {b[0]} AND {b[2]} AND vs.lat BETWEEN {b[1]} AND {b[3]} "
        f"AND vt.lon BETWEEN {b[0]} AND {b[2]} AND vt.lat BETWEEN {b[1]} AND {b[3]}"
    )


def _route_via(conn, vids: list[int]) -> list[int]:
    """pgr_dijkstraVia chained through vids. Returns ordered edge gids
    (skipping the -1 sentinel rows that mark via-point arrivals)."""
    quoted = _edges_sql().replace("'", "''")
    vid_arr = "ARRAY[" + ",".join(str(v) for v in vids) + "]"
    sql = (
        f"SELECT seq, edge FROM pgr_dijkstraVia('{quoted}', {vid_arr}, true) "
        "WHERE edge != -1 ORDER BY seq"
    )
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute("SET work_mem = '256MB'")
        cur.execute(sql)
        return [int(r[1]) for r in cur.fetchall()]


def _aggregate(conn, edge_gids: list[int], label: str) -> None:
    """Per-edge metadata and bucketed totals for a route's edges."""
    if not edge_gids:
        print(f"  [{label}] no edges"); return
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute(
            "SELECT w.length_m, w.highway, w.tracktype, w.surface, "
            "w.cost, w.reverse_cost, w.canopy_frac, "
            "w.forest_local, w.forest_wide, w.water_local, w.waterway_along_edge, "
            "w.view_dominance, w.local_relief, "
            "vs.elev_m AS elev_src, vt.elev_m AS elev_dst "
            "FROM ways w "
            "JOIN ways_vertices_pgr vs ON vs.id = w.source "
            "JOIN ways_vertices_pgr vt ON vt.id = w.target "
            "WHERE w.gid = ANY(%s)",
            (edge_gids,),
        )
        rows = cur.fetchall()

    total_len_m = 0.0
    total_cost  = 0.0
    climb_m     = 0.0
    by_track    : dict[str, float] = {}
    by_highway  : dict[str, float] = {}
    by_surface  : dict[str, float] = {}
    by_grade    : dict[str, float] = {}     # bucketed gradient
    forest_len_m  = 0.0    # length × forest_local
    canopy_len_m  = 0.0    # length × canopy_frac
    waterway_len_m = 0.0   # length × waterway_along_edge

    for (length, highway, tt, surface, cost, rcost, canopy,
         forest_local, forest_wide, water_local, waterway_along, view_dom,
         local_relief, e_src, e_dst) in rows:
        total_len_m += float(length)
        total_cost  += float(cost) if cost is not None else 0.0
        # Climb: only positive elevation deltas.
        if e_src is not None and e_dst is not None:
            d = float(e_dst) - float(e_src)
            if d > 0: climb_m += d
        # Tracktype bucket — "none" for non-track highways.
        tt_key = tt if tt else "(none)"
        by_track[tt_key] = by_track.get(tt_key, 0.0) + float(length)
        # Highway type
        h_key = highway if highway else "(none)"
        by_highway[h_key] = by_highway.get(h_key, 0.0) + float(length)
        # Surface
        s_key = surface if surface else "(untagged)"
        by_surface[s_key] = by_surface.get(s_key, 0.0) + float(length)
        # Gradient bucket (positive direction; downhills lumped with flat)
        if length > 0 and e_src is not None and e_dst is not None:
            grade = (float(e_dst) - float(e_src)) / float(length) * 100.0
        else:
            grade = 0.0
        if grade < 1:    g_key = "0..1% (flat)"
        elif grade < 3:  g_key = "1..3%"
        elif grade < 5:  g_key = "3..5%"
        elif grade < 8:  g_key = "5..8% (climby)"
        elif grade < 12: g_key = "8..12% (steep)"
        else:            g_key = "12%+ (brutal)"
        by_grade[g_key] = by_grade.get(g_key, 0.0) + float(length)
        # Scenic-signal length weighting
        forest_len_m   += float(length) * float(forest_local or 0)
        canopy_len_m   += float(length) * float(canopy or 0)
        waterway_len_m += float(length) * float(waterway_along or 0)

    print(f"\n=== {label} ===")
    print(f"  total length: {total_len_m/1000:7.2f} km")
    print(f"  total climb:  {climb_m:7.0f} m")
    print(f"  total cost:   {total_cost:.0f}")
    print(f"  avg cost/m:   {total_cost/total_len_m:.3f}")
    print(f"  forest-weighted km:   {forest_len_m/1000:6.2f}  ({100*forest_len_m/total_len_m:.1f}% × forest_local)")
    print(f"  canopy km (under tree cover): {canopy_len_m/1000:6.2f}")
    print(f"  waterway-along km:     {waterway_len_m/1000:6.2f}")
    print()
    print("  By tracktype:")
    for k, v in sorted(by_track.items(), key=lambda x: -x[1]):
        print(f"    {k:>15s}: {v/1000:6.2f} km ({100*v/total_len_m:5.1f}%)")
    print("  By highway:")
    for k, v in sorted(by_highway.items(), key=lambda x: -x[1])[:10]:
        print(f"    {k:>15s}: {v/1000:6.2f} km ({100*v/total_len_m:5.1f}%)")
    print("  By gradient bucket:")
    for k in ["0..1% (flat)", "1..3%", "3..5%", "5..8% (climby)",
             "8..12% (steep)", "12%+ (brutal)"]:
        v = by_grade.get(k, 0.0)
        if v > 0:
            print(f"    {k:>15s}: {v/1000:6.2f} km ({100*v/total_len_m:5.1f}%)")


def main() -> None:
    with psycopg.connect(config.PG_DSN) as conn:
        with conn.cursor() as cur:
            graz_v   = _snap(cur, *GRAZ)
            ha_v     = _snap(cur, *HAINBURG)
            mid_v    = _snap(cur, *LEITHA_MID)
            print(f"vids: graz={graz_v}, hainburg={ha_v}, leitha_mid={mid_v}")

        print("\n[1/2] Running FREE scenic route Graz->Hainburg...")
        free_edges = _route_via(conn, [graz_v, ha_v])
        print(f"     {len(free_edges)} edges")

        print("\n[2/2] Running FORCED route Graz->Leithagebirge->Hainburg...")
        forced_edges = _route_via(conn, [graz_v, mid_v, ha_v])
        print(f"     {len(forced_edges)} edges")

        _aggregate(conn, free_edges,   "FREE   Graz->Hainburg (current scenic optimum)")
        _aggregate(conn, forced_edges, "FORCED Graz->Leithagebirge mid->Hainburg")

        # How much extra cost the detour buys us
        with conn.cursor() as cur:
            cur.execute("SELECT SUM(length_m), SUM(cost) FROM ways WHERE gid = ANY(%s)", (free_edges,))
            free_len, free_cost = cur.fetchone()
            cur.execute("SELECT SUM(length_m), SUM(cost) FROM ways WHERE gid = ANY(%s)", (forced_edges,))
            forced_len, forced_cost = cur.fetchone()
        print(f"\n=== Delta (forced - free) ===")
        print(f"  length: +{(forced_len - free_len)/1000:.2f} km")
        print(f"  cost:   +{forced_cost - free_cost:.0f} ({100*(forced_cost - free_cost)/free_cost:+.1f}%)")


if __name__ == "__main__":
    main()
