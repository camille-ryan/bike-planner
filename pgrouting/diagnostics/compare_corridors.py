"""One-off: compare Graz→Wien via Mur valley (Bruck) against Graz→Wien
via eastern Burgenland (Pinkafeld). For each leg, runs pgr_dijkstra
against the bbox-restricted graph (same trick as export_route_compare)
and decomposes the resulting route into V2 cost, length, climb, and
highway-class share.

Output is a single table comparing total stats for the two corridors.
"""
from __future__ import annotations
import psycopg

import config


# Waypoints (lon, lat).
GRAZ      = (15.4395, 47.0707)
BRUCK     = (15.2697, 47.4106)   # Bruck an der Mur — Mur valley corridor anchor
PINKAFELD = (16.1175, 47.3680)   # eastern Burgenland corridor anchor
WIEN      = (16.3725, 48.2082)

CORRIDORS = {
    "mur_valley":  [GRAZ, BRUCK,     WIEN],
    "burgenland":  [GRAZ, PINKAFELD, WIEN],
}

# Same bbox used elsewhere (Graz↔Wien with margin). Inlined into the
# pgr_dijkstra inner SQL to avoid OOMing the 6GB Postgres container.
BBOX = (14.5, 46.8, 17.0, 48.5)


def snap(cur, lon, lat) -> int:
    cur.execute(
        "SELECT id FROM ways_vertices_pgr "
        "ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326) "
        "LIMIT 1",
        (lon, lat),
    )
    return int(cur.fetchone()[0])


def route_leg(conn, start_vid: int, end_vid: int) -> dict:
    """Run pgr_dijkstra between two vids. Return aggregate stats:
    cost, length_m, climb_m (sum of positive elev_dst - elev_src in
    the traversal direction), and per-highway-class share of length.
    """
    b = BBOX
    edges_sql = (
        "SELECT w.gid AS id, w.source, w.target, "
        "       w.cost, w.reverse_cost "
        "FROM ways w "
        "JOIN ways_vertices_pgr vs ON vs.id = w.source "
        "JOIN ways_vertices_pgr vt ON vt.id = w.target "
        f"WHERE vs.lon BETWEEN {b[0]} AND {b[2]} "
        f"  AND vs.lat BETWEEN {b[1]} AND {b[3]} "
        f"  AND vt.lon BETWEEN {b[0]} AND {b[2]} "
        f"  AND vt.lat BETWEEN {b[1]} AND {b[3]}"
    )
    # NB: inline the pgr_dijkstra args. Passing the inner edges_sql as
    # a $1 parameter forced Postgres into a generic plan that OOMed
    # the backend under SIGKILL (psql with the literal form works
    # fine — the optimizer can see and tune around the inner SQL).
    # The values being inlined here are all from our control (ints +
    # floats), so SQL injection isn't a concern.
    sql_quoted = edges_sql.replace("'", "''")
    full_sql = (
        "SELECT p.path_seq, p.node, p.edge, p.cost, "
        "       w.source AS w_source, w.length_m, w.highway, "
        "       w.cycleway, w.bicycle_road, w.surface, "
        "       vs.elev_m AS elev_src, vt.elev_m AS elev_dst "
        f"FROM pgr_dijkstra('{sql_quoted}', {int(start_vid)}, "
        f"                  {int(end_vid)}, true) p "
        "JOIN ways w ON w.gid = p.edge "
        "JOIN ways_vertices_pgr vs ON vs.id = w.source "
        "JOIN ways_vertices_pgr vt ON vt.id = w.target "
        "WHERE p.edge != -1 "
        "ORDER BY p.path_seq"
    )
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute("SET work_mem = '256MB'")
        cur.execute(full_sql)
        rows = cur.fetchall()

    total_cost = 0.0
    total_len = 0.0
    total_climb = 0.0
    by_class: dict[str, float] = {}
    is_cycle_or_bikepri = 0.0
    for (path_seq, node, edge, edge_cost, w_source,
         length_m, highway, cycleway, bicycle_road, surface,
         e_src, e_dst) in rows:
        total_cost += float(edge_cost)
        total_len += float(length_m)
        # Direction handling — node is the start of the edge in
        # traversal direction.
        forward = (int(node) == int(w_source))
        if e_src is not None and e_dst is not None:
            d = (float(e_dst) - float(e_src) if forward
                 else float(e_src) - float(e_dst))
            if d > 0:
                total_climb += d
        cls = highway or "?"
        by_class[cls] = by_class.get(cls, 0.0) + float(length_m)
        if (cls == "cycleway") or (bicycle_road == "yes") or (
                cycleway and cycleway not in ("no", "none", "")):
            is_cycle_or_bikepri += float(length_m)

    return {
        "edges": len(rows),
        "cost": total_cost,
        "length_m": total_len,
        "climb_m": total_climb,
        "by_class": by_class,
        "cycle_or_bikepri_m": is_cycle_or_bikepri,
    }


def main() -> None:
    with psycopg.connect(config.PG_DSN) as conn:
        # Snap each waypoint once.
        with conn.cursor() as cur:
            vids = {}
            for label, coord in (
                ("graz", GRAZ), ("bruck", BRUCK),
                ("pinkafeld", PINKAFELD), ("wien", WIEN),
            ):
                vids[label] = snap(cur, *coord)
        print(f"[corridors] snapped vids: {vids}")

        results: dict[str, dict] = {}
        for name, waypoints in CORRIDORS.items():
            print(f"[corridors] routing {name}: {[w for w in waypoints]}",
                  flush=True)
            legs = []
            for i in range(len(waypoints) - 1):
                a, b = waypoints[i], waypoints[i + 1]
                a_label = {tuple(GRAZ): "graz", tuple(BRUCK): "bruck",
                           tuple(PINKAFELD): "pinkafeld",
                           tuple(WIEN): "wien"}[tuple(a)]
                b_label = {tuple(GRAZ): "graz", tuple(BRUCK): "bruck",
                           tuple(PINKAFELD): "pinkafeld",
                           tuple(WIEN): "wien"}[tuple(b)]
                leg = route_leg(conn, vids[a_label], vids[b_label])
                print(f"[corridors]   leg {a_label}→{b_label}: "
                      f"{leg['edges']} edges, {leg['length_m']/1000:.1f} km, "
                      f"{leg['climb_m']:.0f} m climb, cost={leg['cost']:.0f}",
                      flush=True)
                legs.append(leg)
            # Aggregate across legs.
            agg = {
                "edges": sum(l["edges"] for l in legs),
                "cost": sum(l["cost"] for l in legs),
                "length_m": sum(l["length_m"] for l in legs),
                "climb_m": sum(l["climb_m"] for l in legs),
                "cycle_or_bikepri_m": sum(
                    l["cycle_or_bikepri_m"] for l in legs),
                "by_class": {},
            }
            for l in legs:
                for cls, v in l["by_class"].items():
                    agg["by_class"][cls] = agg["by_class"].get(cls, 0.0) + v
            results[name] = agg

    # Print comparison.
    print("\n=== Corridor comparison ===")
    cols = ["mur_valley", "burgenland"]
    fmt = "{:<32s}  " + "  ".join(["{:>14}"] * len(cols))
    print(fmt.format("", *cols))
    print(fmt.format(
        "edges",
        *[f"{results[c]['edges']:,}" for c in cols]))
    print(fmt.format(
        "length (km)",
        *[f"{results[c]['length_m']/1000:.2f}" for c in cols]))
    print(fmt.format(
        "climb (m)",
        *[f"{results[c]['climb_m']:.0f}" for c in cols]))
    print(fmt.format(
        "V2 cost (sum)",
        *[f"{results[c]['cost']:.0f}" for c in cols]))
    print(fmt.format(
        "cost / km",
        *[f"{results[c]['cost']/(results[c]['length_m']/1000):.2f}"
          for c in cols]))
    print(fmt.format(
        "cycle/bikepri (km)",
        *[f"{results[c]['cycle_or_bikepri_m']/1000:.1f}" for c in cols]))
    print(fmt.format(
        "cycle/bikepri share",
        *[f"{100*results[c]['cycle_or_bikepri_m']/results[c]['length_m']:.0f}%"
          for c in cols]))

    # Highway-class share.
    all_classes = set()
    for c in cols:
        all_classes.update(results[c]["by_class"].keys())
    print("\n=== Highway-class share (km) ===")
    print(fmt.format("class", *cols))
    for cls in sorted(all_classes,
                      key=lambda k: -max(results[c]["by_class"].get(k, 0)
                                         for c in cols)):
        row = [f"{results[c]['by_class'].get(cls, 0)/1000:.1f}" for c in cols]
        if any(float(v) >= 0.5 for v in row):
            print(fmt.format(cls, *row))


if __name__ == "__main__":
    main()
