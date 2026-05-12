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


_BATCH = 200_000


def _recompute_one(row: tuple) -> tuple[int, float, float]:
    """Return (gid, new_cost, new_reverse_cost) for a single edge row."""
    (gid, length_m, is_ferry, reverse_cost_in,
     highway, surface, tracktype, oneway, bicycle, cycleway,
     bicycle_road, access, curv_fwd, curv_rev,
     elev_src, elev_dst) = row

    if elev_src is None or elev_dst is None or length_m <= 0:
        grade_pct_fwd = 0.0
    else:
        grade_pct_fwd = (float(elev_dst) - float(elev_src)) / float(length_m) * 100.0

    fwd_factor = bike_edge_cost(
        highway=highway, surface=surface, tracktype=tracktype,
        bicycle=bicycle, cycleway=cycleway, access=access,
        bicycle_road=bicycle_road, is_ferry=is_ferry,
        grade_pct=grade_pct_fwd, curv=float(curv_fwd),
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
        )
        new_reverse_cost = (float(rev_factor) * float(length_m)
                            if rev_factor is not None
                            else -1.0)
    return (gid, new_cost, new_reverse_cost)


def recompute(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM ways")
        n_edges = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM ways_vertices_pgr WHERE elev_m IS NULL")
        n_null_elev = int(cur.fetchone()[0])
    print(f"[recompute] {n_edges:,} edges; "
          f"{n_null_elev:,} vertices have NULL elevation (will use grade=0 there)")

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE _new_cost (
                gid          bigint PRIMARY KEY,
                cost         double precision NOT NULL,
                reverse_cost double precision NOT NULL
            ) ON COMMIT DROP
        """)

    # Server-side cursor — `name` plus `itersize` streams without
    # buffering 240M rows in client memory.
    with conn.cursor(name="recompute_scan") as scur:
        scur.itersize = _BATCH
        scur.execute("""
            SELECT
                w.gid, w.length_m, w.is_ferry, w.reverse_cost,
                w.highway, w.surface, w.tracktype, w.oneway,
                w.bicycle, w.cycleway, w.bicycle_road, w.access,
                w.curv_fwd, w.curv_rev,
                vs.elev_m AS elev_src,
                vt.elev_m AS elev_dst
            FROM ways w
            JOIN ways_vertices_pgr vs ON vs.id = w.source
            JOIN ways_vertices_pgr vt ON vt.id = w.target
        """)

        processed = 0
        updated   = 0
        skipped_no_cost = 0
        while True:
            batch = scur.fetchmany(_BATCH)
            if not batch:
                break
            triples = []
            for row in batch:
                gid, new_cost, new_rev = _recompute_one(row)
                if new_cost is None:
                    skipped_no_cost += 1
                    continue
                triples.append((gid, new_cost, new_rev))

            if triples:
                with conn.cursor() as cur:
                    cur.execute("TRUNCATE _new_cost")
                    with cur.copy(
                        "COPY _new_cost (gid, cost, reverse_cost) FROM STDIN"
                    ) as cp:
                        for t in triples:
                            cp.write_row(t)
                    cur.execute("""
                        UPDATE ways w
                        SET cost = n.cost, reverse_cost = n.reverse_cost
                        FROM _new_cost n
                        WHERE w.gid = n.gid
                    """)
                    updated += cur.rowcount
                conn.commit()

            processed += len(batch)
            print(f"[recompute]   processed={processed:,} "
                  f"updated={updated:,} no_cost={skipped_no_cost:,}")

    print(f"[recompute] done. processed={processed:,} updated={updated:,} "
          f"no_cost={skipped_no_cost:,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    args = p.parse_args()
    with psycopg.connect(config.PG_DSN) as conn:
        recompute(conn)
