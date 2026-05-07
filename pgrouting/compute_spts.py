"""Chainless SPT preprocess: per-city multi-source Dijkstra from polygon.

Architecture:
  1. Load the full road graph into scipy CSR (one-shot, in-memory).
  2. For each anchor city, fetch the vertex set inside its polygon, run
     scipy multi-source Dijkstra with `min_only=True` and `limit=
     SPT_MAX_COST`. Save (node_global, parent_local, cost) as
     `<city_idx>.npz`.
  3. After all per-city SPTs are written, derive city_adjacency by
     checking which other cities' polygons each SPT reaches. Write
     `cities.json` and `city_graph.json`.

There is **no global K=1 partition step** — no Bellman-Ford waves, no
`visited` table, no `city_adjacency` SQL table. Each per-city Dijkstra
is independent. For routing, the API picks a chain of cities by
following `city_graph.json` (directed edges = "city A's SPT reaches
city B's polygon, with cost X"), and walks the gradient of the
farthest reachable city in the chain.

Memory: the CSR is loaded once and reused across all per-city
Dijkstras. At AT scale (~22M vertices, ~50M edges) it's ~1.3 GB. At
corridor scale (~110M vertices, ~250M edges) it's ~6 GB — tight; if
that becomes a problem we'd switch to per-city bbox-bounded edge
fetches instead of one global CSR.
"""
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import psycopg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


# Per-city Dijkstra cost cap. ~100 km of cycleway at the default. Larger
# values produce SPTs that reach further (fewer chain hops at routing
# time, more disk per city); smaller values are cheaper to compute and
# store. Override via SPT_MAX_COST env var.
SPT_MAX_COST = float(os.environ.get("SPT_MAX_COST", 100_000))


def _load_graph(conn: psycopg.Connection) -> tuple[np.ndarray, csr_matrix]:
    """Pull the full directed graph from postgres into a scipy CSR.

    Edges with `cost >= 0` contribute (source -> target). Edges with
    `reverse_cost >= 0` additionally contribute (target -> source). A
    negative direction-cost means that direction is blocked (oneway).

    Returned `node_global` is sorted ascending; `csr` is indexed by
    local positions = `searchsorted(node_global, vid)`.

    Memory: rows are streamed via a server-side cursor in chunks and
    converted to typed numpy arrays per chunk. fetchall() on the full
    ways table would create ~80 bytes/row × N tuples — ~8 GB at AT
    scale, ~40 GB at corridor — and OOM the container. Chunked fetch
    keeps the per-chunk Python tuple buffer tiny and lets the typed
    numpy chunks dominate (4 bytes/edge end-state).
    """
    print("[spts] loading edges into CSR (chunked fetch)...")
    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    fwd_chunks: list[np.ndarray] = []
    rev_chunks: list[np.ndarray] = []
    total_rows = 0
    CHUNK = 200_000

    with conn.cursor(name="edge_cur") as cur:
        cur.itersize = CHUNK
        cur.execute("""
            SELECT source, target, cost, reverse_cost
            FROM   ways
            WHERE  cost >= 0 OR reverse_cost >= 0
        """)
        while True:
            rows = cur.fetchmany(CHUNK)
            if not rows:
                break
            chunk = np.asarray(rows, dtype=np.float64)
            src_chunks.append(chunk[:, 0].astype(np.int64))
            dst_chunks.append(chunk[:, 1].astype(np.int64))
            fwd_chunks.append(chunk[:, 2].astype(np.float32))
            rev_chunks.append(chunk[:, 3].astype(np.float32))
            total_rows += len(rows)
            del rows, chunk
    print(f"[spts]   fetched {total_rows:,} ways rows in chunks")

    src   = np.concatenate(src_chunks);  src_chunks.clear()
    dst   = np.concatenate(dst_chunks);  dst_chunks.clear()
    fwd_c = np.concatenate(fwd_chunks);  fwd_chunks.clear()
    rev_c = np.concatenate(rev_chunks);  rev_chunks.clear()

    fwd_mask = fwd_c >= 0
    rev_mask = rev_c >= 0
    e_src  = np.concatenate([src[fwd_mask], dst[rev_mask]])
    e_dst  = np.concatenate([dst[fwd_mask], src[rev_mask]])
    e_cost = np.concatenate([fwd_c[fwd_mask], rev_c[rev_mask]])
    del src, dst, fwd_c, rev_c, fwd_mask, rev_mask

    node_global = np.unique(np.concatenate([e_src, e_dst]))
    n = len(node_global)
    src_local = np.searchsorted(node_global, e_src).astype(np.int32)
    dst_local = np.searchsorted(node_global, e_dst).astype(np.int32)
    del e_src, e_dst
    csr = csr_matrix((e_cost, (src_local, dst_local)), shape=(n, n), dtype=np.float32)
    print(f"[spts]   {n:,} unique vertices, {len(e_cost):,} directed edges, "
          f"CSR ~{(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes) // (1024*1024)} MB")
    return node_global, csr


def _bbox_vertices(
    conn: psycopg.Connection, lon: float, lat: float, radius_m: float = 1000.0,
) -> np.ndarray:
    """All vertex ids within `radius_m` meters of (lon, lat).

    Used as a fallback "synthetic polygon" for anchors without an OSM
    admin_level boundary — instead of seeding the SPT from the single
    nearest vertex (which often lands on a disconnected stub when the
    nearest is a parking-lot fragment or excluded private road), seed
    from every vertex in a small bbox around the place=town node so the
    multi-source Dijkstra has many entry points into the routable
    network. Robust to bad snap geometry the same way polygon seeding
    is for major cities.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT v.id FROM ways_vertices_pgr v
            WHERE ST_DWithin(
                v.the_geom::geography,
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                %s
            )
        """, (lon, lat, radius_m))
        return np.asarray(
            sorted(int(r[0]) for r in cur.fetchall()), dtype=np.int64,
        )


def _polygon_vertices(conn: psycopg.Connection, anchor_id: int) -> np.ndarray:
    """All vertex ids inside the given anchor's polygon (sorted)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT v.id
            FROM   ways_vertices_pgr v
            JOIN   anchors a ON a.id = %s
            WHERE  a.geom_boundary IS NOT NULL
              AND  ST_Contains(a.geom_boundary, v.the_geom)
        """, (anchor_id,))
        rows = cur.fetchall()
    return np.asarray(sorted(int(r[0]) for r in rows), dtype=np.int64)


def _per_city_spt(
    node_global: np.ndarray, csr: csr_matrix, sources_global: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Multi-source bounded Dijkstra from `sources_global` over the CSR.

    Returns `(keep_global, parent_local, cost)` where `parent_local`
    indexes into `keep_global` and unreachable rows are filtered out.
    """
    s_local_all = np.searchsorted(node_global, sources_global)
    in_range = s_local_all < len(node_global)
    matched = np.zeros_like(in_range)
    matched[in_range] = node_global[s_local_all[in_range]] == sources_global[in_range]
    s_local = s_local_all[matched].astype(np.int32)
    if len(s_local) == 0:
        return None

    cost, predecessors, _ = dijkstra(
        csgraph=csr, indices=s_local,
        return_predecessors=True, directed=True, limit=SPT_MAX_COST,
        min_only=True,
    )
    reachable = np.isfinite(cost)
    if not reachable.any():
        return None
    keep = np.flatnonzero(reachable)
    keep_global = node_global[keep]
    local_to_kept = np.full(len(node_global), -9999, dtype=np.int32)
    local_to_kept[keep] = np.arange(len(keep), dtype=np.int32)
    pred = predecessors[keep]
    parent_kept = np.where(pred >= 0, local_to_kept[pred], -9999).astype(np.int32)
    cost_kept = cost[keep].astype(np.float32)
    return keep_global.astype(np.int32), parent_kept, cost_kept


def _fetch_anchors(conn: psycopg.Connection) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, name, place, population, country,
                   ST_X(geom), ST_Y(geom),
                   snap_vertex_id, geom_boundary IS NOT NULL
            FROM   anchors
            WHERE  snap_vertex_id IS NOT NULL
            ORDER  BY id
        """)
        rows = cur.fetchall()
    return [{
        "anchor_id": int(r[0]), "name": r[1], "place": r[2],
        "population": r[3], "country": r[4],
        "lon": float(r[5]), "lat": float(r[6]),
        "snap_vertex_id": int(r[7]),
        "has_polygon": bool(r[8]),
    } for r in rows]


def _write_cities(out_dir: Path, anchors: list[dict]) -> None:
    cities = [{
        "city_idx": i,
        "anchor_id": a["anchor_id"],
        "name": a["name"], "place": a["place"],
        "population": a["population"], "country": a["country"],
        "lon": a["lon"], "lat": a["lat"],
        "snap_vertex_id": a["snap_vertex_id"],
        "has_polygon": a["has_polygon"],
    } for i, a in enumerate(anchors)]
    with open(out_dir / "cities.json", "w") as fh:
        json.dump(cities, fh, ensure_ascii=False, indent=1)
    print(f"[spts] wrote cities.json with {len(cities):,} entries")


def _build_city_graph(
    out_dir: Path, anchors: list[dict],
    polygon_sets: dict[int, np.ndarray],
) -> None:
    """Derive directed city_graph from per-city SPT overlaps.

    Edge (A, B, w): city A's SPT reaches city B's polygon at minimum
    cost w. Asymmetric in general (oneway intensives, mountain passes).
    Used by the API to plan city chains: from your current vertex,
    follow edges in this graph greedily toward the destination city.
    """
    spt_dir = out_dir / "spt"
    print("[spts] deriving city_graph from SPT overlaps...")
    edges = []
    for ci_a in range(len(anchors)):
        spt_path = spt_dir / f"{ci_a}.npz"
        if not spt_path.exists():
            continue
        data = np.load(spt_path)
        a_node_global = data["node_global"]
        a_cost = data["cost"]
        for ci_b, b_polys in polygon_sets.items():
            if ci_b == ci_a or len(b_polys) == 0:
                continue
            # Locate B's polygon vertices in A's reachable set.
            idx = np.searchsorted(a_node_global, b_polys)
            in_range = idx < len(a_node_global)
            matched = np.zeros_like(in_range)
            matched[in_range] = a_node_global[idx[in_range]] == b_polys[in_range]
            if not matched.any():
                continue
            hit_costs = a_cost[idx[matched]]
            edges.append((ci_a, ci_b, float(hit_costs.min())))
    cg = {
        "from_city": [e[0] for e in edges],
        "to_city":   [e[1] for e in edges],
        "weight":    [e[2] for e in edges],
    }
    with open(out_dir / "city_graph.json", "w") as fh:
        json.dump(cg, fh)
    print(f"[spts] wrote city_graph.json with {len(edges):,} directed edges")


def _priority_anchor_ids(
    conn: psycopg.Connection,
    polyline_points: list[tuple[float, float]] | None,
) -> list[int]:
    """Return anchor.id values sorted by priority order.

    If `polyline_points` is a list of (lon, lat) waypoints, anchors
    closer to the polyline through those points come first — pass a
    multi-waypoint polyline (e.g. Graz, Wien, Brno, Praha, Berlin,
    Hamburg, København) so anchors along the actual cycle-tour route
    finish ahead of the long tail. A simple straight line works too —
    just pass two points.

    If `polyline_points` is None, falls back to plain id order.
    """
    if not polyline_points:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM anchors WHERE snap_vertex_id IS NOT NULL ORDER BY id")
            return [int(r[0]) for r in cur.fetchall()]

    pts_sql = ", ".join(
        f"ST_MakePoint({lon}, {lat})" for lon, lat in polyline_points
    )
    # Bucket-based priority: every anchor within 30 km of the polyline
    # ties at bucket 0, sorted by along-line position so we sweep
    # from start to end. 30-80 km is bucket 1, 80-150 km bucket 2,
    # rest bucket 3. This pulls "in-corridor" towns to the front
    # regardless of their exact perpendicular distance to the line —
    # so a town 25 km off the segment (a real cycle-tour stopover) is
    # priority along with a hamlet directly under the line.
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH route AS (
                SELECT ST_SetSRID(ST_MakeLine(ARRAY[{pts_sql}]), 4326) AS line
            )
            SELECT a.id
            FROM   anchors a, route r
            WHERE  a.snap_vertex_id IS NOT NULL
            ORDER  BY
                CASE
                    WHEN ST_Distance(a.geom::geography, r.line::geography) <  30000 THEN 0
                    WHEN ST_Distance(a.geom::geography, r.line::geography) <  80000 THEN 1
                    WHEN ST_Distance(a.geom::geography, r.line::geography) < 150000 THEN 2
                    ELSE 3
                END ASC,
                -- within bucket, sweep along the line from start to end
                ST_LineLocatePoint(r.line, a.geom) ASC,
                a.id ASC
        """)
        return [int(r[0]) for r in cur.fetchall()]


def run(conn: psycopg.Connection, out_dir: Path) -> dict:
    """Top-level: load graph, per-city Dijkstra, write metadata + adjacency.

    Anchors are processed in priority order if SPT_PRIORITY_LINE is set
    in the env — value is "lon1,lat1,lon2,lat2" defining a great-circle
    line. Anchors closer to that line get processed first; the rest
    follow in id order. The npz file name is always `<id - 1>.npz`
    regardless of processing order, so resume + city_graph derivation
    work the same.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    spt_dir = out_dir / "spt"
    spt_dir.mkdir(parents=True, exist_ok=True)

    anchors = _fetch_anchors(conn)   # canonical id order
    print(f"[spts] {len(anchors):,} anchors to process")

    by_id = {a["anchor_id"]: a for a in anchors}

    line_env = os.environ.get("SPT_PRIORITY_LINE")
    polyline_points: list[tuple[float, float]] | None = None
    if line_env:
        try:
            parts = [float(x) for x in line_env.split(",")]
            if len(parts) >= 4 and len(parts) % 2 == 0:
                polyline_points = list(zip(parts[0::2], parts[1::2]))
                pretty = " -> ".join(f"({lo},{la})" for lo, la in polyline_points)
                print(f"[spts] priority polyline ({len(polyline_points)} waypoints): "
                      f"{pretty}  — anchors near this line first")
        except ValueError:
            pass

    ordered_ids = _priority_anchor_ids(conn, polyline_points)

    node_global, csr = _load_graph(conn)

    # Pre-fetch polygon vertex sets — keyed by ci = anchor.id - 1 so
    # the file naming and city_graph indexing stay stable regardless
    # of processing order. Anchors without an OSM admin boundary fall
    # back to a 1 km bbox of vertices around the place node, so the
    # multi-source Dijkstra still has many entry points into the
    # routable network and avoids the trapped-snap-vertex failure
    # mode (single-vertex seeds land on a parking-lot stub etc.).
    print("[spts] fetching polygon / bbox vertex sets...")
    polygon_sets: dict[int, np.ndarray] = {}
    bbox_count = 0
    for a in anchors:
        ci = a["anchor_id"] - 1
        if a["has_polygon"]:
            polygon_sets[ci] = _polygon_vertices(conn, a["anchor_id"])
        else:
            polygon_sets[ci] = _bbox_vertices(conn, a["lon"], a["lat"])
            bbox_count += 1
            # If even the bbox returned nothing (anchor in a black-hole
            # region of the graph), keep the single-vertex fallback so
            # the script doesn't crash on an empty source set.
            if len(polygon_sets[ci]) == 0:
                polygon_sets[ci] = np.asarray(
                    [a["snap_vertex_id"]], dtype=np.int64,
                )
    total_poly_v = sum(len(v) for v in polygon_sets.values())
    print(f"[spts]   {total_poly_v:,} total source vertices across "
          f"{sum(1 for a in anchors if a['has_polygon']):,} polygons + "
          f"{bbox_count:,} bbox-fallbacks")

    # Per-city Dijkstra in priority order.
    written = 0
    skipped = 0
    for processed_idx, aid in enumerate(ordered_ids):
        anchor = by_id[aid]
        ci = aid - 1
        out_path = spt_dir / f"{ci}.npz"
        if out_path.exists():
            skipped += 1
            continue
        sources = polygon_sets[ci]
        spt = _per_city_spt(node_global, csr, sources)
        if spt is None:
            skipped += 1
            continue
        node_arr, parent, cost = spt
        np.savez(out_path,
                 node_global=node_arr,
                 parent_local=parent,
                 cost=cost)
        written += 1
        if written % 25 == 0 or processed_idx + 1 == len(ordered_ids):
            print(f"[spts] dijkstra {processed_idx + 1:,}/{len(ordered_ids):,}  "
                  f"last: {anchor['name']} (sources={len(sources):,}, "
                  f"reached={len(node_arr):,})")
    print(f"[spts] wrote {written:,} per-city SPTs (skipped {skipped:,})")

    _write_cities(out_dir, anchors)
    _build_city_graph(out_dir, anchors, polygon_sets)

    return {"anchors": len(anchors), "spts_written": written}
