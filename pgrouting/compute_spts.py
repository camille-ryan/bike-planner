"""Chainless SPT preprocess: per-anchor multi-source Dijkstra over a
30 km geographic-radius subgraph, sliced from a global in-memory CSR.

Architecture (chainless v2.2, 2026-05-08):
  1. Load the global directed road graph from postgres into RAM as a
     scipy CSR + a coordinate array. ~5-10 min one-time cost.
  2. Build a scipy.spatial.cKDTree on the 3D unit-sphere embedding of
     the coordinates so euclidean ball queries map directly to
     great-circle radii (chord distance). ~3-5 min one-time cost.
  3. For each anchor (uniform 30 km radius — no city/ferry-touch
     differentiation any more; cross-ferry chain edges are added as
     a post-step):
       - kdtree.query_ball_point at 30 km → sub_idx.
       - Slice global CSR → sub_csr.
       - Compute is_frontier: vertices in sub_idx whose original
         out-degree exceeds their sub_csr out-degree (= they have
         road edges leading outside the 30 km region — true geographic
         frontier leaves, distinct from interior dead-end leaves).
       - Multi-source Dijkstra from 1 km bbox + snap_vertex_id seeds.
       - Save (node_global, parent_local, cost, is_frontier) as
         `<city_idx>.npz`.
  4. After all per-anchor SPTs:
       - Derive city_graph from pairwise SPT-overlap.
       - Augment city_graph with ferry chain edges (long ferry edges
         in the road graph connect anchor pairs whose 30 km SPTs
         can't span them on land).
       - Write `cities.json` and `city_graph.json`.
"""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import psycopg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


# Uniform geographic radius for all anchors. Cities and ferry-touching
# anchors used to get 100 km here, but that made paired-SPT crescents
# pathologically degenerate when consecutive anchors had wildly
# different reach. With uniform 30 km, ferry crossings are handled by
# explicitly augmenting city_graph with ferry chain edges below.
SPT_RADIUS_M = float(os.environ.get("SPT_RADIUS_M", 30_000))

# Minimum ferry-edge length to count as a "real" ferry for chain-edge
# augmentation. Most OSM `route=ferry` edges in central Europe are
# < 500 m river or lake crossings (~70% in our corridor). The 5 km
# threshold isolates sea / long-water crossings (Femern Belt,
# Trelleborg, Mols-Linien, etc.) from inland noise.
MIN_FERRY_LENGTH_M = float(os.environ.get("MIN_FERRY_LENGTH_M", 5_000))

# Earth radius for the chord-length conversion used by the cKDTree.
R_EARTH_M = 6_371_000.0

# Seed bbox radius (uniform across all anchors).
SEED_BBOX_RADIUS_M = 1000.0


# `_radius_for_anchor` and `_ferry_touching_anchor_ids` removed: every
# anchor now gets the uniform SPT_RADIUS_M (30 km). Ferries are handled
# in `_add_ferry_chain_edges` below as a city_graph augmentation, not
# by widening the ferry-touching anchors' SPT reach.


def _load_global_graph(
    conn: psycopg.Connection,
) -> tuple[np.ndarray, np.ndarray, csr_matrix]:
    """Stream all vertices + edges from postgres into RAM.

    Returns (node_global, coords_xyz, csr). Raw lon/lat are NOT kept
    globally (~900 MB) — for topology saves we recover lon/lat from
    coords_xyz on demand per anchor (cheap, only kept vertices).
    """
    print("[spts] loading global vertex set...", flush=True)
    t0 = time.time()
    id_chunks: list[np.ndarray] = []
    lon_chunks: list[np.ndarray] = []
    lat_chunks: list[np.ndarray] = []
    rows_seen = 0
    with conn.cursor(name="vert_cursor") as cur:
        cur.itersize = 1_000_000
        cur.execute(
            "SELECT id, ST_X(the_geom), ST_Y(the_geom) "
            "FROM ways_vertices_pgr ORDER BY id"
        )
        while True:
            rows = cur.fetchmany(1_000_000)
            if not rows:
                break
            arr = np.asarray(rows, dtype=np.float64)
            id_chunks.append(arr[:, 0].astype(np.int64))
            lon_chunks.append(arr[:, 1].astype(np.float32))
            lat_chunks.append(arr[:, 2].astype(np.float32))
            rows_seen += len(rows)
            print(f"[spts]   vertices: {rows_seen:,}", flush=True)
    node_global = np.concatenate(id_chunks); del id_chunks
    lon = np.concatenate(lon_chunks); del lon_chunks
    lat = np.concatenate(lat_chunks); del lat_chunks
    print(f"[spts]   {len(node_global):,} vertices in {time.time()-t0:.1f}s",
          flush=True)

    # 3D unit-sphere embedding so euclidean ball queries = great-circle chords.
    # We discard raw lon/lat after embedding — recovered on demand from
    # coords_xyz in the per-anchor topology save.
    print("[spts] computing 3D-spherical embedding...", flush=True)
    t1 = time.time()
    lat_r = np.radians(lat, dtype=np.float32)
    lon_r = np.radians(lon, dtype=np.float32)
    coslat = np.cos(lat_r)
    coords_xyz = np.empty((len(node_global), 3), dtype=np.float32)
    coords_xyz[:, 0] = coslat * np.cos(lon_r)
    coords_xyz[:, 1] = coslat * np.sin(lon_r)
    coords_xyz[:, 2] = np.sin(lat_r)
    del lat, lon, lat_r, lon_r, coslat
    print(f"[spts]   embedding done in {time.time()-t1:.1f}s "
          f"({coords_xyz.nbytes / 1e9:.2f} GB)", flush=True)

    print("[spts] loading global edge set...", flush=True)
    t2 = time.time()
    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    fwd_chunks: list[np.ndarray] = []
    rev_chunks: list[np.ndarray] = []
    edges_seen = 0
    with conn.cursor(name="edge_cursor") as cur:
        cur.itersize = 1_000_000
        cur.execute("""
            SELECT source, target, cost, reverse_cost
            FROM   ways
            WHERE  cost >= 0 OR reverse_cost >= 0
        """)
        while True:
            rows = cur.fetchmany(1_000_000)
            if not rows:
                break
            arr = np.asarray(rows, dtype=np.float64)
            src_chunks.append(arr[:, 0].astype(np.int64))
            dst_chunks.append(arr[:, 1].astype(np.int64))
            fwd_chunks.append(arr[:, 2].astype(np.float32))
            rev_chunks.append(arr[:, 3].astype(np.float32))
            edges_seen += len(rows)
            print(f"[spts]   edges: {edges_seen:,}", flush=True)
    src = np.concatenate(src_chunks); del src_chunks
    dst = np.concatenate(dst_chunks); del dst_chunks
    fwd = np.concatenate(fwd_chunks); del fwd_chunks
    rev = np.concatenate(rev_chunks); del rev_chunks
    print(f"[spts]   {len(src):,} edge rows in {time.time()-t2:.1f}s, "
          f"building CSR...", flush=True)

    t3 = time.time()
    src_local = np.searchsorted(node_global, src).astype(np.int32)
    dst_local = np.searchsorted(node_global, dst).astype(np.int32)
    del src, dst

    fwd_mask = fwd >= 0
    rev_mask = rev >= 0
    e_src  = np.concatenate([src_local[fwd_mask], dst_local[rev_mask]])
    e_dst  = np.concatenate([dst_local[fwd_mask], src_local[rev_mask]])
    e_cost = np.concatenate([fwd[fwd_mask], rev[rev_mask]])
    del fwd, rev, src_local, dst_local

    n = len(node_global)
    csr = csr_matrix(
        (e_cost, (e_src, e_dst)),
        shape=(n, n), dtype=np.float32,
    )
    del e_src, e_dst, e_cost
    print(f"[spts]   global CSR built ({csr.nnz:,} directed edges) "
          f"in {time.time()-t3:.1f}s "
          f"(data {csr.data.nbytes / 1e9:.2f} GB + "
          f"indices {csr.indices.nbytes / 1e9:.2f} GB + "
          f"indptr {csr.indptr.nbytes / 1e9:.2f} GB)", flush=True)
    return node_global, coords_xyz, csr


def _build_kdtree(coords_xyz: np.ndarray) -> cKDTree:
    """Build a cKDTree on the 3D-spherical embedding of all vertices.

    leafsize=64 (vs default 16) gives a smaller in-memory tree by
    keeping more points per leaf (~4× fewer internal nodes). Marginally
    slower per-query but saves significant RAM at 114 M points scale.
    Default balanced_tree=True / compact_nodes=True for a compact tree.
    """
    print("[spts] building cKDTree on 3D-spherical coords...", flush=True)
    t0 = time.time()
    tree = cKDTree(coords_xyz, leafsize=64)
    print(f"[spts]   cKDTree built in {time.time()-t0:.1f}s", flush=True)
    return tree


def _anchor_xyz(lon: float, lat: float) -> np.ndarray:
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    coslat = math.cos(lat_r)
    return np.array([
        coslat * math.cos(lon_r),
        coslat * math.sin(lon_r),
        math.sin(lat_r),
    ], dtype=np.float32)


def _chord_for_radius_m(radius_m: float) -> float:
    """Great-circle radius in meters → chord length on the unit sphere."""
    return 2.0 * math.sin(radius_m / R_EARTH_M / 2.0)


def _query_subgraph_idx(
    kdtree: cKDTree, anchor_xyz: np.ndarray, geo_radius_m: float,
) -> np.ndarray:
    """Sorted local indices of vertices within geo_radius_m of anchor."""
    chord = _chord_for_radius_m(geo_radius_m)
    raw = kdtree.query_ball_point(anchor_xyz, r=chord)
    if not raw:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.asarray(raw, dtype=np.int64))


def _per_anchor_spt(
    sub_node_global: np.ndarray, sub_csr: csr_matrix,
    sources_global: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Multi-source Dijkstra over the per-anchor sliced subgraph."""
    s_local_all = np.searchsorted(sub_node_global, sources_global)
    in_range = s_local_all < len(sub_node_global)
    matched = np.zeros_like(in_range)
    matched[in_range] = (
        sub_node_global[s_local_all[in_range]] == sources_global[in_range]
    )
    s_local = s_local_all[matched].astype(np.int32)
    if len(s_local) == 0:
        return None

    cost, predecessors, _ = dijkstra(
        csgraph=sub_csr, indices=s_local,
        return_predecessors=True, directed=True,
        min_only=True,
    )
    reachable = np.isfinite(cost)
    if not reachable.any():
        return None
    keep = np.flatnonzero(reachable)
    keep_global = sub_node_global[keep]
    local_to_kept = np.full(len(sub_node_global), -9999, dtype=np.int32)
    local_to_kept[keep] = np.arange(len(keep), dtype=np.int32)
    pred = predecessors[keep]
    parent_kept = np.where(pred >= 0, local_to_kept[pred], -9999).astype(np.int32)
    cost_kept = cost[keep].astype(np.float32)
    # Return `keep` (positions in sub_node_global of reachable vertices)
    # so callers can index per-subgraph arrays (is_frontier, edge_cost
    # for the slice CSR, etc.) onto the SPT's reachable subset.
    return keep_global.astype(np.int32), parent_kept, cost_kept, keep


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
    print(f"[spts] wrote cities.json with {len(cities):,} entries", flush=True)


def _build_city_graph(
    out_dir: Path, anchors: list[dict],
    conn: psycopg.Connection | None = None,
) -> None:
    """Derive directed city_graph from per-anchor SPT overlaps.

    With uniform 30 km radius, no two SPTs span the open water of a
    long ferry crossing. To keep ferries usable as chain edges, after
    the overlap-based edges we also augment with synthetic ferry chain
    edges via `_add_ferry_chain_edges`.
    """
    spt_dir = out_dir / "spt"
    print("[spts] deriving city_graph from SPT overlaps...", flush=True)
    polygon_vertex_sets: dict[int, np.ndarray] = {}
    for ci in range(len(anchors)):
        path = spt_dir / f"{ci}.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            ng = data["node_global"]
            cost = data["cost"]
            polygon_vertex_sets[ci] = ng[cost == 0]
    print(f"[spts]   recovered seeds for {len(polygon_vertex_sets):,} cities "
          f"from npzs", flush=True)

    edges = []
    for ci_a in polygon_vertex_sets:
        with np.load(spt_dir / f"{ci_a}.npz") as data:
            a_node_global = data["node_global"]
            a_cost = data["cost"]
        for ci_b, b_polys in polygon_vertex_sets.items():
            if ci_b == ci_a or len(b_polys) == 0:
                continue
            idx = np.searchsorted(a_node_global, b_polys)
            in_range = idx < len(a_node_global)
            matched = np.zeros_like(in_range)
            matched[in_range] = a_node_global[idx[in_range]] == b_polys[in_range]
            if not matched.any():
                continue
            edges.append((ci_a, ci_b, float(a_cost[idx[matched]].min())))
    print(f"[spts]   {len(edges):,} overlap-based directed edges", flush=True)

    if conn is not None:
        ferry_edges = _add_ferry_chain_edges(conn, anchors, out_dir)
        # Avoid duplicates (overlap and ferry could agree for short ferries
        # where 30 km radii happen to span the crossing — unlikely with
        # MIN_FERRY_LENGTH_M = 5000 but defensive).
        existing = {(a, b) for a, b, _ in edges}
        for a, b, w in ferry_edges:
            if (a, b) not in existing:
                edges.append((a, b, w))
                existing.add((a, b))
        print(f"[spts]   +{len(ferry_edges):,} ferry-augmented edges", flush=True)

    cg = {
        "from_city": [e[0] for e in edges],
        "to_city":   [e[1] for e in edges],
        "weight":    [e[2] for e in edges],
    }
    with open(out_dir / "city_graph.json", "w") as fh:
        json.dump(cg, fh)
    print(f"[spts] wrote city_graph.json with {len(edges):,} directed edges",
          flush=True)


def _add_ferry_chain_edges(
    conn: psycopg.Connection, anchors: list[dict], out_dir: Path,
) -> list[tuple[int, int, float]]:
    """Return synthetic chain edges crossing long ferries.

    For each ferry edge of length ≥ MIN_FERRY_LENGTH_M, identify the
    chain anchors that "own" each endpoint (= anchor whose SPT covers
    the endpoint with minimum cost). If those owners differ, add
    chain edges in both directions: (owner_A, owner_B, w) and
    (owner_B, owner_A, w_reverse). The walker's ferry-fake paired SPT
    handles the actual crossing at routing time.
    """
    spt_dir = out_dir / "spt"
    print(f"[spts] scanning for long (≥ {int(MIN_FERRY_LENGTH_M):,} m) ferry edges...",
          flush=True)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT source, target, cost, reverse_cost, length_m
            FROM   ways
            WHERE  is_ferry AND length_m >= %s
              AND  (cost >= 0 OR reverse_cost >= 0)
        """, (MIN_FERRY_LENGTH_M,))
        ferries = [(int(r[0]), int(r[1]), float(r[2]), float(r[3]), float(r[4]))
                   for r in cur.fetchall()]
    if not ferries:
        print(f"[spts]   no long ferries found", flush=True)
        return []
    print(f"[spts]   {len(ferries):,} long ferry edges", flush=True)

    endpoints = set()
    for s, t, *_ in ferries:
        endpoints.add(s); endpoints.add(t)

    # Find owning anchor (= min-cost SPT) for each endpoint.
    owners: dict[int, tuple[int, float]] = {}
    for ci in range(len(anchors)):
        path = spt_dir / f"{ci}.npz"
        if not path.exists():
            continue
        with np.load(path) as d:
            ng = np.asarray(d["node_global"])
            cost = np.asarray(d["cost"])
        for vt in endpoints:
            pos = int(np.searchsorted(ng, vt))
            if pos < len(ng) and int(ng[pos]) == vt:
                cv = float(cost[pos])
                if vt not in owners or cv < owners[vt][1]:
                    owners[vt] = (ci, cv)

    out: list[tuple[int, int, float]] = []
    for s, t, c_st, c_ts, _length in ferries:
        a = owners.get(s, (None,))[0]
        b = owners.get(t, (None,))[0]
        if a is None or b is None or a == b:
            continue
        # chain_graph edge weight ≈ ferry-edge cost itself. (Routing's
        # ferry-fake paired SPT walks the actual subgraph, so this weight
        # is just for chain-Dijkstra ranking — the absolute value matters
        # less than relative ordering.)
        if c_st >= 0:
            out.append((a, b, c_st))
        if c_ts >= 0:
            out.append((b, a, c_ts))
    return out


def _priority_anchor_ids(
    conn: psycopg.Connection,
    polyline_points: list[tuple[float, float]] | None,
) -> list[int]:
    """Return anchor.id values sorted by priority — anchors close to
    the polyline first."""
    if not polyline_points:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM anchors WHERE snap_vertex_id IS NOT NULL "
                "ORDER BY id"
            )
            return [int(r[0]) for r in cur.fetchall()]

    pts_sql = ", ".join(
        f"ST_MakePoint({lon}, {lat})" for lon, lat in polyline_points
    )
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
                ST_LineLocatePoint(r.line, a.geom) ASC,
                a.id ASC
        """)
        return [int(r[0]) for r in cur.fetchall()]


def run(conn: psycopg.Connection, out_dir: Path) -> dict:
    """Top-level: load globals, build kdtree, per-anchor slice + SPT,
    derive adjacency."""
    out_dir.mkdir(parents=True, exist_ok=True)
    spt_dir = out_dir / "spt"
    spt_dir.mkdir(parents=True, exist_ok=True)

    anchors = _fetch_anchors(conn)
    print(f"[spts] {len(anchors):,} anchors to process "
          f"(uniform {int(SPT_RADIUS_M):,} m radius)", flush=True)
    by_id = {a["anchor_id"]: a for a in anchors}

    line_env = os.environ.get("SPT_PRIORITY_LINE")
    polyline_points: list[tuple[float, float]] | None = None
    if line_env:
        try:
            parts = [float(x) for x in line_env.split(",")]
            if len(parts) >= 4 and len(parts) % 2 == 0:
                polyline_points = list(zip(parts[0::2], parts[1::2]))
                pretty = " -> ".join(f"({lo},{la})" for lo, la in polyline_points)
                print(f"[spts] priority polyline ({len(polyline_points)} "
                      f"waypoints): {pretty}  — anchors near this line first",
                      flush=True)
        except ValueError:
            pass
    ordered_ids = _priority_anchor_ids(conn, polyline_points)

    # Skip the heavy globals load if every requested SPT already exists.
    pending = [aid for aid in ordered_ids
               if not (spt_dir / f"{aid - 1}.npz").exists()]
    if not pending:
        print(f"[spts] all {len(ordered_ids):,} SPTs already on disk; "
              f"skipping global graph load", flush=True)
        node_global = csr = kdtree = None
    else:
        print(f"[spts] {len(pending):,}/{len(ordered_ids):,} SPTs pending; "
              f"loading global graph...", flush=True)
        node_global, coords_xyz, csr = _load_global_graph(conn)
        kdtree = _build_kdtree(coords_xyz)
        # cKDTree retains a reference to coords_xyz via kdtree.data, so
        # dropping our local doesn't free the array, but it does drop
        # one redundant reference. The topology-output step reads from
        # kdtree.data, not coords_xyz directly, so the local can go.
        import gc
        del coords_xyz
        gc.collect()
        # Per-anchor `orig_out_deg` is computed on demand from
        # csr.indptr[sub_idx + 1] - csr.indptr[sub_idx]. We don't keep
        # a global out-degree array (would be another ~456 MB).

    # Phase C: topology output directory (one shared file per anchor;
    # profile-specific data goes under spt_dir/<anchor>.npz).
    topology_dir = out_dir.parent.parent / "road_topology"
    topology_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    bbox_chord = _chord_for_radius_m(SEED_BBOX_RADIUS_M)
    t_loop = time.time()
    for processed_idx, aid in enumerate(ordered_ids):
        anchor = by_id[aid]
        ci = aid - 1
        out_path = spt_dir / f"{ci}.npz"
        if out_path.exists():
            skipped += 1
            continue

        t_a = time.time()
        anc_xyz = _anchor_xyz(anchor["lon"], anchor["lat"])
        sub_idx = _query_subgraph_idx(kdtree, anc_xyz, SPT_RADIUS_M)
        if len(sub_idx) == 0:
            skipped += 1
            continue

        sub_node_global = node_global[sub_idx]
        sub_csr = csr[sub_idx, :][:, sub_idx]

        # is_frontier per subgraph vertex (precise: orig out-degree > slice).
        sub_out_deg = np.diff(sub_csr.indptr)
        orig_out_deg_local = csr.indptr[sub_idx + 1] - csr.indptr[sub_idx]
        is_frontier_subgraph = orig_out_deg_local > sub_out_deg

        # Seeds: 1 km bbox + snap_vertex_id (uniform across all anchors).
        bbox_local = kdtree.query_ball_point(anc_xyz, r=bbox_chord)
        if bbox_local:
            bbox_global = node_global[np.asarray(bbox_local, dtype=np.int64)]
        else:
            bbox_global = np.empty(0, dtype=np.int64)
        sources = np.unique(np.concatenate([
            bbox_global,
            np.asarray([anchor["snap_vertex_id"]], dtype=np.int64),
        ]))

        spt = _per_anchor_spt(sub_node_global, sub_csr, sources)
        if spt is None:
            skipped += 1
            continue
        node_arr, parent, cost, keep_in_subgraph = spt
        # is_frontier indexed onto the SPT's reachable subset.
        is_frontier_spt = is_frontier_subgraph[keep_in_subgraph].astype(np.bool_)

        # Phase A: per-profile SPT npz with is_frontier byproduct.
        # Phase C: edge_cost (sub_csr's data, only for kept vertices) is
        # also saved here so offline rerouting can run a local Dijkstra.
        # The kept-subset's CSR slice is sub_csr[keep_in_subgraph][:, keep_in_subgraph].
        kept_csr = sub_csr[keep_in_subgraph, :][:, keep_in_subgraph]
        np.savez(out_path,
                 node_global=node_arr,
                 parent_local=parent,
                 cost=cost,
                 is_frontier=is_frontier_spt,
                 # Edge data for offline rerouting (per-profile, since
                 # bike-cost weights would differ per profile).
                 edge_indptr=kept_csr.indptr.astype(np.int32),
                 edge_indices=kept_csr.indices.astype(np.int32),
                 edge_cost=kept_csr.data.astype(np.float32))

        # Phase C: shared topology file (lon/lat per kept vertex). One
        # per anchor, reused across profiles. Topology has no cost data.
        # Skip if it already exists from a prior run.
        # Lon/lat are recovered from coords_xyz (the 3D unit-sphere
        # embedding) rather than kept globally as separate arrays —
        # saves ~900 MB of host memory during the cKDTree build.
        topology_path = topology_dir / f"{ci}.npz"
        if not topology_path.exists():
            kept_global_idx = sub_idx[keep_in_subgraph]
            # Read from kdtree.data, which is the cKDTree's owned copy
            # of the 3D unit-sphere coordinates. No extra global array
            # held by run() besides the kdtree itself.
            xyz_kept = kdtree.data[kept_global_idx]
            kept_lon = np.degrees(np.arctan2(xyz_kept[:, 1], xyz_kept[:, 0])).astype(np.float32)
            kept_lat = np.degrees(np.arcsin(np.clip(xyz_kept[:, 2], -1.0, 1.0))).astype(np.float32)
            np.savez(
                topology_path,
                node_global=node_arr,
                lon=kept_lon,
                lat=kept_lat,
            )

        written += 1
        dt = time.time() - t_a
        n_frontier = int(is_frontier_spt.sum())
        print(f"[spts] {processed_idx + 1:,}/{len(ordered_ids):,}  "
              f"{anchor['name']}  (sub={len(sub_idx):,}, srcs={len(sources):,}, "
              f"reached={len(node_arr):,}, frontier={n_frontier:,}, "
              f"max_cost={cost.max():.0f}, {dt:.2f}s)", flush=True)
    elapsed = time.time() - t_loop
    print(f"[spts] wrote {written:,} per-anchor SPTs "
          f"(skipped {skipped:,}) in {elapsed:.1f}s", flush=True)

    _write_cities(out_dir, anchors)

    total_done = sum(
        1 for a in anchors
        if (spt_dir / f"{a['anchor_id'] - 1}.npz").exists()
    )
    if total_done == len(anchors):
        _build_city_graph(out_dir, anchors, conn=conn)
    else:
        print(f"[spts] {total_done:,}/{len(anchors):,} npz on disk — "
              f"city_graph deferred until full corridor completes",
              flush=True)

    return {"anchors": len(anchors), "spts_written": written}
