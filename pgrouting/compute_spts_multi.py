"""Multi-profile per-anchor SPT compute — DESIGN SKETCH (not wired yet).

Replacement for `compute_spts.py` that loads the global edge graph from
postgres ONCE into an in-memory CSR with all 5 profile-weight columns
attached, then runs (anchors × profiles) dijkstras against the shared
structure.

Win vs old pipeline: pgr_dijkstra materialized `edges_sql` per call —
30 SPTs paid the 22M-edge load tax 30×. Here we pay it once
(~1.2 GB resident) then dispatch dijkstras cheaply via
scipy.sparse.csgraph.dijkstra (cython, multi-source via min_only=True,
radius-capped via limit=).

═══════════════════════════════════════════════════════════════════════
SPT direction: BACKWARD ("cost from this point to the city center").
═══════════════════════════════════════════════════════════════════════

The CSR we build is the TRANSPOSED original graph. Concretely, for
each ways row (s, t, cost_<p>, reverse_cost_<p>):

    Original graph: edge (s,t,cost_<p>) and edge (t,s,reverse_cost_<p>)
    Our CSR (transposed): edge (t,s,cost_<p>) and edge (s,t,reverse_cost_<p>)

Running dijkstra from anchor `a` in this CSR computes the shortest
path a→...→x in the transposed graph, which equals the shortest path
x→...→a in the original graph.

Output semantics per (anchor, profile) SPT:
  cost[x]         = shortest cost to travel FROM x TO the anchor
  parent_local[x] = NEXT HOP from x TOWARD the anchor (in original)

This makes the per-anchor npz directly usable as "drop the user's
start point into anchor a's SPT, walk parent pointers, you arrive at
a." That is the entry path into the paired-SPT structure.

Per-anchor output schema is otherwise unchanged from compute_spts.py
so `build_paired_spts.py` / `build_paired_corridor.py` keep consuming
it — but those scripts may have assumed forward semantics; flagged in
the TODOs below.

Open questions marked TODO. Not runnable as-is.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psycopg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import config

# 5 V3 production profiles. Each must have cost_<name> and
# reverse_cost_<name> columns in ways (created by schema.sql + populated
# by recompute_cost.py).
PROFILES = ("direct", "vineyard_lover", "forest_lover", "views", "water")

# Default per-anchor radius. Same default as compute_spts.py (30 km).
SPT_RADIUS_M_DEFAULT = 30_000.0

# Minimum ferry length to treat as a "real" sea/long-water crossing
# warranting a synthetic chain edge. Matches V1's MIN_FERRY_LENGTH_M.
MIN_FERRY_LENGTH_M = 5_000.0

CACHE_DIR_DEFAULT = config.DATA_DIR / "spt_cache"
OUTPUT_ROOT_DEFAULT = config.SPT_DIR


# ─────────────────────────────────────────────────────────────────────
# Graph loader: postgres ways → CSR (cached on disk)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class MultiProfileGraph:
    """Shared CSR (TRANSPOSED original graph) + per-profile weight columns.

    Each ways row (s, t, cost_<p>, reverse_cost_<p>) contributes two
    directed CSR entries:
      (csr_src=t, csr_dst=s, weight=cost_<p>)          — original s→t
      (csr_src=s, csr_dst=t, weight=reverse_cost_<p>)  — original t→s

    dijkstra from anchor a in this CSR yields cost[x] = cost of x→a in
    the original graph (i.e., a BACKWARD SPT). See module docstring.

    weights is (n_edges, n_profiles) float32. edge_gid maps each CSR
    edge back to postgres ways.gid (for polyline lookup later in
    build_paired_corridor).

    vid is the sorted dense → global vertex id map. Global → dense
    lookups use np.searchsorted(vid, …) — we deliberately do NOT keep
    a dict (~400 MB at 8 M nodes). lon/lat are per-dense-vertex.
    """

    indptr: np.ndarray         # int64, (n_nodes + 1,)
    indices: np.ndarray        # int32, (n_edges,)        — dense target idx
    weights: np.ndarray        # float32, (n_edges, n_profiles)
    edge_gid: np.ndarray       # int64, (n_edges,)        — ways.gid
    vid: np.ndarray            # int64, (n_nodes,) sorted — dense → global
    lon: np.ndarray            # float64, (n_nodes,)
    lat: np.ndarray            # float64, (n_nodes,)
    profiles: tuple[str, ...]  # column order in `weights`

    def slice_profile(self, profile: str) -> csr_matrix:
        """Cheap: returns a scipy csr_matrix backed by the shared
        indptr/indices and the column-sliced weights — no copy of the
        topology arrays."""
        col = self.profiles.index(profile)
        return csr_matrix(
            (self.weights[:, col], self.indices, self.indptr),
            shape=(len(self.vid), len(self.vid)),
        )

    def vid_to_dense(self, global_vids: np.ndarray | list[int] | int) -> np.ndarray:
        """Map global vids → dense indices via searchsorted. Returns
        an int32 array (or scalar if input is scalar). Caller is
        responsible for verifying matches if vids may be missing."""
        arr = np.asarray(global_vids, dtype=np.int64)
        return np.searchsorted(self.vid, arr).astype(np.int32)


_PROFILE_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
# "Unreachable" stand-in for NULL / negative costs. Must be float32 inf
# (not a finite sentinel like 1e15) — under spatial-slice dijkstra with
# no cost cap, finite sentinels get summed into long paths and inflate
# real cost values into nonsensical ranges. With inf, sentinel-traversed
# paths stay at inf and are naturally excluded by the `isfinite` mask.
_SENTINEL_COST = np.float32(np.inf)


def load_graph(
    dsn: str,
    cache_path: Path = CACHE_DIR_DEFAULT / "graph_multi.npz",
    profiles: tuple[str, ...] = PROFILES,
    force: bool = False,
) -> MultiProfileGraph:
    """Postgres ways → MultiProfileGraph, cached on disk.

    First call: ~5–10 min (query + CSR build + npz write).
    Subsequent calls: ~10 s (npz mmap load).
    """
    if cache_path.exists() and not force:
        return _load_cache(cache_path, profiles)

    # SQL-injection guard: profile names are interpolated into the
    # SELECT clause. Reject anything other than `[a-z_][a-z0-9_]*`.
    for p in profiles:
        if not _PROFILE_NAME_RE.match(p):
            raise ValueError(f"unsafe profile name: {p!r}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[load_graph] building cache at {cache_path}", flush=True)

    with psycopg.connect(dsn) as conn:
        vid, lon, lat = _stream_vertices(conn)
        gid_raw, src_raw, dst_raw, cost_raw = _stream_edges(conn, profiles)

    # Map global vids → dense indices (vid is sorted ASC from postgres).
    print("[load_graph] mapping vids → dense indices...", flush=True)
    t = time.time()
    src_dense = np.searchsorted(vid, src_raw).astype(np.int32)
    dst_dense = np.searchsorted(vid, dst_raw).astype(np.int32)
    del src_raw, dst_raw
    print(f"[load_graph]   done in {time.time()-t:.1f}s", flush=True)

    # Split cost_raw into per-profile fwd/rev. Layout is
    # [cost_p0, rev_p0, cost_p1, rev_p1, …]; copy so we can free cost_raw.
    fwd_cost = cost_raw[:, 0::2].copy()
    rev_cost = cost_raw[:, 1::2].copy()
    del cost_raw

    # Replace NULL (→ NaN from psycopg) and negative sentinels with the
    # "unreachable" float32 sentinel. dijkstra will see these as edges
    # with cost ~1e15 and simply never relax through them — cheaper than
    # building per-profile topology with edges actually omitted.
    _sanitize_costs_inplace(fwd_cost)
    _sanitize_costs_inplace(rev_cost)

    # Build TRANSPOSED CSR entries. For each ways row (s, t):
    #   Entry A: (csr_src=t, csr_dst=s, weight=cost_p)         [orig s→t]
    #   Entry B: (csr_src=s, csr_dst=t, weight=reverse_cost_p) [orig t→s]
    print("[load_graph] building transposed CSR entries...", flush=True)
    t = time.time()
    e_src = np.concatenate([dst_dense, src_dense])
    e_dst = np.concatenate([src_dense, dst_dense])
    e_weights = np.concatenate([fwd_cost, rev_cost], axis=0).astype(np.float32, copy=False)
    e_gid = np.concatenate([gid_raw, gid_raw])
    del src_dense, dst_dense, fwd_cost, rev_cost, gid_raw
    n_edges = len(e_src)
    print(f"[load_graph]   {n_edges:,} directed CSR entries "
          f"in {time.time()-t:.1f}s", flush=True)

    # Sort by csr_src to build CSR layout (stable for reproducibility).
    print("[load_graph] sorting by csr_src for CSR layout...", flush=True)
    t = time.time()
    order = np.argsort(e_src, kind="stable")
    indices = e_dst[order].astype(np.int32)
    weights = e_weights[order]
    edge_gid = e_gid[order]
    sorted_src = e_src[order]
    del e_src, e_dst, e_weights, e_gid, order

    n_nodes = len(vid)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.add.at(indptr[1:], sorted_src, 1)
    np.cumsum(indptr, out=indptr)
    del sorted_src
    print(f"[load_graph]   CSR layout built in {time.time()-t:.1f}s", flush=True)

    # Persist cache. np.savez (uncompressed) so future loads can mmap.
    print(f"[load_graph] saving cache to {cache_path}...", flush=True)
    t = time.time()
    np.savez(
        cache_path,
        indptr=indptr,
        indices=indices,
        weights=weights,
        edge_gid=edge_gid,
        vid=vid,
        lon=lon,
        lat=lat,
        profiles=np.array(list(profiles)),
    )
    print(f"[load_graph]   cache saved ({cache_path.stat().st_size/1e9:.2f} GB) "
          f"in {time.time()-t:.1f}s", flush=True)

    return MultiProfileGraph(
        indptr=indptr, indices=indices, weights=weights,
        edge_gid=edge_gid, vid=vid, lon=lon, lat=lat, profiles=profiles,
    )


def _stream_vertices(
    conn: psycopg.Connection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream ways_vertices_pgr into (vid, lon, lat) sorted by id."""
    print("[load_graph] loading vertices...", flush=True)
    t = time.time()
    id_chunks: list[np.ndarray] = []
    lon_chunks: list[np.ndarray] = []
    lat_chunks: list[np.ndarray] = []
    rows_seen = 0
    with conn.cursor(name="vert_cursor") as cur:
        cur.itersize = 1_000_000
        # NO ORDER BY: after the 2026-06 spatial CLUSTER reorder of
        # ways_vertices_pgr, ORDER BY id forces a pkey-driven scan
        # that fetches heap rows in scattered (post-spatial) order →
        # random IO on every row, ~1h per million. Without ORDER BY,
        # postgres does a fast seq scan in physical heap order. The
        # caller doesn't depend on order — id_chunks gets concatenated
        # then later mapped vid→dense via argsort.
        cur.execute(
            "SELECT id, ST_X(the_geom), ST_Y(the_geom) "
            "FROM ways_vertices_pgr"
        )
        while True:
            rows = cur.fetchmany(1_000_000)
            if not rows:
                break
            arr = np.asarray(rows, dtype=np.float64)
            id_chunks.append(arr[:, 0].astype(np.int64))
            lon_chunks.append(arr[:, 1].astype(np.float64))
            lat_chunks.append(arr[:, 2].astype(np.float64))
            rows_seen += len(rows)
            print(f"[load_graph]   vertices: {rows_seen:,}", flush=True)
    vid = np.concatenate(id_chunks)
    lon = np.concatenate(lon_chunks)
    lat = np.concatenate(lat_chunks)
    # Sort in Python by vid (was previously ORDER BY id in the SELECT;
    # moved to here so the postgres scan can be sequential in physical
    # heap order — much faster than random-IO id-pkey scan post-CLUSTER).
    # Downstream `searchsorted` assumes vid is sorted ASC.
    t_sort = time.time()
    order = np.argsort(vid, kind="stable")
    vid = vid[order]
    lon = lon[order]
    lat = lat[order]
    del order
    print(f"[load_graph]   sorted vid in Python in {time.time()-t_sort:.1f}s",
          flush=True)
    print(f"[load_graph]   {len(vid):,} vertices in {time.time()-t:.1f}s",
          flush=True)
    return vid, lon, lat


def _stream_edges(
    conn: psycopg.Connection, profiles: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream ways into (gid, src, dst, cost_raw).

    cost_raw is (n_rows, 2*n_profiles) float32 in column order
    [cost_p0, reverse_cost_p0, cost_p1, reverse_cost_p1, …].
    NULL → NaN (sanitised later); negatives kept as-is (sanitised later).
    """
    cost_select = ", ".join(
        f"cost_{p}, reverse_cost_{p}" for p in profiles
    )
    print(f"[load_graph] loading edges ({len(profiles)} profiles × 2 cost cols)...",
          flush=True)
    t = time.time()
    gid_chunks: list[np.ndarray] = []
    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    cost_chunks: list[np.ndarray] = []
    rows_seen = 0
    next_report = 5_000_000
    with conn.cursor(name="edge_cursor") as cur:
        cur.itersize = 500_000
        cur.execute(f"""
            SELECT gid, source, target, {cost_select}
            FROM ways
            WHERE source IS NOT NULL AND target IS NOT NULL
        """)
        while True:
            rows = cur.fetchmany(500_000)
            if not rows:
                break
            arr = np.asarray(rows, dtype=np.float64)
            gid_chunks.append(arr[:, 0].astype(np.int64))
            src_chunks.append(arr[:, 1].astype(np.int64))
            dst_chunks.append(arr[:, 2].astype(np.int64))
            cost_chunks.append(arr[:, 3:].astype(np.float32))
            rows_seen += len(rows)
            if rows_seen >= next_report:
                print(f"[load_graph]   ways rows: {rows_seen:,}", flush=True)
                next_report += 5_000_000
    gid = np.concatenate(gid_chunks)
    src = np.concatenate(src_chunks)
    dst = np.concatenate(dst_chunks)
    cost = np.concatenate(cost_chunks)
    print(f"[load_graph]   {len(gid):,} ways rows in {time.time()-t:.1f}s",
          flush=True)
    return gid, src, dst, cost


def _sanitize_costs_inplace(w: np.ndarray) -> None:
    """Replace NaN and negative values (V1 'no edge' sentinel) with
    `_SENTINEL_COST`. dijkstra will see these as expensive edges and
    never relax through them — cheaper than per-profile topology."""
    bad = ~np.isfinite(w) | (w < 0.0)
    np.copyto(w, _SENTINEL_COST, where=bad)


def _load_cache(path: Path, profiles: tuple[str, ...]) -> MultiProfileGraph:
    """Memory-map the CSR cache. Cache must contain all requested
    profiles (as a subset, in any order — slice_profile resolves the
    column by name). Force-rebuild if any requested profile is missing.
    """
    print(f"[load_graph] mmap cache {path}...", flush=True)
    t = time.time()
    z = np.load(path, mmap_mode="r")
    cached_profiles = tuple(str(p) for p in z["profiles"].tolist())
    missing = [p for p in profiles if p not in cached_profiles]
    if missing:
        raise RuntimeError(
            f"cache at {path} is missing requested profile(s) {missing}; "
            f"cache has {cached_profiles}. Rerun with --force-cache to rebuild."
        )
    # Expose the cache's full profile tuple — slice_profile uses
    # profiles.index(name) to find the weight column. Requesting only a
    # subset is OK; we just don't use the other columns.
    g = MultiProfileGraph(
        indptr=z["indptr"],
        indices=z["indices"],
        weights=z["weights"],
        edge_gid=z["edge_gid"],
        vid=z["vid"],
        lon=z["lon"],
        lat=z["lat"],
        profiles=cached_profiles,
    )
    print(f"[load_graph]   mmap done in {time.time()-t:.1f}s "
          f"({len(g.vid):,} nodes, {len(g.indices):,} edges, "
          f"cache profiles: {cached_profiles})",
          flush=True)
    return g


# ─────────────────────────────────────────────────────────────────────
# SPT compute — one anchor, one profile
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SPTResult:
    """Backward SPT for one anchor under one profile.

    Schema matches compute_spts.py's npz format. SEMANTIC change:
    `cost` is the cost FROM each vertex TO the anchor (not from
    anchor outward), and `parent_local` is the next hop TOWARD the
    anchor. build_paired_spts.py may need adjustment if it assumed
    forward semantics — see module TODO list.
    """
    node_global: np.ndarray   # int32  — global vids of reachable verts
    parent_local: np.ndarray  # int32  — next hop toward anchor (kept-set idx); <0 = anchor seed
    cost: np.ndarray          # float32 — cost from this vertex TO the anchor
    is_frontier: np.ndarray   # bool   — true geographic frontier leaves
    # CSR slice of kept subgraph (for offline rerouting inside corridor).
    # Edge direction matches the parent walk: u → parent[u].
    edge_indptr: np.ndarray
    edge_indices: np.ndarray
    edge_cost: np.ndarray


def compute_spt(
    graph: MultiProfileGraph,
    spatial_global_vids: np.ndarray,
    seed_global_vids: list[int] | np.ndarray,
    profile: str,
    max_cost_m: float | None = None,
) -> SPTResult:
    """One anchor's BACKWARD SPT under one profile, over a SPATIAL slice.

    `spatial_global_vids` is the pre-fetched set of vertices within the
    anchor's geographic radius (typically 30 km, fetched by caller via
    postgres ST_DWithin). The CSR is sliced to this subgraph and
    dijkstra runs WITHOUT a cost cap — the spatial slice itself bounds
    the work. This matches V1's compute_spts.py semantic: subgraph is
    geographic, costs reflect the profile.

    Cost-bounded dijkstra (V2 first attempt) is the wrong semantic for
    scenic profiles: their per-edge penalties make `cost-as-distance`
    arithmetic incoherent, so a 30 km *cost* cap collapses to a near-
    zero geographic radius. V1 avoided this by capping the topology.

    `max_cost_m` is optional and applied on top of the spatial cap —
    leave at None for unlimited (i.e., dijkstra runs to completion on
    the slice).

    Multi-source: `seed_global_vids` is the 1 km seed bbox around the
    anchor. scipy's `indices=` + `min_only=True` treats them as one
    virtual super-source.
    """
    # ── 1. Spatial subgraph slice (dense indices of spatial vids).
    spatial_dense = np.searchsorted(
        graph.vid, np.asarray(spatial_global_vids, dtype=np.int64),
    ).astype(np.int32)
    # spatial_dense is sorted ASC (since spatial_global_vids comes
    # sorted by id from postgres). Required for searchsorted-based
    # seed-local lookup below.

    csr = graph.slice_profile(profile)
    sub_csr = csr[spatial_dense, :][:, spatial_dense]
    sub_n = len(spatial_dense)

    # ── 2. Map seeds → sub-local indices, validating membership.
    seeds_global = np.asarray(seed_global_vids, dtype=np.int64)
    seeds_dense_pos = np.searchsorted(graph.vid, seeds_global)
    in_g = seeds_dense_pos < len(graph.vid)
    seeds_dense = np.where(in_g, seeds_dense_pos, 0)
    in_graph = in_g & (graph.vid[seeds_dense] == seeds_global)
    if not in_graph.any():
        raise ValueError(
            f"no valid seed vertices found among {len(seeds_global)} requested"
        )

    sub_local_pos = np.searchsorted(spatial_dense, seeds_dense[in_graph])
    in_sub = sub_local_pos < sub_n
    sub_local_safe = np.where(in_sub, sub_local_pos, 0)
    in_spatial = in_sub & (spatial_dense[sub_local_safe] == seeds_dense[in_graph])
    seeds_local = sub_local_pos[in_spatial].astype(np.int32)
    if len(seeds_local) == 0:
        raise ValueError(
            "anchor seeds are not inside the spatial subgraph — "
            "did the spatial radius shrink below the seed bbox radius?"
        )

    # ── 3. Dijkstra on the spatial slice. No cost cap (or generous one).
    limit = np.inf if max_cost_m is None else max_cost_m
    cost_local, predecessors, _ = dijkstra(
        sub_csr, directed=True, indices=seeds_local,
        return_predecessors=True, min_only=True, limit=limit,
    )

    reachable = np.isfinite(cost_local)
    kept_sub = np.flatnonzero(reachable).astype(np.int32)
    n_kept = len(kept_sub)
    if n_kept == 0:
        return SPTResult(
            node_global=np.empty(0, dtype=np.int32),
            parent_local=np.empty(0, dtype=np.int32),
            cost=np.empty(0, dtype=np.float32),
            is_frontier=np.empty(0, dtype=bool),
            edge_indptr=np.zeros(1, dtype=np.int32),
            edge_indices=np.empty(0, dtype=np.int32),
            edge_cost=np.empty(0, dtype=np.float32),
        )

    # ── 4. Remap sub-local → kept-local for parent_local.
    sub_to_kept = np.full(sub_n, -1, dtype=np.int32)
    sub_to_kept[kept_sub] = np.arange(n_kept, dtype=np.int32)
    pred = predecessors[kept_sub]
    pred_safe = np.where(pred >= 0, pred, 0)
    parent_local = np.where(
        pred >= 0,
        sub_to_kept[pred_safe],
        np.int32(-1),
    ).astype(np.int32)

    # ── 5. is_frontier (V1 definition): vertex has at least one edge
    # leaving the spatial slice. Computed per-spatial-vertex from
    # out-degree mismatch, then projected to the kept set.
    sub_out_deg = np.diff(sub_csr.indptr)
    orig_out_deg = graph.indptr[spatial_dense + 1] - graph.indptr[spatial_dense]
    is_frontier_sub = orig_out_deg > sub_out_deg
    is_frontier = is_frontier_sub[kept_sub]

    # ── 6. Edge subgraph slice over kept vertices.
    kept_csr = sub_csr[kept_sub, :][:, kept_sub]

    return SPTResult(
        node_global=graph.vid[spatial_dense[kept_sub]].astype(np.int32),
        parent_local=parent_local,
        cost=cost_local[kept_sub].astype(np.float32),
        is_frontier=is_frontier,
        edge_indptr=kept_csr.indptr.astype(np.int32),
        edge_indices=kept_csr.indices.astype(np.int32),
        edge_cost=kept_csr.data.astype(np.float32),
    )


# ─────────────────────────────────────────────────────────────────────
# cities.json + city_graph.json
# ─────────────────────────────────────────────────────────────────────

def _write_cities(out_dir: Path, anchors: list[dict]) -> None:
    """Shared across profiles — topology doesn't depend on weights."""
    cities = [{
        "city_idx": a["city_idx"],
        "anchor_id": a.get("anchor_id", a["city_idx"]),
        "name": a["name"],
        "place": a.get("place"),
        "population": a.get("population"),
        "country": a.get("country"),
        "lon": a["lon"],
        "lat": a["lat"],
        "snap_vertex_id": a["snap_vertex_id"],
        "has_polygon": a.get("has_polygon", False),
    } for a in anchors]
    path = out_dir / "cities.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(cities, fh, ensure_ascii=False, indent=1)
    print(f"[cities] wrote {path} ({len(cities):,} entries)", flush=True)


def _build_city_graph_for_profile(
    profile_dir: Path,
    anchors: list[dict],
    profile: str,
    conn: psycopg.Connection | None = None,
) -> None:
    """Derive directed city_graph from SPT overlaps for one profile.

    Edge (ci_from, ci_to, w): w = cost(ci_from → ci_to) under the
    original graph for this profile. To produce that under BACKWARD
    SPT semantics, we look up ci_from's seed vertices inside ci_to's
    backward SPT — its cost values there equal cost(seed → anchor) =
    cost(ci_from → ci_to). This is the inner/outer loop flip from V1
    documented in the audit notes below. The chain consumer
    (build_paired_spts._chain_dijkstra) is direction-agnostic and
    needs no changes.

    Bidirectional with asymmetric weights falls out for free: every
    ordered pair (from, to) is visited and emits its own edge.
    """
    spt_dir = profile_dir / "spt"
    print(f"[city_graph:{profile}] deriving from SPT overlaps...", flush=True)

    seeds: dict[int, np.ndarray] = {}
    for a in anchors:
        ci = a["city_idx"]
        if (spt_dir / f"{ci}.npz").exists():
            seeds[ci] = np.asarray(a["seed_vids"], dtype=np.int64)
    print(f"[city_graph:{profile}]   {len(seeds):,} cities with SPTs on disk",
          flush=True)

    # OUTER = ci_to (load to's SPT once); INNER = ci_from (cheap lookup).
    # That's the I/O-efficient layout: each SPT npz is opened exactly
    # once per profile.
    edges: list[tuple[int, int, float]] = []
    for ci_to in seeds:
        with np.load(spt_dir / f"{ci_to}.npz") as data:
            to_ng = np.asarray(data["node_global"])
            to_cost = np.asarray(data["cost"])
        for ci_from, from_seeds in seeds.items():
            if ci_from == ci_to:
                continue
            idx = np.searchsorted(to_ng, from_seeds)
            in_range = idx < len(to_ng)
            matched = np.zeros_like(in_range, dtype=bool)
            matched[in_range] = to_ng[idx[in_range]] == from_seeds[in_range]
            if not matched.any():
                continue
            edges.append((
                ci_from, ci_to,
                float(to_cost[idx[matched]].min()),
            ))
    print(f"[city_graph:{profile}]   {len(edges):,} overlap-based directed edges",
          flush=True)

    if conn is not None:
        ferry_edges = _add_ferry_chain_edges(conn, anchors, spt_dir, profile)
        existing = {(a, b) for a, b, _ in edges}
        added = 0
        for a, b, w in ferry_edges:
            if (a, b) not in existing:
                edges.append((a, b, w))
                existing.add((a, b))
                added += 1
        print(f"[city_graph:{profile}]   +{added:,} ferry-augmented edges",
              flush=True)

    cg = {
        "from_city": [e[0] for e in edges],
        "to_city":   [e[1] for e in edges],
        "weight":    [e[2] for e in edges],
    }
    out_path = profile_dir / "city_graph.json"
    with open(out_path, "w") as fh:
        json.dump(cg, fh)
    print(f"[city_graph:{profile}] wrote {out_path} ({len(edges):,} edges)",
          flush=True)


def _add_ferry_chain_edges(
    conn: psycopg.Connection,
    anchors: list[dict],
    spt_dir: Path,
    profile: str,
) -> list[tuple[int, int, float]]:
    """Ferry chain edges — ported from V1 with the profile cost column.

    Ferry costs are intrinsic (per-edge cost_<profile> / reverse_cost_<profile>
    from postgres), so direction-correct under either SPT semantic.
    Owner of each endpoint = anchor whose SPT contains it with min cost
    — geometrically equivalent to V1 for symmetric road edges.
    """
    if not _PROFILE_NAME_RE.match(profile):
        raise ValueError(f"unsafe profile name {profile!r}")
    cost_col = f"cost_{profile}"
    rev_col = f"reverse_cost_{profile}"
    print(f"[ferry:{profile}] scanning long ferries "
          f"(≥{int(MIN_FERRY_LENGTH_M):,} m)...", flush=True)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT source, target,
                   COALESCE({cost_col}, 1e15),
                   COALESCE({rev_col}, 1e15),
                   length_m
            FROM ways
            WHERE is_ferry AND length_m >= %s
        """, (MIN_FERRY_LENGTH_M,))
        ferries = [
            (int(r[0]), int(r[1]), float(r[2]), float(r[3]), float(r[4]))
            for r in cur.fetchall()
        ]
    if not ferries:
        print(f"[ferry:{profile}]   no long ferries found", flush=True)
        return []
    print(f"[ferry:{profile}]   {len(ferries):,} long ferries", flush=True)

    endpoints: set[int] = set()
    for s, t, *_ in ferries:
        endpoints.add(s); endpoints.add(t)

    # Min-cost owner per endpoint, across all anchors' SPTs.
    owners: dict[int, tuple[int, float]] = {}
    for a in anchors:
        ci = a["city_idx"]
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
        # Profile-specific sentinel: 1e15 means "this profile considers
        # the ferry effectively unusable in this direction" — skip.
        if c_st < 1e14:
            out.append((a, b, c_st))
        if c_ts < 1e14:
            out.append((b, a, c_ts))
    return out


# ─────────────────────────────────────────────────────────────────────
# Anchor fetch (with 1 km bbox seed vids)
# ─────────────────────────────────────────────────────────────────────

def _fetch_spatial_vids(
    conn: psycopg.Connection, lon: float, lat: float, radius_m: float,
) -> np.ndarray:
    """Sorted int64 array of ways_vertices_pgr ids within `radius_m`
    of (lon, lat). Approximated as a SQUARE bbox in degrees instead
    of a true geographic circle.

    Why bbox: the original `ST_DWithin(::geography, ::geography, m)`
    forced postgres into Parallel Seq Scan (~286 sec/anchor at 4-country
    scale) because the geometry-typed GIST can't serve geography
    distance. A bbox query using the geometry GIST + the covering
    `ways_vertices_pgr_geom_id_idx (INCLUDE id)` runs in ~1 sec.

    Tradeoff: the bbox is a circumscribed square (~27% more vertices
    than the circle). Those extras get loaded into the SPT slice but
    Dijkstra's natural reach limit skips any that aren't connected
    within cost — no correctness loss, just slightly more SPT compute.

    Latitude correction: 1° lat ≈ 111 km globally; 1° lon ≈
    111·cos(lat) km. For radius_m=30000 at 50°N, that's 0.27° lat ×
    0.42° lon. We use the looser of the two (lon) for a square bbox
    so we never miss vertices."""
    deg_per_km = 1.0 / 111.0
    # Loosen for latitude (lon spacing shrinks toward poles).
    lat_cos = max(0.1, np.cos(np.radians(lat)))
    half_deg = (radius_m / 1000.0) * deg_per_km / lat_cos
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id FROM ways_vertices_pgr
            WHERE the_geom && ST_Expand(
                ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s
            )
            ORDER BY id
        """, (float(lon), float(lat), float(half_deg)))
        return np.fromiter(
            (int(r[0]) for r in cur.fetchall()), dtype=np.int64,
        )


def _fetch_anchors_with_seeds(
    conn: psycopg.Connection,
    seed_radius_m: float = 1000.0,
    name_filter: str | None = None,
) -> list[dict]:
    """Fetch anchors + derive seed vids (1 km bbox ∪ snap_vertex_id).

    `city_idx` follows V1's convention: anchors.id - 1 (so existing
    paired/corridor npz file naming stays consistent). `name_filter`
    is an ILIKE substring; if None, returns all anchors with a
    `snap_vertex_id`.
    """
    with conn.cursor() as cur:
        if name_filter is None:
            cur.execute("""
                SELECT id, name, place, population, country,
                       ST_X(geom), ST_Y(geom), snap_vertex_id,
                       geom_boundary IS NOT NULL
                FROM anchors
                WHERE snap_vertex_id IS NOT NULL
                ORDER BY id
            """)
        else:
            cur.execute("""
                SELECT id, name, place, population, country,
                       ST_X(geom), ST_Y(geom), snap_vertex_id,
                       geom_boundary IS NOT NULL
                FROM anchors
                WHERE snap_vertex_id IS NOT NULL
                  AND name ILIKE %s
                ORDER BY id
            """, (f"%{name_filter}%",))
        rows = cur.fetchall()

    # 2026-06: re-instated 1 km bbox seed-set using the fast
    # `the_geom && ST_Expand(...)` pattern (~50 ms per anchor with
    # the covering `ways_vertices_pgr_geom_id_idx (INCLUDE id)`).
    # Earlier I'd dropped this because the old ST_DWithin geography
    # form took 200 sec/anchor — but with the bbox pattern it's
    # cheap. It IS a correctness requirement: ~3-4% of snap_vertex_id
    # values land on tiny disconnected components in the bikeable
    # subgraph (islands, piers, isolated footbridges). The 1 km bbox
    # union picks up neighboring main-component vertices so Dijkstra
    # propagates correctly. Observed broken-without-it: Bregenz,
    # Gänserndorf, Krems an der Donau, Leonding.
    anchors: list[dict] = []
    for r in rows:
        anchor_id, name, place, pop, country = r[0], r[1], r[2], r[3], r[4]
        lon, lat, snap_vid, has_poly = r[5], r[6], r[7], r[8]
        nearby = _fetch_spatial_vids(conn, float(lon), float(lat), seed_radius_m)
        seed_vids = np.unique(np.concatenate([
            np.array([int(snap_vid)], dtype=np.int64), nearby
        ]))
        anchors.append({
            "city_idx": int(anchor_id) - 1,   # V1 convention
            "anchor_id": int(anchor_id),
            "name": name,
            "place": place,
            "population": pop,
            "country": country,
            "lon": float(lon),
            "lat": float(lat),
            "snap_vertex_id": int(snap_vid),
            "has_polygon": bool(has_poly),
            "seed_vids": seed_vids,
        })
    return anchors


# ─────────────────────────────────────────────────────────────────────
# Driver: load once, dispatch all (anchor × profile) SPTs
# ─────────────────────────────────────────────────────────────────────

def run(
    profiles: tuple[str, ...] = PROFILES,
    anchor_filter: str | None = None,
    max_radius_m: float = SPT_RADIUS_M_DEFAULT,
    force: bool = False,
    cache_path: Path | None = None,
    force_cache: bool = False,
    dsn: str | None = None,
    output_root: Path = OUTPUT_ROOT_DEFAULT,
) -> None:
    """Programmatic entry point. Called by both __main__ and
    `main.cmd_spts_multi`.

    With `anchor_filter` unset: full run — every anchor × every profile,
    plus cities.json + per-profile city_graph.json writes.

    With `anchor_filter` set (ILIKE substring): only matching anchors
    are computed, and cities.json + city_graph.json writes are SKIPPED
    (those need the full anchor set). Useful for smoke testing and
    targeted re-runs.

    `max_radius_m` is the GEOGRAPHIC radius for the per-anchor spatial
    slice — same semantic as V1's SPT_RADIUS_M. The slice is fetched
    via postgres ST_DWithin, then dijkstra runs on the slice without a
    cost cap (so scenic-profile penalties don't collapse reach).
    """
    if dsn is None:
        dsn = config.PG_DSN
    if cache_path is None:
        cache_path = CACHE_DIR_DEFAULT / "graph_multi.npz"

    print(f"[spts-multi] profiles: {profiles}", flush=True)
    print(f"[spts-multi] anchor filter: {anchor_filter or '(all)'}", flush=True)
    print(f"[spts-multi] max_radius_m: {max_radius_m:,.0f} (geographic)", flush=True)
    print(f"[spts-multi] cache_path: {cache_path}", flush=True)

    with psycopg.connect(dsn) as conn:
        anchors = _fetch_anchors_with_seeds(conn, name_filter=anchor_filter)

    if not anchors:
        raise SystemExit(
            f"no anchors matched filter {anchor_filter!r}; "
            "check spelling or drop --anchor for a full run."
        )

    print(f"[spts-multi] fetched {len(anchors):,} anchors:", flush=True)
    for a in anchors:
        print(f"[spts-multi]   ci={a['city_idx']:>4} {a['name']:<28} "
              f"({a['lon']:>7.4f}, {a['lat']:>6.4f}) "
              f"snap={a['snap_vertex_id']:>9}  seeds={len(a['seed_vids']):>3,}",
              flush=True)

    if force:
        for profile in profiles:
            spt_dir = output_root / profile / "spt"
            for a in anchors:
                p_npz = spt_dir / f"{a['city_idx']}.npz"
                if p_npz.exists():
                    print(f"[spts-multi]   removing {p_npz}", flush=True)
                    p_npz.unlink()

    graph = load_graph(
        dsn, cache_path=cache_path, profiles=profiles, force=force_cache,
    )

    # Spatial subgraph fetch INTERLEAVED with SPT compute — fetch each
    # anchor's vids on-demand, compute SPT, free vids before moving to
    # next anchor. The old "fetch all 3,212 then compute" pattern OOMed
    # python at 9.9 GB because 3,212 × ~1.3M int64 vids = ~32 GB. By
    # interleaving, peak memory stays bounded at one anchor's worth.

    for profile in profiles:
        profile_dir = output_root / profile
        spt_out = profile_dir / "spt"
        spt_out.mkdir(parents=True, exist_ok=True)
        print(f"[spts-multi:{profile}] computing {len(anchors):,} anchors "
              f"(spatial fetch interleaved, max_radius_m={max_radius_m:,.0f})",
              flush=True)
        # Open one connection for the whole loop's spatial fetches.
        with psycopg.connect(dsn) as fetch_conn:
            for a in anchors:
                out_path = spt_out / f"{a['city_idx']}.npz"
                if out_path.exists():
                    print(f"[spts-multi:{profile}]   {a['city_idx']:>4} "
                          f"{a['name']:<28} (skip: exists)", flush=True)
                    continue
                t = time.time()
                spatial_vids = _fetch_spatial_vids(
                    fetch_conn, a["lon"], a["lat"], max_radius_m,
                )
                t_fetch = time.time() - t
                spt = compute_spt(
                    graph,
                    spatial_vids,
                    a["seed_vids"],
                    profile,
                )
                np.savez(
                    out_path,
                    node_global=spt.node_global,
                    parent_local=spt.parent_local,
                    cost=spt.cost,
                    is_frontier=spt.is_frontier,
                    edge_indptr=spt.edge_indptr,
                    edge_indices=spt.edge_indices,
                    edge_cost=spt.edge_cost,
                )
                max_c = float(spt.cost.max()) if len(spt.cost) else 0.0
                print(f"[spts-multi:{profile}]   {a['city_idx']:>4} "
                      f"{a['name']:<28} sub={len(spatial_vids):>6,} "
                      f"reached={len(spt.node_global):>7,} "
                      f"frontier={int(spt.is_frontier.sum()):>5,} "
                      f"max_cost={max_c:>9.0f}  fetch={t_fetch:.2f}s "
                      f"total={time.time()-t:.2f}s", flush=True)
                # Free spatial_vids before next anchor — keeps peak
                # memory bounded.
                del spatial_vids, spt

        if anchor_filter is None:
            _write_cities(profile_dir, anchors)
            with psycopg.connect(dsn) as conn:
                _build_city_graph_for_profile(
                    profile_dir, anchors, profile, conn=conn,
                )
        else:
            print(f"[spts-multi:{profile}] (single-anchor mode) skipping "
                  f"cities/city_graph writes — they need the full anchor set",
                  flush=True)

    print("[spts-multi] DONE", flush=True)


# ─────────────────────────────────────────────────────────────────────
# Audit notes (2026-05-17) — what changes vs V1 `compute_spts.py`:
# ─────────────────────────────────────────────────────────────────────
#
# (1) `_build_pair` in build_paired_spts.py: WORKS UNCHANGED. The
#     paired_(A, B) construction is direction-agnostic at the
#     algorithmic level — kept = f_only ∪ b_frontier, walk A.parent
#     toward A's center, terminate at b_frontier (just inside B).
#     Only cost-value interpretation flips (cost-to-anchor instead of
#     cost-from-anchor); same magnitudes for symmetric edges.
#
# (2) `build_paired_corridor.py`: WORKS UNCHANGED. No edge-geometry
#     lookups against ways.gid, no one-way handling concerns. The
#     trunk blob is purely vertex-level (vid, succ, lat, lon).
#
# (3) `_build_city_graph` (compute_spts.py lines 324-339): LOOP
#     DIRECTION MUST FLIP. V1's loop "B's polys in A's forward SPT,
#     min cost" gives cost(A→B). If we keep that loop verbatim under
#     backward SPT, we silently get cost(B→A) for edge (A,B,w) —
#     wrong direction. Fix: iterate (from, to), look up from's polys
#     in to's BACKWARD SPT. Same edge labels emerge, weights are
#     cost(from→to) as intended, chain consumer (`_chain_dijkstra`)
#     unchanged.
#
#     V1's city_graph IS already bidirectional with asymmetric
#     weights (the outer/inner loop visits every ordered pair).
#     Ferry chain edges (compute_spts.py lines 423-426) emit both
#     (a, b, c_st) and (b, a, c_ts) explicitly. No bidirectionality
#     fix needed — just the weight derivation flip.
#
# (4) `is_frontier` (compute_spts.py lines 545-548): port V1's formula
#     `orig_out_deg > sub_out_deg` as-is against the transposed CSR.
#     The interpretation flips (in-degree of original instead of out-
#     degree) but for symmetric road edges — the overwhelming majority
#     for bikes — both definitions agree. Saved per-anchor in the npz
#     exactly as V1 does it; consumer `build_paired_spts._build_pair`
#     uses it unchanged.
#
# CLI wire-in (TODO main.py): `compute-spts-multi [--profiles ...]`
# ─────────────────────────────────────────────────────────────────────

# Replaces existing `compute-spts --profile X` (one profile at a time).
# Default: all 5 profiles. Reuses anchor selection from compute_spts.py
# (1 km seed bbox + snap_vertex_id with the pedestrian-exclusion filter
# already in route_city_pairs._snap_vertex).


# ─────────────────────────────────────────────────────────────────────
# Standalone CLI (for smoke testing + targeted re-runs).
# Full-pipeline wire-in to main.py is a separate todo.
# ─────────────────────────────────────────────────────────────────────

def _cli_main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Multi-profile per-anchor backward SPTs over a "
                    "shared in-memory transposed CSR.",
    )
    p.add_argument("--profile", default=None,
        help="profile name (default: all 5). Comma-separated for a subset.")
    p.add_argument("--anchor", default=None,
        help="anchor name ILIKE substring (default: all anchors).")
    p.add_argument("--max-radius-m", type=float, default=SPT_RADIUS_M_DEFAULT,
        help="per-anchor SPT GEOGRAPHIC radius (default 30000 m).")
    p.add_argument("--force", action="store_true",
        help="overwrite existing per-anchor npz instead of skipping.")
    p.add_argument("--cache-path", default=None)
    p.add_argument("--force-cache", action="store_true",
        help="rebuild the global graph cache even if it exists.")
    args = p.parse_args()

    profiles = (
        tuple(s.strip() for s in args.profile.split(","))
        if args.profile else PROFILES
    )
    cache_path = Path(args.cache_path) if args.cache_path else None
    run(
        profiles=profiles,
        anchor_filter=args.anchor,
        max_radius_m=args.max_radius_m,
        force=args.force,
        cache_path=cache_path,
        force_cache=args.force_cache,
    )


if __name__ == "__main__":
    _cli_main()
