"""Runtime bounded local Dijkstra on the road-graph CSR.

The paired-trunk router walks precomputed successor pointers, which
is fast but requires the walk's start vertex to be present in the
trunk. When it's not (paired-trunk pruning drops non-optimal
starting vertices), we previously fell back to haversine-nearest
straight-line stitches or `_last_mile` LCA overshoots — both
produced visible polyline loops.

This module implements the missing "unpruned-CSR local-Dijkstra
bridge" the trunk_router docstring TODO-flags. It loads per-1°-cell
edge files from `data/cells/<profile>/<cx>_<cy>.npz` on demand,
builds a CSR sub-graph, and runs a bounded multi-destination
Dijkstra to find the shortest road-graph path from a start vertex to
any of a set of target vertices (e.g. all vertices in some paired
trunk).

Cell files are written by `pgrouting/spt/compute_spts_polygon.py`'s
prep step; the schema is a structured `edges` array with (src_id,
dst_id, src_lon, src_lat, dst_lon, dst_lat, cost, reverse_cost).
Cross-cell edges are stored once at the source cell with both
endpoint coords inlined, so loading one cell resolves interior
routing correctly and adjacent cells only need to be added when a
route might cross the boundary.

Cost semantics: costs are the same bike-edge costs the SPT preprocess
used (`cost.py` in the pgrouting side). A `max_cost` limit on
Dijkstra bounds the search radius in km-equivalent units. 50 covers
comfortably more than any first-mile bridge would need.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .settings import DATA_DIR


CELL_DEG = 1.0
CELL_DIR = Path(DATA_DIR) / "cells"


def _cell_key(lon: float, lat: float) -> tuple[int, int]:
    return int(math.floor(lon / CELL_DEG)), int(math.floor(lat / CELL_DEG))


def _cell_path(profile: str, cx: int, cy: int) -> Path:
    return CELL_DIR / profile / f"{cx}_{cy}.npz"


@lru_cache(maxsize=32)
def _load_cell(profile: str, cx: int, cy: int) -> dict | None:
    """Read one cell's edge list and return {'src', 'dst', 'fwd',
    'rev', 'gid_of_local', 'verts'} — parallel arrays keyed by local
    vertex index, plus the global (OSM) id lookup and coord array.

    None if the cell file doesn't exist (edge of coverage), or if it
    exists but has zero edges.
    """
    path = _cell_path(profile, cx, cy)
    if not path.exists():
        return None
    with np.load(path, mmap_mode="r") as z:
        arr = z["edges"]
        n = int(len(arr))
        if n == 0:
            return None
        src_gid = np.asarray(arr["src_id"])
        dst_gid = np.asarray(arr["dst_id"])
        fwd     = np.asarray(arr["cost"],         dtype=np.float32)
        rev     = np.asarray(arr["reverse_cost"], dtype=np.float32)
        sx      = np.asarray(arr["src_lon"])
        sy      = np.asarray(arr["src_lat"])
        tx      = np.asarray(arr["dst_lon"])
        ty      = np.asarray(arr["dst_lat"])
    all_gid = np.concatenate([src_gid, dst_gid])
    unique_gid, inverse = np.unique(all_gid, return_inverse=True)
    src_local = inverse[:n].astype(np.int32)
    dst_local = inverse[n:].astype(np.int32)
    n_verts = len(unique_gid)
    verts = np.empty((n_verts, 2), dtype=np.float64)
    verts[dst_local, 0] = tx; verts[dst_local, 1] = ty
    verts[src_local, 0] = sx; verts[src_local, 1] = sy
    return {
        "src":          src_local,
        "dst":          dst_local,
        "fwd":          fwd,
        "rev":          rev,
        "gid_of_local": unique_gid.astype(np.int64),
        "verts":        verts,
    }


def _merge_cells(cell_a: dict, cell_b: dict) -> dict:
    """Concatenate two loaded cells into a single subgraph. Handles
    the overlap: vertices at cross-cell boundaries appear in both
    files' gid_of_local, so we re-unique by global id."""
    all_gid = np.concatenate([cell_a["gid_of_local"],
                              cell_b["gid_of_local"]])
    unique_gid, inverse = np.unique(all_gid, return_inverse=True)
    na = len(cell_a["gid_of_local"])
    map_a = inverse[:na].astype(np.int32)
    map_b = inverse[na:].astype(np.int32)
    src = np.concatenate([map_a[cell_a["src"]], map_b[cell_b["src"]]])
    dst = np.concatenate([map_a[cell_a["dst"]], map_b[cell_b["dst"]]])
    fwd = np.concatenate([cell_a["fwd"], cell_b["fwd"]])
    rev = np.concatenate([cell_a["rev"], cell_b["rev"]])
    n_verts = len(unique_gid)
    verts = np.empty((n_verts, 2), dtype=np.float64)
    # Populate coords from wherever they were known — either cell can
    # own a given vertex; overlap re-assigns but the values match.
    verts[map_a] = cell_a["verts"]
    verts[map_b] = cell_b["verts"]
    return {
        "src": src, "dst": dst,
        "fwd": fwd, "rev": rev,
        "gid_of_local": unique_gid.astype(np.int64),
        "verts": verts,
    }


def _build_csr(cell: dict) -> csr_matrix:
    """CSR over the cell's bike-routable edges. Negative-cost edges
    are treated as impassable (matches the SPT preprocess), so
    fwd/rev >= 0 masks are applied."""
    src = cell["src"]; dst = cell["dst"]
    fwd = cell["fwd"]; rev = cell["rev"]
    fwd_mask = fwd >= 0
    rev_mask = rev >= 0
    e_src = np.concatenate([src[fwd_mask], dst[rev_mask]])
    e_dst = np.concatenate([dst[fwd_mask], src[rev_mask]])
    e_cost = np.concatenate([fwd[fwd_mask], rev[rev_mask]])
    n = len(cell["gid_of_local"])
    return csr_matrix(
        (e_cost, (e_src, e_dst)),
        shape=(n, n),
        dtype=np.float32,
    )


def local_dijkstra_to_targets(
    profile: str,
    start_lon: float, start_lat: float,
    start_vid: int,
    target_vids: np.ndarray,
    max_cost: float = 50.0,
    target_lonlats: np.ndarray | None = None,
    prefer_lonlat: tuple[float, float] | None = None,
    prefer_radius_km: float = 5.0,
    intercept_lonlat: tuple[float, float] | None = None,
    intercept_bias: float = 0.0,
) -> tuple[list[list[float]], int] | None:
    """Find the shortest bike-graph path from `start_vid` to whichever
    of `target_vids` is closest, bounded by `max_cost`. Returns
    (polyline, reached_target_vid) — the polyline goes from
    start_vid's coord to the reached target's coord (inclusive on
    both ends) on the actual road graph. Returns None if:
      - the cell(s) around start don't exist
      - start_vid isn't a road-graph vertex in the loaded cells
      - no target is reachable within max_cost
      - the required cells aren't on disk (edge of coverage)

    The polyline is the actual road-graph shortest path, no straight-
    line jumps. Costs are the same bike-edge units the SPT preprocess
    uses (`cost.py`), so `max_cost=50` corresponds to roughly ~30–50
    km of route depending on grade/surface.

    `prefer_lonlat` + `target_lonlats` + `prefer_radius_km` — when
    all three are set, targets are filtered to those within
    `prefer_radius_km` of `prefer_lonlat`. Used at chain-hop entries
    to avoid picking a trunk vertex on the "wrong side" of the source
    anchor — one whose successor chain would walk back through where
    we started. If the filtered set is empty, falls back to the full
    target set.

    `intercept_lonlat` + `intercept_bias` + `target_lonlats` — when
    set, the WINNING target is the one that minimizes
    `dijkstra_cost + intercept_bias * geodesic_km_to(intercept)`
    rather than `dijkstra_cost` alone. An A*-flavored intercept-point
    heuristic: prefer entering the trunk closer to the trip's
    destination even if it costs more bike-effort locally. Trades
    off "reach the trunk cheaply" against "avoid trunks whose succ-
    chain from the near-source entry would loop back through where
    we started." Larger `intercept_bias` means stronger pull toward
    intercept coord. Edge costs are ~1000-per-km empirically so
    `intercept_bias=1000` ≈ "one bike-effort unit per meter of
    geodesic remaining." Bias 0 disables the heuristic.
    """
    # Load the source cell + adjacent cells only if the source is
    # near a cell boundary. A 1° cell at central-EU latitudes spans
    # ~110 km E-W and ~70 km N-S; the first-mile bridge we're solving
    # is typically < 20 km, so a single cell nearly always suffices.
    # Cell edges are stored with both endpoints inlined, so a cross-
    # cell edge whose source is in our cell reaches into the neighbor
    # correctly without loading it — the neighbor load only helps if
    # the SHORTEST PATH crosses out of and back into our cell.
    cx, cy = _cell_key(start_lon, start_lat)
    primary = _load_cell(profile, cx, cy)
    if primary is None:
        return None
    graph = primary
    frac_lon = (start_lon / CELL_DEG) - cx
    frac_lat = (start_lat / CELL_DEG) - cy
    BUFFER_FRAC = 0.15
    for (dx, dy, condition) in (
        (-1, 0, frac_lon < BUFFER_FRAC),
        ( 1, 0, frac_lon > 1 - BUFFER_FRAC),
        ( 0,-1, frac_lat < BUFFER_FRAC),
        ( 0, 1, frac_lat > 1 - BUFFER_FRAC),
    ):
        if condition:
            neighbor = _load_cell(profile, cx + dx, cy + dy)
            if neighbor is not None:
                graph = _merge_cells(graph, neighbor)

    # Locate start_vid + target_vids in the merged cell.
    gid_of_local = graph["gid_of_local"]
    start_pos = int(np.searchsorted(gid_of_local, start_vid))
    if (start_pos >= len(gid_of_local)
            or int(gid_of_local[start_pos]) != start_vid):
        return None

    # Optional geographic pre-filter — keep only targets within
    # `prefer_radius_km` of `prefer_lonlat`. Prevents the routing from
    # picking a trunk vertex on the wrong side of the source anchor.
    tgt_arr_raw = np.asarray(target_vids, dtype=np.int64)
    if (prefer_lonlat is not None
            and target_lonlats is not None
            and len(target_lonlats) == len(tgt_arr_raw)):
        plon, plat = float(prefer_lonlat[0]), float(prefer_lonlat[1])
        _R = 6_371_000.0
        _lat_a = math.radians(plat)
        _lat_v = np.radians(target_lonlats[:, 1].astype(np.float64))
        _lon_d = np.radians(target_lonlats[:, 0].astype(np.float64) - plon)
        _hav = (np.sin((_lat_v - _lat_a) / 2) ** 2
                + math.cos(_lat_a) * np.cos(_lat_v)
                * np.sin(_lon_d / 2) ** 2)
        _dist = 2 * _R * np.arcsin(np.sqrt(_hav))
        keep = _dist <= (prefer_radius_km * 1000.0)
        if keep.any():
            tgt_arr_raw = tgt_arr_raw[keep]
    # np.sort (not in-place) because target_vids may be a view into a
    # read-only structured array (e.g. `trunk_arr["vid"]` from the
    # paired-trunks blob).
    tgt_arr = np.sort(tgt_arr_raw)
    positions = np.searchsorted(gid_of_local, tgt_arr)
    in_range = positions < len(gid_of_local)
    matched = np.zeros_like(in_range)
    matched[in_range] = gid_of_local[positions[in_range]] == tgt_arr[in_range]
    target_locals = positions[matched].astype(np.int32)
    if len(target_locals) == 0:
        return None

    csr = _build_csr(graph)
    dist, pred = dijkstra(
        csgraph=csr,
        indices=[start_pos],
        return_predecessors=True,
        directed=True,
        limit=max_cost,
    )
    # Pick the reached target with smallest dist.
    dist0 = dist[0]  # dijkstra returns 2D even for one source
    pred0 = pred[0]
    reached_mask = np.isfinite(dist0[target_locals])
    reached_targets = target_locals[reached_mask]
    if len(reached_targets) == 0:
        return None

    # Score reached targets. Default: pure dijkstra cost (cheapest to
    # reach on the road graph). With `intercept_lonlat`+`intercept_bias`+
    # `target_lonlats`: add a per-target penalty proportional to the
    # geodesic distance from the target to the intercept coord. Effect:
    # bias toward entering the trunk closer to the trip's destination.
    reached_costs = dist0[reached_targets]
    if (intercept_lonlat is not None and intercept_bias > 0
            and target_lonlats is not None
            and len(target_lonlats) == len(target_vids)):
        # target_lonlats was aligned with the pre-filter tgt_arr_raw
        # via the `keep` mask. Rebuild the alignment: the entries in
        # `positions` came from tgt_arr (post-filter, sorted), and
        # matched[i] tells us which tgt_arr[i] survived. We need the
        # coords for the reached targets specifically. Easier path:
        # look each reached vid back up in the ORIGINAL target_vids
        # via searchsorted, then index target_lonlats.
        reached_vids_arr = gid_of_local[reached_targets]
        orig_target_arr = np.asarray(target_vids, dtype=np.int64)
        order = np.argsort(orig_target_arr)
        sorted_orig = orig_target_arr[order]
        opos = np.searchsorted(sorted_orig, reached_vids_arr)
        valid = (opos < len(sorted_orig)) & (sorted_orig[opos] == reached_vids_arr)
        if valid.all():
            orig_idx = order[opos]
            _R = 6_371_000.0
            _lat_a = math.radians(float(intercept_lonlat[1]))
            _lat_v = np.radians(target_lonlats[orig_idx, 1].astype(np.float64))
            _lon_d = np.radians(
                target_lonlats[orig_idx, 0].astype(np.float64)
                - float(intercept_lonlat[0])
            )
            _hav = (np.sin((_lat_v - _lat_a) / 2) ** 2
                    + math.cos(_lat_a) * np.cos(_lat_v)
                    * np.sin(_lon_d / 2) ** 2)
            geodesic_m = 2 * _R * np.arcsin(np.sqrt(_hav))
            reached_costs = reached_costs + intercept_bias * (geodesic_m / 1000.0)
    best_local = int(reached_targets[np.argmin(reached_costs)])

    # Walk predecessors from best_local back to start_pos.
    path_locals = [best_local]
    cur = best_local
    while cur != start_pos:
        p = int(pred0[cur])
        if p < 0:
            return None  # shouldn't happen if best_local was reached
        path_locals.append(p)
        cur = p
    path_locals.reverse()

    verts = graph["verts"]
    polyline = [[float(verts[i][0]), float(verts[i][1])]
                for i in path_locals]
    reached_vid = int(gid_of_local[best_local])
    return polyline, reached_vid
