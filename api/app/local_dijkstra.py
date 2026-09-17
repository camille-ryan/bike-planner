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
    """CSR over the WHOLE cell's bike-routable edges. Negative-cost
    edges are treated as impassable (matches the SPT preprocess), so
    fwd/rev >= 0 masks are applied.

    Cached on the cell dict itself — `_load_cell` is `@lru_cache`d so
    subsequent stitches into the same cell skip the 0.8s coo→csr
    setup.
    """
    if "_csr" in cell:
        return cell["_csr"]
    src = cell["src"]; dst = cell["dst"]
    fwd = cell["fwd"]; rev = cell["rev"]
    fwd_mask = fwd >= 0
    rev_mask = rev >= 0
    e_src = np.concatenate([src[fwd_mask], dst[rev_mask]])
    e_dst = np.concatenate([dst[fwd_mask], src[rev_mask]])
    e_cost = np.concatenate([fwd[fwd_mask], rev[rev_mask]])
    n = len(cell["gid_of_local"])
    csr = csr_matrix(
        (e_cost, (e_src, e_dst)),
        shape=(n, n),
        dtype=np.float32,
    )
    cell["_csr"] = csr
    return csr


def _build_csr_bbox(
    cell: dict,
    bbox: tuple[float, float, float, float],
    must_keep_locals: np.ndarray,
) -> tuple[csr_matrix, np.ndarray] | None:
    """CSR over a rectangular sub-window of the cell (lon_min, lon_max,
    lat_min, lat_max). Returns `(csr, kept_locals)` where
    `kept_locals[i]` is the FULL-CELL local index of CSR row `i`
    (sorted). Callers translate:
      full → csr  via  np.searchsorted(kept_locals, full_idx)
      csr  → full via  kept_locals[csr_idx]

    `must_keep_locals` is a set of full-cell local indices (start +
    target verts) that MUST be included even if outside the bbox, so
    Dijkstra can find them by position.

    Returns None if the crop is empty (no edges survive).

    Cheap enough (2 boolean masks + a compact remap + one coo→csr)
    that it undercuts the full-cell build when the crop is <1% of
    the cell — which is the target case for a first-mile stitch.
    """
    verts = cell["verts"]
    lon_min, lon_max, lat_min, lat_max = bbox
    in_bbox = (
        (verts[:, 0] >= lon_min) & (verts[:, 0] <= lon_max)
        & (verts[:, 1] >= lat_min) & (verts[:, 1] <= lat_max)
    )
    if must_keep_locals.size:
        in_bbox[must_keep_locals] = True
    # Keep only edges whose BOTH endpoints survive — a directed edge
    # into a dropped vertex is unusable, and Dijkstra can't relax
    # through it anyway.
    src = cell["src"]; dst = cell["dst"]
    e_keep = in_bbox[src] & in_bbox[dst]
    if not e_keep.any():
        return None
    src_full = src[e_keep]
    dst_full = dst[e_keep]
    fwd = cell["fwd"][e_keep]
    rev = cell["rev"][e_keep]
    # Remap full-cell locals → compact 0..k-1 in sorted order.
    kept_locals = np.where(in_bbox)[0]  # sorted ascending
    remap = np.full(len(verts), -1, dtype=np.int32)
    remap[kept_locals] = np.arange(len(kept_locals), dtype=np.int32)
    src_c = remap[src_full]
    dst_c = remap[dst_full]
    fwd_mask = fwd >= 0
    rev_mask = rev >= 0
    e_src = np.concatenate([src_c[fwd_mask], dst_c[rev_mask]])
    e_dst = np.concatenate([dst_c[fwd_mask], src_c[rev_mask]])
    e_cost = np.concatenate([fwd[fwd_mask], rev[rev_mask]])
    n = len(kept_locals)
    csr = csr_matrix(
        (e_cost, (e_src, e_dst)),
        shape=(n, n),
        dtype=np.float32,
    )
    return csr, kept_locals


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
    intercept_probe_coords: np.ndarray | None = None,
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

    `intercept_probe_coords` — when passed, use THESE coords per
    target instead of `target_lonlats` for the intercept-bias
    scoring. Purpose: paired trunks are trees with multiple branches,
    and the ENTRY coord is a bad proxy for which branch the trunk
    walk will take. The caller can pre-compute the coord `N` hops
    along the succ-chain from each entry (via the trunk's own
    `next_idx`) and pass that as `intercept_probe_coords`. Then the
    intercept-bias picks whichever entry's SUCC-CHAIN heads toward
    the destination, rather than whichever entry is itself near the
    destination.
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

    verts = graph["verts"]

    # Inner helper: run Dijkstra on a (csr, kept_locals) pair.
    # `kept_locals is None` means the CSR spans the whole cell
    # (identity mapping). Otherwise `kept_locals[i]` is the full-cell
    # local index of CSR row `i`; callers translate full↔csr through
    # it. Encapsulated so the bbox-fast-path and full-cell fallback
    # share the same post-Dijkstra scoring and path-recovery logic.
    def _run(csr, kept_locals):
        if kept_locals is None:
            csr_start = start_pos
            csr_targets = target_locals
        else:
            pos_s = int(np.searchsorted(kept_locals, start_pos))
            if (pos_s >= len(kept_locals)
                    or int(kept_locals[pos_s]) != start_pos):
                return None
            csr_start = pos_s
            pos_t = np.searchsorted(kept_locals, target_locals)
            in_crop = (pos_t < len(kept_locals))
            hit = np.zeros_like(in_crop)
            hit[in_crop] = kept_locals[pos_t[in_crop]] == target_locals[in_crop]
            if not hit.any():
                return None
            csr_targets = pos_t[hit].astype(np.int32)
        dist, pred = dijkstra(
            csgraph=csr,
            indices=[csr_start],
            return_predecessors=True,
            directed=True,
            limit=max_cost,
        )
        dist0 = dist[0]
        pred0 = pred[0]
        reached_mask = np.isfinite(dist0[csr_targets])
        reached_targets = csr_targets[reached_mask]
        if len(reached_targets) == 0:
            return None
        reached_costs = dist0[reached_targets]
        # Intercept-bias scoring (see docstring).
        _bias_coords = (
            intercept_probe_coords
            if intercept_probe_coords is not None
            else target_lonlats
        )
        if (intercept_lonlat is not None and intercept_bias > 0
                and _bias_coords is not None
                and len(_bias_coords) == len(target_vids)):
            reached_full = (reached_targets if kept_locals is None
                            else kept_locals[reached_targets])
            reached_vids_arr = gid_of_local[reached_full]
            orig_target_arr = np.asarray(target_vids, dtype=np.int64)
            order = np.argsort(orig_target_arr)
            sorted_orig = orig_target_arr[order]
            opos = np.searchsorted(sorted_orig, reached_vids_arr)
            valid = (opos < len(sorted_orig)) & (sorted_orig[opos] == reached_vids_arr)
            if valid.all():
                orig_idx = order[opos]
                _R = 6_371_000.0
                _lat_a = math.radians(float(intercept_lonlat[1]))
                _lat_v = np.radians(_bias_coords[orig_idx, 1].astype(np.float64))
                _lon_d = np.radians(
                    _bias_coords[orig_idx, 0].astype(np.float64)
                    - float(intercept_lonlat[0])
                )
                _hav = (np.sin((_lat_v - _lat_a) / 2) ** 2
                        + math.cos(_lat_a) * np.cos(_lat_v)
                        * np.sin(_lon_d / 2) ** 2)
                geodesic_m = 2 * _R * np.arcsin(np.sqrt(_hav))
                reached_costs = reached_costs + intercept_bias * (geodesic_m / 1000.0)
        best_csr = int(reached_targets[np.argmin(reached_costs)])
        # Walk predecessors in CSR space.
        path_csr = [best_csr]
        cur = best_csr
        while cur != csr_start:
            p = int(pred0[cur])
            if p < 0:
                return None
            path_csr.append(p)
            cur = p
        path_csr.reverse()
        # Translate to full-cell for coord lookup.
        path_full = (path_csr if kept_locals is None
                     else [int(kept_locals[i]) for i in path_csr])
        polyline = [[float(verts[i][0]), float(verts[i][1])]
                    for i in path_full]
        reached_vid = int(gid_of_local[path_full[-1]])
        return polyline, reached_vid

    # Fast path: crop to a small bbox around START sized by the
    # geodesic distance to the NEAREST target. A first-mile stitch
    # rarely needs more than ~1.5× the straight-line distance
    # (bike-route slack over geodesic). Sizing the bbox by that
    # distance means:
    #   - trunk passes close to the anchor (0.5 km) → 2 km bbox
    #   - anchor sits inside the SPT with trunk 5 km away → 10 km bbox
    #   - anchor at edge, trunk 25 km away → the full-cell fallback
    # A 3 km floor covers the very-close case with room to bend.
    #
    # Cost of this pre-check: one vectorized haversine over the
    # target verts (already loaded), <1 ms.
    tgt_local_coords = verts[target_locals]
    _R = 6_371_000.0
    _lat_a = math.radians(start_lat)
    _lat_v = np.radians(tgt_local_coords[:, 1].astype(np.float64))
    _lon_d = np.radians(tgt_local_coords[:, 0].astype(np.float64) - start_lon)
    _hav = (np.sin((_lat_v - _lat_a) / 2) ** 2
            + math.cos(_lat_a) * np.cos(_lat_v)
            * np.sin(_lon_d / 2) ** 2)
    _target_km = 2 * _R * np.arcsin(np.sqrt(_hav)) / 1000.0
    nearest_km = float(_target_km.min())
    stitch_radius_km = max(3.0, nearest_km * 1.5 + 2.0)
    _km_per_deg_lat = 111.0
    _km_per_deg_lon = 111.0 * math.cos(math.radians(start_lat))
    _dlat = stitch_radius_km / _km_per_deg_lat
    _dlon = stitch_radius_km / max(_km_per_deg_lon, 1.0)
    bbox = (start_lon - _dlon, start_lon + _dlon,
            start_lat - _dlat, start_lat + _dlat)
    must_keep = np.array([start_pos], dtype=np.int32)
    bbox_build = _build_csr_bbox(graph, bbox, must_keep)
    if bbox_build is not None:
        csr_bbox, kept_locals = bbox_build
        # Only worth the crop overhead if we actually shrunk things.
        # The cell is ~200k verts; a stitch bbox is usually a few
        # thousand. Skip the crop attempt when it barely helps.
        if len(kept_locals) * 2 < len(gid_of_local):
            result = _run(csr_bbox, kept_locals)
            if result is not None:
                return result
            # Crop was built but Dijkstra didn't reach any target
            # inside it — shortest path may exit the bbox. Fall
            # through to the full-cell retry.

    # Fallback: whole-cell CSR (cached on the cell dict so this is
    # a lookup after the first hit).
    csr = _build_csr(graph)
    return _run(csr, None)
