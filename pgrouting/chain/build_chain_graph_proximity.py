"""Combined chain-graph filter: reachability + proximity-based triangle removal.

Replaces bidir_reach_filter.py + dedup_chain_triangles.py.

For each candidate edge (A, B):
  1. Load cells intersecting a bbox around A + all of A's outstanding
     candidates (LRU-cached; reuses across candidates from the same
     source anchor).
  2. Multi-source Dijkstra from every CSR vertex within
     INTERMED_ANCHOR_RADIUS_M of A's centroid.
  3. Find the shortest path to any CSR vertex within
     INTERMED_ANCHOR_RADIUS_M of B's centroid.
  4. Trace the path via predecessors.
  5. Check whether that path passes within
     INTERMED_ANCHOR_RADIUS_M of ANY OTHER (non-pier) anchor C.
     - If yes: drop (A, B); enqueue (A, C) and (C, B).
     - If no: keep (A, B) with cost_m = Dijkstra distance and
       geom = the traced polyline (real road path).

Iterates via a per-round batch: process all outstanding candidates
grouped by source anchor, then process any expansions in the next
round. Expansions are strictly shorter than the dropped edge, so
convergence is guaranteed in a few rounds.

Fixes two problems the old bidir + dedup pipeline had:
  * Anchor snapping: 5 km-disc source set means we no longer depend
    on a single `snap_vertex_id` being present in the loaded
    subgraph. Kutná Hora → Kolín was over-dropped by bidir for this
    reason.
  * Triangle detection: proximity-based intermediate-C check catches
    real-road-passes-through cases that dedup's cost-based test
    missed. Neukirchen b. Heiligen Blut → Klatovy was left in the
    graph despite the actual road running through Nýrsko.

Pier↔pier ferry edges are preserved unconditionally (sea routes
exceed the CSR cap by design).

Env vars:
  SPT_PROFILE                   default views
  CELL_DIR                      default /data/cells/<profile>
  INTERMED_ANCHOR_RADIUS_M      default 5000 (source disc, dest disc,
                                intermediate-C match radius)
  INTERMED_COST_CAP_MULT        default 8.0
  INTERMED_CELL_CACHE_SIZE      default 60
  INTERMED_MAX_BBOX_EDGES       default 50_000_000
  INTERMED_BUFFER_FRAC          default 0.20
  INTERMED_MAX_PATH_LEN         default 20000 (safety cap on trace_path)
  INTERMED_MAX_ROUNDS           default 6 (safety cap on iteration)
"""
from __future__ import annotations

import gc
import heapq
import json
import math
import os
import shutil
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


PROFILE = os.environ.get("SPT_PROFILE", "views")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
IN_GRAPH_ORIG = DATA_DIR / "way_city_graph.json"
IN_GRAPH_CAND = DATA_DIR / "way_city_graph_candidates.json"
IN_ANCHORS = DATA_DIR / "way_city_anchors.geojson"
CELL_DIR = Path(os.environ.get("CELL_DIR", f"/data/cells/{PROFILE}"))
CELL_DEG = 1.0

RADIUS_M         = float(os.environ.get("INTERMED_ANCHOR_RADIUS_M", "5000"))
COST_CAP_MULT    = float(os.environ.get("INTERMED_COST_CAP_MULT", "8.0"))
CELL_CACHE_SIZE  = int(os.environ.get("INTERMED_CELL_CACHE_SIZE", "24"))
MAX_BBOX_EDGES   = int(os.environ.get("INTERMED_MAX_BBOX_EDGES", "50000000"))
BUFFER_FRAC      = float(os.environ.get("INTERMED_BUFFER_FRAC", "0.20"))
MAX_PATH_LEN     = int(os.environ.get("INTERMED_MAX_PATH_LEN", "20000"))
MAX_ROUNDS       = int(os.environ.get("INTERMED_MAX_ROUNDS", "6"))
# Chain edges whose road cost is >K× the anchor-centroid haversine
# are detour edges — the road wraps around water/mountain/border to
# reach the far endpoint. We DROP them here so the chain graph is
# authoritative; downstream stages (polygons, SPT, paired trunks)
# don't need their own filter. Pier↔pier ferry hops are exempt via
# the unconditional-keep fast path in _process_source_batch — sea
# routes always exceed this ratio by design.
#
# The ratio is SCALE-INVARIANT: a legit 160 km rural edge in
# Utah/Nevada has ratio ~1.25 and passes through. In the Europe
# dataset ~1% of edges hit ratio > 4 (p50=1.0, p95=1.6, p99=4.3),
# and every one that does is a cross-water/mountain outlier. See
# feedback_no_hard_edge_caps.md for why this beats a fixed km cap.
DETOUR_RATIO_CUTOFF = float(os.environ.get("INTERMED_DETOUR_RATIO_CUTOFF", "4.0"))

OUT_GRAPH   = DATA_DIR / "way_city_graph.json"
OUT_GEOJSON = DATA_DIR / "way_city_graph.geojson"
OUT_DROPPED = DATA_DIR / "way_city_graph_dropped.json"
OUT_ORPHANS = DATA_DIR / "way_city_graph_orphans.json"
FERRY_EDGES_IN = DATA_DIR / "ferry_edges.json"
R_EARTH_M   = 6_371_000.0
RADIUS_DEG  = RADIUS_M / 111_000.0


def _load_ferry_edge_set() -> set[tuple[int, int]]:
    """Return the set of (src_vid, dst_vid) tuples for every ferry way,
    both directions. Empty set if ferry_edges.json isn't there (older
    augment step; the mask is silently a no-op).

    Rationale: LAND anchors should reach chain neighbors via road only.
    Ferry crossings are chain hops between piers and are added by the
    ferry augment step's pier↔pier BFS. Without this mask, the stage-6
    proximity Dijkstra rides a heavily-weighted sea ferry way and
    manufactures bogus LAND↔PIER chain edges (e.g. Rostock → Gedser).
    Pier↔pier chain edges are handled by an unconditional-keep
    fast-path higher up, so this mask doesn't strand them."""
    if not FERRY_EDGES_IN.exists():
        return set()
    pairs = json.loads(FERRY_EDGES_IN.read_text())
    out: set[tuple[int, int]] = set()
    for s, d in pairs:
        out.add((int(s), int(d)))
        out.add((int(d), int(s)))
    return out


def _haversine_m(lon1, lat1, lon2, lat2):
    lat1_r = math.radians(lat1); lat2_r = math.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
    return 2.0 * R_EARTH_M * math.asin(math.sqrt(a))


def _haversine_m_vec(lon1, lat1, lons, lats):
    """Vectorized haversine — (lon1, lat1) scalar, (lons, lats) arrays."""
    lat1_r = math.radians(lat1)
    lats_r = np.radians(lats.astype(np.float64))
    dlat = lats_r - lat1_r
    dlon = np.radians(lons.astype(np.float64) - lon1)
    a = (np.sin(dlat / 2) ** 2
         + math.cos(lat1_r) * np.cos(lats_r) * np.sin(dlon / 2) ** 2)
    return 2.0 * R_EARTH_M * np.arcsin(np.sqrt(a))


def _is_detour_edge(cost_m: float,
                    a_lon: float, a_lat: float,
                    b_lon: float, b_lat: float) -> bool:
    """True if this chain edge's road-cost exceeds DETOUR_RATIO_CUTOFF
    times the anchor-centroid haversine A↔B. Uses the anchor centroids
    (not geom endpoints) so the ratio reflects real road overhead."""
    hav = _haversine_m(a_lon, a_lat, b_lon, b_lat)
    if hav <= 0 or cost_m <= 0:
        return False
    return (cost_m / hav) > DETOUR_RATIO_CUTOFF


def _bearing_deg(lon1, lat1, lon2, lat2):
    lat1_r = math.radians(lat1); lat2_r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(lat2_r)
    x = (math.cos(lat1_r) * math.sin(lat2_r)
         - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _cells_in_bbox(bbox):
    lo_lon, lo_lat, hi_lon, hi_lat = bbox
    cx0 = int(math.floor(lo_lon / CELL_DEG))
    cy0 = int(math.floor(lo_lat / CELL_DEG))
    cx1 = int(math.ceil(hi_lon / CELL_DEG))
    cy1 = int(math.ceil(hi_lat / CELL_DEG))
    return [(cx, cy) for cx in range(cx0, cx1 + 1)
                      for cy in range(cy0, cy1 + 1)]


class CellCache:
    """LRU cache of raw cell edge arrays keyed on (cx, cy)."""

    def __init__(self, size):
        self.size = size
        self.items = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, cx, cy):
        key = (cx, cy)
        if key in self.items:
            self.hits += 1
            self.items.move_to_end(key)
            return self.items[key]
        path = CELL_DIR / f"{cx}_{cy}.npz"
        if not path.exists():
            self.items[key] = None
        else:
            with np.load(path) as z:
                arr = z["edges"]
                if len(arr) == 0:
                    self.items[key] = None
                else:
                    self.items[key] = {
                        "src":     np.asarray(arr["src_id"]),
                        "dst":     np.asarray(arr["dst_id"]),
                        "fwd":     np.asarray(arr["cost"], dtype=np.float32),
                        "rev":     np.asarray(arr["reverse_cost"], dtype=np.float32),
                        "src_lon": np.asarray(arr["src_lon"], dtype=np.float32),
                        "src_lat": np.asarray(arr["src_lat"], dtype=np.float32),
                        "dst_lon": np.asarray(arr["dst_lon"], dtype=np.float32),
                        "dst_lat": np.asarray(arr["dst_lat"], dtype=np.float32),
                    }
        self.misses += 1
        while len(self.items) > self.size:
            self.items.popitem(last=False)
        self.items.move_to_end(key)
        return self.items[key]


def _load_csr_for_bbox(bbox, cache, ferry_set=None):
    """Load cells intersecting bbox from cache. Filter negative rev
    edges and (if ferry_set is provided) any (src, dst) pair belonging
    to a ferry way. Returns (csr, gid_to_local, coords, n_edges_total).
    coords[local_idx] = (lon, lat). Returns (None, {}, None, 0) if
    the bbox is empty of edges."""
    cells = _cells_in_bbox(bbox)
    chunks = defaultdict(list)   # field name -> list of chunks
    n_edges_running = 0
    for cx, cy in cells:
        c = cache.get(cx, cy)
        if c is None:
            continue
        for k in ("src", "dst", "fwd", "rev",
                  "src_lon", "src_lat", "dst_lon", "dst_lat"):
            chunks[k].append(c[k])
        n_edges_running += len(c["src"])
    if not chunks:
        return None, {}, None, 0

    src_gid = np.concatenate(chunks["src"])
    dst_gid = np.concatenate(chunks["dst"])
    fwd     = np.concatenate(chunks["fwd"])
    rev     = np.concatenate(chunks["rev"])
    src_lon = np.concatenate(chunks["src_lon"])
    src_lat = np.concatenate(chunks["src_lat"])
    dst_lon = np.concatenate(chunks["dst_lon"])
    dst_lat = np.concatenate(chunks["dst_lat"])
    del chunks

    all_gids = np.concatenate([src_gid, dst_gid])
    unique_gids, inv = np.unique(all_gids, return_inverse=True)
    del all_gids
    n_edges = len(src_gid)
    src_local = inv[:n_edges].astype(np.int32, copy=False)
    dst_local = inv[n_edges:].astype(np.int32, copy=False)
    del inv, src_gid, dst_gid
    n_verts = len(unique_gids)

    # Build coord table by assigning src coord + dst coord to their local
    # indices. Later writes for the same vertex overwrite earlier ones,
    # but the coord is consistent per vertex — any assignment is fine.
    coords = np.zeros((n_verts, 2), dtype=np.float32)
    coords[src_local, 0] = src_lon
    coords[src_local, 1] = src_lat
    coords[dst_local, 0] = dst_lon
    coords[dst_local, 1] = dst_lat
    del src_lon, src_lat, dst_lon, dst_lat

    fwd_ok = fwd >= 0
    rev_ok = rev >= 0

    # Mask ferry ways out of the road graph. Ferry crossings are chain
    # hops between piers (added by the augment step's pier↔pier BFS);
    # LAND anchors have no business chaining across water via the
    # Dijkstra's own graph. See _load_ferry_edge_set() for the "why".
    if ferry_set:
        # Need to look at the original (global) vertex IDs, not the
        # remapped locals, because the ferry set is keyed on OSM vids.
        # Rebuild the (src, dst) global ids for this batch.
        src_global = unique_gids[src_local]
        dst_global = unique_gids[dst_local]
        ferry_mask = np.fromiter(
            ((int(s), int(d)) in ferry_set
             for s, d in zip(src_global, dst_global)),
            dtype=bool, count=n_edges,
        )
        fwd_ok &= ~ferry_mask
        rev_ok &= ~ferry_mask
        del ferry_mask, src_global, dst_global

    row = np.concatenate([src_local[fwd_ok], dst_local[rev_ok]])
    col = np.concatenate([dst_local[fwd_ok], src_local[rev_ok]])
    data = np.concatenate([fwd[fwd_ok],       rev[rev_ok]])
    del src_local, dst_local, fwd, rev, fwd_ok, rev_ok
    csr = csr_matrix((data, (row, col)), shape=(n_verts, n_verts))
    del row, col, data
    gid_to_local = {int(g): i for i, g in enumerate(unique_gids)}
    return csr, gid_to_local, coords, n_edges


def _batch_bbox(source, targets, buffer_frac=BUFFER_FRAC):
    """Bounding box of source + every target's centroid, expanded by
    buffer_frac and by RADIUS_DEG (to guarantee the 5-km disc around
    each endpoint lies inside the CSR)."""
    lons = [source["lon"]] + [t["lon"] for t in targets]
    lats = [source["lat"]] + [t["lat"] for t in targets]
    lo_lon, hi_lon = min(lons), max(lons)
    lo_lat, hi_lat = min(lats), max(lats)
    lo_lon = min(lo_lon, hi_lon - 0.1)
    hi_lon = max(hi_lon, lo_lon + 0.1)
    lo_lat = min(lo_lat, hi_lat - 0.1)
    hi_lat = max(hi_lat, lo_lat + 0.1)
    dlon = (hi_lon - lo_lon) * buffer_frac + RADIUS_DEG
    dlat = (hi_lat - lo_lat) * buffer_frac + RADIUS_DEG
    return (lo_lon - dlon, lo_lat - dlat, hi_lon + dlon, hi_lat + dlat)


def _sector_split(source, targets, n_sectors=2):
    """Split targets into n_sectors compass wedges around source."""
    groups = [[] for _ in range(n_sectors)]
    width = 360.0 / n_sectors
    for t in targets:
        bear = _bearing_deg(source["lon"], source["lat"],
                            t["lon"], t["lat"])
        groups[int(bear / width) % n_sectors].append(t)
    return [g for g in groups if g]


def _find_local_verts_in_disc(coords, cx, cy, radius_m):
    """Local vertex indices within radius_m of (cx, cy). Rough-degree
    prefilter then exact haversine."""
    if coords is None or len(coords) == 0:
        return np.empty(0, dtype=np.int64)
    # Rough box prefilter — cheap.
    r_deg = radius_m / 111_000.0
    lo_lon = cx - r_deg / max(math.cos(math.radians(cy)), 0.1)
    hi_lon = cx + r_deg / max(math.cos(math.radians(cy)), 0.1)
    lo_lat = cy - r_deg
    hi_lat = cy + r_deg
    m = ((coords[:, 0] >= lo_lon) & (coords[:, 0] <= hi_lon)
         & (coords[:, 1] >= lo_lat) & (coords[:, 1] <= hi_lat))
    cand = np.flatnonzero(m)
    if len(cand) == 0:
        return cand
    # Exact haversine on the small candidate set.
    d = _haversine_m_vec(cx, cy, coords[cand, 0], coords[cand, 1])
    return cand[d <= radius_m]


def _trace_path(preds, dest):
    """Walk preds back from dest until preds[cur] < 0 (source hit or
    unreachable-of-source). Returns path (source → dest) as an array
    of local vertex indices."""
    path = [int(dest)]
    cur = int(dest)
    for _ in range(MAX_PATH_LEN):
        p = int(preds[cur])
        if p < 0:
            break
        path.append(p)
        cur = p
    path.reverse()
    return np.array(path, dtype=np.int64)


def _find_intermediate_anchor(path_coords, a_ref, b_ref,
                              anchor_records, anchor_kdtree, radius_m):
    """Return an anchor dict whose location is within radius_m of any
    point on the path, excluding A, B, and ferry piers. None if no
    such anchor exists.

    Uses a KDTree over anchors and iterates candidates near the path
    bbox — cheap because most paths span ≤ 30 km × 30 km."""
    if len(path_coords) == 0:
        return None
    r_deg = radius_m / 111_000.0
    # Coarse bbox around the whole path + radius; query anchors inside.
    lo = path_coords.min(axis=0) - r_deg
    hi = path_coords.max(axis=0) + r_deg
    # KDTree range query: use a bounding rectangle via query_ball_point
    # around the midpoint with the diagonal radius.
    mid = (lo + hi) / 2.0
    diag_deg = np.linalg.norm(hi - lo) / 2.0 + r_deg
    idxs = anchor_kdtree.query_ball_point(mid, r=diag_deg)
    if not idxs:
        return None

    # For every anchor in the bbox, check whether ANY path vertex is
    # within radius_m. Vectorize per-anchor to keep the tight loop
    # cheap.
    for ai in idxs:
        c = anchor_records[ai]
        if c["ref"] == a_ref or c["ref"] == b_ref:
            continue
        if c["ref"].startswith("ferry:"):
            continue
        # Rough per-anchor prefilter (bbox check on the path).
        d_lon = path_coords[:, 0] - c["lon"]
        d_lat = path_coords[:, 1] - c["lat"]
        rough = np.maximum(np.abs(d_lon), np.abs(d_lat))
        near = np.flatnonzero(rough <= r_deg * 1.4142)
        if len(near) == 0:
            continue
        d = _haversine_m_vec(c["lon"], c["lat"],
                             path_coords[near, 0], path_coords[near, 1])
        if d.min() <= radius_m:
            return c
    return None


def _canonical_key(a_ref, b_ref):
    """Canonical undirected pair key."""
    return tuple(sorted((a_ref, b_ref)))


def _process_source_batch(source, targets, cache, anchor_records,
                          anchor_kdtree, ref_to_anchor, ferry_set=None,
                          depth=0, max_depth=4):
    """Run one Dijkstra from source's 5-km disc; return (kept, dropped,
    expansions). Recursively sector-splits on MAX_BBOX_EDGES overrun.
    ferry_set masks ferry ways out of the Dijkstra graph — see
    _load_ferry_edge_set() for the rationale."""
    kept: list[dict] = []
    dropped: list[dict] = []
    expansions: list[dict] = []

    # Pier↔pier ferry edges — keep unconditionally.
    source_is_pier = source["ref"].startswith("ferry:")
    testable: list[dict] = []
    for t in targets:
        if source_is_pier and t["ref"].startswith("ferry:"):
            kept.append({
                "a": source["ref"], "b": t["ref"],
                "a_name": source.get("name"), "b_name": t.get("name"),
                "cost_m": _haversine_m(source["lon"], source["lat"],
                                       t["lon"], t["lat"]),
                "geom": [[source["lon"], source["lat"]],
                         [t["lon"], t["lat"]]],
                "_is_ferry_pier_pair": True,
            })
        else:
            testable.append(t)
    if not testable:
        return kept, dropped, expansions

    bbox = _batch_bbox(source, testable)
    n_cells_est = sum(1 for _ in _cells_in_bbox(bbox))
    est_edges = n_cells_est * 800_000    # rough upper cell size
    if est_edges > MAX_BBOX_EDGES and depth < max_depth:
        groups = _sector_split(source, testable, n_sectors=2 ** (depth + 1))
        for g in groups:
            k, d, e = _process_source_batch(source, g, cache,
                                            anchor_records, anchor_kdtree,
                                            ref_to_anchor,
                                            ferry_set=ferry_set,
                                            depth=depth + 1,
                                            max_depth=max_depth)
            kept.extend(k); dropped.extend(d); expansions.extend(e)
        return kept, dropped, expansions

    csr, gid_to_local, coords, n_edges = _load_csr_for_bbox(
        bbox, cache, ferry_set=ferry_set)
    if csr is None:
        for t in testable:
            dropped.append({
                "a": source["ref"], "b": t["ref"],
                "cost_m": _haversine_m(source["lon"], source["lat"],
                                       t["lon"], t["lat"]),
                "_reason": "empty subgraph",
            })
        return kept, dropped, expansions

    src_verts = _find_local_verts_in_disc(coords, source["lon"], source["lat"],
                                          RADIUS_M)
    if len(src_verts) == 0:
        for t in testable:
            dropped.append({
                "a": source["ref"], "b": t["ref"],
                "cost_m": _haversine_m(source["lon"], source["lat"],
                                       t["lon"], t["lat"]),
                "_reason": "source has no verts in disc",
            })
        del csr, coords; gc.collect()
        return kept, dropped, expansions

    # Cost cap = COST_CAP_MULT × max haversine over the batch.
    havs = [_haversine_m(source["lon"], source["lat"], t["lon"], t["lat"])
            for t in testable]
    cap = COST_CAP_MULT * max(havs) if havs else 0.0

    # scipy min_only=True + return_predecessors=True → 3-tuple:
    # (dist_per_vertex, pred_per_vertex, source_per_vertex). We don't
    # need `sources` for the path trace (walking preds back until <0
    # reaches whichever source it was), so discard it.
    dist, preds, _ = dijkstra(csr, indices=src_verts,
                              return_predecessors=True,
                              min_only=True, limit=cap)
    del csr

    for t, hav in zip(testable, havs):
        dst_verts = _find_local_verts_in_disc(coords, t["lon"], t["lat"],
                                              RADIUS_M)
        if len(dst_verts) == 0:
            dropped.append({
                "a": source["ref"], "b": t["ref"],
                "cost_m": hav,
                "_reason": "target has no verts in disc",
            })
            continue
        dists_to_t = dist[dst_verts]
        best_i = int(np.argmin(dists_to_t))
        best_dest = int(dst_verts[best_i])
        best_dist = float(dists_to_t[best_i])
        if not math.isfinite(best_dist):
            dropped.append({
                "a": source["ref"], "b": t["ref"],
                "cost_m": hav,
                "_reason": f"unreachable within {COST_CAP_MULT}×hav",
            })
            continue

        path_locs = _trace_path(preds, best_dest)
        path_coords = coords[path_locs]

        c_hit = _find_intermediate_anchor(
            path_coords, source["ref"], t["ref"],
            anchor_records, anchor_kdtree, RADIUS_M)
        if c_hit is not None:
            dropped.append({
                "a": source["ref"], "b": t["ref"],
                "cost_m": best_dist,
                "_reason": f"intermediate:{c_hit['ref']}",
            })
            expansions.append({"a": source["ref"], "b": c_hit["ref"],
                               "a_name": source.get("name"),
                               "b_name": c_hit.get("name")})
            expansions.append({"a": c_hit["ref"], "b": t["ref"],
                               "a_name": c_hit.get("name"),
                               "b_name": t.get("name")})
        else:
            # Cap geom length so accumulating `kept_all` doesn't blow
            # memory. Strided downsample to at most GEOM_TARGET_VERTS
            # points, always keeping the endpoints so the polyline
            # still starts/ends on the source & destination anchors.
            GEOM_TARGET_VERTS = 200
            n = len(path_coords)
            if n > GEOM_TARGET_VERTS:
                stride = max(1, n // (GEOM_TARGET_VERTS - 1))
                idx = list(range(0, n - 1, stride)) + [n - 1]
                pc = path_coords[idx]
            else:
                pc = path_coords
            # tolist() + explicit floats is smaller than a numpy array
            # held in kept_all, but each entry adds a few hundred bytes
            # of Python overhead.  ~200 verts × ~64 B/entry ≈ 12 KB / edge
            # × ~7000 edges ≈ 85 MB total — manageable.
            geom = [[float(lon), float(lat)]
                    for lon, lat in pc.tolist()]
            # Floor cost_m at the anchor-centroid haversine. The
            # multi-source Dijkstra effectively measures road distance
            # from A's 5 km disc EDGE to B's 5 km disc EDGE — for close
            # anchors that under-counts by up to 2 × RADIUS_M. Chain-
            # Dijkstra would then treat the leg as nearly free and
            # pick nonsense routes. Take max of the two so cost is at
            # least the true centroid-to-centroid straight-line
            # distance.
            cost_m = max(best_dist, hav)
            # Detour filter — chain graph is the authority on which
            # edges are valid. If the road wraps around water/mountain
            # to reach B (cost / centroid-haversine > cutoff), drop
            # the edge. Pier↔pier hops bypass this via the earlier
            # unconditional-keep fast path.
            if _is_detour_edge(cost_m,
                               source["lon"], source["lat"],
                               t["lon"], t["lat"]):
                dropped.append({
                    "a": source["ref"], "b": t["ref"],
                    "cost_m": cost_m,
                    "_reason": f"detour ratio>{DETOUR_RATIO_CUTOFF}",
                })
                continue
            kept.append({
                "a": source["ref"], "b": t["ref"],
                "a_name": source.get("name"), "b_name": t.get("name"),
                "cost_m": cost_m,
                "geom": geom,
            })

    del preds, dist, coords
    gc.collect()
    return kept, dropped, expansions


def _load_anchors():
    with open(IN_ANCHORS) as fh:
        gj = json.load(fh)
    records: list[dict] = []
    for f in gj["features"]:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        records.append({
            "ref": p["ref"],
            "name": p.get("name"),
            "kind": p.get("kind"),
            "country": p.get("country"),
            "lon": float(lon),
            "lat": float(lat),
        })
    return records


def _load_candidates():
    """Load candidate edges. Prefer /data/way_city_graph_candidates.json
    if it exists (snapshot from a prior run); else read
    /data/way_city_graph.json (crow-flies + ferry augmented) and
    snapshot it."""
    if IN_GRAPH_CAND.exists():
        return json.loads(IN_GRAPH_CAND.read_text())
    if not IN_GRAPH_ORIG.exists():
        raise SystemExit(f"neither {IN_GRAPH_CAND} nor {IN_GRAPH_ORIG} exists")
    edges = json.loads(IN_GRAPH_ORIG.read_text())
    shutil.copy(IN_GRAPH_ORIG, IN_GRAPH_CAND)
    print(f"[intermed] snapshotted input → {IN_GRAPH_CAND.name}", flush=True)
    return edges


def _write_geojson(edges: list[dict]) -> None:
    features = []
    for e in edges:
        geom = e.get("geom") or [[0.0, 0.0], [0.0, 0.0]]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": geom},
            "properties": {
                "a": e["a"], "b": e["b"],
                "a_name": e.get("a_name"), "b_name": e.get("b_name"),
                "cost_m": e.get("cost_m"),
            },
        })
    OUT_GEOJSON.write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}, ensure_ascii=False))


def _write_orphans(kept: list[dict], anchor_records: list[dict]) -> None:
    degree: dict[str, int] = defaultdict(int)
    for e in kept:
        degree[e["a"]] += 1
        degree[e["b"]] += 1
    orphans = [{"ref": a["ref"], "name": a.get("name"),
                "kind": a.get("kind"), "country": a.get("country"),
                "lon": a["lon"], "lat": a["lat"]}
               for a in anchor_records if degree.get(a["ref"], 0) == 0]
    OUT_ORPHANS.write_text(json.dumps({
        "count": len(orphans),
        "total_anchors": len(anchor_records),
        "orphans": orphans,
    }, ensure_ascii=False, indent=2))
    return len(orphans)


def main() -> None:
    t0 = time.time()
    print(f"[intermed] profile={PROFILE}  radius={RADIUS_M:.0f}m  "
          f"cost_cap={COST_CAP_MULT}×hav  buffer={BUFFER_FRAC*100:.0f}%",
          flush=True)
    anchor_records = _load_anchors()
    ref_to_anchor = {a["ref"]: a for a in anchor_records}
    anchor_kdtree = cKDTree(
        np.array([(a["lon"], a["lat"]) for a in anchor_records]))
    candidates = _load_candidates()
    print(f"[intermed] {len(anchor_records):,} anchors, "
          f"{len(candidates):,} candidate edges", flush=True)

    ferry_set = _load_ferry_edge_set()
    print(f"[intermed] ferry-mask: {len(ferry_set):,} directed pairs "
          f"({'active' if ferry_set else 'no ferry_edges.json — mask disabled'})",
          flush=True)

    cache = CellCache(CELL_CACHE_SIZE)
    kept_all: list[dict] = []
    dropped_all: list[dict] = []
    seen: set = set()
    todo: list[dict] = list(candidates)

    for rnd in range(1, MAX_ROUNDS + 1):
        # Deduplicate & filter already-seen pairs.
        fresh = []
        for e in todo:
            key = _canonical_key(e["a"], e["b"])
            if key in seen:
                continue
            seen.add(key)
            fresh.append(e)
        if not fresh:
            print(f"[intermed] round {rnd}: nothing new — converged", flush=True)
            break
        # Group by source anchor. For an undirected candidate we treat
        # the endpoint that has an anchor record first as the source.
        by_source: dict[str, list[dict]] = defaultdict(list)
        for e in fresh:
            src = e["a"] if e["a"] in ref_to_anchor else e["b"]
            oth = e["b"] if src == e["a"] else e["a"]
            by_source[src].append({
                "ref": oth,
                **({"name": e.get("b_name") if src == e["a"]
                                            else e.get("a_name")}),
            })
        print(f"[intermed] round {rnd}: {len(fresh):,} pairs across "
              f"{len(by_source):,} source anchors", flush=True)

        rnd_expansions: list[dict] = []
        t_rnd = time.time()
        last_log = t_rnd
        n_done = 0
        n_srcs = len(by_source)
        for src_ref, tgts in sorted(by_source.items()):
            source = ref_to_anchor.get(src_ref)
            if source is None:
                for tgt in tgts:
                    dropped_all.append({
                        "a": src_ref, "b": tgt["ref"],
                        "_reason": "source anchor missing",
                    })
                n_done += 1
                continue
            target_anchors = []
            for tgt in tgts:
                t_anchor = ref_to_anchor.get(tgt["ref"])
                if t_anchor is None:
                    dropped_all.append({
                        "a": src_ref, "b": tgt["ref"],
                        "_reason": "target anchor missing",
                    })
                else:
                    target_anchors.append(t_anchor)
            if target_anchors:
                k, d, exp = _process_source_batch(
                    source, target_anchors, cache,
                    anchor_records, anchor_kdtree, ref_to_anchor,
                    ferry_set=ferry_set)
                kept_all.extend(k)
                dropped_all.extend(d)
                rnd_expansions.extend(exp)
            n_done += 1
            # Aggressive GC on every source-anchor batch — the CSR +
            # dist + preds arrays for a wide bbox can be several
            # hundred MB, and without an explicit collect Python's
            # allocator can hold onto them long enough that the
            # process peaks above the WSL VM limit.
            gc.collect()

            if time.time() - last_log >= 30:
                elapsed = time.time() - t_rnd
                hits = cache.hits; misses = cache.misses
                hit_rate = 100 * hits / max(hits + misses, 1)
                try:
                    with open("/proc/self/status") as fh:
                        rss_kb = int(
                            next(l for l in fh
                                 if l.startswith("VmRSS:")).split()[1])
                    rss_gb = rss_kb / 1_048_576
                except Exception:
                    rss_gb = -1.0
                print(f"[intermed]   {n_done}/{n_srcs} src  "
                      f"kept {len(kept_all):,}  drop {len(dropped_all):,}  "
                      f"expansions {len(rnd_expansions):,}  "
                      f"cache hit {hit_rate:.0f}%  "
                      f"rss={rss_gb:.2f}GB  "
                      f"{elapsed:.0f}s", flush=True)
                last_log = time.time()

        print(f"[intermed] round {rnd} done in {time.time()-t_rnd:.0f}s  "
              f"kept={len(kept_all):,}  dropped={len(dropped_all):,}  "
              f"expansions={len(rnd_expansions):,}", flush=True)
        todo = rnd_expansions
    else:
        print(f"[intermed] hit MAX_ROUNDS={MAX_ROUNDS} without convergence; "
              f"{len(todo):,} expansions remain unprocessed", flush=True)

    # Dedupe kept — an undirected pair may have been produced from both
    # sides (once as source→target, once as target→source). Keep the
    # smallest cost / longest geom per canonical pair.
    by_pair: dict[tuple, dict] = {}
    for e in kept_all:
        key = _canonical_key(e["a"], e["b"])
        prev = by_pair.get(key)
        if prev is None or e.get("cost_m", 1e18) < prev.get("cost_m", 1e18):
            by_pair[key] = e
    kept_unique = list(by_pair.values())

    OUT_GRAPH.write_text(json.dumps(kept_unique, ensure_ascii=False))
    OUT_DROPPED.write_text(json.dumps(dropped_all, ensure_ascii=False))
    _write_geojson(kept_unique)
    n_orphans = _write_orphans(kept_unique, anchor_records)

    print(f"[intermed] DONE in {time.time()-t0:.0f}s", flush=True)
    print(f"[intermed]   kept:     {len(kept_unique):,} chain edges",
          flush=True)
    print(f"[intermed]   dropped:  {len(dropped_all):,}", flush=True)
    print(f"[intermed]   orphans:  {n_orphans} anchors (in "
          f"{OUT_ORPHANS.name})", flush=True)
    print(f"[intermed]   cache hit rate: "
          f"{100*cache.hits/max(cache.hits+cache.misses, 1):.0f}%",
          flush=True)


if __name__ == "__main__":
    main()
