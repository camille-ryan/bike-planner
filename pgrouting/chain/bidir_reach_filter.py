"""Filter candidate chain edges by road-network reachability + reweight.

**Per-anchor pair-scope approach** (task #54, v3):

For each source anchor A:
  1. Compute a bbox = union of {A + all A's candidate targets} lonlats,
     expanded by BUFFER_FRAC (default 20 %). Tight bbox around A's
     immediate reach — density-adaptive without any tile grid.
  2. Load cells intersecting that bbox into a scipy CSR (with an LRU
     cache of individual cell arrays so consecutive anchors that share
     cells don't re-decompress them).
  3. Run one bounded scipy Dijkstra from A, cap =
     COST_CAP_MULT × max(haversine(A, B_i)) across A's candidates.
  4. For each candidate B: check if any of B's snap_vids is in the
     reached set with dist ≤ COST_CAP_MULT × haversine(A, B). Reachable
     → keep, reweight cost_m to actual road distance. Not reachable →
     drop.

Advantages over the earlier fixed-tile grid:
  - Each anchor's bbox adapts to its local candidate reach, so dense
    urban regions load only a small local subgraph — no OOM even in
    Berlin/Frankfurt.
  - When the bbox is unusually large, we split A's candidates by
    compass sector and process each sector separately (guaranteed
    smaller bbox). Never drops any edge from raw density.
  - Sequential per-anchor checkpoints so any interruption resumes.

Env vars:
  SPT_PROFILE                (default views)
  CELL_DIR                   (default /data/cells/<profile>)
  CROW_MAX_EDGE_KM           (default 100)   — crow-flies edge cap
  BIDIR_BUFFER_FRAC          (default 0.20)  — bbox expansion factor
  BIDIR_COST_CAP_MULT        (default 2.0)   — road cost ≤ mult × haversine
  BIDIR_MAX_BBOX_EDGES       (default 50000000)
                             — force sector split if bbox load exceeds
  BIDIR_CELL_CACHE_SIZE      (default 60)    — LRU cell cache slots
"""
from __future__ import annotations

import gc
import json
import math
import os
import shutil
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import psycopg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import config


PROFILE = os.environ.get("SPT_PROFILE", "views")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
IN_GRAPH_ORIG = DATA_DIR / "way_city_graph.json"
IN_GRAPH_CAND = DATA_DIR / "way_city_graph_candidates.json"
IN_ANCHORS = DATA_DIR / "way_city_anchors.geojson"
CELL_DIR = Path(os.environ.get("CELL_DIR", f"/data/cells/{PROFILE}"))
CELL_DEG = 1.0

MAX_EDGE_KM = float(os.environ.get("CROW_MAX_EDGE_KM", "100"))
MAX_EDGE_M = MAX_EDGE_KM * 1000.0
BUFFER_FRAC = float(os.environ.get("BIDIR_BUFFER_FRAC", "0.20"))
COST_CAP_MULT = float(os.environ.get("BIDIR_COST_CAP_MULT", "2.0"))
MAX_BBOX_EDGES = int(os.environ.get("BIDIR_MAX_BBOX_EDGES", "50000000"))
CELL_CACHE_SIZE = int(os.environ.get("BIDIR_CELL_CACHE_SIZE", "60"))

OUT_GRAPH   = DATA_DIR / "way_city_graph.json"
OUT_DROPPED = DATA_DIR / "way_city_graph_dropped.json"
CKPT_DIR    = DATA_DIR / "bidir_ckpt"
R_EARTH_M   = 6_371_000.0


def _haversine_m(lon1, lat1, lon2, lat2):
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
    return 2.0 * R_EARTH_M * math.asin(math.sqrt(a))


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
    """LRU cache of raw cell edge arrays keyed on (cx, cy). Values are
    dicts of numpy arrays (src, dst, fwd, rev). Total capacity in slot
    count so consecutive anchors sharing cells reuse decompressed
    data."""
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
                        "src": np.asarray(arr["src_id"]),
                        "dst": np.asarray(arr["dst_id"]),
                        "fwd": np.asarray(arr["cost"], dtype=np.float32),
                        "rev": np.asarray(arr["reverse_cost"], dtype=np.float32),
                    }
        self.misses += 1
        # LRU eviction.
        while len(self.items) > self.size:
            self.items.popitem(last=False)
        self.items.move_to_end(key)
        return self.items[key]


def _load_csr_for_bbox(bbox, cache):
    """Load cells intersecting bbox from cache. Filter negative rev
    edges. Return (csr, gid_to_local, n_edges_total). Returns
    (None, {}, 0) if the bbox is empty of edges."""
    cells = _cells_in_bbox(bbox)
    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    fwd_chunks: list[np.ndarray] = []
    rev_chunks: list[np.ndarray] = []
    n_edges_running = 0
    for cx, cy in cells:
        c = cache.get(cx, cy)
        if c is None:
            continue
        n = len(c["src"])
        n_edges_running += n
        src_chunks.append(c["src"])
        dst_chunks.append(c["dst"])
        fwd_chunks.append(c["fwd"])
        rev_chunks.append(c["rev"])
    if not src_chunks:
        return None, {}, 0
    src_gid = np.concatenate(src_chunks); del src_chunks
    dst_gid = np.concatenate(dst_chunks); del dst_chunks
    fwd = np.concatenate(fwd_chunks);     del fwd_chunks
    rev = np.concatenate(rev_chunks);     del rev_chunks
    all_gids = np.concatenate([src_gid, dst_gid])
    unique_gids, inv = np.unique(all_gids, return_inverse=True)
    del all_gids
    n_edges = len(src_gid)
    src_local = inv[:n_edges].astype(np.int32, copy=False)
    dst_local = inv[n_edges:].astype(np.int32, copy=False)
    del inv, src_gid, dst_gid
    n_verts = len(unique_gids)
    fwd_ok = fwd >= 0
    rev_ok = rev >= 0
    row = np.concatenate([src_local[fwd_ok], dst_local[rev_ok]])
    col = np.concatenate([dst_local[fwd_ok], src_local[rev_ok]])
    data = np.concatenate([fwd[fwd_ok],       rev[rev_ok]])
    del src_local, dst_local, fwd, rev, fwd_ok, rev_ok
    csr = csr_matrix((data, (row, col)), shape=(n_verts, n_verts))
    del row, col, data
    gid_to_local = {int(g): i for i, g in enumerate(unique_gids)}
    return csr, gid_to_local, n_edges


def _pair_bbox(a_city, candidates, ref_to_city, buffer_frac=BUFFER_FRAC):
    """Bounding box of A and all its candidate targets, expanded by
    buffer_frac on each side."""
    lons = [a_city["lon"]]; lats = [a_city["lat"]]
    for c in candidates:
        b = ref_to_city.get(c["b"])
        if b is None:
            continue
        lons.append(b["lon"]); lats.append(b["lat"])
    lo_lon, hi_lon = min(lons), max(lons)
    lo_lat, hi_lat = min(lats), max(lats)
    # Ensure a minimum bbox (avoid a zero-degree bbox when A and B
    # coincide, unlikely but be defensive).
    lo_lon = min(lo_lon, hi_lon - 0.1)
    hi_lon = max(hi_lon, lo_lon + 0.1)
    lo_lat = min(lo_lat, hi_lat - 0.1)
    hi_lat = max(hi_lat, lo_lat + 0.1)
    dlon = (hi_lon - lo_lon) * buffer_frac
    dlat = (hi_lat - lo_lat) * buffer_frac
    return (lo_lon - dlon, lo_lat - dlat, hi_lon + dlon, hi_lat + dlat)


def _sector_split(a_city, candidates, ref_to_city, n_sectors=2):
    """Split candidates into n_sectors compass sectors around A. Used
    to shrink the bbox when a full-candidate bbox overloads memory."""
    groups: list[list[dict]] = [[] for _ in range(n_sectors)]
    width = 360.0 / n_sectors
    for c in candidates:
        b = ref_to_city.get(c["b"])
        if b is None:
            groups[0].append(c)
            continue
        bear = _bearing_deg(a_city["lon"], a_city["lat"],
                            b["lon"], b["lat"])
        groups[int(bear / width) % n_sectors].append(c)
    return [g for g in groups if g]


def _process_anchor(a_city, candidates, ref_to_city, cache, kept,
                    dropped, depth=0, max_depth=4):
    """Bounded dijkstra from A over a bbox covering A + candidates.
    On MAX_BBOX_EDGES overrun, split by sector and recurse. Depth-limited
    to avoid pathological infinite splits."""
    # Split off pier↔pier ferry edges and pass them straight to `kept`
    # — sea routes exceed 2× straight-line so the road-reachability
    # test can't validate them, and they're semantically guaranteed
    # (real ferry way between the two piers). Pier↔land edges DO get
    # tested (heuristic KDTree links might fail on real road access).
    a_is_pier = a_city["ref"].startswith("ferry:")
    testable: list[dict] = []
    for e in candidates:
        both_piers = a_is_pier and e["b"].startswith("ferry:")
        if both_piers:
            kept.append(e)
        else:
            testable.append(e)
    if not testable:
        return
    candidates = testable

    bbox = _pair_bbox(a_city, candidates, ref_to_city)
    csr, gid_to_local, n_edges = _load_csr_for_bbox(bbox, cache)
    if csr is None:
        for e in candidates:
            dropped.append({**e, "_reason": "empty subgraph"})
        return
    if n_edges > MAX_BBOX_EDGES and depth < max_depth:
        del csr, gid_to_local
        gc.collect()
        groups = _sector_split(a_city, candidates, ref_to_city,
                               n_sectors=2 ** (depth + 1))
        for grp in groups:
            _process_anchor(a_city, grp, ref_to_city, cache,
                            kept, dropped, depth=depth + 1,
                            max_depth=max_depth)
        return

    a_snap = int(a_city["snap_vertex_id"])
    a_local = gid_to_local.get(a_snap)
    if a_local is None:
        for e in candidates:
            dropped.append({**e, "_reason": "source snap not in bbox"})
        return

    # Cost cap = COST_CAP_MULT × max haversine to any candidate.
    hav_by_edge = []
    for e in candidates:
        b = ref_to_city.get(e["b"])
        if b is None:
            hav_by_edge.append(0.0)
            continue
        hav_by_edge.append(_haversine_m(
            a_city["lon"], a_city["lat"], b["lon"], b["lat"]))
    cap = COST_CAP_MULT * max(hav_by_edge) if hav_by_edge else 0.0

    dist = dijkstra(csr, indices=a_local, limit=cap,
                    return_predecessors=False)
    del csr

    for e, hav in zip(candidates, hav_by_edge):
        b = ref_to_city.get(e["b"])
        if b is None:
            dropped.append({**e, "_reason": "target not in cities"})
            continue
        b_snaps = b.get("snap_vids") or [b["snap_vertex_id"]]
        best = float("inf")
        for b_snap in b_snaps:
            loc = gid_to_local.get(int(b_snap))
            if loc is None:
                continue
            d = float(dist[loc])
            if d < best:
                best = d
        if not math.isfinite(best) or best > COST_CAP_MULT * hav:
            dropped.append({**e, "_reason": f"unreachable within {COST_CAP_MULT}×hav"})
            continue
        kept.append({**e, "cost_m": best})
    del dist, gid_to_local
    gc.collect()


def main() -> None:
    t0 = time.time()
    print(f"[bidir-filter] profile={PROFILE}  buffer={BUFFER_FRAC:.0%}  "
          f"cap={COST_CAP_MULT}×hav  max_bbox_edges={MAX_BBOX_EDGES:,}  "
          f"cell_cache={CELL_CACHE_SIZE}", flush=True)

    if not IN_GRAPH_CAND.exists():
        shutil.copyfile(IN_GRAPH_ORIG, IN_GRAPH_CAND)
        print(f"[bidir-filter] snapshotted candidate graph → "
              f"{IN_GRAPH_CAND.name}", flush=True)
    edges: list[dict] = json.loads(IN_GRAPH_CAND.read_text())
    anchors_fc = json.loads(IN_ANCHORS.read_text())

    ref_to_city: dict[str, dict] = {}
    for f in anchors_fc["features"]:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        ref_to_city[p["ref"]] = {
            "ref": p["ref"], "name": p.get("name"),
            "lon": float(lon), "lat": float(lat),
            "snap_vertex_id": None, "snap_vids": None,
        }
    for ref, c in ref_to_city.items():
        if ref.startswith("ferry:"):
            vid = int(ref.split(":", 1)[1])
            c["snap_vertex_id"] = vid
            c["snap_vids"] = [vid]

    db_ids = [int(r.split(":", 1)[1]) for r in ref_to_city
              if r.startswith("db:")]
    if db_ids:
        print(f"[bidir-filter] querying postgres for "
              f"{len(db_ids):,} db anchor snap vertices…", flush=True)
        with psycopg.connect(config.PG_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, snap_vertex_id FROM anchors "
                    "WHERE id = ANY(%s) AND snap_vertex_id IS NOT NULL",
                    (db_ids,),
                )
                for aid, snap_vid in cur.fetchall():
                    ref = f"db:{aid}"
                    if ref in ref_to_city:
                        ref_to_city[ref]["snap_vertex_id"] = int(snap_vid)
                        ref_to_city[ref]["snap_vids"] = [int(snap_vid)]

    unsnapped = [(ref, c) for ref, c in ref_to_city.items()
                 if c["snap_vertex_id"] is None]
    if unsnapped:
        print(f"[bidir-filter] snapping {len(unsnapped):,} villages via "
              f"postgres KNN…", flush=True)
        with psycopg.connect(config.PG_DSN) as conn:
            with conn.cursor() as cur:
                refs = [ref for ref, _ in unsnapped]
                lons = [c["lon"] for _, c in unsnapped]
                lats = [c["lat"] for _, c in unsnapped]
                cur.execute(
                    """
                    SELECT v.ref, snap.id
                    FROM UNNEST(%s::text[], %s::float8[], %s::float8[])
                         AS v(ref, lon, lat)
                    CROSS JOIN LATERAL (
                      SELECT id FROM ways_vertices_pgr
                      ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint(v.lon, v.lat), 4326)
                      LIMIT 1
                    ) snap
                    """, (refs, lons, lats))
                for ref, snap_vid in cur.fetchall():
                    ref_to_city[ref]["snap_vertex_id"] = int(snap_vid)
                    ref_to_city[ref]["snap_vids"] = [int(snap_vid)]

    edges_by_source = defaultdict(list)
    for e in edges:
        edges_by_source[e["a"]].append(e)

    # Sort sources by tile of their lonlat, so consecutive anchors are
    # geographically near — cell LRU cache hits.
    sources_sorted = sorted(
        edges_by_source.keys(),
        key=lambda ref: (
            int(math.floor(ref_to_city[ref]["lon"])),
            int(math.floor(ref_to_city[ref]["lat"])),
        ) if ref in ref_to_city else (0, 0),
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []
    dropped: list[dict] = []
    cache = CellCache(CELL_CACHE_SIZE)

    print(f"[bidir-filter] {len(edges):,} candidate edges, "
          f"{len(sources_sorted):,} source anchors", flush=True)

    n_processed = 0
    for a_ref in sources_sorted:
        a_city = ref_to_city.get(a_ref)
        if a_city is None or a_city["snap_vertex_id"] is None:
            for e in edges_by_source[a_ref]:
                # Never drop a pier↔pier ferry edge just because we
                # can't snap the source vertex — the ferry way itself
                # is the ground-truth connection.
                if (a_ref.startswith("ferry:")
                        and e["b"].startswith("ferry:")):
                    kept.append(e)
                else:
                    dropped.append({**e, "_reason": "no snap for source"})
            n_processed += 1
            continue
        # Per-anchor checkpoint keyed on ref (URL-safe).
        safe_ref = a_ref.replace("/", "_").replace(":", "-")
        ckpt_path = CKPT_DIR / f"anchor_{safe_ref}.json"
        if ckpt_path.exists():
            data = json.loads(ckpt_path.read_text())
            kept.extend(data.get("kept", []))
            dropped.extend(data.get("dropped", []))
            n_processed += 1
            continue
        len_kept_before = len(kept)
        len_dropped_before = len(dropped)
        t_a = time.time()
        _process_anchor(a_city, edges_by_source[a_ref], ref_to_city,
                        cache, kept, dropped)
        ckpt_path.write_text(json.dumps({
            "anchor":  a_ref,
            "kept":    kept[len_kept_before:],
            "dropped": dropped[len_dropped_before:],
        }, ensure_ascii=False))
        n_processed += 1
        if n_processed % 50 == 0:
            hit_rate = 100 * cache.hits / max(cache.hits + cache.misses, 1)
            print(f"[bidir-filter]   {n_processed}/{len(sources_sorted)} "
                  f"anchors  |  kept {len(kept):,}  dropped {len(dropped):,}  "
                  f"|  cache hit {hit_rate:.0f}%  "
                  f"|  {time.time()-t0:.0f}s elapsed  last {time.time()-t_a:.1f}s",
                  flush=True)

    print(f"[bidir-filter] DONE in {time.time()-t0:.1f}s — "
          f"kept {len(kept):,} / {len(edges):,} "
          f"({100*len(kept)//max(len(edges),1)}%), "
          f"dropped {len(dropped):,}", flush=True)
    OUT_GRAPH.write_text(json.dumps(kept, ensure_ascii=False))
    OUT_DROPPED.write_text(json.dumps(dropped, ensure_ascii=False))
    print(f"[bidir-filter] wrote {OUT_GRAPH}", flush=True)


if __name__ == "__main__":
    main()
