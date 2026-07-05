"""Filter candidate chain edges by road-network reachability + reweight.

Approach
--------
For each candidate edge (A, B) in the crow-flies chain graph, verify
that there's an actual bike-routable road path A → B with cost ≤
2 × haversine(A, B). If yes, keep the edge and set weight to the real
road distance. If no, drop it.

Bounds the reachability test at 2× straight-line distance so a
coastal / island anchor doesn't get a "chain-neighbor" 100 km away
across water (path exists only via a long ferry detour that no bike
route would sensibly take).

Scalable via tiling
-------------------
Never loads the full road graph. Tiles the anchor set into 3°×3°
cores with a 2° buffer, and processes each tile independently:

1. Load cells intersecting the tile's buffered bbox into a scipy CSR
   (~1-3 GB per tile regardless of total geography size).
2. Group candidate edges by source anchor whose location falls in
   this tile's core.
3. For each such source anchor A, run one bounded Dijkstra from A's
   snap vertex with `limit = 2 × max(haversine(A, B) for B in
   A's candidates)`. Amortizes the search across all of A's candidates.
4. For each candidate (A, B): look up min(dist[B_snap] for B_snap in
   B.snap_vids). If ≤ 2 × haversine(A, B) → keep, reweight; else drop.
5. Free the tile's CSR before moving to the next tile.

Buffer of 2° at these latitudes ≈ 200 km — always exceeds the max
possible cost cap (2 × MAX_EDGE_KM = 200 km) so bounded Dijkstra from
any source in the core cannot escape the loaded region.

Runs deterministically in memory bounded by tile buffer size. Works at
4-country scale today and 40-country scale later without change.

Env vars
--------
- SPT_PROFILE                (default views)
- CELL_DIR                   (default /data/cells/<profile>)
- CROW_MAX_EDGE_KM           (default 100)  — from crow_flies_chain_graph
- BIDIR_TILE_CORE_DEG        (default 3.0)
- BIDIR_TILE_BUFFER_DEG      (default 2.0)
- BIDIR_COST_CAP_MULT        (default 2.0)  — road cost ≤ mult × haversine
"""
from __future__ import annotations

import gc
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import config


PROFILE = os.environ.get("SPT_PROFILE", "views")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
IN_GRAPH = DATA_DIR / "way_city_graph.json"
IN_ANCHORS = DATA_DIR / "way_city_anchors.geojson"
CELL_DIR = Path(os.environ.get("CELL_DIR", f"/data/cells/{PROFILE}"))
CELL_DEG = 1.0

MAX_EDGE_KM = float(os.environ.get("CROW_MAX_EDGE_KM", "100"))
MAX_EDGE_M = MAX_EDGE_KM * 1000.0
CORE_DEG = float(os.environ.get("BIDIR_TILE_CORE_DEG", "3.0"))
BUFFER_DEG = float(os.environ.get("BIDIR_TILE_BUFFER_DEG", "2.0"))
COST_CAP_MULT = float(os.environ.get("BIDIR_COST_CAP_MULT", "2.0"))
# Skip tiles whose loaded subgraph exceeds this many edges. Big tiles
# push CSR + Dijkstra memory past our 11 GB WSL VM budget. Skipped
# tiles' edges are marked dropped-with-reason so a later rerun with
# smaller cores can fill them in.
MAX_TILE_EDGES = int(os.environ.get("BIDIR_MAX_TILE_EDGES", "70000000"))

OUT_GRAPH = DATA_DIR / "way_city_graph.json"
OUT_DROPPED = DATA_DIR / "way_city_graph_dropped.json"
# Per-tile checkpoint dir. Each tile's results are saved to its own
# file so an OOM mid-run doesn't lose completed work — just relaunch
# and completed tiles are skipped.
CKPT_DIR = DATA_DIR / "bidir_ckpt"
R_EARTH_M = 6_371_000.0


def _haversine_m(lon1, lat1, lon2, lat2):
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
    return 2.0 * R_EARTH_M * math.asin(math.sqrt(a))


def _tile_of(lon: float, lat: float) -> tuple[int, int]:
    return int(math.floor(lon / CORE_DEG)), int(math.floor(lat / CORE_DEG))


def _cells_in_bbox(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    lo_lon, lo_lat, hi_lon, hi_lat = bbox
    cx0 = int(math.floor(lo_lon / CELL_DEG))
    cy0 = int(math.floor(lo_lat / CELL_DEG))
    cx1 = int(math.ceil(hi_lon / CELL_DEG))
    cy1 = int(math.ceil(hi_lat / CELL_DEG))
    return [(cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1)]


def _load_csr_for_bbox(bbox: tuple[float, float, float, float]):
    """Load all bike-routable edges in cells intersecting bbox. Return
    (csr, gid_to_local) where csr is a scipy sparse forward-cost matrix
    over the LOCAL vertex indexing.
    """
    # Skip huge tiles early — bail BEFORE materializing the edge arrays.
    # This prevents OOM during load itself. We peek at each cell's edge
    # count and abort once the running total exceeds the tile cap.
    max_edges = int(os.environ.get("BIDIR_MAX_TILE_EDGES", "70000000"))
    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    fwd_chunks: list[np.ndarray] = []
    rev_chunks: list[np.ndarray] = []
    n_cells_hit = 0
    running_edges = 0
    for cx, cy in _cells_in_bbox(bbox):
        path = CELL_DIR / f"{cx}_{cy}.npz"
        if not path.exists():
            continue
        n_cells_hit += 1
        with np.load(path, mmap_mode="r") as z:
            arr = z["edges"]
            if len(arr) == 0:
                continue
            running_edges += len(arr)
            if running_edges > max_edges:
                # Return special sentinel so caller can log + skip.
                return "TILE_TOO_BIG", {}, n_cells_hit
            src_chunks.append(np.asarray(arr["src_id"]))
            dst_chunks.append(np.asarray(arr["dst_id"]))
            fwd_chunks.append(np.asarray(arr["cost"], dtype=np.float32))
            rev_chunks.append(np.asarray(arr["reverse_cost"], dtype=np.float32))
    if not src_chunks:
        return None, {}, 0
    src_gid = np.concatenate(src_chunks); del src_chunks
    dst_gid = np.concatenate(dst_chunks); del dst_chunks
    fwd = np.concatenate(fwd_chunks);     del fwd_chunks
    rev = np.concatenate(rev_chunks);     del rev_chunks
    # Intern global vertex IDs to local 0..N-1 (int32 fits — n_verts
    # per tile is < 2 billion).
    all_gids = np.concatenate([src_gid, dst_gid])
    unique_gids, inv = np.unique(all_gids, return_inverse=True)
    del all_gids
    n_edges = len(src_gid)
    src_local = inv[:n_edges].astype(np.int32, copy=False)
    dst_local = inv[n_edges:].astype(np.int32, copy=False)
    del inv, src_gid, dst_gid
    n_verts = len(unique_gids)
    # Build directed CSR with both fwd (src→dst) and rev (dst→src, with
    # the reverse_cost from cell) entries. Cells encode "one-way, can't
    # bike this direction" as negative reverse_cost; filter those out
    # so dijkstra never sees negative weights.
    fwd_ok = fwd >= 0
    rev_ok = rev >= 0
    row = np.concatenate([src_local[fwd_ok], dst_local[rev_ok]])
    col = np.concatenate([dst_local[fwd_ok], src_local[rev_ok]])
    data = np.concatenate([fwd[fwd_ok],       rev[rev_ok]])
    del src_local, dst_local, fwd, rev, fwd_ok, rev_ok
    csr = csr_matrix((data, (row, col)), shape=(n_verts, n_verts))
    del row, col, data
    gid_to_local = {int(g): i for i, g in enumerate(unique_gids)}
    return csr, gid_to_local, n_cells_hit


def main() -> None:
    t0 = time.time()
    print(f"[bidir-filter] profile={PROFILE}  core={CORE_DEG}° buffer={BUFFER_DEG}°  "
          f"cap={COST_CAP_MULT}×haversine", flush=True)

    edges: list[dict] = json.loads(IN_GRAPH.read_text())
    anchors_fc = json.loads(IN_ANCHORS.read_text())
    # Build ref → {lon, lat, snap_vertex_id} from postgres for db anchors
    # and directly from ref for ferry piers ("ferry:<vid>" = vid IS the
    # snap vertex on the ferry way).
    ref_to_city: dict[str, dict] = {}
    for f in anchors_fc["features"]:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        ref_to_city[p["ref"]] = {
            "ref": p["ref"], "name": p.get("name"),
            "lon": float(lon), "lat": float(lat),
            "snap_vertex_id": None, "snap_vids": None,
        }
    # Ferry piers: ref = "ferry:<vid>" where vid is the snap vertex.
    for ref, c in ref_to_city.items():
        if ref.startswith("ferry:"):
            vid = int(ref.split(":", 1)[1])
            c["snap_vertex_id"] = vid
            c["snap_vids"] = [vid]

    # db anchors: query postgres anchors.snap_vertex_id.
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
    # Snap remaining anchors (villages, mostly) via postgres KNN on
    # ways_vertices_pgr geom index.
    unsnapped = [(ref, c) for ref, c in ref_to_city.items()
                 if c["snap_vertex_id"] is None]
    if unsnapped:
        print(f"[bidir-filter] snapping {len(unsnapped):,} remaining anchors "
              f"(villages) via postgres KNN…", flush=True)
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
                    """,
                    (refs, lons, lats),
                )
                for ref, snap_vid in cur.fetchall():
                    ref_to_city[ref]["snap_vertex_id"] = int(snap_vid)
                    ref_to_city[ref]["snap_vids"] = [int(snap_vid)]
    n_no_snap = sum(1 for c in ref_to_city.values()
                    if c["snap_vertex_id"] is None)
    if n_no_snap:
        print(f"[bidir-filter] WARN: {n_no_snap:,} anchors still have no "
              f"snap_vertex_id — will drop their edges", flush=True)

    print(f"[bidir-filter] {len(edges):,} candidate edges, "
          f"{len(ref_to_city):,} anchors", flush=True)

    # For each edge, decide which tile "owns" this source A.
    # We tile by source's lon/lat.
    # Then group edges by source ref, then by tile.
    edges_by_source: dict[str, list[dict]] = defaultdict(list)
    for e in edges:
        edges_by_source[e["a"]].append(e)
    # Anchor tile.
    source_tiles: dict[str, tuple[int, int]] = {}
    for a_ref in edges_by_source:
        c = ref_to_city.get(a_ref)
        if c is None:
            continue
        source_tiles[a_ref] = _tile_of(float(c["lon"]), float(c["lat"]))

    tiles: dict[tuple[int, int], list[str]] = defaultdict(list)
    for a_ref, tile in source_tiles.items():
        tiles[tile].append(a_ref)

    print(f"[bidir-filter] {len(tiles)} non-empty source tiles", flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []
    dropped: list[dict] = []
    for ti, (tile, anchors_here) in enumerate(sorted(tiles.items()), 1):
        tile_lon, tile_lat = tile
        ckpt_path = CKPT_DIR / f"tile_{tile_lon:+04d}_{tile_lat:+04d}.json"
        if ckpt_path.exists():
            data = json.loads(ckpt_path.read_text())
            kept.extend(data.get("kept", []))
            dropped.extend(data.get("dropped", []))
            print(f"[bidir-filter]   tile {ti}/{len(tiles)} {tile}: "
                  f"RESUMED from checkpoint "
                  f"(kept {len(data.get('kept', []))}, "
                  f"dropped {len(data.get('dropped', []))})",
                  flush=True)
            continue
        core = (tile_lon * CORE_DEG, tile_lat * CORE_DEG,
                (tile_lon + 1) * CORE_DEG, (tile_lat + 1) * CORE_DEG)
        buf = (core[0] - BUFFER_DEG, core[1] - BUFFER_DEG,
               core[2] + BUFFER_DEG, core[3] + BUFFER_DEG)
        t_tile = time.time()
        len_kept_before = len(kept)
        len_dropped_before = len(dropped)
        loaded = _load_csr_for_bbox(buf)
        if loaded is None or loaded[0] is None:
            print(f"[bidir-filter]   tile {ti}/{len(tiles)} {tile}: "
                  f"empty subgraph — skipping ({len(anchors_here)} anchors)",
                  flush=True)
            for a_ref in anchors_here:
                for e in edges_by_source[a_ref]:
                    dropped.append({**e, "_reason": "empty tile subgraph"})
            ckpt_path.write_text(json.dumps({
                "tile":    [tile_lon, tile_lat],
                "kept":    [],
                "dropped": dropped[len_dropped_before:],
            }, ensure_ascii=False))
            continue
        # Early-exit sentinel from the loader when the tile would blow
        # our memory budget. Defer the tile's edges and move on.
        # (`isinstance` avoids scipy CSR's overloaded __eq__ which does
        # elementwise comparison and blows up on string args.)
        if isinstance(loaded[0], str) and loaded[0] == "TILE_TOO_BIG":
            n_cells_seen = loaded[2] if isinstance(loaded, tuple) else 0
            print(f"[bidir-filter]   tile {ti}/{len(tiles)} {tile}: "
                  f"SKIPPED — would exceed MAX_TILE_EDGES "
                  f"(saw {n_cells_seen} cells before bail) — "
                  f"{len(anchors_here)} anchors deferred",
                  flush=True)
            for a_ref in anchors_here:
                for e in edges_by_source[a_ref]:
                    dropped.append({**e, "_reason": "tile too big at load"})
            ckpt_path.write_text(json.dumps({
                "tile":    [tile_lon, tile_lat],
                "kept":    [],
                "dropped": dropped[len_dropped_before:],
            }, ensure_ascii=False))
            gc.collect()
            continue
        csr, gid_to_local, n_cells_hit = loaded
        n_edges = csr.nnz // 2
        if n_edges > MAX_TILE_EDGES:
            print(f"[bidir-filter]   tile {ti}/{len(tiles)} {tile}: "
                  f"SKIPPED — {n_edges:,} edges > {MAX_TILE_EDGES:,} "
                  f"cap ({len(anchors_here)} anchors deferred)",
                  flush=True)
            for a_ref in anchors_here:
                for e in edges_by_source[a_ref]:
                    dropped.append({**e, "_reason": "tile too big"})
            ckpt_path.write_text(json.dumps({
                "tile":    [tile_lon, tile_lat],
                "kept":    [],
                "dropped": dropped[len_dropped_before:],
            }, ensure_ascii=False))
            del csr, gid_to_local
            gc.collect()
            continue
        print(f"[bidir-filter]   tile {ti}/{len(tiles)} {tile}: "
              f"{csr.shape[0]:,} verts, {n_edges:,} edges, "
              f"{n_cells_hit} cells, {len(anchors_here)} sources, "
              f"loaded in {time.time()-t_tile:.1f}s", flush=True)

        # For each source anchor in this tile core, run bounded Dijkstra.
        for a_ref in anchors_here:
            a_city = ref_to_city[a_ref]
            a_snap = int(a_city["snap_vertex_id"])
            a_lon = float(a_city["lon"]); a_lat = float(a_city["lat"])
            a_local = gid_to_local.get(a_snap)
            candidates = edges_by_source[a_ref]

            if a_local is None:
                # Source's snap vid isn't in this tile's subgraph
                # (shouldn't happen if bbox+buffer covers core, but be
                # defensive).
                for e in candidates:
                    dropped.append({**e, "_reason": "source snap not in cells"})
                continue

            # Cost cap = COST_CAP_MULT × max haversine to any candidate.
            hav_by_edge: list[float] = []
            for e in candidates:
                b_city = ref_to_city.get(e["b"])
                if b_city is None:
                    hav_by_edge.append(0.0)
                    continue
                hav_by_edge.append(_haversine_m(
                    a_lon, a_lat,
                    float(b_city["lon"]), float(b_city["lat"])))
            cap = COST_CAP_MULT * max(hav_by_edge)

            dist = dijkstra(csr, indices=a_local, limit=cap,
                            return_predecessors=False)
            # numpy allocates a fresh 20M-float array per call; help the
            # allocator by dropping the previous ref before the next
            # iteration.
            gc.collect()

            for e, hav in zip(candidates, hav_by_edge):
                b_city = ref_to_city.get(e["b"])
                if b_city is None:
                    dropped.append({**e, "_reason": "target not in cities"})
                    continue
                b_snaps = b_city.get("snap_vids") or [b_city["snap_vertex_id"]]
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

        # Save checkpoint for this tile before freeing memory. If a
        # future tile OOMs, relaunch and completed tiles resume.
        ckpt_path.write_text(json.dumps({
            "tile":    [tile_lon, tile_lat],
            "kept":    kept[len_kept_before:],
            "dropped": dropped[len_dropped_before:],
        }, ensure_ascii=False))
        # Free before next tile.
        del csr, gid_to_local
        gc.collect()
        print(f"[bidir-filter]     tile done in {time.time()-t_tile:.1f}s — "
              f"kept {len(kept):,}, dropped {len(dropped):,} so far  "
              f"(checkpoint: {ckpt_path.name})",
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
