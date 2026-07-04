"""Pair-Dijkstra chain edges over a paved + unclassified subgraph.

Replaces the Voronoi-on-fragmented-subgraph chain graph with explicit
per-pair shortest-path search. For each anchor A, we find its K nearest
other anchors by euclidean distance and run a bounded Dijkstra from A's
paved-snap to each B's paved-snap. A chain edge (A, B) exists iff the
search finds a finite-cost path.

The paved subgraph (motorway + trunk + primary + secondary + tertiary +
unclassified — i.e., every reasonably drivable class) is much denser
than the curated way-graph subgraph and is robust against OSM tagging
gaps. The Salzachklamm-style breaks vanish because at least one tagged
class crosses them. Skipping residential/service/track keeps Dijkstra
from routing through cul-de-sacs and parking lots.

The cost cap (LIMIT_RATIO × euclidean) keeps each SSSP cheap and
prevents wraparound routes for genuinely-unreachable pairs.

Outputs (replaces the static build's chain graph):
  /data/way_city_graph.json
  /data/way_city_graph.geojson

Anchors GeoJSON is reused as-is from the prior selection (e.g., the
bottom-up output). This script does not re-select anchors.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import psycopg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial import cKDTree

import config


# 16 overlapping 45°-wide sectors (centers spaced 22.5° apart). For each
# sector, the nearest anchor whose 5 km disc intersects the sector wedge
# is a chain candidate. Overlap (22.5°) plus disc tolerance prevents
# anchors near a sector boundary from being missed.
N_SECTORS         = 16
SECTOR_WIDTH_DEG  = 45.0
DISC_RADIUS_M     = 5_000.0
# Hard cap on directional-neighbor distance. Doesn't matter for Austria
# (everywhere has anchors within ~50 km), but stops the algorithm from
# proposing a phantom chain edge across hundreds of km of Outback /
# Nevada / Yukon / Sahara where the next settlement is too far for a
# bike tour to plausibly hop. 300 km ≈ a long day's drive; longer than
# any reasonable single-segment bike route.
MAX_ARC_M         = 300_000.0

# Every drivable highway class (excluding bike-unfriendly motorway_link
# and trunk_link only because those are stripped at ingest along with
# motorway). Residential + *_link classes are needed because OSM
# downgrades major roads through towns, and link slip roads bridge
# named-road components. Skipping service/track/footway/path/etc. so
# routes don't snake through parking lots.
PAVED_HIGHWAYS  = ("motorway", "trunk", "primary", "secondary",
                   "tertiary", "unclassified", "residential",
                   "living_street",
                   "primary_link", "secondary_link", "tertiary_link",
                   "trunk_link", "motorway_link", "road")
# Multi-source attach: every drivable vertex within this radius of an
# anchor's centroid becomes a zero-cost source for the SSSP. Robust
# against the "anchor snapped to the wrong stub" failure, since the
# super-source effectively lets the search start from anywhere in town.
SNAP_RADIUS_M   = 1000.0
# Geographic component-bridging on the paved subgraph itself. The OSM
# source data is continuous along major roads, but osm2pgrouting drops
# the occasional segment (typically a bridge/tunnel/roundabout that
# acquires a foot=no or access=no tag), leaving small disconnected
# islands. For any two paved vertices within this distance of each
# other but in different connected components, add a zero-cost phantom
# edge. 50 m is short enough that we only bridge "missing single
# segment" gaps, not unrelated parallel roads.
COMPONENT_BRIDGE_M = 200.0
LIMIT_RATIO     = 2.0        # Dijkstra cost limit = ratio × euclidean(A,B)
# Indirect-edge filter: drop chain edge (A, B) if its path passes within
# this distance of any third anchor C. C is then "on the route" and
# A→C, C→B are the proper chain edges. 2 km is tight enough to avoid
# false positives where C is just spatially close to the corridor but
# not actually on it.
INDIRECT_THRESHOLD_M = 2_000.0

# Spatial tiling. The full paved subgraph (~36M edges across 4 countries,
# more at continental scale) doesn't fit in RAM as a Python list of
# tuples — the streaming load peaked ~14 GB and OOM-killed WSL. Instead
# we partition anchors into geographic tiles and process each tile with
# a bbox-filtered SELECT that pulls only edges in (tile + MAX_ARC_M
# buffer). Per-tile peak memory is O(tile area) and independent of how
# many countries we add.
#
# Tile size = 3° (~330 km lat, ~200-240 km lon at central-European
# latitudes). Buffer = MAX_ARC_M in degrees ≈ 2.7°, rounded up to 3°.
# Loaded bbox per tile is ~9°×9° (~700-900 km square). Empirically a
# 4-country graph at this scale loads ~5-12M edges per tile (~1-2 GB
# numpy peak), which fits inside Postgres + Python + bike-api alongside.
ANCHOR_TILE_SIZE_DEG = 3.0
BBOX_BUFFER_DEG      = 3.0

OUT_DIR             = Path("/data")
OUT_GRAPH_JSON      = OUT_DIR / "way_city_graph.json"
OUT_GRAPH_GEOJSON   = OUT_DIR / "way_city_graph.geojson"
ANCHORS_GEOJSON_IN  = OUT_DIR / "way_city_anchors.geojson"

R_EARTH_M = 6_371_000.0


def _lonlat_to_xyz(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    coslat = np.cos(lat_r)
    return np.column_stack([
        R_EARTH_M * coslat * np.cos(lon_r),
        R_EARTH_M * coslat * np.sin(lon_r),
        R_EARTH_M * np.sin(lat_r),
    ])


def _chord_for_arc(arc_m: float) -> float:
    return 2.0 * R_EARTH_M * np.sin(arc_m / (2.0 * R_EARTH_M))


def _load_paved_subgraph_bbox(conn: psycopg.Connection,
                              bbox: tuple[float, float, float, float]):
    """Streamed pull of paved-class edges whose source vertex lies in
    `bbox = (lon_min, lat_min, lon_max, lat_max)`. The source-vertex
    filter is sufficient to load every edge needed for SSSP from any
    anchor inside the bbox's inner core, provided the bbox includes a
    MAX_ARC_M buffer — an edge whose source is just outside the bbox
    can't be on the shortest path of an SSSP that's already exceeded
    the limit by the time it would reach that source.

    Streams directly into per-batch numpy chunks (no Python tuple list)
    to keep peak memory ~3-4x lower than the prior implementation.

    Returns:
      src, dst:    (n_edges,) int64 local vertex indices
      length:      (n_edges,) float64 meters
      verts:       (n_verts, 2) float64 (lon, lat)
    """
    t = time.time()
    # Read from the pre-baked ways_paved table (see build_ways_paved.py).
    # ways_paved already has the highway-filter applied and the JOINs
    # denormalized inline, with a GIST index on src_pt. Per-tile cost
    # drops from ~50 min (against the live JOIN) to seconds.
    sql = """
        SELECT src_id, dst_id, length_m,
               src_lon, src_lat, dst_lon, dst_lat
        FROM ways_paved
        WHERE src_pt && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
          AND ST_Contains(ST_MakeEnvelope(%s, %s, %s, %s, 4326), src_pt)
    """
    params = [bbox[0], bbox[1], bbox[2], bbox[3],
              bbox[0], bbox[1], bbox[2], bbox[3]]

    # Stream into list-of-batch-numpy-chunks. Each batch becomes a
    # numpy array immediately so Python tuple overhead never crosses
    # batch boundaries.
    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    len_chunks: list[np.ndarray] = []
    sx_chunks:  list[np.ndarray] = []
    sy_chunks:  list[np.ndarray] = []
    tx_chunks:  list[np.ndarray] = []
    ty_chunks:  list[np.ndarray] = []

    total = 0
    with conn.cursor(name="paved_subgraph_cursor_bbox") as cur:
        cur.itersize = 200_000
        cur.execute(sql, params)
        while True:
            batch = cur.fetchmany(200_000)
            if not batch:
                break
            n = len(batch)
            src_chunks.append(np.fromiter((r[0] for r in batch), dtype=np.int64, count=n))
            dst_chunks.append(np.fromiter((r[1] for r in batch), dtype=np.int64, count=n))
            len_chunks.append(np.fromiter((r[2] for r in batch), dtype=np.float64, count=n))
            sx_chunks.append(np.fromiter((r[3] for r in batch), dtype=np.float64, count=n))
            sy_chunks.append(np.fromiter((r[4] for r in batch), dtype=np.float64, count=n))
            tx_chunks.append(np.fromiter((r[5] for r in batch), dtype=np.float64, count=n))
            ty_chunks.append(np.fromiter((r[6] for r in batch), dtype=np.float64, count=n))
            total += n

    if total == 0:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty((0, 2), dtype=np.float64))

    src_gid = np.concatenate(src_chunks); src_chunks.clear()
    dst_gid = np.concatenate(dst_chunks); dst_chunks.clear()
    length  = np.concatenate(len_chunks); len_chunks.clear()
    sx      = np.concatenate(sx_chunks);  sx_chunks.clear()
    sy      = np.concatenate(sy_chunks);  sy_chunks.clear()
    tx      = np.concatenate(tx_chunks);  tx_chunks.clear()
    ty      = np.concatenate(ty_chunks);  ty_chunks.clear()

    print(f"[pairs] bbox load: {total:,} edges in {time.time()-t:.1f}s "
          f"(bbox lon {bbox[0]:.1f}..{bbox[2]:.1f}, "
          f"lat {bbox[1]:.1f}..{bbox[3]:.1f})", flush=True)

    # Intern global vertex IDs (the postgres `ways.source`/`ways.target`
    # gids) into a contiguous local index space [0, n_verts). Numpy-only
    # — np.unique gives a sorted gid array and per-row inverse indices
    # in one O(N log N) call.
    all_gid = np.concatenate([src_gid, dst_gid])
    unique_gid, inverse = np.unique(all_gid, return_inverse=True)
    n_edges = len(src_gid)
    src_local = inverse[:n_edges].astype(np.int64, copy=False)
    dst_local = inverse[n_edges:].astype(np.int64, copy=False)
    del all_gid, inverse, src_gid, dst_gid

    # Per-unique-gid (lon, lat). Take the source-side coords when the
    # gid appears as a source, target-side otherwise. Avoid a Python
    # loop by indexing via np.searchsorted.
    n_verts = len(unique_gid)
    verts = np.empty((n_verts, 2), dtype=np.float64)
    # First write target coords (so they exist for vertices that only
    # appear as targets), then overwrite with source coords (preferred).
    verts[dst_local, 0] = tx; verts[dst_local, 1] = ty
    verts[src_local, 0] = sx; verts[src_local, 1] = sy

    return src_local, dst_local, length, verts


def _bearings_deg(lat1: np.ndarray, lon1: np.ndarray,
                  lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized initial bearing from (lat1, lon1) to (lat2, lon2), in
    degrees [0, 360). All inputs in degrees."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360.0) % 360.0


def _sector_neighbors(ai: int, lats: np.ndarray, lons: np.ndarray,
                      dist_m: np.ndarray) -> list[int]:
    """For anchor `ai`, pick at most one neighbor per sector. Returns the
    deduped union across all N_SECTORS sectors.

    A candidate j qualifies for sector S if the angular distance from
    bearing(ai → j) to the sector center is ≤ (SECTOR_WIDTH_DEG/2) + α,
    where α is the half-angle subtended at ai by j's DISC_RADIUS_M disc.
    The disc tolerance prevents anchors that sit just outside a sector
    boundary from being missed.
    """
    bearings = _bearings_deg(
        np.full(len(lats), lats[ai]),
        np.full(len(lons), lons[ai]),
        lats, lons,
    )
    # Disc half-angle as seen from ai. For nearby anchors the disc
    # subtends a large angle (we cap at 90°). Avoid divide-by-zero at
    # self by clipping distance.
    safe_d = np.maximum(dist_m, 1.0)
    disc_sin = np.clip(DISC_RADIUS_M / safe_d, 0.0, 1.0)
    disc_half_deg = np.degrees(np.arcsin(disc_sin))

    half_w = SECTOR_WIDTH_DEG / 2.0
    sector_centers = np.arange(N_SECTORS) * (360.0 / N_SECTORS)

    # Hard cap: don't consider anchors beyond MAX_ARC_M in any direction.
    within_range = dist_m <= MAX_ARC_M

    picks: set[int] = set()
    for c in sector_centers:
        delta = np.abs(bearings - c)
        delta = np.minimum(delta, 360.0 - delta)
        qualifies = within_range & (delta <= half_w + disc_half_deg)
        qualifies[ai] = False           # exclude self
        if not qualifies.any():
            continue
        # Nearest qualifying candidate by distance.
        d = np.where(qualifies, dist_m, np.inf)
        picks.add(int(np.argmin(d)))
    return sorted(picks)


def _load_anchors() -> list[dict]:
    """Anchors from a prior selection step. We expect way_city_anchors.geojson
    written by select_anchors_bottom_up.py or build_way_graph.py."""
    fc = json.loads(ANCHORS_GEOJSON_IN.read_text())
    anchors: list[dict] = []
    for f in fc["features"]:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        anchors.append({
            "ref":  p["ref"],
            "name": p["name"],
            "kind": p.get("kind"),
            "place": p.get("place"),
            "population": p.get("population"),
            "lon":  float(lon),
            "lat":  float(lat),
        })
    return anchors


def _assign_tiles(anchors: list[dict],
                  tile_size_deg: float
                  ) -> dict[tuple[int, int], list[int]]:
    """Bucket anchors by lon/lat floor-divided by tile_size_deg.
    Returns {(tile_x, tile_y): [anchor_idx, ...]} keyed on integer
    tile coordinates. Snap to a fixed origin so re-runs hit identical
    tile boundaries.
    """
    tiles: dict[tuple[int, int], list[int]] = {}
    for i, a in enumerate(anchors):
        tx = int(np.floor(a["lon"] / tile_size_deg))
        ty = int(np.floor(a["lat"] / tile_size_deg))
        tiles.setdefault((tx, ty), []).append(i)
    return tiles


def _tile_bbox(tile_key: tuple[int, int], tile_size_deg: float,
               buffer_deg: float) -> tuple[float, float, float, float]:
    tx, ty = tile_key
    lon_min = tx * tile_size_deg - buffer_deg
    lat_min = ty * tile_size_deg - buffer_deg
    lon_max = (tx + 1) * tile_size_deg + buffer_deg
    lat_max = (ty + 1) * tile_size_deg + buffer_deg
    return (lon_min, lat_min, lon_max, lat_max)


def _process_tile(conn: psycopg.Connection,
                  anchors: list[dict],
                  home_anchor_idx: list[int],
                  sector_neighbors_per_anchor: list[list[int]],
                  dist_mat: np.ndarray,
                  bbox: tuple[float, float, float, float],
                  seen: set[tuple[str, str]],
                  ) -> list[dict]:
    """Compute chain edges for `home_anchor_idx` anchors using a
    paved subgraph restricted to `bbox`. `seen` is a shared set that
    dedups (a, b) pairs across tiles."""
    src, dst, length, verts = _load_paved_subgraph_bbox(conn, bbox)
    if len(verts) == 0:
        return []

    # Heal osm2pgrouting fragmentation within the loaded bbox.
    base_csr = csr_matrix(
        (np.concatenate([length, length]),
         (np.concatenate([src, dst]), np.concatenate([dst, src]))),
        shape=(len(verts), len(verts)),
    )
    _, comp_label = connected_components(base_csr, directed=False)
    del base_csr
    main_comp = int(np.argmax(np.bincount(comp_label)))

    v_xyz = _lonlat_to_xyz(verts[:, 0], verts[:, 1])
    chord_bridge = _chord_for_arc(COMPONENT_BRIDGE_M)
    island_mask = comp_label != main_comp
    island_idx = np.flatnonzero(island_mask)
    full_tree = cKDTree(v_xyz)
    bridge_src: list[int] = []
    bridge_dst: list[int] = []
    if len(island_idx):
        d_nn, idx_nn = full_tree.query(v_xyz[island_idx],
                                       k=10, distance_upper_bound=chord_bridge)
        for row, src_v in enumerate(island_idx):
            for k in range(d_nn.shape[1]):
                d = d_nn[row, k]
                if not np.isfinite(d) or d > chord_bridge:
                    break
                other = int(idx_nn[row, k])
                if other == int(src_v):
                    continue
                if comp_label[other] == comp_label[src_v]:
                    continue
                bridge_src.append(int(src_v))
                bridge_dst.append(other)
    n_bridge = len(bridge_src)
    if n_bridge:
        src = np.concatenate([src, np.asarray(bridge_src, dtype=np.int64)])
        dst = np.concatenate([dst, np.asarray(bridge_dst, dtype=np.int64)])
        length = np.concatenate([length,
                                 np.zeros(n_bridge, dtype=np.float64)])

    # Multi-source attach every anchor that *could* be queried in this
    # tile — that's home anchors AND any anchor that's a sector-neighbor
    # of a home anchor. The buffer guarantees those neighbors' positions
    # fall inside the loaded subgraph.
    needed_idx: set[int] = set(home_anchor_idx)
    for ai in home_anchor_idx:
        needed_idx.update(sector_neighbors_per_anchor[ai])
    needed_list = sorted(needed_idx)

    needed_xyz = _lonlat_to_xyz(
        np.array([anchors[i]["lon"] for i in needed_list]),
        np.array([anchors[i]["lat"] for i in needed_list]),
    )
    vtree = cKDTree(v_xyz)
    chord_snap = _chord_for_arc(SNAP_RADIUS_M)
    nearby = vtree.query_ball_point(needed_xyz, r=chord_snap)
    snaps: dict[int, list[int]] = {}
    for k, ai in enumerate(needed_list):
        snaps[ai] = [int(v) for v in nearby[k]]

    # Build CSR with one super-source per attached anchor. Super-source
    # vertex id = n_verts + position-in-needed_list.
    n_verts = len(verts)
    super_of: dict[int, int] = {ai: n_verts + k
                                for k, ai in enumerate(needed_list)}
    extra_src: list[int] = []
    extra_dst: list[int] = []
    for ai, vs in snaps.items():
        sup = super_of[ai]
        for vid in vs:
            extra_src.append(sup); extra_dst.append(vid)
            extra_src.append(vid); extra_dst.append(sup)
    extra_src_arr = np.asarray(extra_src, dtype=np.int64)
    extra_dst_arr = np.asarray(extra_dst, dtype=np.int64)
    extra_cost = np.zeros(len(extra_src), dtype=np.float64)

    rows_csr = np.concatenate([src, dst, extra_src_arr])
    cols_csr = np.concatenate([dst, src, extra_dst_arr])
    data_csr = np.concatenate([length, length, extra_cost])
    n_total = n_verts + len(needed_list)
    csr = csr_matrix((data_csr, (rows_csr, cols_csr)),
                     shape=(n_total, n_total))
    del rows_csr, cols_csr, data_csr, extra_src_arr, extra_dst_arr

    edges: list[dict] = []
    for ai in home_anchor_idx:
        if not snaps.get(ai):
            continue
        nbrs = sector_neighbors_per_anchor[ai]
        if not nbrs:
            continue
        max_chord = float(max(dist_mat[ai, bi] for bi in nbrs))
        cost_limit = min(LIMIT_RATIO * max_chord + 5_000.0, MAX_ARC_M)
        distv, predv = dijkstra(csr, indices=super_of[ai],
                                return_predecessors=True,
                                limit=cost_limit)
        anchor_a = anchors[ai]
        a_super = super_of[ai]

        for bi in nbrs:
            if bi == ai or not snaps.get(bi):
                continue
            key = (min(anchor_a["ref"], anchors[bi]["ref"]),
                   max(anchor_a["ref"], anchors[bi]["ref"]))
            if key in seen:
                continue
            b_super = super_of[bi]
            cost = float(distv[b_super])
            if not np.isfinite(cost):
                continue

            path: list[int] = []
            cur = b_super
            safety = n_total + 4
            while cur != a_super and safety > 0:
                path.append(cur)
                nxt = int(predv[cur])
                if nxt == -9999:
                    path = None
                    break
                cur = nxt
                safety -= 1
            if path is None:
                continue
            path.append(a_super)
            path.reverse()
            path = [v for v in path if v < n_verts]

            coords: list[tuple[float, float]] = []
            prev = None
            for v in path:
                ll = (float(verts[v, 0]), float(verts[v, 1]))
                if ll != prev:
                    coords.append(ll)
                    prev = ll

            edges.append({
                "a":      anchor_a["ref"],
                "a_name": anchor_a["name"],
                "b":      anchors[bi]["ref"],
                "b_name": anchors[bi]["name"],
                "cost_m": cost,
                "geom":   coords,
            })
            seen.add(key)
    return edges


def main() -> None:
    t0 = time.time()
    print(f"[pairs] paved set: {PAVED_HIGHWAYS}", flush=True)
    print(f"[pairs] {N_SECTORS} sectors × {SECTOR_WIDTH_DEG:.0f}° "
          f"(centers {360.0/N_SECTORS:.1f}° apart, "
          f"disc {DISC_RADIUS_M:.0f}m)  "
          f"snap_radius={SNAP_RADIUS_M:.0f}m  limit_ratio={LIMIT_RATIO}",
          flush=True)
    print(f"[pairs] tile {ANCHOR_TILE_SIZE_DEG:.1f}° + buffer "
          f"{BBOX_BUFFER_DEG:.1f}°", flush=True)

    anchors = _load_anchors()
    print(f"[pairs] {len(anchors):,} anchors loaded from "
          f"{ANCHORS_GEOJSON_IN.name}", flush=True)

    # Sector neighbors are pure geometry — compute once globally before
    # any graph load. (N² over ~2k anchors = ~4M cells, trivial.)
    a_xyz = _lonlat_to_xyz(
        np.array([a["lon"] for a in anchors]),
        np.array([a["lat"] for a in anchors]),
    )
    a_lats = np.array([a["lat"] for a in anchors])
    a_lons = np.array([a["lon"] for a in anchors])
    dist_mat = np.linalg.norm(
        a_xyz[:, None, :] - a_xyz[None, :, :], axis=2
    )
    sector_neighbors_per_anchor: list[list[int]] = []
    total_picks = 0
    for ai in range(len(anchors)):
        nbrs = _sector_neighbors(ai, a_lats, a_lons, dist_mat[ai])
        sector_neighbors_per_anchor.append(nbrs)
        total_picks += len(nbrs)
    print(f"[pairs] sector picks: {total_picks:,} (avg "
          f"{total_picks/len(anchors):.1f} per anchor, max {N_SECTORS})",
          flush=True)

    # Tile anchors and process each tile separately to bound memory.
    tiles = _assign_tiles(anchors, ANCHOR_TILE_SIZE_DEG)
    print(f"[pairs] {len(tiles)} non-empty anchor tiles "
          f"(size {ANCHOR_TILE_SIZE_DEG:.1f}°)", flush=True)

    chain_edges: list[dict] = []
    seen: set[tuple[str, str]] = set()
    t_tiles = time.time()
    with psycopg.connect(config.PG_DSN) as conn:
        for ti, (tkey, home_anchors) in enumerate(sorted(tiles.items())):
            bbox = _tile_bbox(tkey, ANCHOR_TILE_SIZE_DEG, BBOX_BUFFER_DEG)
            print(f"[pairs] tile {ti+1}/{len(tiles)} {tkey} "
                  f"({len(home_anchors)} home anchors)…", flush=True)
            tile_t = time.time()
            tile_edges = _process_tile(
                conn, anchors, home_anchors,
                sector_neighbors_per_anchor, dist_mat, bbox, seen,
            )
            chain_edges.extend(tile_edges)
            print(f"[pairs]   +{len(tile_edges):,} edges "
                  f"({len(chain_edges):,} total) "
                  f"in {time.time()-tile_t:.1f}s", flush=True)
    print(f"[pairs] all tiles done in {time.time()-t_tiles:.1f}s: "
          f"{len(chain_edges):,} chain edges before filter",
          flush=True)

    # `in_buffer` semantics: an anchor with no snaps in any tile has no
    # incident edges, so it's effectively orphaned. Reconstruct from
    # which anchors appear in any chain_edge.
    incident_refs = {e["a"] for e in chain_edges} | {e["b"] for e in chain_edges}
    in_buffer = np.array([a["ref"] in incident_refs for a in anchors])
    n_orphan = int((~in_buffer).sum())
    print(f"[pairs] post-tile: {len(anchors)-n_orphan} anchors incident, "
          f"{n_orphan} orphan", flush=True)

    # --- indirect-edge filter ------------------------------------------
    # For each candidate (A, B), check whether the path passes near a
    # third anchor C that is a *mutual* sector-neighbor of A and B.
    # When both endpoints already point at C, A→C and C→B are the
    # proper chain edges and A→B is redundant. If C is only on one
    # side's sector list, the filter would create a disconnect; keep
    # A→B. After the filter, if any anchor would end up isolated (0
    # direct edges), restore its single shortest filtered edge as a
    # safety net so the dense-cluster case (Wels in Upper Austria) can't
    # orphan a real town.
    t_filt = time.time()
    in_buf_idx = np.flatnonzero(in_buffer)
    in_buf_xyz = a_xyz[in_buf_idx]
    in_buf_refs = [anchors[i]["ref"] for i in in_buf_idx]
    ref_to_orig_idx = {r: i for r, i in zip(in_buf_refs, in_buf_idx.tolist())}
    sector_set_per_orig = {
        i: set(sector_neighbors_per_anchor[i]) for i in in_buf_idx.tolist()
    }
    anchor_tree = cKDTree(in_buf_xyz)
    chord_third = _chord_for_arc(INDIRECT_THRESHOLD_M)

    direct: list[dict] = []
    dropped: list[dict] = []
    for edge in chain_edges:
        a_orig = ref_to_orig_idx[edge["a"]]
        b_orig = ref_to_orig_idx[edge["b"]]
        a_sec = sector_set_per_orig[a_orig]
        b_sec = sector_set_per_orig[b_orig]
        # Mutual neighbors of A and B — these are the only candidates
        # that, if found on the path, make A→B truly redundant.
        mutual = a_sec & b_sec
        mutual.discard(a_orig)
        mutual.discard(b_orig)
        if not mutual:
            direct.append(edge)
            continue

        geom_xyz = _lonlat_to_xyz(
            np.array([p[0] for p in edge["geom"]]),
            np.array([p[1] for p in edge["geom"]]),
        )
        nn_d, nn_i = anchor_tree.query(geom_xyz, k=1)
        is_indirect = False
        for d, ni in zip(nn_d, nn_i):
            if d > chord_third:
                continue
            third_orig = int(in_buf_idx[int(ni)])
            if third_orig in mutual:
                is_indirect = True
                break
        if is_indirect:
            dropped.append(edge)
        else:
            direct.append(edge)
    print(f"[pairs] indirect-edge filter "
          f"(threshold {INDIRECT_THRESHOLD_M:.0f}m, mutual-only): "
          f"dropped {len(dropped):,} indirect → {len(direct):,} direct  "
          f"in {time.time()-t_filt:.1f}s", flush=True)

    # Safety net: restore the shortest filtered edge for any anchor
    # that would otherwise be isolated.
    deg: dict[str, int] = {}
    for e in direct:
        deg[e["a"]] = deg.get(e["a"], 0) + 1
        deg[e["b"]] = deg.get(e["b"], 0) + 1
    in_buf_refs_set = set(in_buf_refs)
    isolated_refs = in_buf_refs_set - set(deg)
    if isolated_refs and dropped:
        by_ref: dict[str, list[dict]] = {}
        for e in dropped:
            for r in (e["a"], e["b"]):
                by_ref.setdefault(r, []).append(e)
        restored = 0
        already = set()
        for ref in isolated_refs:
            cands = by_ref.get(ref) or []
            if not cands:
                continue
            best = min(cands, key=lambda e: e["cost_m"])
            key = (best["a"], best["b"])
            if key in already:
                continue
            direct.append(best)
            already.add(key)
            restored += 1
        if restored:
            print(f"[pairs]   safety net: restored {restored:,} edges "
                  f"to prevent orphaning ({len(isolated_refs)} anchors "
                  f"were at risk)", flush=True)
    chain_edges = direct

    chain_edges.sort(key=lambda e: (e["a"], e["b"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_GRAPH_JSON, "w") as fh:
        json.dump(chain_edges, fh, ensure_ascii=False)
    print(f"[pairs] wrote {OUT_GRAPH_JSON} ({len(chain_edges):,} edges)",
          flush=True)

    fc_edges = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString",
                             "coordinates": [list(c) for c in e["geom"]]},
                "properties": {
                    "a":       e["a"], "a_name": e["a_name"],
                    "b":       e["b"], "b_name": e["b_name"],
                    "cost_km": round(e["cost_m"]/1000.0, 2),
                },
            }
            for e in chain_edges
        ],
    }
    with open(OUT_GRAPH_GEOJSON, "w") as fh:
        json.dump(fc_edges, fh, ensure_ascii=False)
    print(f"[pairs] wrote {OUT_GRAPH_GEOJSON}", flush=True)

    # Re-write anchors with in_graph=true for every anchor that ended up
    # incident to ≥ 1 chain edge after the indirect-edge filter +
    # safety net. The web app fades anchors with in_graph=false.
    participants = {e["a"] for e in chain_edges} | {e["b"] for e in chain_edges}
    fc_anchors_in = json.loads(ANCHORS_GEOJSON_IN.read_text())
    n_in = 0
    for feat in fc_anchors_in["features"]:
        in_graph = feat["properties"]["ref"] in participants
        feat["properties"]["in_graph"] = in_graph
        if in_graph:
            n_in += 1
    with open(ANCHORS_GEOJSON_IN, "w") as fh:
        json.dump(fc_anchors_in, fh, ensure_ascii=False)
    print(f"[pairs] re-wrote {ANCHORS_GEOJSON_IN.name} with in_graph: "
          f"{n_in:,} in-graph / {len(fc_anchors_in['features']):,} anchors",
          flush=True)

    print(f"[pairs] DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
