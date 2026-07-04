"""Way-based city graph (experiment).

Replaces the SPT-overlap city_graph with one derived from highway
topology. Two anchors are connected iff they are Voronoi-adjacent on
the subgraph of {motorway, trunk, primary} edges — i.e., no other
anchor sits between them along the corridor of those roads.

Algorithm
---------
1. Pull the subgraph of `ways` with highway in HIGHWAY_SET, plus the
   union of their endpoint vertices from ways_vertices_pgr. Build a
   local 0..N-1 vertex indexing for scipy CSR.
2. Anchor set = `anchors` table (cities + towns + previously snapped
   POIs) ∪ all OSM `place=village` nodes loaded from VILLAGES_PATH.
3. Snap each anchor to the nearest subgraph vertex within
   ANCHOR_BUFFER_M (chord ≈ arc at these scales). Anchors with no
   subgraph vertex in range are orphans — written out separately for
   inspection.
4. Add a virtual super-source vertex connected to every kept anchor's
   snap vertex with a directed zero-cost edge. Run a single-source
   Dijkstra from super-source over the (subgraph + super-edges).
   pred[v] for v != super now backpointers along the shortest path to
   v's nearest anchor.
5. Label each vertex by its nearest anchor by walking pred[] back to
   the snap. Iterate in dist-order so each vertex inherits its label
   from its already-labeled predecessor.
6. For each subgraph edge (u, v) with label[u] != label[v], anchors
   label[u] and label[v] are "chain-adjacent on the highway corridor".
   Edge cost = dist[u] + edge_len(u,v) + dist[v]. Keep the minimum
   over all Voronoi-boundary edges between a given pair.
7. Reconstruct each chain edge's geometry by walking pred[] from the
   chosen boundary edge back to each anchor's snap, then chaining.

This naturally handles US50-style long-corridor cases: if no anchor
sits between two settlements 100 mi apart on the same road, they're
Voronoi-adjacent regardless of distance.

Outputs (under /data/):
  way_city_graph.json            list of {a, b, cost_m, geom: [[lon,lat],...]}
  way_city_graph.geojson         LineString FeatureCollection
  way_city_anchors.geojson       Point FC for kept anchors (in_graph flag)
  way_city_anchors_orphans.geojson  Point FC for anchors outside the buffer
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import psycopg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import config


SEED_HIGHWAYS         = ("motorway", "trunk", "primary", "secondary")
# Classes eligible to be flood-filled into the subgraph when they share
# a name or ref with an in-set way at the connecting vertex. Anything
# not in this list is excluded (footway, path, track, service, etc.).
PROMOTABLE_HIGHWAYS   = (
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street", "road",
    "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link",
)
ANCHOR_BUFFER_M   = 4828.0          # ~3 mi

# Tiling for the subgraph load. The 4-country subgraph (seed +
# named-promotable + ferries) is ~20-30 M rows, ~2-4 GB of Python
# objects — too big to load at once alongside the API. We tile:
# every core tile is TILE_CORE_DEG × TILE_CORE_DEG; each per-tile
# load pulls edges whose endpoint lies in a buffered bbox
# (TILE_CORE_DEG + 2 * TILE_BUFFER_DEG on a side). The buffer must
# exceed the longest chain edge so Voronoi neighbours of any core
# anchor are fully resolved inside the tile — 1.5° ≈ 150 km at these
# latitudes, well above the observed ~50 km max chain edge.
#
# Anchors near a tile's edges are computed by both the tile that owns
# them (their core tile) AND any tile whose buffer reaches them; we
# emit every observed edge and dedupe canonically at merge time. This
# keeps per-tile memory bounded (~500 MB peak) at the cost of ~2× work
# on the buffer overlap.
TILE_CORE_DEG   = 3.0
TILE_BUFFER_DEG = 1.5
# After the name/ref flood-fill, bridge endpoints of same-name-or-ref
# in-set ways that lie within this many meters of each other but aren't
# connected via a shared graph vertex. Handles OSM tagging gaps at town
# crossings (e.g., a roundabout segment without ref breaks B311 into
# multiple components). Wide enough to span town traverses, narrow
# enough that two unrelated same-named roads can't false-bridge.
BRIDGE_GAP_M      = 200.0
VILLAGES_PATH     = Path("/data/osm/austria-villages.geojsonseq")

OUT_DIR                = Path("/data")
OUT_GRAPH_JSON         = OUT_DIR / "way_city_graph.json"
OUT_GRAPH_GEOJSON      = OUT_DIR / "way_city_graph.geojson"
OUT_NODES_GEOJSON      = OUT_DIR / "way_city_anchors.geojson"
OUT_ORPHANS_GEOJSON    = OUT_DIR / "way_city_anchors_orphans.geojson"

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


def _load_subgraph_rows_from_cells(bbox):
    """Fast path: read raw subgraph rows from pre-exported cell files
    (see ingest/subgraph_export.py). Returns a list of 14-tuples in the
    same column order the postgres cursor yielded, so the flood-fill
    code below is unchanged.

    Loads every cell that intersects `bbox` expanded by 1 cell on each
    side — enough padding to catch edges whose source is just outside
    the query bbox but whose target is inside.
    """
    import os as _os
    cells_dir = Path(_os.environ.get("SUBGRAPH_CELLS_DIR",
                                     "/data/subgraph_cells"))
    manifest_path = cells_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    highway_table = manifest["highway_table"]
    lo_lon, lo_lat, hi_lon, hi_lat = bbox
    clo_lon = int(math.floor(lo_lon)) - 1
    clo_lat = int(math.floor(lo_lat)) - 1
    chi_lon = int(math.ceil(hi_lon))  + 1
    chi_lat = int(math.ceil(hi_lat))  + 1

    rows: list[tuple] = []
    n_files = 0
    for clon in range(clo_lon, chi_lon + 1):
        for clat in range(clo_lat, chi_lat + 1):
            path = cells_dir / f"{clon:+04d}_{clat:+04d}.npz"
            if not path.exists():
                continue
            n_files += 1
            with np.load(path, allow_pickle=False) as data:
                n = len(data["source"])
                if n == 0:
                    continue
                source     = data["source"]
                target     = data["target"]
                length_m   = data["length_m"]
                osm_way_id = data["osm_way_id"]
                hw_idx     = data["hw_idx"]
                is_ferry_a = data["is_ferry"]
                src_lon    = data["src_lon"]
                src_lat    = data["src_lat"]
                dst_lon    = data["dst_lon"]
                dst_lat    = data["dst_lat"]
                name_table = data["name_table"]
                name_idx   = data["name_idx"]
                ref_table  = data["ref_table"]
                ref_idx    = data["ref_idx"]
                for k in range(n):
                    hi = int(hw_idx[k])
                    hw = highway_table[hi] if 0 <= hi < len(highway_table) else ""
                    rows.append((
                        int(source[k]), int(target[k]), float(length_m[k]),
                        int(osm_way_id[k]), hw,
                        str(name_table[name_idx[k]]),
                        str(ref_table[ref_idx[k]]),
                        int(source[k]),
                        float(src_lon[k]), float(src_lat[k]),
                        int(target[k]),
                        float(dst_lon[k]), float(dst_lat[k]),
                        bool(is_ferry_a[k]),
                    ))
    return rows


def _load_subgraph(conn: psycopg.Connection,
                   bbox: tuple[float, float, float, float] | None = None):
    """Build the flood-filled subgraph.

    Step 1: pull seed edges (highway in SEED_HIGHWAYS) and a pool of
            candidate edges (highway in PROMOTABLE_HIGHWAYS, with a
            name or ref tag joined from way_tags).
    Step 2: BFS over osm_way_ids — a candidate way joins the subgraph
            if it touches an already-in-set way at a shared vertex AND
            shares name or ref with it.
    Step 3: emit the union (seed edges + promoted edges) as the local
            vertex-indexed (src, dst, length) plus the lon/lat array.

    If a bbox is given, prefers pre-exported cell files at
    `/data/subgraph_cells/` — a scalable flat-file mirror of the
    postgres subgraph produced by `ingest/subgraph_export.py`. Falls
    back to a postgres bbox query if cells aren't available.

    Returns the same (src, dst, length, verts, local_to_global) tuple
    as the simple version, so the downstream Dijkstra/Voronoi code is
    unchanged.
    """
    t = time.time()
    # Fast path: exported cell files. Skip postgres entirely.
    if bbox is not None:
        cell_rows = _load_subgraph_rows_from_cells(bbox)
        if cell_rows is not None:
            print(f"[way-graph]   loaded {len(cell_rows):,} rows from "
                  f"subgraph_cells in {time.time()-t:.1f}s", flush=True)
            rows = cell_rows
            return _process_rows_into_subgraph(rows, t)

    seed_ph    = ",".join(["%s"] * len(SEED_HIGHWAYS))
    promote_ph = ",".join(["%s"] * len(PROMOTABLE_HIGHWAYS))
    bbox_clause = ""
    if bbox is not None:
        # Filter to edges whose EITHER endpoint sits inside the envelope.
        # `vs.the_geom && env` is the GIST-indexed bbox test. Edges that
        # straddle the tile boundary are kept because at least one of
        # their vertices is inside.
        bbox_clause = (
            "  AND (vs.the_geom && ST_MakeEnvelope(%s,%s,%s,%s,4326)"
            "       OR vt.the_geom && ST_MakeEnvelope(%s,%s,%s,%s,4326))"
        )
    sql = f"""
        SELECT w.source, w.target, w.length_m, w.osm_way_id, w.highway,
               COALESCE(t.name, '') AS name, COALESCE(t.ref, '') AS ref,
               vs.id, ST_X(vs.the_geom), ST_Y(vs.the_geom),
               vt.id, ST_X(vt.the_geom), ST_Y(vt.the_geom),
               w.is_ferry
        FROM ways w
        JOIN ways_vertices_pgr vs ON vs.id = w.source
        JOIN ways_vertices_pgr vt ON vt.id = w.target
        LEFT JOIN way_tags t ON t.osm_way_id = w.osm_way_id
        WHERE w.length_m > 0.0
          AND (
              w.highway IN ({seed_ph})
              OR w.is_ferry
              OR (
                  w.highway IN ({promote_ph})
                  AND (COALESCE(t.name, '') <> '' OR COALESCE(t.ref, '') <> '')
              )
          )
        {bbox_clause}
    """
    params = list(SEED_HIGHWAYS) + list(PROMOTABLE_HIGHWAYS)
    if bbox is not None:
        params.extend(bbox)   # vs envelope
        params.extend(bbox)   # vt envelope
    rows: list[tuple] = []
    with conn.cursor(name="way_graph_subgraph") as cur:
        # Server-side cursor + itersize streams in pages so postgres
        # doesn't have to materialize ~1 M joined rows in shared memory.
        cur.itersize = 200_000
        cur.execute(sql, params)
        while True:
            batch = cur.fetchmany(200_000)
            if not batch:
                break
            rows.extend(batch)
    print(f"[way-graph] pulled {len(rows):,} candidate rows "
          f"(seed + named-promotable) in {time.time()-t:.1f}s", flush=True)

    return _process_rows_into_subgraph(rows, t)


def _process_rows_into_subgraph(rows, t_start):
    """Consumes raw rows (14-tuples in postgres column order) and does
    the flood-fill + geographic bridging + local-vertex indexing.

    Extracted so both the postgres cursor path and the cell-file path
    hit the same downstream logic without duplication.
    """
    t = t_start
    seed_set: set[int] = set()    # osm_way_ids of seed-class ways
    by_way: dict[int, dict] = {}  # osm_way_id → {name, ref, vertices: set, rows: list of idx}

    n_seed_rows = 0
    for i, r in enumerate(rows):
        (sv, tv, lm, oid, hw, name, ref, sg, sx, sy, tg, tx, ty, is_ferry) = r
        oid = int(oid)
        is_seed = (hw in SEED_HIGHWAYS) or bool(is_ferry)
        if is_seed:
            n_seed_rows += 1
            seed_set.add(oid)
        slot = by_way.get(oid)
        if slot is None:
            slot = {"name": name or "", "ref": ref or "",
                    "vertices": set(), "rows": []}
            by_way[oid] = slot
        slot["vertices"].add(int(sg))
        slot["vertices"].add(int(tg))
        slot["rows"].append(i)

    print(f"[way-graph]   seed: {n_seed_rows:,} rows / {len(seed_set):,} "
          f"OSM ways  |  candidate pool: {len(by_way) - len(seed_set):,} "
          f"named-promotable OSM ways",
          flush=True)

    # Vertex → set of osm_way_ids touching it (across seed + candidates).
    vertex_to_ways: dict[int, list[int]] = {}
    for oid, slot in by_way.items():
        for v in slot["vertices"]:
            vertex_to_ways.setdefault(v, []).append(oid)

    # BFS flood-fill from seed.
    in_set = set(seed_set)
    frontier = list(seed_set)
    passes = 0
    while frontier:
        passes += 1
        next_frontier: list[int] = []
        for oid in frontier:
            slot = by_way[oid]
            name_w, ref_w = slot["name"], slot["ref"]
            if not name_w and not ref_w:
                continue   # nothing to match on — seed way without tags
            for v in slot["vertices"]:
                for oid2 in vertex_to_ways.get(v, ()):
                    if oid2 in in_set:
                        continue
                    s2 = by_way[oid2]
                    name2, ref2 = s2["name"], s2["ref"]
                    if (name_w and name_w == name2) or (ref_w and ref_w == ref2):
                        in_set.add(oid2)
                        next_frontier.append(oid2)
        print(f"[way-graph]   flood pass {passes}: +{len(next_frontier):,} "
              f"ways (total in-set {len(in_set):,})", flush=True)
        frontier = next_frontier

    n_promoted = len(in_set) - len(seed_set)
    print(f"[way-graph] flood-fill done in {passes} passes  "
          f"(+{n_promoted:,} ways promoted)", flush=True)

    # Build the local vertex indexing and emit src/dst/length only for
    # rows whose osm_way_id is in_set.
    global_to_local: dict[int, int] = {}
    verts_lonlat: list[tuple[float, float]] = []
    local_to_global: list[int] = []

    def _intern(gid: int, lon: float, lat: float) -> int:
        idx = global_to_local.get(gid)
        if idx is None:
            idx = len(local_to_global)
            global_to_local[gid] = idx
            local_to_global.append(gid)
            verts_lonlat.append((lon, lat))
        return idx

    src_list: list[int] = []
    dst_list: list[int] = []
    len_list: list[float] = []
    # For each kept vertex, remember the name/ref tags of in-set ways
    # incident to it. Used by the geographic-bridge step that follows.
    vert_tags: dict[int, set[tuple[str, str]]] = {}
    for r in rows:
        (sv, tv, lm, oid, hw, name, ref, sg, sx, sy, tg, tx, ty, _isf) = r
        if int(oid) not in in_set:
            continue
        s_loc = _intern(int(sg), float(sx), float(sy))
        t_loc = _intern(int(tg), float(tx), float(ty))
        src_list.append(s_loc)
        dst_list.append(t_loc)
        len_list.append(float(lm))
        key = (name or "", ref or "")
        if key != ("", ""):
            vert_tags.setdefault(s_loc, set()).add(key)
            vert_tags.setdefault(t_loc, set()).add(key)

    print(f"[way-graph] {len(src_list):,} subgraph edges  /  "
          f"{len(verts_lonlat):,} vertices  before bridging "
          f"(in {time.time()-t:.1f}s)", flush=True)

    # === Geographic bridging ============================================
    # Same name/ref segments are sometimes split by an untagged roundabout
    # or interchange (e.g., B311 enters Sankt Johann via a roundabout
    # without ref → ref-based flood stops). We close those gaps by adding
    # phantom edges between same-name-or-ref vertices that lie within
    # BRIDGE_GAP_M of each other AND belong to different connected
    # components of the current in-set. Union-find keeps the bridge count
    # to (components - 1) per name/ref group; without it, two adjacent
    # vertices on the same road would each generate redundant bridges.
    t_b = time.time()
    n_bridged = 0
    if len(vert_tags) > 1:
        # Union-find over the current subgraph: every real edge unions
        # its endpoints. A candidate bridge is accepted only if it joins
        # two distinct components.
        n_verts_pre = len(verts_lonlat)
        parent = list(range(n_verts_pre))
        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def _union(a: int, b: int) -> bool:
            ra, rb = _find(a), _find(b)
            if ra == rb:
                return False
            parent[ra] = rb
            return True
        for a, b in zip(src_list, dst_list):
            _union(a, b)

        # Pre-extract name/ref sets per tagged vertex for the match check.
        v_names: dict[int, set[str]] = {}
        v_refs:  dict[int, set[str]] = {}
        for v, tags in vert_tags.items():
            ns = {n for n, _ in tags if n}
            rs = {r_ for _, r_ in tags if r_}
            if ns: v_names[v] = ns
            if rs: v_refs[v]  = rs

        tag_verts = sorted(vert_tags.keys())
        v_arr = np.asarray(tag_verts, dtype=np.int64)
        v_xyz = _lonlat_to_xyz(
            np.asarray([verts_lonlat[i][0] for i in tag_verts]),
            np.asarray([verts_lonlat[i][1] for i in tag_verts]),
        )
        tree = cKDTree(v_xyz)
        chord_gap = _chord_for_arc(BRIDGE_GAP_M)
        pairs = tree.query_pairs(r=chord_gap, output_type="ndarray")
        # Sort by chord distance ASC so the shortest candidate per
        # component pair is the one that wins union.
        if len(pairs):
            xyz_a = v_xyz[pairs[:, 0]]
            xyz_b = v_xyz[pairs[:, 1]]
            chord = np.linalg.norm(xyz_a - xyz_b, axis=1)
            order = np.argsort(chord, kind="stable")
            pairs = pairs[order]
            chord = chord[order]

        n_skipped_same_comp = 0
        n_skipped_no_match = 0
        for k in range(len(pairs)):
            i = int(pairs[k, 0]); j = int(pairs[k, 1])
            va = int(v_arr[i]);   vb = int(v_arr[j])
            if _find(va) == _find(vb):
                n_skipped_same_comp += 1
                continue
            shared = False
            na = v_names.get(va); nb = v_names.get(vb)
            if na and nb and (na & nb):
                shared = True
            else:
                ra = v_refs.get(va); rb = v_refs.get(vb)
                if ra and rb and (ra & rb):
                    shared = True
            if not shared:
                n_skipped_no_match += 1
                continue
            d = float(chord[k])
            src_list.append(va); dst_list.append(vb); len_list.append(d)
            _union(va, vb)
            n_bridged += 1

        print(f"[way-graph] geographic bridging: +{n_bridged:,} phantom edges "
              f"(gap≤{BRIDGE_GAP_M:.0f} m)  | skipped "
              f"{n_skipped_same_comp:,} same-component, "
              f"{n_skipped_no_match:,} no-tag-match  "
              f"in {time.time()-t_b:.1f}s", flush=True)

    src = np.asarray(src_list, dtype=np.int64)
    dst = np.asarray(dst_list, dtype=np.int64)
    length = np.asarray(len_list, dtype=np.float64)
    verts = np.asarray(verts_lonlat, dtype=np.float64)
    l2g = np.asarray(local_to_global, dtype=np.int64)
    print(f"[way-graph] FINAL: {len(src):,} subgraph edges  /  "
          f"{len(verts):,} vertices  in {time.time()-t:.1f}s",
          flush=True)
    return src, dst, length, verts, l2g


def _load_db_anchors(conn: psycopg.Connection) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, name, place, population, country,
                   ST_X(geom), ST_Y(geom)
            FROM anchors
            WHERE snap_vertex_id IS NOT NULL
            ORDER BY id
        """)
        return [{
            "kind":       "db",
            "ref":        f"db:{int(r[0])}",
            "name":       r[1] or "?",
            "place":      r[2] or "?",
            "population": r[3],
            "country":    r[4],
            "lon":        float(r[5]),
            "lat":        float(r[6]),
        } for r in cur.fetchall()]


def _load_village_anchors() -> list[dict]:
    out: list[dict] = []
    for seq, line in enumerate(open(VILLAGES_PATH)):
        # geojsonseq uses RS (0x1e) as record prefix; strip non-JSON
        # leading bytes before parsing.
        line = line.strip().lstrip("\x1e").strip()
        if not line:
            continue
        f = json.loads(line)
        lon, lat = f["geometry"]["coordinates"]
        props = f.get("properties", {})
        try:
            pop = int(props["population"]) if props.get("population") else None
        except (TypeError, ValueError):
            pop = None
        # Upstream extract dropped OSM @id from many records; fall back
        # to sequence index so each village has a unique ref.
        osm_id = str(f.get("id") or props.get("@id") or "")
        ref = f"osm:{osm_id}" if osm_id else f"osm:seq-{seq}"
        out.append({
            "kind":       "village",
            "ref":        ref,
            "name":       props.get("name") or "?",
            "place":      "village",
            "population": pop,
            "country":    None,
            "lon":        float(lon),
            "lat":        float(lat),
        })
    return out


MAX_SNAPS_PER_ANCHOR = 8   # cap to keep super-edges and label noise bounded


def _compute_components(src: np.ndarray, dst: np.ndarray, n_verts: int) -> list[int]:
    """Union-find component labels for the in-set subgraph. Precompute
    once per subgraph; reuse across many anchor-set iterations."""
    parent = list(range(n_verts))
    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(src, dst):
        ra, rb = _find(int(a)), _find(int(b))
        if ra != rb:
            parent[ra] = rb
    # Final path-compression pass so _find(x) is O(1) after this.
    for i in range(n_verts):
        _find(i)
    return parent


def _snap_anchors(anchors: list[dict], verts: np.ndarray,
                  src: np.ndarray, dst: np.ndarray,
                  components: list[int] | None = None):
    """Snap each anchor to the nearest subgraph vertex *per connected
    component* of the in-set subgraph, within ANCHOR_BUFFER_M.

    Cities sit at junctions of multiple major corridors that often
    appear as separate connected components (different OSM `ref`s, or
    same ref split by an untagged section). A single-vertex snap forces
    the anchor to pick one road and miss the others. Multi-snap gives
    the anchor a presence in each reachable component, so Voronoi
    boundaries form correctly along all incident corridors.

    `components` is the output of _compute_components(); if not supplied,
    it's computed inline. Iterative drivers should pass a precomputed
    array to avoid redundant union-finds.

    Returns:
        snap_records: list[(anchor_idx, snap_local_vid, dist_m)]
        per_anchor_count: dict[anchor_idx, int]
        best_dist:    dict[anchor_idx, float] (closest snap per anchor)
        nearest_road: list[float] — closest in-set vertex distance per
                      anchor (defined even for orphans).
    """
    n_verts = len(verts)
    if components is None:
        components = _compute_components(src, dst, n_verts)

    a_lon = np.array([a["lon"] for a in anchors])
    a_lat = np.array([a["lat"] for a in anchors])
    a_xyz = _lonlat_to_xyz(a_lon, a_lat)
    v_xyz = _lonlat_to_xyz(verts[:, 0], verts[:, 1])
    tree = cKDTree(v_xyz)
    chord_buf = _chord_for_arc(ANCHOR_BUFFER_M)

    # query_ball_point returns, for each anchor, the list of vertex
    # indices within chord_buf. May be empty for anchors far from any
    # in-set road. Also compute a 1-NN distance for reporting orphans.
    nearby = tree.query_ball_point(a_xyz, r=chord_buf)
    nn_dist, _ = tree.query(a_xyz, k=1)
    nearest_road = [float(x) for x in nn_dist]

    snap_records: list[tuple[int, int, float]] = []
    per_anchor_count: dict[int, int] = {}
    best_dist: dict[int, float] = {}

    for ai, vid_list in enumerate(nearby):
        if not vid_list:
            continue
        # Distance from this anchor to each candidate vertex (chord).
        cand = np.asarray(vid_list, dtype=np.int64)
        d = np.linalg.norm(v_xyz[cand] - a_xyz[ai], axis=1)
        order = np.argsort(d)
        # Pick the closest vertex per connected component.
        per_comp: dict[int, tuple[int, float]] = {}
        for k in order:
            vid = int(cand[k])
            comp = components[vid]
            if comp in per_comp:
                continue
            per_comp[comp] = (vid, float(d[k]))
            if len(per_comp) >= MAX_SNAPS_PER_ANCHOR:
                break
        for vid, dist_m in per_comp.values():
            snap_records.append((ai, vid, dist_m))
        per_anchor_count[ai] = len(per_comp)
        best_dist[ai] = min(d for _, d in per_comp.values())

    return snap_records, per_anchor_count, best_dist, nearest_road


def compute_chain_graph(anchors: list[dict],
                        src: np.ndarray, dst: np.ndarray,
                        edge_len: np.ndarray, verts: np.ndarray,
                        components: list[int] | None = None,
                        verbose: bool = True) -> dict:
    """Snap anchors → super-source Dijkstra → Voronoi labels → chain edges.

    This is the per-iteration core, factored out so hierarchical-
    refinement drivers can call it many times against the same
    subgraph with different anchor sets. The subgraph (src, dst,
    edge_len, verts) and its component labels stay constant; only
    `anchors` varies.

    Returns:
        {
          "chain_edges":      list of {a, a_name, b, b_name, cost_m, geom},
          "kept_anchors":     list[dict] (subset that snapped),
          "kept_idx":         list[int] (indices into `anchors`),
          "orphan_idx":       list[int] (indices into `anchors`),
          "per_anchor_count": dict[orig_idx -> n_snaps],
          "best_dist":        dict[orig_idx -> closest snap m],
          "nearest_road":     list[float] per `anchors` index,
          "participant_refs": set[str] (refs that appear in chain_edges),
        }
    """
    if components is None:
        components = _compute_components(src, dst, len(verts))
    snap_records, per_anchor_count, best_dist, nearest_road = _snap_anchors(
        anchors, verts, src, dst, components=components)
    kept_idx = sorted(per_anchor_count.keys())
    orphan_idx = [i for i in range(len(anchors)) if i not in per_anchor_count]
    kept_anchors = [anchors[i] for i in kept_idx]
    # Map anchor's index in `anchors` → its position in kept_anchors.
    anchor_to_kept = {a_i: ki for ki, a_i in enumerate(kept_idx)}
    # Build the snap arrays (one entry per snap, possibly multiple per anchor).
    snap_kept_idx = np.asarray([anchor_to_kept[r[0]] for r in snap_records],
                               dtype=np.int64)
    snap_vids     = np.asarray([r[1] for r in snap_records], dtype=np.int64)
    snap_dists    = np.asarray([r[2] for r in snap_records], dtype=np.float64)
    n_total_snaps = len(snap_records)
    avg_snaps = n_total_snaps / max(len(kept_anchors), 1)
    if verbose:
        print(f"[way-graph] {len(kept_anchors):,} anchors snapped, "
              f"{len(orphan_idx):,} orphans (no road within buffer)",
              flush=True)
        print(f"[way-graph] multi-snap: {n_total_snaps:,} super-edges "
              f"across {len(kept_anchors):,} anchors "
              f"(avg {avg_snaps:.2f} components per anchor, "
              f"cap {MAX_SNAPS_PER_ANCHOR})",
              flush=True)

    # Build CSR: original subgraph (undirected → both directions) + a
    # virtual super-source vertex (index n_verts) with directed zero-
    # cost edges to every kept-anchor snap. Multi-snap means an anchor
    # may have multiple super-edges, one per connected component within
    # ANCHOR_BUFFER_M of its centroid.
    n_verts = len(verts)
    super_idx = n_verts

    rows_csr = np.concatenate([src, dst, np.full(n_total_snaps, super_idx,
                                                 dtype=np.int64)])
    cols_csr = np.concatenate([dst, src, snap_vids])
    data_csr = np.concatenate([edge_len, edge_len,
                               np.zeros(n_total_snaps, dtype=np.float64)])
    csr = csr_matrix((data_csr, (rows_csr, cols_csr)),
                     shape=(n_verts + 1, n_verts + 1))
    if verbose:
        print(f"[way-graph] CSR: {n_verts+1:,} vertices, "
              f"{len(rows_csr):,} edges (incl. {n_total_snaps:,} super-edges)",
              flush=True)

    t = time.time()
    distv, predv = dijkstra(csr, indices=super_idx, directed=True,
                            return_predecessors=True)
    if verbose:
        print(f"[way-graph] Dijkstra done in {time.time()-t:.1f}s", flush=True)
        n_reachable = int(np.isfinite(distv[:n_verts]).sum())
        print(f"[way-graph] {n_reachable:,}/{n_verts:,} subgraph vertices "
              f"reachable from anchor snaps", flush=True)

    # snap_vid → kept-anchor index. With multi-snap, an anchor may map
    # multiple snap_vids back to itself (one per component). If two
    # different anchors land on the same vertex (rare), the closer one
    # wins.
    snap_to_anchor: dict[int, int] = {}
    snap_to_dist:   dict[int, float] = {}
    for sv, ki, dm in zip(snap_vids.tolist(), snap_kept_idx.tolist(),
                          snap_dists.tolist()):
        cur = snap_to_dist.get(sv)
        if cur is None or dm < cur:
            snap_to_anchor[sv] = ki
            snap_to_dist[sv]   = dm

    # Propagate labels by walking pred[] in dist order. Each vertex
    # inherits its predecessor's label, except snap vertices (whose
    # predecessor is super-source) which take their own anchor.
    labels = np.full(n_verts + 1, -1, dtype=np.int32)
    order = np.argsort(distv, kind="stable")
    for v in order:
        if v == super_idx:
            continue
        if not np.isfinite(distv[v]):
            break  # remaining are unreachable; argsort puts inf at end
        p = int(predv[v])
        if p == super_idx:
            labels[v] = snap_to_anchor.get(int(v), -1)
        elif p == -9999:
            labels[v] = -1
        else:
            labels[v] = labels[p]

    # Voronoi-boundary edges → chain adjacency. Track min-cost witness
    # so we can reconstruct geometry.
    pair_cost: dict[tuple[int, int], float] = {}
    pair_witness: dict[tuple[int, int], tuple[int, int]] = {}
    for i in range(len(src)):
        u = int(src[i]); v = int(dst[i])
        la = int(labels[u]); lb = int(labels[v])
        if la < 0 or lb < 0 or la == lb:
            continue
        cost = float(distv[u]) + float(edge_len[i]) + float(distv[v])
        key = (min(la, lb), max(la, lb))
        # Orient witness so u side maps to key[0].
        uw, vw = (u, v) if la == key[0] else (v, u)
        if cost < pair_cost.get(key, float("inf")):
            pair_cost[key] = cost
            pair_witness[key] = (uw, vw)

    participant_idx = {a for k in pair_cost for a in k}
    participant_refs = {kept_anchors[i]["ref"] for i in participant_idx}
    if verbose:
        print(f"[way-graph] {len(pair_cost):,} chain edges between "
              f"{len(participant_idx):,} distinct anchors", flush=True)

    # Geometry reconstruction. pred chain walks back from each witness
    # endpoint to its anchor snap (where pred == super_idx).
    def _walk_to_snap(v0: int) -> list[int]:
        path = [v0]
        cur = v0
        # Hard cap loop length defensively. Real chains are ≤ a few
        # thousand vertices on a long Austrian corridor.
        for _ in range(n_verts + 1):
            p = int(predv[cur])
            if p == super_idx or p == -9999:
                return path
            path.append(p)
            cur = p
        raise RuntimeError(f"pred walk did not terminate from vertex {v0}")

    chain_edges = []
    for (a_idx, b_idx), cost in pair_cost.items():
        uw, vw = pair_witness[(a_idx, b_idx)]
        path_a = _walk_to_snap(uw)   # uw → snap(a)
        path_b = _walk_to_snap(vw)   # vw → snap(b)
        full = list(reversed(path_a)) + path_b
        coords: list[tuple[float, float]] = []
        prev = None
        for vi in full:
            ll = (float(verts[vi, 0]), float(verts[vi, 1]))
            if ll != prev:
                coords.append(ll)
                prev = ll
        chain_edges.append({
            "a":      kept_anchors[a_idx]["ref"],
            "a_name": kept_anchors[a_idx]["name"],
            "b":      kept_anchors[b_idx]["ref"],
            "b_name": kept_anchors[b_idx]["name"],
            "cost_m": cost,
            "geom":   coords,
        })

    chain_edges.sort(key=lambda e: (e["a"], e["b"]))

    return {
        "chain_edges":      chain_edges,
        "kept_anchors":     kept_anchors,
        "kept_idx":         kept_idx,
        "orphan_idx":       orphan_idx,
        "per_anchor_count": per_anchor_count,
        "best_dist":        best_dist,
        "nearest_road":     nearest_road,
        "participant_refs": participant_refs,
    }


def _write_outputs(anchors: list[dict], result: dict) -> None:
    """Emit chain_edges + nodes + orphans GeoJSON files."""
    chain_edges    = result["chain_edges"]
    kept_anchors   = result["kept_anchors"]
    kept_idx       = result["kept_idx"]
    orphan_idx     = result["orphan_idx"]
    per_anchor_cnt = result["per_anchor_count"]
    best_dist      = result["best_dist"]
    nearest_road   = result["nearest_road"]
    participant_refs = result["participant_refs"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_GRAPH_JSON, "w") as fh:
        json.dump(chain_edges, fh, ensure_ascii=False)
    print(f"[way-graph] wrote {OUT_GRAPH_JSON} ({len(chain_edges):,} edges)",
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
    print(f"[way-graph] wrote {OUT_GRAPH_GEOJSON}", flush=True)

    fc_nodes = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [kept_anchors[i]["lon"],
                                             kept_anchors[i]["lat"]]},
                "properties": {
                    "ref":         kept_anchors[i]["ref"],
                    "name":        kept_anchors[i]["name"],
                    "kind":        kept_anchors[i]["kind"],
                    "place":       kept_anchors[i]["place"],
                    "population":  kept_anchors[i]["population"],
                    "country":     kept_anchors[i].get("country"),
                    "vid":         kept_anchors[i].get("vid"),
                    "_protected":  kept_anchors[i].get("_protected", False),
                    "in_graph":    kept_anchors[i]["ref"] in participant_refs,
                    "snap_dist_m": round(float(best_dist[kept_idx[i]]), 1),
                    "n_snaps":     per_anchor_cnt[kept_idx[i]],
                },
            }
            for i in range(len(kept_anchors))
        ],
    }
    with open(OUT_NODES_GEOJSON, "w") as fh:
        json.dump(fc_nodes, fh, ensure_ascii=False)
    n_in_graph = sum(1 for f in fc_nodes["features"]
                     if f["properties"]["in_graph"])
    print(f"[way-graph] wrote {OUT_NODES_GEOJSON} "
          f"({n_in_graph:,} in-graph / {len(kept_anchors):,} kept)",
          flush=True)

    fc_orphans = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [anchors[i]["lon"],
                                             anchors[i]["lat"]]},
                "properties": {
                    "name":           anchors[i]["name"],
                    "kind":           anchors[i]["kind"],
                    "place":          anchors[i]["place"],
                    "nearest_road_m": round(nearest_road[i], 1),
                },
            }
            for i in orphan_idx
        ],
    }
    with open(OUT_ORPHANS_GEOJSON, "w") as fh:
        json.dump(fc_orphans, fh, ensure_ascii=False)
    print(f"[way-graph] wrote {OUT_ORPHANS_GEOJSON} "
          f"({len(orphan_idx):,} anchors > buffer)", flush=True)


def _load_anchors_geojson(path: Path) -> list[dict]:
    """Load anchors from a way_city_anchors.geojson written by an
    earlier selection step (e.g., select_anchors_bottom_up.py). Used
    when we want compute_chain_graph to run against a pre-selected
    anchor set (including protected ferry piers) rather than
    re-selecting from postgres."""
    fc = json.loads(path.read_text())
    out: list[dict] = []
    for f in fc["features"]:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        out.append({
            "kind":       p.get("kind"),
            "ref":        p["ref"],
            "name":       p["name"],
            "place":      p.get("place"),
            "population": p.get("population"),
            "country":    p.get("country"),
            "vid":        p.get("vid"),
            "_protected": p.get("_protected", False),
            "lon":        float(lon),
            "lat":        float(lat),
        })
    return out


def _tile_bboxes(anchors: list[dict]) -> list[dict]:
    """Return a list of tile descriptors covering every anchor.

    Each descriptor is {'core': bbox, 'buffer': bbox} where a bbox is
    (lon_min, lat_min, lon_max, lat_max). Only tiles whose core
    contains at least one anchor are emitted.
    """
    import math
    if not anchors:
        return []
    lons = [float(a["lon"]) for a in anchors]
    lats = [float(a["lat"]) for a in anchors]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    # Snap the tile grid to whole degrees so tile edges are stable
    # across runs even if a few anchors shift.
    x0 = math.floor(lon_min / TILE_CORE_DEG) * TILE_CORE_DEG
    y0 = math.floor(lat_min / TILE_CORE_DEG) * TILE_CORE_DEG
    x1 = math.ceil(lon_max / TILE_CORE_DEG) * TILE_CORE_DEG
    y1 = math.ceil(lat_max / TILE_CORE_DEG) * TILE_CORE_DEG

    tiles: list[dict] = []
    x = x0
    while x < x1:
        y = y0
        while y < y1:
            core = (x, y, x + TILE_CORE_DEG, y + TILE_CORE_DEG)
            n_in_core = sum(
                1 for a in anchors
                if core[0] <= a["lon"] < core[2] and core[1] <= a["lat"] < core[3]
            )
            if n_in_core > 0:
                buf = (
                    core[0] - TILE_BUFFER_DEG,
                    core[1] - TILE_BUFFER_DEG,
                    core[2] + TILE_BUFFER_DEG,
                    core[3] + TILE_BUFFER_DEG,
                )
                tiles.append({"core": core, "buffer": buf,
                              "n_core_anchors": n_in_core})
            y += TILE_CORE_DEG
        x += TILE_CORE_DEG
    return tiles


def _anchors_in_bbox(anchors: list[dict],
                     bbox: tuple[float, float, float, float]) -> tuple[list[dict], list[int]]:
    """Return (anchors_in_bbox, original_indices). Half-open on max
    edges so an anchor never falls in two tiles' cores."""
    out: list[dict] = []
    orig_idx: list[int] = []
    lo_lon, lo_lat, hi_lon, hi_lat = bbox
    for i, a in enumerate(anchors):
        if lo_lon <= a["lon"] < hi_lon and lo_lat <= a["lat"] < hi_lat:
            out.append(a)
            orig_idx.append(i)
    return out, orig_idx


def main() -> None:
    import gc
    import os
    t0 = time.time()
    print(f"[way-graph] seed highways: {SEED_HIGHWAYS}", flush=True)
    print(f"[way-graph] promotable:    {PROMOTABLE_HIGHWAYS}", flush=True)
    print(f"[way-graph] flood-fill match: same name OR same ref + shared vertex",
          flush=True)
    print(f"[way-graph] anchor buffer: {ANCHOR_BUFFER_M:.0f} m "
          f"(~{ANCHOR_BUFFER_M/1609.344:.2f} mi)", flush=True)
    print(f"[way-graph] tile: core {TILE_CORE_DEG}° + buffer {TILE_BUFFER_DEG}°",
          flush=True)

    # Load ALL anchors first (pure file read — cheap and needed for
    # tile placement).
    if OUT_NODES_GEOJSON.exists() and os.environ.get("WAY_GRAPH_USE_PRESELECTED_ANCHORS", "1") == "1":
        anchors_all = _load_anchors_geojson(OUT_NODES_GEOJSON)
        print(f"[way-graph] anchors: {len(anchors_all):,} loaded from "
              f"{OUT_NODES_GEOJSON.name} (pre-selected)", flush=True)
    else:
        with psycopg.connect(config.PG_DSN) as conn:
            anchors_all = _load_db_anchors(conn)
        print(f"[way-graph] anchors: {len(anchors_all):,} db "
              f"(villages disabled)", flush=True)

    tiles = _tile_bboxes(anchors_all)
    print(f"[way-graph] {len(tiles)} non-empty tile(s) covering all anchors",
          flush=True)

    # Accumulate results across tiles. Chain edges are deduped
    # canonically (min_ref, max_ref) → keep min cost witness. Anchor
    # participation is a set union across tiles.
    merged_edges: dict[tuple[str, str], dict] = {}
    participant_refs: set[str] = set()
    orig_orphan_refs: set[str] = set(a["ref"] for a in anchors_all)  # start pessimistic
    best_dist_by_orig: dict[int, float] = {}
    n_snaps_by_orig:   dict[int, int]   = {}
    # `nearest_road` per original anchor: we keep the minimum across tiles.
    nearest_road_by_orig: dict[int, float] = {}

    for ti, td in enumerate(tiles, 1):
        core = td["core"]; buf = td["buffer"]
        anchors_tile, orig_idx = _anchors_in_bbox(anchors_all, buf)
        print(f"[way-graph] tile {ti}/{len(tiles)} core="
              f"({core[0]:g},{core[1]:g})..({core[2]:g},{core[3]:g}) "
              f"buffered anchors={len(anchors_tile):,} "
              f"(of which core={td['n_core_anchors']:,})",
              flush=True)

        with psycopg.connect(config.PG_DSN) as conn:
            src, dst, edge_len, verts, _l2g = _load_subgraph(conn, bbox=buf)

        if len(src) == 0:
            print(f"[way-graph]   tile {ti}: empty subgraph — skip", flush=True)
            del src, dst, edge_len, verts, _l2g
            gc.collect()
            continue

        components = _compute_components(src, dst, len(verts))
        result = compute_chain_graph(anchors_tile, src, dst, edge_len, verts,
                                     components=components, verbose=True)

        # Take from the per-tile result whatever is best across tiles.
        for local_ki, orig_i in enumerate(orig_idx):
            if local_ki in result["per_anchor_count"]:
                orig_orphan_refs.discard(anchors_all[orig_i]["ref"])
                # Keep the minimum snap distance across tiles.
                d_here = result["best_dist"][local_ki]
                d_cur  = best_dist_by_orig.get(orig_i)
                if d_cur is None or d_here < d_cur:
                    best_dist_by_orig[orig_i] = d_here
                    n_snaps_by_orig[orig_i]   = result["per_anchor_count"][local_ki]

        for local_i, nr in enumerate(result["nearest_road"]):
            orig_i = orig_idx[local_i]
            cur = nearest_road_by_orig.get(orig_i)
            if cur is None or nr < cur:
                nearest_road_by_orig[orig_i] = nr

        # Emit edges. Keep the min-cost witness per canonical key.
        for e in result["chain_edges"]:
            key = tuple(sorted((e["a"], e["b"])))
            keep = merged_edges.get(key)
            if keep is None or e["cost_m"] < keep["cost_m"]:
                # Canonicalize a/b to key order for stable output.
                if key[0] == e["a"]:
                    merged_edges[key] = e
                else:
                    merged_edges[key] = {
                        "a": e["b"], "a_name": e["b_name"],
                        "b": e["a"], "b_name": e["a_name"],
                        "cost_m": e["cost_m"],
                        "geom": list(reversed(e["geom"])),
                    }

        participant_refs |= result["participant_refs"]

        print(f"[way-graph]   tile {ti}: +{len(result['chain_edges']):,} edges "
              f"(merged total: {len(merged_edges):,})", flush=True)

        del src, dst, edge_len, verts, _l2g, components, result
        gc.collect()

    # Synthesize a "result" dict compatible with _write_outputs.
    chain_edges = list(merged_edges.values())
    kept_idx = sorted(best_dist_by_orig.keys())
    kept_anchors = [anchors_all[i] for i in kept_idx]
    orphan_idx = [i for i in range(len(anchors_all))
                  if anchors_all[i]["ref"] in orig_orphan_refs]
    # Rebuild `per_anchor_count` / `best_dist` keyed on position-in-kept
    # (that's what _write_outputs indexes with).
    per_anchor_count = {i: n_snaps_by_orig.get(i, 0) for i in kept_idx}
    best_dist = {i: best_dist_by_orig.get(i, 0.0) for i in kept_idx}
    nearest_road = [nearest_road_by_orig.get(i, float("inf"))
                    for i in range(len(anchors_all))]
    # Convert dicts keyed by orig-anchor-idx into arrays keyed by
    # position-in-kept, as _write_outputs expects (it indexes with
    # kept_idx[i] into per_anchor_count / best_dist).
    result = {
        "chain_edges":       chain_edges,
        "kept_anchors":      kept_anchors,
        "kept_idx":          kept_idx,
        "orphan_idx":        orphan_idx,
        "per_anchor_count":  per_anchor_count,
        "best_dist":         best_dist,
        "nearest_road":      nearest_road,
        "participant_refs":  participant_refs,
    }
    print(f"[way-graph] merged: {len(chain_edges):,} chain edges, "
          f"{len(kept_anchors):,} kept anchors, "
          f"{len(orphan_idx):,} orphans",
          flush=True)

    _write_outputs(anchors_all, result)
    print(f"[way-graph] DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
