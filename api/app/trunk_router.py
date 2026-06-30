"""Paired-trunk router. Loads the entire trunk DB (blob schema) into
memory at startup, then serves routing requests as in-memory walks.

Architecture:
  1. Snap user's start/end (lon, lat) to road graph vertices (postgres).
  2. Pick start_city / end_city = nearest anchor by (lon, lat).
  3. Plan a city sequence via Dijkstra on city_graph restricted to pairs
     that actually have trunks in the DB (the 33 degenerate skips at
     build time are absent).
  4. Walk each chain edge's trunk by following `next_idx` from the
     current entry vertex until reaching a trunk root (succ == -1).
     The root vertex becomes the entry into the next chain edge's trunk.
  5. If an entry vertex isn't directly present in a trunk, fall back to
     the geographically nearest trunk vertex (haversine). This stands in
     for the unpruned-CSR local-Dijkstra bridge until that's implemented.
  6. Materialize the route polyline from the lat/lon embedded in each
     trunk row — no postgres roundtrip required.

Preload memory: ~5.7 GB of blobs as numpy arrays + ~1.9 GB of next_idx
arrays = ~7.5 GB resident. Fits comfortably alongside postgres on a
16 GB host. Startup load is ~30-60 s warm (one cold pass through the
DB file).
"""
from __future__ import annotations

import heapq
import json
import math
import sqlite3
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import psycopg
from scipy.spatial import cKDTree

from . import db as db_mod
from .settings import SPT_DIR


TRUNK_DTYPE = np.dtype([
    ("vid",  "<i8"),
    ("succ", "<i8"),
    ("lat",  "<f4"),
    ("lon",  "<f4"),
])
NULL_SENTINEL = np.int64(-1)

_MAX_WALK_STEPS = 200_000


class _ProfileData:
    """All per-profile artifacts the router needs, loaded once at startup.

    Holds:
      - cities, city_kdtree, snap_vid_by_city
      - chain_adj (forward city_graph, filtered to buildable pairs)
      - trunks: dict[(src, dst), (arr, next_idx)] — corridor cache
    """

    def __init__(self, profile: str):
        self.profile = profile
        base = SPT_DIR / profile

        if not (base / "city_graph.json").exists():
            raise FileNotFoundError(
                f"no city_graph.json for profile '{profile}' at {base}"
            )
        db_path = base / "paired_trunks.db"
        if not db_path.exists():
            raise FileNotFoundError(
                f"no paired_trunks.db for profile '{profile}' at {db_path}"
            )

        t0 = time.time()
        with open(base / "cities.json") as fh:
            self.cities: list[dict] = json.load(fh)
        self.city_kdtree = cKDTree(
            np.array([(c["lon"], c["lat"]) for c in self.cities])
        )
        self.snap_vid_by_city: dict[int, int] = {
            int(c["city_idx"]): int(c["snap_vertex_id"])
            for c in self.cities
            if c.get("snap_vertex_id") is not None
        }

        # Preload all trunk blobs into a dict. `immutable=1` skips WAL/
        # SHM file creation (which the :ro mount blocks) and tells SQLite
        # the DB won't change — fine since we read everything upfront and
        # never re-query at runtime. CAVEAT: if a paired rebuild is
        # writing to the SAME DB file concurrently, immutable=1's
        # change-detection skip will surface as "database disk image is
        # malformed". Restart the API only when no rebuild is in flight,
        # or wait for each per-profile rebuild to finish before requesting
        # routes for that profile.
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro&immutable=1", uri=True,
        )
        conn.execute("PRAGMA mmap_size = 8589934592")  # 8 GB mmap window
        self.trunks: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        n_total_rows = 0
        for src, dst, n, blob in conn.execute(
            "SELECT src_city, dst_city, n_rows, blob FROM trunk_blobs"
        ):
            arr = np.frombuffer(blob, dtype=TRUNK_DTYPE)
            # Precompute next_idx for each row (successor's index in the
            # same array, or -1 if it's a trunk root / orphan).
            pos = np.searchsorted(arr["vid"], arr["succ"])
            in_range = pos < len(arr)
            pos_c = np.clip(pos, 0, len(arr) - 1)
            matched = in_range & (arr["vid"][pos_c] == arr["succ"])
            is_root = arr["succ"] == NULL_SENTINEL
            next_idx = np.where(matched & ~is_root, pos, -1).astype(np.int64)
            self.trunks[(int(src), int(dst))] = (arr, next_idx)
            n_total_rows += n
        conn.close()

        # Filter the city_graph to pairs we actually have trunks for.
        # The remaining `chain_adj` edges are guaranteed to be walkable.
        with open(base / "city_graph.json") as fh:
            cg = json.load(fh)
        self.chain_adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
        n_kept = 0; n_dropped = 0
        for fa, tb, w in zip(cg["from_city"], cg["to_city"], cg["weight"]):
            a, b = int(fa), int(tb)
            if (a, b) in self.trunks:
                self.chain_adj[a].append((b, float(w)))
                n_kept += 1
            else:
                n_dropped += 1

        elapsed = time.time() - t0
        print(
            f"[trunk_router] preloaded profile '{profile}': "
            f"{len(self.trunks):,} trunks, {n_total_rows:,} vertices, "
            f"city_graph kept {n_kept:,} edges (dropped {n_dropped:,} "
            f"degenerate) in {elapsed:.1f}s",
            flush=True,
        )


@lru_cache(maxsize=8)
def _load_profile(profile: str) -> _ProfileData:
    return _ProfileData(profile)


def preload(profile: str) -> None:
    """Force the profile to load. Call from FastAPI startup."""
    _load_profile(profile)


# ---------------------------------------------------------------------
# Postgres / snap helpers
# ---------------------------------------------------------------------

_NON_BIKE_HIGHWAY = (
    "pedestrian", "footway", "steps", "platform", "corridor",
    "elevator", "escalator",
)


def _snap_to_vertex(
    conn: psycopg.Connection, lon: float, lat: float,
    search_radius_m: float = 20_000.0,
) -> int:
    """Nearest ways_vertices_pgr.id that lies on the bike-routable road
    network. We REQUIRE the snapped vertex to touch at least one edge
    with a bike-routable `highway` tag — plain nearest-neighbor was
    landing on pedestrian-only orphans (5 m pebblestone footpaths very
    close to city centers) that aren't in any anchor's SPT, breaking
    the entire route. Same exclusion list as
    pgrouting/route_city_pairs._snap_vertex.
    """
    expand_deg = max(0.25, search_radius_m / 50_000.0)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT v.id
            FROM ways_vertices_pgr v
            WHERE v.the_geom && ST_Expand(
                ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                %s
            )
              AND EXISTS (
                SELECT 1 FROM ways w
                WHERE (w.source = v.id OR w.target = v.id)
                  AND w.highway <> ALL(%s::text[])
              )
            ORDER BY v.the_geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326)
            LIMIT 1
            """,
            (lon, lat, expand_deg, list(_NON_BIKE_HIGHWAY), lon, lat),
        )
        row = cur.fetchone()
    if not row:
        raise RuntimeError(
            f"no bike-routable road vertex within ~{search_radius_m/1000:.0f} km "
            f"of ({lon}, {lat})"
        )
    return int(row[0])


def _nearest_city(prof: _ProfileData, lon: float, lat: float) -> int:
    _, idx = prof.city_kdtree.query([lon, lat], k=1)
    return int(prof.cities[idx]["city_idx"])


# ---------------------------------------------------------------------
# city_graph Dijkstra
# ---------------------------------------------------------------------

def _city_graph_dijkstra(
    adj: dict[int, list[tuple[int, float]]], src: int, dst: int,
) -> list[int] | None:
    if src == dst:
        return [src]
    dist = {src: 0.0}
    parent: dict[int, int] = {}
    heap: list[tuple[float, int]] = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            path = [u]
            while u != src:
                u = parent[u]; path.append(u)
            return list(reversed(path))
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd; parent[v] = u
                heapq.heappush(heap, (nd, v))
    return None


# ---------------------------------------------------------------------
# Trunk walking
# ---------------------------------------------------------------------

def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6_371_000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _walk(
    arr: np.ndarray, next_idx: np.ndarray, entry_vid: int,
    bridge_target: tuple[float, float] | None,
) -> tuple[list[int], float] | None:
    """Walk a trunk by chasing next_idx pointers from entry_vid.
    Returns (path_indices, bridge_m) or None if the walk fails.
    """
    bridge_m = 0.0
    pos = int(np.searchsorted(arr["vid"], entry_vid))
    if pos >= len(arr) or int(arr["vid"][pos]) != entry_vid:
        if bridge_target is None or len(arr) == 0:
            return None
        tlat, tlon = bridge_target
        R = 6_371_000.0
        lat_a = math.radians(tlat)
        lat_v = np.radians(arr["lat"].astype(np.float64))
        lon_diff = np.radians(arr["lon"].astype(np.float64) - tlon)
        a = (np.sin((lat_v - lat_a) / 2) ** 2
             + math.cos(lat_a) * np.cos(lat_v)
             * np.sin(lon_diff / 2) ** 2)
        dist = 2 * R * np.arcsin(np.sqrt(a))
        pos = int(np.argmin(dist))
        bridge_m = float(dist[pos])

    out: list[int] = []
    i = pos
    steps = 0
    while i >= 0 and steps < _MAX_WALK_STEPS:
        out.append(i)
        i = int(next_idx[i])
        steps += 1
    if steps >= _MAX_WALK_STEPS:
        return None
    return out, bridge_m


# ---------------------------------------------------------------------
# Last-mile: per-anchor SPT walk to/from a specific vertex
# ---------------------------------------------------------------------
#
# The trunk's `paired_(A, B)` walks from any vertex in A's catchment
# toward A's center and terminates at the B-frontier (just inside B's
# catchment from A's side). It never reaches the user's actual
# destination vertex `end_vid`. For that, we run a parent-walk on the
# destination city's per-anchor backward SPT npz:
#
#   * Walk parent from `end_vid` toward city center → set of vertices E.
#   * Walk parent from the chain terminus (or start_vid for short trips)
#     toward city center, stopping when we hit a vertex in E.
#   * That vertex is the LCA. Stitch: terminus → … → LCA → … → end_vid
#     (the second half is read backward from the end_vid walk).
#
# This isn't strictly the shortest path (worst case it overshoots to
# city center then back to end_vid) but it's a single-traversal stitch
# that needs only `parent_local` from the npz — no scipy dijkstra at
# request time.

@lru_cache(maxsize=32)
def _load_anchor_spt(profile: str, city_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-anchor backward SPT npz (node_global, parent_local).

    Cached LRU(32) so popular destination cities stay hot. node_global
    is upcast to int64 for clean searchsorted against int64 vids.

    Accepts either NPZ schema:
      - compute_spts_multi: keys node_global + parent_local
      - compute_spts_polygon: keys node_global + parent
    """
    path = SPT_DIR / profile / "spt" / f"{city_idx}.npz"
    with np.load(path) as d:
        ng = np.asarray(d["node_global"]).astype(np.int64)
        par_key = "parent_local" if "parent_local" in d.files else "parent"
        par = np.asarray(d[par_key]).astype(np.int32)
    # If parent indices are sorted in the NPZ's original (Dijkstra-output)
    # order but node_global was later sorted ASC for searchsorted, the
    # parent[] indices are stale. Polygon NPZs save node_global in
    # Dijkstra-output order; if it's not strictly ascending, sort + reindex.
    if len(ng) > 1 and ng[1] < ng[0]:
        order = np.argsort(ng, kind="stable")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        valid = (par >= 0) & (par < len(par))
        new_par = par.copy()
        new_par[valid] = inv[par[valid]]
        ng = ng[order]
        par = new_par[order].astype(np.int32)
    return ng, par


def _fetch_vertex_coords(
    conn: psycopg.Connection, vids: list[int],
) -> dict[int, tuple[float, float]]:
    """Bulk-fetch (lon, lat) for a list of global vids. Used by the
    last-mile reconstruction to attach geometry to vertices that
    aren't already inside a preloaded trunk."""
    if not vids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, ST_X(the_geom), ST_Y(the_geom)
            FROM ways_vertices_pgr
            WHERE id = ANY(%s::bigint[])
            """,
            (vids,),
        )
        return {int(r[0]): (float(r[1]), float(r[2])) for r in cur.fetchall()}


def _last_mile(
    profile: str, city_idx: int, entry_vid: int, end_vid: int,
    conn: psycopg.Connection,
) -> list[list[float]] | None:
    """Stitch parent walks within one anchor's SPT to produce coords
    from `entry_vid` to `end_vid`. Returns [[lon, lat], …] or None if
    either endpoint isn't in this SPT or the walks don't converge.

    Algorithm: walk `parent_local` from end_vid → seed (collect set E),
    then walk from entry_vid until we hit a vertex in E. Concatenate
    the entry walk up-to-LCA with the reversed end walk LCA-to-end.
    """
    try:
        ng, par = _load_anchor_spt(profile, city_idx)
    except FileNotFoundError:
        return None
    if len(ng) == 0:
        return None

    def pos_of(vid: int) -> int:
        p = int(np.searchsorted(ng, vid))
        if p >= len(ng) or int(ng[p]) != vid:
            return -1
        return p

    end_pos = pos_of(end_vid)
    if end_pos < 0:
        return None
    entry_pos = pos_of(entry_vid)
    if entry_pos < 0:
        return None

    # Walk end_vid → seed, building path indices and a position→list-idx
    # map for cheap LCA lookup.
    end_walk: list[int] = []
    end_walk_pos: dict[int, int] = {}
    i = end_pos
    steps = 0
    while i >= 0 and steps < _MAX_WALK_STEPS:
        if i in end_walk_pos:
            break  # cycle protection (shouldn't happen in a tree)
        end_walk_pos[i] = len(end_walk)
        end_walk.append(i)
        i = int(par[i])
        steps += 1

    # Walk entry_vid → seed until we hit a vertex that's also on the end walk.
    entry_walk: list[int] = []
    i = entry_pos
    steps = 0
    lca_in_end: int | None = None
    while i >= 0 and steps < _MAX_WALK_STEPS:
        if i in end_walk_pos:
            lca_in_end = end_walk_pos[i]
            entry_walk.append(i)
            break
        entry_walk.append(i)
        i = int(par[i])
        steps += 1

    if lca_in_end is None:
        # No shared index. Most common cause: the SPT is multi-source
        # (Wien's seeds = 1 km bbox + snap_vertex_id), and the two walks
        # converged on DIFFERENT seeds. Each seed is a separate parent-
        # tree root (parent_local = -1) with no edge to the others, so
        # the walks can't meet by index. Fall back: glue entry_walk's
        # last vertex directly to end_walk's last vertex. The seeds are
        # all within 1 km of the city center, so the visual jump is
        # small. (Pathologically, if either walk hit max steps early,
        # the glue might be larger — acceptable.)
        full_idx_path = entry_walk + list(reversed(end_walk))
    else:
        # LCA found cleanly. Drop the duplicate join vertex so coords
        # don't repeat.
        full_idx_path = entry_walk + list(reversed(end_walk[:lca_in_end]))

    # Fetch lon/lat for every vid in the path. One postgres roundtrip.
    vids = [int(ng[p]) for p in full_idx_path]
    coords_by_vid = _fetch_vertex_coords(conn, vids)
    coords: list[list[float]] = []
    for v in vids:
        c = coords_by_vid.get(v)
        if c is None:
            continue  # missing coord — skip rather than break the route
        coords.append([c[0], c[1]])
    return coords


# ---------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------

def _decimate_polyline(
    coords: list[list[float]], min_step_m: float,
) -> list[list[float]]:
    """Keep coords[0]; drop subsequent coords closer than `min_step_m`
    (haversine) to the last kept point. Always keep coords[-1] so the
    line still reaches the destination.

    For a 1,300 km route with ~one point per 35 m (35 K coords),
    `min_step_m=100` drops it to ~13 K — visually identical at any
    realistic zoom, ~3× smaller payload, ~3× faster client parse.
    """
    if len(coords) <= 2 or min_step_m <= 0:
        return coords
    out = [coords[0]]
    last_lon, last_lat = coords[0]
    for i in range(1, len(coords) - 1):
        lon, lat = coords[i]
        if _haversine_m(last_lon, last_lat, lon, lat) >= min_step_m:
            out.append(coords[i])
            last_lon, last_lat = lon, lat
    out.append(coords[-1])
    return out


def route(
    start: tuple[float, float], end: tuple[float, float], profile: str,
    simplify_m: float = 100.0,
) -> dict:
    """Plan a Graz→Cph-style route and return a GeoJSON Feature with
    a LineString geometry + diagnostic properties.

    `simplify_m`: drop polyline points closer together than this many
    meters before returning (default 100 m — visually equivalent to
    full-fidelity, ~3× smaller payload). Pass 0 to disable.
    """
    prof = _load_profile(profile)

    t0 = time.time()
    with db_mod.connect() as conn:
        start_vid = _snap_to_vertex(conn, start[0], start[1])
        end_vid   = _snap_to_vertex(conn, end[0],   end[1])
    t_snap = time.time() - t0

    start_city = _nearest_city(prof, start[0], start[1])
    end_city   = _nearest_city(prof, end[0],   end[1])

    t1 = time.time()
    chain = _city_graph_dijkstra(prof.chain_adj, start_city, end_city)
    t_chain = time.time() - t1
    if chain is None:
        raise RuntimeError(
            f"no city_graph path from city {start_city} to city {end_city}"
        )

    # NEW (polygon-SPT era): trunks are bracketed with synthetic
    # anchor-center vids (-2-ci for city ci). Every (A, B) trunk
    # starts with SYNTH_A and ends with SYNTH_B, so chain joins are
    # deterministic — no SPT-membership skip-ahead, no haversine
    # fallback. The first leg's entry vid is SYNTH_(start_city) and
    # the last leg ends at SYNTH_(end_city). User-supplied start/end
    # lonlat become first-mile / last-mile straight segments to the
    # anchor centers (the polygon SPT's region semantics already
    # absorb up to ~1 km of slack around each anchor).
    SYNTH = lambda ci: -2 - int(ci)

    t2 = time.time()
    coords: list[list[float]] = []
    bridges: list[dict] = []

    # First-mile: user start → start_city center (straight segment).
    sc_info = next((c for c in prof.cities
                    if int(c["city_idx"]) == int(start_city)), None)
    if sc_info is None:
        raise RuntimeError(f"start_city {start_city} not in cities.json")
    coords.append([float(start[0]), float(start[1])])
    coords.append([float(sc_info["lon"]), float(sc_info["lat"])])
    first_mile_m = _haversine_m(
        float(start[0]), float(start[1]),
        float(sc_info["lon"]), float(sc_info["lat"]),
    )
    if first_mile_m > 0:
        bridges.append({"leg": "first_mile", "from_city": None,
                        "to_city": int(start_city),
                        "distance_m": round(first_mile_m, 1)})

    # Chain walk: every leg's trunk is entered via SYNTH_(from_city)
    # and ends at SYNTH_(to_city) — searchsorted-clean joins.
    chain_terminus_vid = SYNTH(start_city)
    for i in range(0, len(chain) - 1):
        a, b = chain[i], chain[i + 1]
        trunk = prof.trunks.get((a, b))
        if trunk is None:
            raise RuntimeError(f"missing trunk for ({a}, {b})")
        arr, next_idx = trunk
        walk = _walk(arr, next_idx, chain_terminus_vid,
                     bridge_target=None)
        if walk is None:
            raise RuntimeError(
                f"trunk walk failed at leg {i} (city {a} → {b}), "
                f"entry_vid={chain_terminus_vid}"
            )
        idxs, bridge_m = walk
        if bridge_m > 0:
            bridges.append({
                "leg": i, "from_city": int(a), "to_city": int(b),
                "distance_m": round(bridge_m, 1),
            })
        for k in idxs:
            coords.append([float(arr["lon"][k]), float(arr["lat"][k])])
        chain_terminus_vid = int(arr["vid"][idxs[-1]])
    t_walk = time.time() - t2

    # Last-mile: end_city center → user end (straight segment).
    ec_info = next((c for c in prof.cities
                    if int(c["city_idx"]) == int(end_city)), None)
    if ec_info is None:
        raise RuntimeError(f"end_city {end_city} not in cities.json")
    last_mile_m = _haversine_m(
        float(ec_info["lon"]), float(ec_info["lat"]),
        float(end[0]), float(end[1]),
    )
    coords.append([float(end[0]), float(end[1])])
    if last_mile_m > 0:
        bridges.append({"leg": "last_mile", "from_city": int(end_city),
                        "to_city": None,
                        "distance_m": round(last_mile_m, 1)})
    t3 = time.time(); t_last_mile = time.time() - t3

    # Cheap gross-length stat for the response (over the full polyline,
    # before simplification — that's the geometrically correct length).
    gross_m = 0.0
    for k in range(1, len(coords)):
        gross_m += _haversine_m(
            coords[k-1][0], coords[k-1][1], coords[k][0], coords[k][1],
        )

    full_vertex_count = len(coords)
    if simplify_m > 0:
        coords = _decimate_polyline(coords, simplify_m)

    # Human-readable chain names for UI display.
    name_by_idx = {int(c["city_idx"]): c["name"] for c in prof.cities}
    chain_names = [name_by_idx.get(ci, str(ci)) for ci in chain]

    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
        "properties": {
            "profile":            profile,
            "chain_length":       len(chain),
            "chain_names":        chain_names,
            "leg_count":          len(chain) - 1,
            "vertex_count":       len(coords),
            "vertex_count_full":  full_vertex_count,
            "simplify_m":         simplify_m,
            "gross_length_m":     round(gross_m, 1),
            "bridges":            bridges,
            "start_city_idx":     start_city,
            "end_city_idx":       end_city,
            "snap_start_vid":     start_vid,
            "snap_end_vid":       end_vid,
            "timings_ms": {
                "snap":          round(t_snap * 1000, 1),
                "chain":         round(t_chain * 1000, 1),
                "walk":          round(t_walk * 1000, 1),
                "last_mile":     round(t_last_mile * 1000, 1),
                "total":         round((time.time() - t0) * 1000, 1),
            },
        },
    }
