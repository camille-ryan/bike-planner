"""Paired-trunk router. Serves routing requests as in-memory walks,
lazy-loading trunk blobs from the paired_trunks.db on first access.

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

Memory model (lazy):
  Startup preloads only the SET of valid (src_city, dst_city) keys
  (~14k rows × 16 B = ~240 KB) plus cities.json + city_graph.json.
  Trunk blobs are decoded on first `.get()` and held in an LRU
  bounded by `_TRUNK_CACHE_MAX_ENTRIES` (default 512, override via
  env var TRUNK_CACHE_MAX_ENTRIES). Steady-state RSS scales with
  the cache, not the DB size. A cold trunk fetch is ~1 SQLite row
  read + ~10-50 ms of next_idx precompute; warm queries touch the
  cached entry with no I/O.
"""
from __future__ import annotations

import heapq
import json
import math
import os
import sqlite3
import threading
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import psycopg
from scipy.spatial import cKDTree

from . import db as db_mod
from .local_dijkstra import local_dijkstra_to_targets
from .settings import SPT_DIR


TRUNK_DTYPE = np.dtype([
    ("vid",  "<i8"),
    ("succ", "<i8"),
    ("lat",  "<f4"),
    ("lon",  "<f4"),
])
NULL_SENTINEL = np.int64(-1)

_MAX_WALK_STEPS = 200_000

# How many decoded trunks to keep resident. Each entry is one
# corridor's (vid, succ, lat, lon) blob view + a parallel int64
# next_idx array — very roughly 100-500 KB per common pair, with a
# long tail up to a few MB for very long corridors. A cache of 512
# entries budgets ~250 MB in the worst case and easily covers the
# working set of any single multi-leg route.
_TRUNK_CACHE_MAX_ENTRIES = int(os.environ.get("TRUNK_CACHE_MAX_ENTRIES", "512"))


def _decode_trunk_blob(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode one trunk-DB blob into (arr, next_idx). The blob layout
    matches TRUNK_DTYPE; next_idx maps each row to its successor's
    index in the same array (or -1 if the row is a trunk root)."""
    arr = np.frombuffer(blob, dtype=TRUNK_DTYPE)
    pos = np.searchsorted(arr["vid"], arr["succ"])
    in_range = pos < len(arr)
    pos_c = np.clip(pos, 0, len(arr) - 1)
    matched = in_range & (arr["vid"][pos_c] == arr["succ"])
    is_root = arr["succ"] == NULL_SENTINEL
    next_idx = np.where(matched & ~is_root, pos, -1).astype(np.int64)
    return arr, next_idx


class _TrunkStore:
    """Lazy in-memory cache over the paired_trunks.db.

    On construction, materializes only the set of valid (src, dst)
    keys — the SELECT is ~50 ms for 14k rows and lets callers filter
    the city_graph and short-circuit misses without SQL.

    `.get((src, dst))` returns the decoded (arr, next_idx) pair on
    demand, keeping the last `_TRUNK_CACHE_MAX_ENTRIES` entries
    resident under an LRU. Cold hits are a single SQL row read plus
    the numpy next_idx precompute.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        # sqlite3.Connection isn't safe for concurrent use from more
        # than one thread. FastAPI runs sync endpoints in a threadpool,
        # so two overlapping requests could race the same connection.
        # Serialize DB touches through a lock.
        self._conn_lock = threading.Lock()
        rows = conn.execute("SELECT src_city, dst_city FROM trunk_blobs")
        self.pairs: frozenset[tuple[int, int]] = frozenset(
            (int(a), int(b)) for a, b in rows
        )
        # Statistics for the /trunk/cache-stats endpoint / logs — the
        # LRU is otherwise a black box.
        self._hits = 0
        self._misses = 0
        # Bind the lru_cache to the instance without leaking `self`
        # into the cache key (functools.lru_cache would keep every
        # `self` alive forever).
        self._cached_fetch = lru_cache(maxsize=_TRUNK_CACHE_MAX_ENTRIES)(
            self._fetch_uncached
        )

    def _fetch_uncached(self, key: tuple[int, int]
                        ) -> tuple[np.ndarray, np.ndarray] | None:
        with self._conn_lock:
            row = self._conn.execute(
                "SELECT blob FROM trunk_blobs "
                "WHERE src_city=? AND dst_city=?",
                key,
            ).fetchone()
        if row is None:
            return None
        return _decode_trunk_blob(row[0])

    def get(self, key: tuple[int, int]
            ) -> tuple[np.ndarray, np.ndarray] | None:
        # Cheap presence check before touching the LRU — a chain-graph
        # miss is common (Dijkstra probes non-existent edges) and we
        # don't want those cluttering the cache.
        if key not in self.pairs:
            return None
        info = self._cached_fetch.cache_info()
        result = self._cached_fetch(key)
        info2 = self._cached_fetch.cache_info()
        if info2.hits > info.hits:
            self._hits += 1
        elif info2.misses > info.misses:
            self._misses += 1
        return result

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self.pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def cache_stats(self) -> dict:
        info = self._cached_fetch.cache_info()
        return {
            "n_pairs":     len(self.pairs),
            "resident":    info.currsize,
            "max_size":    info.maxsize,
            "hits":        self._hits,
            "misses":      self._misses,
        }


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
        # ref (e.g. "db:2", "ferry:34988792") -> city_idx. Enables
        # /trunk/route to accept an anchor by ref, skipping the
        # postgres snap + kd-tree nearest-anchor step entirely (used
        # for city-name tour planning, see task #38).
        self.city_idx_by_ref: dict[str, int] = {
            c["ref"]: int(c["city_idx"])
            for c in self.cities
            if c.get("ref")
        }

        # Open the trunk DB once and keep the connection open for the
        # life of the profile. `immutable=1` skips WAL/SHM creation
        # (blocked by the :ro mount) and tells SQLite the file won't
        # change — required for the lazy-load fast path. CAVEAT: if a
        # paired rebuild is writing to the SAME DB file concurrently,
        # immutable=1's change-detection skip will surface as
        # "database disk image is malformed". Restart the API only
        # when no rebuild is in flight, or wait for each per-profile
        # rebuild to finish before requesting routes for that profile.
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro&immutable=1",
            uri=True,
            check_same_thread=False,
        )
        conn.execute("PRAGMA mmap_size = 8589934592")  # 8 GB mmap window
        self.trunks = _TrunkStore(conn)

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
            f"[trunk_router] indexed profile '{profile}': "
            f"{len(self.trunks):,} trunk pairs (lazy; cache_max="
            f"{_TRUNK_CACHE_MAX_ENTRIES}), city_graph kept "
            f"{n_kept:,} edges (dropped {n_dropped:,} degenerate) "
            f"in {elapsed:.1f}s",
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


@lru_cache(maxsize=32)
def _load_anchor_spt_coords(profile: str, city_idx: int
                            ) -> tuple[np.ndarray, cKDTree] | None:
    """Load coords_lonlat from an anchor's polygon SPT + build a
    KDTree for nearest-vertex snap. Cached LRU(32).

    Used to snap raw lat/lon endpoints to a vertex that's guaranteed
    to be in this anchor's polygon SPT — postgres's `_snap_to_vertex`
    can pick a road vertex that isn't in the SPT (different bike-
    routable filter than the cell-file edge set), which breaks the
    parent-walk stitch. Snapping here restores the guarantee.
    """
    path = SPT_DIR / profile / "spt" / f"{city_idx}.npz"
    if not path.exists():
        return None
    with np.load(path) as d:
        ng     = np.asarray(d["node_global"]).astype(np.int64)
        coords = np.asarray(d["coords_lonlat"]).astype(np.float64)
    if len(ng) == 0:
        return None
    # Sort by ng ascending so callers can share the sorted order.
    if len(ng) > 1 and ng[1] < ng[0]:
        order = np.argsort(ng, kind="stable")
        ng = ng[order]
        coords = coords[order]
    return ng, coords, cKDTree(coords)


def _snap_coord_to_spt(profile: str, city_idx: int,
                       lon: float, lat: float) -> int | None:
    """Return the vid of the vertex in `city_idx`'s polygon SPT that's
    closest (by lat/lon) to the given coord, or None if no NPZ."""
    loaded = _load_anchor_spt_coords(profile, city_idx)
    if loaded is None:
        return None
    ng, _coords, tree = loaded
    _, idx = tree.query([lon, lat], k=1)
    return int(ng[int(idx)])


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
    start: tuple[float, float] | None, end: tuple[float, float] | None,
    profile: str,
    simplify_m: float = 100.0,
    start_ref: str | None = None,
    end_ref: str | None = None,
    via: list[tuple[str | None, tuple[float, float] | None]] | None = None,
) -> dict:
    """Plan a Graz→Cph-style route and return a GeoJSON Feature with
    a LineString geometry + diagnostic properties.

    Endpoints can be given either as a `(lon, lat)` tuple OR as an
    anchor `ref` string (e.g. "db:2", "ferry:34988792"). Passing a
    ref skips the postgres snap + nearest-anchor search entirely and
    routes directly from/to that anchor's snap vertex — the intended
    mode for city-name tour planning where there's no first- or
    last-mile bridge (task #38).

    `via`: optional list of intermediate stops as (ref, coord) tuples
    (each element has exactly one of the two set). The route runs
    pairwise chain-Dijkstra start→via[0]→via[1]→…→end, concatenates
    the chains (dedup at each waypoint), and walks the merged chain
    end-to-end. No per-waypoint first/last-mile bridges — a single
    first-mile stitch at start, a single last-mile stitch at end
    (task #48).

    `simplify_m`: drop polyline points closer together than this many
    meters before returning (default 100 m — visually equivalent to
    full-fidelity, ~3× smaller payload). Pass 0 to disable.
    """
    prof = _load_profile(profile)

    # Resolve endpoints. Ref mode short-circuits both snap and nearest;
    # lat/lon mode does postgres snap + kd-tree nearest as before. Each
    # end is resolved independently, so mixing (from_ref + to as lat/lon
    # or vice versa) is supported for partial tour planning.
    t0 = time.time()
    if start_ref is not None:
        start_city = prof.city_idx_by_ref.get(start_ref)
        if start_city is None:
            raise RuntimeError(f"unknown start ref '{start_ref}'")
        _sc = prof.cities[start_city]
        start = (float(_sc["lon"]), float(_sc["lat"]))
        start_vid = int(prof.snap_vid_by_city[start_city])
    if end_ref is not None:
        end_city = prof.city_idx_by_ref.get(end_ref)
        if end_city is None:
            raise RuntimeError(f"unknown end ref '{end_ref}'")
        _ec = prof.cities[end_city]
        end = (float(_ec["lon"]), float(_ec["lat"]))
        end_vid = int(prof.snap_vid_by_city[end_city])

    if (start_ref is None or end_ref is None):
        with db_mod.connect() as conn:
            if start_ref is None:
                start_vid = _snap_to_vertex(conn, start[0], start[1])
            if end_ref is None:
                end_vid   = _snap_to_vertex(conn, end[0],   end[1])
    t_snap = time.time() - t0

    if start_ref is None:
        start_city = _nearest_city(prof, start[0], start[1])
    if end_ref is None:
        end_city   = _nearest_city(prof, end[0],   end[1])

    # Resolve intermediate waypoints (task #48). Each becomes a forced
    # chain anchor between start_city and end_city.
    via_cities: list[int] = []
    if via:
        for (v_ref, v_coord) in via:
            if v_ref is not None:
                vc = prof.city_idx_by_ref.get(v_ref)
                if vc is None:
                    raise RuntimeError(f"unknown via ref '{v_ref}'")
                via_cities.append(int(vc))
            elif v_coord is not None:
                via_cities.append(int(_nearest_city(prof, v_coord[0], v_coord[1])))
            else:
                raise RuntimeError("via stop needs either ref or coord")

    t1 = time.time()
    # Pairwise chain-Dijkstra through all forced stops. Concatenate
    # chains with dedup at each waypoint boundary — the last city of
    # one leg's chain equals the first of the next.
    stops = [start_city] + via_cities + [end_city]
    chain: list[int] = []
    for k in range(len(stops) - 1):
        sub = _city_graph_dijkstra(prof.chain_adj, stops[k], stops[k + 1])
        if sub is None:
            raise RuntimeError(
                f"no city_graph path from city {stops[k]} to city {stops[k+1]}"
            )
        if k == 0:
            chain.extend(sub)
        else:
            # Dedup the boundary city (last of previous == first of this).
            if chain and sub and chain[-1] == sub[0]:
                chain.extend(sub[1:])
            else:
                chain.extend(sub)
    t_chain = time.time() - t1

    # V1-style paired-SPT walk (post-2026-07 rebuild): each (A, B)
    # trunk is a pruned slice of A.SPT covering the corridor into B.
    # `succ` points to A.SPT.parent (toward A-seed) with NULL_SENTINEL
    # wherever the parent falls outside kept — which happens exactly
    # at B-frontier vertices. Walking succ from any kept vertex thus
    # gradient-descends toward A-seed and terminates naturally at the
    # first B-frontier encountered. That terminating vid is in A∩B,
    # so it's also in B.SPT, and (typically) F-only for the next pair
    # (B, C) — chain joins are deterministic at real vids, no SYNTH.
    #
    # SKIP FIRST PAIR: user_snap is deep inside chain[0]'s territory
    # (probably even at chain[0]'s seed). Walking chain[0].parent
    # would either terminate immediately or drag us to chain[0]-seed
    # away from B. The first pair we USE is (chain[1], chain[2]);
    # user_snap sits in chain[1]'s F-only region and the walk carries
    # us toward chain[2]-frontier. For chains of length ≤2 the loop
    # is empty and the route is just the first/last mile straight.

    t2 = time.time()
    coords: list[list[float]] = [[float(start[0]), float(start[1])]]
    bridges: list[dict] = []

    chain_terminus_vid = start_vid
    chain_terminus_coord = (float(start[1]), float(start[0]))  # (lat, lon) for _walk

    # SKIP_MAX (also used by the in-loop skip-lookahead below).
    SKIP_MAX = 4

    # First-mile skip-lookahead (task #47). Extension of task #46: if
    # user's start_vid is directly present in some downstream trunk
    # (chain[k], chain[k+1]) for k in 1..SKIP_MAX, jump the walk loop
    # past legs 1..k-1 entirely. Chain-Dijkstra often plans through
    # short ferry-pier hops early in the chain whose polygon trunks
    # don't cover the direct corridor from start — those legs would
    # each bridge. If the start vid is already a valid entry to a
    # LATER trunk, we skip the pier-detour bridges entirely.
    walk_start_i = 1
    if len(chain) >= 3:
        for k in range(1, SKIP_MAX + 1):
            if k + 1 >= len(chain):
                break
            a_k, b_k = chain[k], chain[k + 1]
            trunk_k = prof.trunks.get((a_k, b_k))
            if trunk_k is None:
                continue
            arr_k = trunk_k[0]
            pos = int(np.searchsorted(arr_k["vid"], start_vid))
            if pos < len(arr_k) and int(arr_k["vid"][pos]) == start_vid:
                walk_start_i = k
                break
        # Mark skipped legs (1 .. walk_start_i - 1) so the client can
        # filter them from the paired-trunks visualization.
        for skipped in range(1, walk_start_i):
            bridges.append({
                "leg": skipped, "from_city": int(chain[skipped]),
                "to_city": int(chain[skipped + 1]),
                "distance_m": 0.0, "skipped": True,
            })

    # First-mile stitch: walk fm_start_vid's parent chain in chain[1]'s
    # polygon SPT, stopping at the first vertex that's a member of
    # trunk(chain[1], chain[2]). Use that as chain_terminus_vid so the
    # main walk loop's succ chain takes over naturally from there.
    if len(chain) >= 3 and walk_start_i == 1:
        a1, b1 = chain[1], chain[2]
        trunk_ab = prof.trunks.get((a1, b1))
        if trunk_ab is not None:
            arr_ab, next_idx_ab = trunk_ab
            trunk_vids = arr_ab["vid"]
            stitched = False
            # Best salvage candidate across stitch_city attempts,
            # in case the primary parent walk fails for every one.
            # Used only if no stitch_city succeeded via primary walk.
            best_salvage: dict | None = None
            for stitch_city in (int(a1), int(chain[0])):
                fm_start_vid = _snap_coord_to_spt(
                    profile, stitch_city, start[0], start[1],
                )
                if fm_start_vid is None:
                    continue
                try:
                    ng_s, par_s = _load_anchor_spt(profile, stitch_city)
                except FileNotFoundError:
                    continue
                loaded_coords = _load_anchor_spt_coords(profile, stitch_city)
                if loaded_coords is None:
                    continue
                _ng_c, spt_coords_arr, _tree = loaded_coords
                # Walk parent chain; stop at first trunk-membership hit.
                entry_pos = int(np.searchsorted(ng_s, fm_start_vid))
                if entry_pos >= len(ng_s) or int(ng_s[entry_pos]) != fm_start_vid:
                    continue
                i = entry_pos
                stitch_positions: list[int] = []
                trunk_entry_vid = -1
                trunk_entry_pos = -1
                for _ in range(_MAX_WALK_STEPS):
                    stitch_positions.append(i)
                    cur_vid = int(ng_s[i])
                    tpos = int(np.searchsorted(trunk_vids, cur_vid))
                    if tpos < len(trunk_vids) and int(trunk_vids[tpos]) == cur_vid:
                        trunk_entry_vid = cur_vid
                        trunk_entry_pos = tpos
                        break
                    p = int(par_s[i])
                    if p < 0:
                        break
                    i = p
                if trunk_entry_vid < 0:
                    # Primary parent walk terminated at an SPT seed
                    # without hitting the target trunk. Compute a
                    # salvage candidate for this stitch_city: the
                    # trunk vertex geographically closest to where
                    # the walk ended. Track it in `best_salvage` but
                    # don't apply yet — we want the MINIMUM bridge
                    # across all stitch_city attempts (some anchor's
                    # SPT might overlap the trunk far better than
                    # another's). Falls through to `continue` for
                    # this stitch_city, then the salvage is applied
                    # after the loop if no primary walk succeeded.
                    end_coord = spt_coords_arr[i]
                    lat_a = math.radians(float(end_coord[1]))
                    lat_v = np.radians(arr_ab["lat"].astype(np.float64))
                    lon_d = np.radians(
                        arr_ab["lon"].astype(np.float64)
                        - float(end_coord[0])
                    )
                    hav = (np.sin((lat_v - lat_a) / 2) ** 2
                           + math.cos(lat_a) * np.cos(lat_v)
                           * np.sin(lon_d / 2) ** 2)
                    best = int(np.argmin(hav))
                    salvage_bridge_m = float(
                        2 * 6_371_000.0 * np.arcsin(np.sqrt(hav[best])))
                    if (best_salvage is None
                            or salvage_bridge_m < best_salvage["bridge_m"]):
                        best_salvage = {
                            "bridge_m":         salvage_bridge_m,
                            "trunk_vid":        int(arr_ab["vid"][best]),
                            "trunk_pos":        best,
                            "stitch_positions": list(stitch_positions),
                            "spt_coords":       spt_coords_arr,
                        }
                    continue
                # Emit stitch coords (parent walk up to the trunk entry).
                # Drop the final one because the trunk walk emits it as
                # its first vertex — avoid duplicate. For salvage, the
                # last SPT vertex isn't the trunk entry, so we lose
                # its coord in the polyline but the missing point is
                # < 500 m from the trunk entry — invisible at map scale.
                for spos in stitch_positions[:-1]:
                    c = spt_coords_arr[spos]
                    coords.append([float(c[0]), float(c[1])])
                chain_terminus_vid = trunk_entry_vid
                chain_terminus_coord = (
                    float(arr_ab["lat"][trunk_entry_pos]),
                    float(arr_ab["lon"][trunk_entry_pos]),
                )
                stitched = True
                break

            if not stitched:
                # Real routing on the road-graph CSR. Loads per-1°
                # cell edge files at query time (LRU-cached). Uses an
                # A*-flavored intercept-point bias so the winning
                # trunk entry is the one with min `dijkstra_cost +
                # BIAS * geodesic_km_to(trip_end)`.
                #
                # KEY: iterate over legs 1..SKIP_MAX+1, not just
                # leg 1. Because paired-SPT polygons overlap in
                # geographic space (especially for wetland-detour
                # corridors), the source vertex is sometimes actually
                # CLOSER to a downstream trunk than to leg 1's trunk.
                # In the Senftenberg→Halbe case, Senftenberg is
                # geographically nearer to the (Lübbenau, Halbe)
                # trunk than to the (Vetschau, Lübbenau) trunk it
                # was walking, because the Lübbenau→Halbe path
                # extends south. Entering the far trunk directly
                # bypasses the wetland detour on the intermediate
                # legs.
                #
                # Score each candidate leg by dijkstra polyline km +
                # geodesic-remaining to the trip end. Pick the min.
                best_dijk_cand = None  # (score, k, poly, reached_vid, arr, next_idx)
                for k in range(1, SKIP_MAX + 2):
                    if k + 1 >= len(chain):
                        break
                    trunk_k = prof.trunks.get(
                        (int(chain[k]), int(chain[k + 1])))
                    if trunk_k is None:
                        continue
                    arr_k, next_idx_k = trunk_k
                    trunk_k_lonlats = np.column_stack([
                        arr_k["lon"].astype(np.float64),
                        arr_k["lat"].astype(np.float64),
                    ])
                    dijk_k = local_dijkstra_to_targets(
                        profile,
                        float(start[0]), float(start[1]),
                        int(start_vid),
                        arr_k["vid"],
                        max_cost=200_000.0,
                        target_lonlats=trunk_k_lonlats,
                        intercept_lonlat=(float(end[0]), float(end[1])),
                        intercept_bias=1000.0,
                    )
                    if dijk_k is None:
                        continue
                    poly_k, reached_k = dijk_k
                    poly_km = 0.0
                    for _j in range(1, len(poly_k)):
                        poly_km += _haversine_m(
                            poly_k[_j - 1][0], poly_k[_j - 1][1],
                            poly_k[_j][0],     poly_k[_j][1],
                        ) / 1000.0
                    remain_km = _haversine_m(
                        poly_k[-1][0], poly_k[-1][1],
                        float(end[0]),  float(end[1]),
                    ) / 1000.0
                    score = poly_km + remain_km
                    if best_dijk_cand is None or score < best_dijk_cand[0]:
                        best_dijk_cand = (score, k, poly_k, reached_k,
                                          arr_k, next_idx_k)
                if best_dijk_cand is not None:
                    _sc, k_win, dijk_poly, reached_vid, arr_ab, next_idx_ab = \
                        best_dijk_cand
                    for c in dijk_poly[1:-1]:
                        coords.append([float(c[0]), float(c[1])])
                    tpos = int(np.searchsorted(arr_ab["vid"], reached_vid))
                    if (tpos < len(arr_ab)
                            and int(arr_ab["vid"][tpos]) == reached_vid):
                        chain_terminus_vid = int(reached_vid)
                        chain_terminus_coord = (
                            float(arr_ab["lat"][tpos]),
                            float(arr_ab["lon"][tpos]),
                        )
                        # If we entered a later leg, skip the earlier
                        # ones. Also mark them as skipped for the viz.
                        for skipped in range(walk_start_i, k_win):
                            bridges.append({
                                "leg": skipped,
                                "from_city": int(chain[skipped]),
                                "to_city":   int(chain[skipped + 1]),
                                "distance_m": 0.0, "skipped": True,
                            })
                        walk_start_i = k_win
                        stitched = True

            if not stitched and best_salvage is not None:
                # Fallback: min-bridge salvage across stitch_city
                # attempts. Straight-line bridge of up to 30 km —
                # visible but bounded. Used when the local Dijkstra
                # failed (cells missing at edge of coverage, or
                # start_vid not in the loaded subgraph).
                SALVAGE_MAX_BRIDGE_M = 30_000.0
                if best_salvage["bridge_m"] <= SALVAGE_MAX_BRIDGE_M:
                    for spos in best_salvage["stitch_positions"][:-1]:
                        c = best_salvage["spt_coords"][spos]
                        coords.append([float(c[0]), float(c[1])])
                    trunk_entry_vid = best_salvage["trunk_vid"]
                    trunk_entry_pos = best_salvage["trunk_pos"]
                    chain_terminus_vid = trunk_entry_vid
                    chain_terminus_coord = (
                        float(arr_ab["lat"][trunk_entry_pos]),
                        float(arr_ab["lon"][trunk_entry_pos]),
                    )
                    stitched = True

            if not stitched:
                # Fallback to the prior bidirectional-LCA stitch. Can
                # produce a V-shape but at least gets us onto the trunk.
                R = 6_371_000.0
                lat_a = math.radians(start[1])
                lat_v = np.radians(arr_ab["lat"].astype(np.float64))
                lon_diff = np.radians(arr_ab["lon"].astype(np.float64) - start[0])
                hav = (np.sin((lat_v - lat_a) / 2) ** 2
                       + math.cos(lat_a) * np.cos(lat_v)
                       * np.sin(lon_diff / 2) ** 2)
                hav_dist = 2 * R * np.arcsin(np.sqrt(hav))
                b_front_mask = (next_idx_ab == NULL_SENTINEL)
                if b_front_mask.any():
                    b_front_pos = np.flatnonzero(b_front_mask)
                    pos_T = int(b_front_pos[np.argmin(hav_dist[b_front_pos])])
                else:
                    pos_T = int(np.argmin(hav_dist))
                T_vid = int(arr_ab["vid"][pos_T])
                if T_vid != start_vid:
                    fm_coords = None
                    for stitch_city in (int(a1), int(chain[0])):
                        fm_start_vid = _snap_coord_to_spt(
                            profile, stitch_city, start[0], start[1],
                        )
                        if fm_start_vid is None:
                            continue
                        try:
                            with db_mod.connect() as conn:
                                fm_coords = _last_mile(
                                    profile, stitch_city,
                                    fm_start_vid, T_vid, conn,
                                )
                        except Exception as exc:  # noqa: BLE001
                            print(f"[trunk_router] _first_mile fallback stitch "
                                  f"failed for city {stitch_city}: {exc}",
                                  flush=True)
                            fm_coords = None
                        if fm_coords:
                            break
                    if fm_coords:
                        for c in fm_coords[:-1]:
                            coords.append([float(c[0]), float(c[1])])
                        chain_terminus_vid = T_vid
                        chain_terminus_coord = (
                            float(arr_ab["lat"][pos_T]),
                            float(arr_ab["lon"][pos_T]),
                        )

    # In-loop skip-lookahead (task #46): before entering trunk i,
    # check if the incoming chain_terminus_vid is already a vertex in
    # some LATER trunk (i+k, i+k+1) for k in 1..SKIP_MAX. If so, skip
    # legs i..i+k-1 entirely and walk from that later trunk. This
    # handles chain-Dijkstra plans that route through short ferry-pier
    # hops whose polygon trunks don't cover the corridor between
    # chain-neighbors — the terminus falls in the DOWNSTREAM trunk's
    # kept set as an interior vertex, so bridging through intermediate
    # trunks (which snap-to-terminus, walk 1 vert, snap again) is
    # dead weight.
    # SKIP_MAX is defined above (near the first-mile block).
    i = walk_start_i
    while i < len(chain) - 1:
        a, b = chain[i], chain[i + 1]
        trunk = prof.trunks.get((a, b))
        if trunk is None:
            raise RuntimeError(f"missing trunk for ({a}, {b})")
        arr, next_idx = trunk

        # Only worth trying to skip when we DON'T have a clean entry
        # into the current trunk. If the current entry is already in
        # the trunk, no bridge would result — no need to look ahead.
        entry_pos = int(np.searchsorted(arr["vid"], chain_terminus_vid))
        entry_present = (entry_pos < len(arr)
                         and int(arr["vid"][entry_pos]) == chain_terminus_vid)
        if not entry_present:
            best_k = 0
            for k in range(1, SKIP_MAX + 1):
                if i + k >= len(chain) - 1:
                    break
                a_ahead, b_ahead = chain[i + k], chain[i + k + 1]
                trunk_ahead = prof.trunks.get((a_ahead, b_ahead))
                if trunk_ahead is None:
                    break
                arr_ahead = trunk_ahead[0]
                pos_a = int(np.searchsorted(arr_ahead["vid"], chain_terminus_vid))
                if (pos_a < len(arr_ahead)
                        and int(arr_ahead["vid"][pos_a]) == chain_terminus_vid):
                    best_k = k
                    break   # first (nearest) hit wins — don't overshoot chain plan
            if best_k > 0:
                for skipped in range(best_k):
                    j = i + skipped
                    bridges.append({
                        "leg": j, "from_city": int(chain[j]),
                        "to_city": int(chain[j + 1]),
                        "distance_m": 0.0, "skipped": True,
                    })
                i += best_k
                continue   # re-loop with the new i, entry_present will now be True

        walk = _walk(arr, next_idx, chain_terminus_vid,
                     bridge_target=chain_terminus_coord)
        if walk is None:
            raise RuntimeError(
                f"trunk walk failed at leg {i} (city {a} → {b}), "
                f"entry_vid={chain_terminus_vid}"
            )
        idxs, bridge_m = walk

        # Truncation fallback (pre-skip belt-and-suspenders): scan
        # visited positions from END backwards; truncate at last
        # vertex present in NEXT trunk so the following hop enters
        # cleanly. If skip fired above, this only prunes tail.
        if i + 2 < len(chain):
            next_a, next_b = chain[i + 1], chain[i + 2]
            next_trunk = prof.trunks.get((next_a, next_b))
            if next_trunk is not None:
                next_arr, _ = next_trunk
                next_vids = next_arr["vid"]
                visited_vids = arr["vid"][idxs]
                pos = np.searchsorted(next_vids, visited_vids)
                pos_c = np.clip(pos, 0, len(next_vids) - 1)
                in_next = (
                    (pos < len(next_vids))
                    & (next_vids[pos_c] == visited_vids)
                )
                if in_next.any():
                    last_in = int(np.flatnonzero(in_next).max())
                    if last_in < len(idxs) - 1:
                        idxs = idxs[: last_in + 1]

        if bridge_m > 0:
            a_ref = prof.cities[int(a)]["ref"]
            b_ref = prof.cities[int(b)]["ref"]
            # Classify: `ferry_leg` iff BOTH endpoints are ferry piers
            # (chain-Dijkstra picked a pier-to-pier ferry hop; the
            # straight-line is the ferry crossing itself). A pier↔land
            # bridge is a `gap`, not a ferry — the pier should have
            # road access to the land anchor, so a bridge means the
            # paired trunk didn't cover it (routing failure).
            kind = ("ferry_leg"
                    if a_ref.startswith("ferry:") and b_ref.startswith("ferry:")
                    else "gap")
            from_coord = ([float(coords[-1][0]), float(coords[-1][1])]
                          if coords else None)
            to_coord = [float(arr["lon"][idxs[0]]),
                        float(arr["lat"][idxs[0]])]
            bridges.append({
                "leg": i, "from_city": int(a), "to_city": int(b),
                "distance_m": round(bridge_m, 1),
                "kind": kind,
                "from_lonlat": from_coord,
                "to_lonlat":   to_coord,
            })
        for k in idxs:
            coords.append([float(arr["lon"][k]), float(arr["lat"][k])])
        chain_terminus_vid = int(arr["vid"][idxs[-1]])
        chain_terminus_coord = (
            float(arr["lat"][idxs[-1]]), float(arr["lon"][idxs[-1]]),
        )
        i += 1
    t_walk = time.time() - t2

    # First-mile: user start → first walked coord (straight bridge).
    if len(coords) > 1:
        first_mile_m = _haversine_m(
            float(start[0]), float(start[1]),
            coords[1][0], coords[1][1],
        )
        if first_mile_m > 0:
            bridges.insert(0, {"leg": "first_mile", "from_city": None,
                               "to_city": int(start_city),
                               "distance_m": round(first_mile_m, 1),
                               "kind": "gap",
                               "from_lonlat": [float(start[0]), float(start[1])],
                               "to_lonlat":   [coords[1][0], coords[1][1]]})

    # Last-mile (task #36 option-a: parent-walk concat).
    # `_last_mile` stitches parent-chains inside end_city's polygon
    # SPT to route from `chain_terminus_vid` to `end_vid`. Falls
    # back to a straight bridge if that anchor's SPT is unavailable
    # or the two walks don't converge on a shared vertex.
    t3 = time.time()
    last_walked = coords[-1]
    lm_coords: list[list[float]] | None = None
    if chain_terminus_vid != end_vid:
        # Mirror the first-mile fallback pattern: chain[-1]'s polygon
        # SPT is the natural home for last-mile stitch, but postgres
        # can snap end to a road vertex that's inside the polygon
        # geometrically but not reached by chain[-1]-seed's Dijkstra
        # (small disconnected road pockets, or vertices beyond where
        # the SPT expanded). Fall back to chain[-2] whose polygon
        # covers its own side + reaches into chain[-1]'s area.
        stitch_cities = [int(end_city)]
        if len(chain) >= 2:
            stitch_cities.append(int(chain[-2]))
        for stitch_city in stitch_cities:
            # Re-snap end coord to nearest vertex IN this stitch city's
            # polygon SPT — postgres's `_snap_to_vertex` uses a
            # different bike-routable filter than the polygon-SPT cell
            # edge set, so `end_vid` can be off by one road vertex
            # (a few meters geographically but a hard "not in SPT" for
            # the parent walk).
            lm_end_vid = _snap_coord_to_spt(
                profile, stitch_city, end[0], end[1],
            )
            if lm_end_vid is None:
                continue
            try:
                with db_mod.connect() as conn:
                    lm_coords = _last_mile(
                        profile, stitch_city, int(chain_terminus_vid),
                        lm_end_vid, conn,
                    )
            except Exception as exc:  # noqa: BLE001 — never let last-mile kill the route
                print(f"[trunk_router] _last_mile failed for city "
                      f"{stitch_city}: {exc}", flush=True)
                lm_coords = None
            if lm_coords:
                break
    if lm_coords:
        # Drop the first coord if it duplicates last_walked (LCA at
        # chain_terminus_vid) — _last_mile emits from entry to end
        # inclusive, and chain_terminus is already in `coords`.
        skip_first = (
            abs(lm_coords[0][0] - last_walked[0]) < 1e-7
            and abs(lm_coords[0][1] - last_walked[1]) < 1e-7
        )
        for c in lm_coords[1 if skip_first else 0:]:
            coords.append([float(c[0]), float(c[1])])
        # Also append the anchor coord itself if the last routed
        # vertex isn't already there (rare — anchor snap_vertex is
        # usually the last stitch vertex).
        if (abs(coords[-1][0] - end[0]) > 1e-7
                or abs(coords[-1][1] - end[1]) > 1e-7):
            coords.append([float(end[0]), float(end[1])])
    else:
        # Straight bridge fallback (pre-#36 behaviour).
        last_mile_m = _haversine_m(
            float(last_walked[0]), float(last_walked[1]),
            float(end[0]), float(end[1]),
        )
        coords.append([float(end[0]), float(end[1])])
        if last_mile_m > 0:
            bridges.append({"leg": "last_mile", "from_city": int(end_city),
                            "to_city": None,
                            "distance_m": round(last_mile_m, 1),
                            "kind": "gap",
                            "from_lonlat": [last_walked[0], last_walked[1]],
                            "to_lonlat":   [float(end[0]), float(end[1])]})
    t_last_mile = time.time() - t3

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
            "chain_city_idx":     [int(ci) for ci in chain],
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
