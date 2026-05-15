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
        # the DB won't change while we have it open — fine since we read
        # everything upfront and never re-query at runtime.
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


@lru_cache(maxsize=4)
def _load_profile(profile: str) -> _ProfileData:
    return _ProfileData(profile)


def preload(profile: str) -> None:
    """Force the profile to load. Call from FastAPI startup."""
    _load_profile(profile)


# ---------------------------------------------------------------------
# Postgres / snap helpers
# ---------------------------------------------------------------------

def _snap_to_vertex(
    conn: psycopg.Connection, lon: float, lat: float,
    search_radius_m: float = 20_000.0,
) -> int:
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
            ORDER BY v.the_geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326)
            LIMIT 1
            """,
            (lon, lat, expand_deg, lon, lat),
        )
        row = cur.fetchone()
    if not row:
        raise RuntimeError(
            f"no road vertex within ~{search_radius_m/1000:.0f} km "
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

    # Walk each chain edge's trunk.
    t2 = time.time()
    coords: list[list[float]] = []
    bridges: list[dict] = []
    entry_vid = (
        prof.snap_vid_by_city.get(start_city)
        if start_city != end_city else start_vid
    ) or start_vid
    cur_lat = start[1]; cur_lon = start[0]

    for i in range(len(chain) - 1):
        a, b = chain[i], chain[i + 1]
        trunk = prof.trunks.get((a, b))
        if trunk is None:
            # Should not happen because chain_adj is filtered to
            # buildable pairs, but be defensive.
            raise RuntimeError(f"missing trunk for ({a}, {b})")
        arr, next_idx = trunk
        walk = _walk(arr, next_idx, entry_vid, bridge_target=(cur_lat, cur_lon))
        if walk is None:
            raise RuntimeError(
                f"trunk walk failed at leg {i} (city {a} → {b}), "
                f"entry_vid={entry_vid}"
            )
        idxs, bridge_m = walk
        if bridge_m > 0:
            bridges.append({
                "leg": i,
                "from_city": a, "to_city": b,
                "distance_m": round(bridge_m, 1),
            })
        for k in idxs:
            coords.append([float(arr["lon"][k]), float(arr["lat"][k])])
        entry_vid = int(arr["vid"][idxs[-1]])
        cur_lat = float(arr["lat"][idxs[-1]])
        cur_lon = float(arr["lon"][idxs[-1]])
    t_walk = time.time() - t2

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
                "total":         round((time.time() - t0) * 1000, 1),
            },
        },
    }
