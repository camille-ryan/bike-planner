"""Chainless SPT router.

Reads the per-city SPTs produced by `pgrouting/compute_spts.py`:

    data/spt/<profile>/
      cities.json
      city_graph.json
      spt/<city_idx>.npz       per-city subgraph SPT (chainless)

There is **no** global node-coords array or Voronoi cell assignment in
this layout — cities' SPTs overlap. Coordinates and snap-to-nearest are
served by Postgres (GIST index on `ways_vertices_pgr.the_geom`).

Algorithm:
  1. Snap requested start/end lon/lats to global graph vertices via
     Postgres (`<->` KNN operator on the spatial index).
  2. Pick start_city / end_city as the cities.json anchor closest to
     each endpoint (KDTree on anchor coords).
  3. Plan a city sequence on the **reversed** city_graph from
     start_city to end_city. The reversal is load-bearing: a forward
     edge (A, B, w) in city_graph means "A.SPT covers B.polygon", but
     to walk a leg from A's polygon into B's polygon we need
     "B.SPT covers A.polygon" — that's the reversed edge.
  4. Walk gradients leg by leg: in c_{i+1}.SPT, walk parent_local
     pointers from current_vertex until we hit a seed (cost == 0,
     which marks the multi-source roots of c_{i+1}.SPT — i.e.
     c_{i+1}'s polygon vertices).
  5. Final leg: LCA inside end_city.SPT to connect the last seed-arrival
     to the end vertex (both reachable in end_city.SPT).
  6. Materialize coords from `ways_vertices_pgr` in one batch query.

Per-city SPTs are mmap-loaded on demand and cached in-process.
"""
from __future__ import annotations

import heapq
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import psycopg
from scipy.spatial import cKDTree

from . import db
from .settings import SPT_DIR


_MAX_LEG_STEPS = 200_000


# ---------------------------------------------------------------------
# Profile-scoped singletons. Loaded once per profile via _load_profile.
# ---------------------------------------------------------------------

class _ProfileData:
    def __init__(self, profile: str):
        self.profile = profile
        base = SPT_DIR / profile
        if not (base / "city_graph.json").exists():
            raise FileNotFoundError(
                f"no chainless SPT data for profile '{profile}' at {base} "
                f"(city_graph.json missing — preprocess incomplete?)"
            )

        with open(base / "cities.json") as fh:
            self.cities: list[dict] = json.load(fh)
        # cKDTree over anchor (lon, lat) for nearest-city lookup.
        self.city_kdtree = cKDTree(
            np.array([(c["lon"], c["lat"]) for c in self.cities])
        )

        with open(base / "city_graph.json") as fh:
            cg = json.load(fh)
        # `chain_adj`: for routing chain c_0 → c_1 → ... → c_k, each
        # step c_i → c_{i+1} requires c_{i+1}.SPT to cover c_i.polygon.
        # The original city_graph has forward edge (A, B, w) =
        # "A.SPT covers B.polygon". Reversing it gives the right
        # adjacency for Dijkstra'ing chain-feasible paths.
        self.chain_adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for fa, tb, w in zip(cg["from_city"], cg["to_city"], cg["weight"]):
            # forward (fa, tb): fa.SPT covers tb.polygon
            # to step into tb, need tb.SPT to cover (something) — but
            # this edge tells us about fa.SPT, not tb.SPT. Reverse:
            self.chain_adj[int(tb)].append((int(fa), float(w)))

        self.spt_dir = base / "spt"

    @lru_cache(maxsize=128)
    def spt(self, city_idx: int) -> dict[str, np.ndarray]:
        """Fully load a per-city SPT into RAM (no mmap). 9P/WSL2 page
        faults made the per-step lookahead in route() pay 1-3 ms per
        searchsorted; loading into anonymous memory once costs ~10-50 ms
        per SPT but turns subsequent searchsorts into pure CPU ~5 µs.

        128-entry LRU; a Graz→Cph chain has ~80 SPTs ≈ ~2 GB RAM peak.
        """
        path = self.spt_dir / f"{city_idx}.npz"
        if not path.exists():
            raise FileNotFoundError(f"no SPT for city_idx={city_idx}")
        with np.load(path) as f:
            return {
                "node_global":  np.asarray(f["node_global"]),
                "parent_local": np.asarray(f["parent_local"]),
                "cost":         np.asarray(f["cost"]),
            }


@lru_cache(maxsize=4)
def _load_profile(profile: str) -> _ProfileData:
    return _ProfileData(profile)


# ---------------------------------------------------------------------
# Postgres helpers
# ---------------------------------------------------------------------

def _snap_to_vertex(
    conn: psycopg.Connection, lon: float, lat: float,
    search_radius_m: float = 20_000.0,
) -> int:
    """Find the global vertex id nearest to (lon, lat)."""
    expand_deg = max(0.25, search_radius_m / 50_000.0)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT v.id
            FROM ways_vertices_pgr v
            WHERE v.the_geom && ST_Expand(
                ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                %s
            )
            ORDER BY v.the_geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326)
            LIMIT 1
        """, (lon, lat, expand_deg, lon, lat))
        row = cur.fetchone()
    if not row:
        raise RuntimeError(
            f"no road vertex within ~{search_radius_m/1000:.0f} km "
            f"of ({lon}, {lat})"
        )
    return int(row[0])


def _coords_for_vertices(
    conn: psycopg.Connection, global_ids: list[int],
) -> list[list[float]]:
    """Materialize [[lon, lat], ...] for an ordered list of global vertex
    ids. Order-preserving via UNNEST WITH ORDINALITY."""
    if not global_ids:
        return []
    ids = list(map(int, global_ids))
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ST_X(v.the_geom), ST_Y(v.the_geom)
            FROM ways_vertices_pgr v
            JOIN unnest(%s::bigint[]) WITH ORDINALITY AS u(vid, ord)
              ON v.id = u.vid
            ORDER BY u.ord
        """, (ids,))
        return [[float(r[0]), float(r[1])] for r in cur.fetchall()]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _local_idx(spt: dict, global_node: int) -> int:
    """Return the local index of `global_node` in `spt['node_global']`,
    or -1 if not present."""
    arr = spt["node_global"]
    pos = int(np.searchsorted(arr, global_node))
    if pos >= len(arr) or int(arr[pos]) != int(global_node):
        return -1
    return pos


def _city_graph_dijkstra(
    adj: dict[int, list[tuple[int, float]]], src: int, dst: int,
) -> list[int] | None:
    """Plain Dijkstra on a sparse graph. Returns the city-id sequence
    src ... dst, or None if no path."""
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
                u = parent[u]
                path.append(u)
            return list(reversed(path))
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    return None


def _walk_into_seeds(
    spt: dict, start_global: int,
    must_be_in: np.ndarray | None = None,
) -> np.ndarray | None:
    """Walk parent_local in `spt` from `start_global` until we land on a
    seed vertex (cost == 0, marking a multi-source SPT root). Returns
    the global-vertex sequence as int64 array, ending with the seed.

    `must_be_in`, if provided, is a sorted int32 array of vertex ids
    (e.g., the next-next city's `node_global`) that the landing seed
    must also appear in. Use this for look-ahead between consecutive
    legs — chain_adj guarantees an overlap exists, but the overlap
    may not be the seed we'd walk to by default. With must_be_in, we
    keep walking past incompatible seeds until either the chain
    terminates (returning the last seed found, even if incompatible —
    caller can re-route) or we land on a compatible seed.

    Returns None if `start_global` isn't in this SPT.
    """
    node_global = spt["node_global"]
    parent_local = spt["parent_local"]
    cost = spt["cost"]

    pos = int(np.searchsorted(node_global, start_global))
    if pos >= len(node_global) or int(node_global[pos]) != int(start_global):
        return None

    def _in_must(global_id: int) -> bool:
        if must_be_in is None:
            return True
        idx = int(np.searchsorted(must_be_in, global_id))
        return idx < len(must_be_in) and int(must_be_in[idx]) == int(global_id)

    chain = np.empty(_MAX_LEG_STEPS, dtype=np.int32)
    chain[0] = pos
    n = 1
    cur = pos
    last_compatible_n: int | None = None
    if float(cost[pos]) == 0.0 and _in_must(int(node_global[pos])):
        last_compatible_n = 1

    while n < _MAX_LEG_STEPS:
        nxt = int(parent_local[cur])
        if nxt < 0 or nxt == cur:
            break
        chain[n] = nxt
        cur = nxt
        n += 1
        if float(cost[cur]) == 0.0:
            if _in_must(int(node_global[cur])):
                last_compatible_n = n
                break  # found a compatible seed, stop early
            # incompatible seed; continue walking only if there's a parent
            # to follow (most multi-source seeds have parent_local == -9999,
            # so this typically terminates the walk)

    # Prefer a compatible seed if we found one. Otherwise return the
    # walk as far as we got — caller will likely fail on the next leg
    # but we hand back what we have so the error message is precise.
    end = last_compatible_n if last_compatible_n is not None else n
    return np.asarray(node_global)[chain[:end]].astype(np.int64)


def _walk_to_root(spt: dict, start_local: int) -> list[int]:
    """Walk parent_local from `start_local` to the SPT tree root (a
    seed of the multi-source SPT). Returns the local-index sequence
    [start, ..., root]. The root is the closest seed to `start_local`."""
    chain: list[int] = []
    cur = int(start_local)
    parent = spt["parent_local"]
    seen = set()
    while cur >= 0 and cur not in seen:
        chain.append(cur)
        seen.add(cur)
        nxt = int(parent[cur])
        if nxt < 0 or nxt == cur:
            break
        cur = nxt
    return chain


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

def route(
    start: tuple[float, float], end: tuple[float, float],
    profile: str,
) -> dict:
    """Plan a route from `start` to `end` using chainless per-city SPTs.

    Returns a dict-shaped GeoJSON Feature with LineString coords and
    a `properties` dict including the city-name sequence and total
    track length.
    """
    import time
    t0 = time.time()
    prof = _load_profile(profile)
    t_prof = time.time()

    with db.connect() as conn:
        s_vid = _snap_to_vertex(conn, start[0], start[1])
        e_vid = _snap_to_vertex(conn, end[0], end[1])
        t_snap = time.time()

        # Pick start_city / end_city as the nearest anchor by
        # (lon, lat). KDTree is over geographic coords; for lon/lat
        # not too far apart this is a reasonable approximation of
        # great-circle nearest. KDTree.query returns (dist, idx).
        _, s_city = prof.city_kdtree.query([start[0], start[1]])
        _, e_city = prof.city_kdtree.query([end[0],   end[1]])
        s_city, e_city = int(s_city), int(e_city)

        # Same-city: walk both endpoints toward their tree roots in
        # the city's SPT. They may be in different trees (multi-source
        # SPT), so we don't get a continuous bike-route; instead we
        # produce two segments meeting at the polygon's interior.
        # Acceptable approximation for v1; full local Dijkstra is the
        # right fix later.
        if s_city == e_city:
            spt = prof.spt(s_city)
            sl = _local_idx(spt, s_vid)
            el = _local_idx(spt, e_vid)
            if sl < 0 or el < 0:
                raise RuntimeError(
                    f"endpoint not present in city {prof.cities[s_city]['name']}'s SPT "
                    f"(s_vid={s_vid} sl={sl}, e_vid={e_vid} el={el})"
                )
            s_chain = _walk_to_root(spt, sl)
            e_chain = _walk_to_root(spt, el)
            global_path = (
                [int(spt["node_global"][i]) for i in s_chain] +
                [int(spt["node_global"][i]) for i in reversed(e_chain)]
            )
            return _build_feature(
                conn, global_path, [prof.cities[s_city]["name"]],
            )

        # Multi-city: plan chain on the chain-adjacency graph (reversed
        # city_graph; see _ProfileData.__init__).
        city_path = _city_graph_dijkstra(prof.chain_adj, s_city, e_city)
        if not city_path:
            raise RuntimeError(
                f"no chain-feasible city path from "
                f"{prof.cities[s_city]['name']} to {prof.cities[e_city]['name']}"
            )
        t_chain = time.time()

        # Per-step look-1-ahead walk. The chain Dijkstra picks the city
        # sequence; the walk physically traces a road-graph path that
        # passes through each chain city's SPT-coverage zone but not
        # necessarily through their centers.
        #
        # Algorithm: maintain target_idx = which chain city's SPT we're
        # currently walking parents in. Before each parent step, peek
        # at chain[target_idx+1].SPT; if it covers current, advance
        # target_idx. This avoids:
        #   - The U-turn pathology of pure leg-by-leg walking. B's
        #     seeds cluster at B's center (1 km bbox around B's
        #     snap_vertex), so walking B.SPT.parent always terminates
        #     at B's center. Look-1-ahead switches to chain[i+1] before
        #     reaching chain[i]'s center, because chain[i+1].SPT was
        #     selected specifically for covering chain[i]'s polygon.
        #   - The O(chain) per-recheck scan of the prior algorithm,
        #     which was the dominant cost on long routes.
        spts = [prof.spt(ci) for ci in city_path]
        t_load = time.time()
        end_idx = len(city_path) - 1

        end_local_in_dest = _local_idx(spts[-1], e_vid)
        if end_local_in_dest < 0:
            raise RuntimeError(
                f"e_vid={e_vid} not in destination ({prof.cities[e_city]['name']}) SPT"
            )

        # Initial target: first chain city whose SPT contains s_vid.
        # In the common case this is chain[0] (start city).
        target_idx = -1
        target_local = -1
        for i in range(end_idx + 1):
            local = _local_idx(spts[i], s_vid)
            if local >= 0:
                target_idx = i
                target_local = local
                break
        if target_idx < 0:
            raise RuntimeError(
                f"start vertex {s_vid} not in any chain city's SPT"
            )
        target_spt = spts[target_idx]

        full_path: list[int] = [s_vid]
        current = s_vid
        steps = 0

        # SWITCH_INTERVAL: how many parent steps to take in target_spt
        # before re-checking look-1-ahead. The python-level _local_idx
        # call costs ~250 µs (numpy.searchsorted overhead dominates),
        # so per-step look-ahead would cost ~10 s on a 36 K-step walk.
        # Batching to K=20 caps overshoot at ~600 m (~30 m/edge × 20)
        # and reduces look-aheads to ~1.8 K calls.
        SWITCH_INTERVAL = 20

        while current != e_vid and steps < _MAX_LEG_STEPS:
            # Look-1-ahead: advance target_idx as far as consecutive
            # chain cities cover current. Steady state: 1 failed call.
            while target_idx < end_idx:
                ahead_local = _local_idx(spts[target_idx + 1], current)
                if ahead_local < 0:
                    break
                target_idx += 1
                target_spt = spts[target_idx]
                target_local = ahead_local

            # Walk up to SWITCH_INTERVAL parent steps in target_spt.
            # Hot loop — bind dict lookups outside.
            phase_parent = target_spt["parent_local"]
            phase_node_global = target_spt["node_global"]
            cur_local = target_local
            hit_seed = False
            for _ in range(SWITCH_INTERVAL):
                nxt_local = int(phase_parent[cur_local])
                if nxt_local < 0 or nxt_local == cur_local:
                    hit_seed = True
                    break
                cur_local = nxt_local
                full_path.append(int(phase_node_global[cur_local]))
                steps += 1
            target_local = cur_local
            current = int(phase_node_global[cur_local])

            if hit_seed:
                if target_idx == end_idx:
                    break  # at destination polygon; final segment connects e_vid
                # Look-1-ahead missed already this iteration; try farther
                # chain cities as a recovery (chain coverage gap).
                recovered = False
                for j in range(target_idx + 2, end_idx + 1):
                    jl = _local_idx(spts[j], current)
                    if jl >= 0:
                        target_idx = j
                        target_spt = spts[j]
                        target_local = jl
                        recovered = True
                        break
                if recovered:
                    continue
                full_chain = " → ".join(prof.cities[c]["name"] for c in city_path)
                raise RuntimeError(
                    f"gradient walk dead-ended at "
                    f"{prof.cities[city_path[target_idx]]['name']} "
                    f"(chain idx {target_idx}/{end_idx}); "
                    f"no farther city in chain covers vid={current}. "
                    f"Full chain: {full_chain}"
                )

        # Final segment: connect from wherever the chain walk ended to
        # e_vid. Both should be in destination.SPT; walk parents from
        # e_vid to its tree root and append in reverse. May leave a
        # visual gap if `current` and e_vid are in different trees of
        # the multi-source SPT — fix later with a local Dijkstra.
        if current != e_vid:
            dest_spt = spts[-1]
            e_chain = _walk_to_root(dest_spt, end_local_in_dest)
            for li in reversed(e_chain):
                full_path.append(int(dest_spt["node_global"][li]))
        t_walk = time.time()

        city_names = [prof.cities[c]["name"] for c in city_path]
        feat = _build_feature(conn, full_path, city_names)
        t_feat = time.time()
        print(
            f"[route] prof={t_prof-t0:.3f}s "
            f"snap={t_snap-t_prof:.3f}s "
            f"chain={t_chain-t_snap:.3f}s ({len(city_path)} cities) "
            f"load_spts={t_load-t_chain:.3f}s "
            f"walk={t_walk-t_load:.3f}s ({steps} steps, {len(full_path):,} pts) "
            f"feat={t_feat-t_walk:.3f}s "
            f"TOTAL={t_feat-t0:.3f}s",
            flush=True,
        )
        return feat


def _build_feature(
    conn: psycopg.Connection,
    global_path: list[int], city_names: list[str],
) -> dict:
    coords = _coords_for_vertices(conn, global_path)
    total_m = 0.0
    if len(coords) >= 2:
        lons = np.array([c[0] for c in coords], dtype=np.float64)
        lats = np.array([c[1] for c in coords], dtype=np.float64)
        dlat = np.deg2rad(np.diff(lats))
        dlon = np.deg2rad(np.diff(lons))
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.deg2rad(lats[:-1])) * np.cos(np.deg2rad(lats[1:])) *
             np.sin(dlon / 2) ** 2)
        total_m = float(2 * 6_371_000.0 * np.arcsin(np.sqrt(a)).sum())
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {
            "creator": "spt-router",
            "cities": city_names,
            "track-length": int(total_m),
            "node-count": len(global_path),
        },
    }
