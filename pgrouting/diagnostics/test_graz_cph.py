"""End-to-end smoke test for the new paired-trunk DB.

Plans Graz → Copenhagen via city_graph Dijkstra, then for each chain
edge (A, B) walks the trunk DB (`paired_trunks.db`) by following
`successor` from a chosen entry vertex until NULL. The trunk root of
(A, B) is then used as the entry into (B, C) for the next leg.

Reports:
- Chain length (number of anchors) and ordered names
- Per-leg trunk size, walk steps, end (lat, lon)
- Total polyline length / point count
- Where (if anywhere) the chain handoff fails (entry vertex not in next
  trunk — requires local Dijkstra bridge that's out of scope for this
  smoke test)
"""
from __future__ import annotations
import heapq
import json
import math
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import config


# Matches build_trunk_blob_db.py exactly.
TRUNK_DTYPE = np.dtype([
    ("vid",  "<i8"),
    ("succ", "<i8"),
    ("lat",  "<f4"),
    ("lon",  "<f4"),
])
NULL_SENTINEL = np.int64(-1)


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6_371_000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _city_idx(cities: list[dict], name: str) -> int:
    for c in cities:
        if c["name"] == name:
            return int(c["city_idx"])
    raise KeyError(name)


def _chain_dijkstra(
    adj: dict[int, list[tuple[int, float]]], src: int, dst: int,
) -> list[int]:
    """Forward city_graph Dijkstra. Edge (A, B, w) means "A's SPT
    covers B's seed, with cost w" — that's the natural forward
    direction for paired trunks: trunk(A, B) hands off A's reach into
    B's territory."""
    dist = {src: 0.0}; parent: dict[int, int] = {}
    heap: list[tuple[float, int]] = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            path = [u]
            while u != src:
                u = parent[u]; path.append(u)
            return list(reversed(path))
        if d > dist.get(u, float("inf")): continue
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd; parent[v] = u
                heapq.heappush(heap, (nd, v))
    raise RuntimeError(f"no city_graph path {src} → {dst}")


def _load_trunk(
    db: sqlite3.Connection, src: int, dst: int,
) -> dict[int, tuple[int | None, float, float]]:
    """Batch-load an entire (src, dst) trunk as {vid: (succ, lat, lon)}.

    One SQL roundtrip per leg instead of one per successor step.
    Typical trunk = ~16 K rows; load + dict-build is sub-millisecond.
    """
    return {
        int(r[0]): (None if r[1] is None else int(r[1]),
                    float(r[2]), float(r[3]))
        for r in db.execute(
            "SELECT vertex_id, successor, lat, lon FROM trunks "
            "WHERE src_city = ? AND dst_city = ?",
            (src, dst),
        )
    }


def _load_trunk_blob(
    db: sqlite3.Connection, src: int, dst: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a packed trunk (one BLOB row per pair). Returns (arr, next_idx)
    or None if the pair is missing.

    `arr` is a structured array with TRUNK_DTYPE, sorted by vid.
    `next_idx[k]` is the index in `arr` of arr['succ'][k], or -1 if
    that vertex is a trunk root (succ == NULL_SENTINEL) or its
    successor isn't in the trunk (shouldn't happen — sanity).
    """
    row = db.execute(
        "SELECT blob FROM trunk_blobs "
        "WHERE src_city = ? AND dst_city = ?",
        (src, dst),
    ).fetchone()
    if row is None:
        return None
    arr = np.frombuffer(row[0], dtype=TRUNK_DTYPE)
    # Resolve successor IDs to positions in the same array via
    # searchsorted (arr['vid'] is sorted ascending by construction).
    succs = arr["succ"]
    pos = np.searchsorted(arr["vid"], succs)
    in_range = pos < len(arr)
    pos_clipped = np.clip(pos, 0, len(arr) - 1)
    matched = in_range & (arr["vid"][pos_clipped] == succs)
    is_root = succs == NULL_SENTINEL
    next_idx = np.where(matched & ~is_root, pos, -1).astype(np.int64)
    return arr, next_idx


def _walk_trunk_blob(
    arr: np.ndarray, next_idx: np.ndarray, entry_vid: int,
    bridge_target: tuple[float, float] | None = None,
    max_steps: int = 200_000,
) -> tuple[list[tuple[int, float, float]], str, float]:
    """Walk a packed trunk by chasing next_idx pointers from entry_vid.
    Bridge fallback uses haversine over arr['lat'], arr['lon'] if the
    entry isn't directly present.
    """
    bridge_m = 0.0
    pos = int(np.searchsorted(arr["vid"], entry_vid))
    if pos >= len(arr) or int(arr["vid"][pos]) != entry_vid:
        if bridge_target is None or len(arr) == 0:
            return [], (
                "trunk empty" if len(arr) == 0
                else f"vertex {entry_vid} not in trunk and no bridge target"
            ), 0.0
        # Haversine to all rows, vectorized.
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

    path_idx: list[int] = []
    i = pos
    steps = 0
    while i >= 0 and steps < max_steps:
        path_idx.append(i)
        i = int(next_idx[i])
        steps += 1
    if steps >= max_steps:
        return [], f"max_steps={max_steps} exceeded", bridge_m
    # Materialize (vid, lat, lon) for the walk.
    vids = arr["vid"][path_idx]
    lats = arr["lat"][path_idx]
    lons = arr["lon"][path_idx]
    out = [(int(vids[k]), float(lats[k]), float(lons[k])) for k in range(len(path_idx))]
    tag = "ok" if bridge_m == 0 else f"ok+bridge:{bridge_m:.0f}m"
    return out, tag, bridge_m


def _walk_trunk(
    trunk: dict[int, tuple[int | None, float, float]],
    entry_vid: int,
    bridge_target: tuple[float, float] | None = None,
    max_steps: int = 200_000,
) -> tuple[list[tuple[int, float, float]], str, float]:
    """Walk an in-memory trunk dict from entry_vid by `successor` until
    NULL. If entry_vid isn't present and `bridge_target=(lat, lon)` is
    given, fall back to the trunk vertex closest to that point
    (haversine scan over the dict, ~16 K entries).
    """
    path: list[tuple[int, float, float]] = []
    bridge_m = 0.0

    if entry_vid not in trunk:
        if bridge_target is None or not trunk:
            return path, (
                "trunk empty" if not trunk
                else f"vertex {entry_vid} not in trunk and no bridge target"
            ), 0.0
        tlat, tlon = bridge_target
        best = None; best_d = float("inf")
        for vid, (_, lat, lon) in trunk.items():
            d = _haversine_m(lon, lat, tlon, tlat)
            if d < best_d:
                best_d = d; best = vid
        entry_vid = best
        bridge_m = best_d

    cur = entry_vid
    steps = 0
    while steps < max_steps:
        node = trunk.get(cur)
        if node is None:
            return path, f"walk broke at vid={cur} after {steps} steps", bridge_m
        succ, lat, lon = node
        path.append((cur, lat, lon))
        if succ is None:
            tag = "ok" if bridge_m == 0 else f"ok+bridge:{bridge_m:.0f}m"
            return path, tag, bridge_m
        cur = succ
        steps += 1
    return path, f"max_steps={max_steps} exceeded", bridge_m


def main() -> None:
    profile_dir = config.SPT_DIR / "lht"
    db_path = profile_dir / "paired_trunks.db"
    cities = json.load(open(profile_dir / "cities.json"))
    cg = json.load(open(profile_dir / "city_graph.json"))

    # Restrict city_graph to pairs actually present in the trunk DB —
    # the 33 degenerate skips at build time are missing from the DB
    # and a chain that traverses one of them has no walkable trunk.
    db_for_pairs = sqlite3.connect(db_path)
    buildable: set[tuple[int, int]] = {
        (int(r[0]), int(r[1]))
        for r in db_for_pairs.execute(
            "SELECT DISTINCT src_city, dst_city FROM trunks"
        ).fetchall()
    }
    db_for_pairs.close()

    chain_adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    skipped_edges = 0
    for fa, tb, w in zip(cg["from_city"], cg["to_city"], cg["weight"]):
        a, b = int(fa), int(tb)
        if (a, b) not in buildable:
            skipped_edges += 1
            continue
        chain_adj[a].append((b, float(w)))
    print(f"[test] city_graph: {len(buildable):,} buildable pairs, "
          f"{skipped_edges:,} degenerate edges filtered out")

    graz = _city_idx(cities, "Graz")
    cph  = _city_idx(cities, "København")
    print(f"[test] Graz   = anchor {graz}  "
          f"(lon={cities[graz]['lon']:.4f}, lat={cities[graz]['lat']:.4f}, "
          f"snap_vid={cities[graz].get('snap_vertex_id')})")
    print(f"[test] Cph    = anchor {cph}  "
          f"(lon={cities[cph]['lon']:.4f}, lat={cities[cph]['lat']:.4f}, "
          f"snap_vid={cities[cph].get('snap_vertex_id')})")

    t0 = time.time()
    chain = _chain_dijkstra(chain_adj, graz, cph)
    t_chain = time.time() - t0
    print(f"[test] city_graph Dijkstra: {len(chain)} anchors in "
          f"{t_chain*1000:.1f} ms")
    print(f"[test] chain: {' → '.join(cities[i]['name'] for i in chain[:6])} "
          f"... → {' → '.join(cities[i]['name'] for i in chain[-3:])}")

    db = sqlite3.connect(db_path)
    db.execute("PRAGMA query_only = 1")
    db.execute("PRAGMA cache_size = -262144")    # 256 MB page cache
    db.execute("PRAGMA mmap_size  =  1073741824") # 1 GB mmap window

    def run_chain(label: str) -> tuple[list, float, float, int, float]:
        full_path: list[tuple[int, float, float]] = []
        entry_vid = int(cities[graz]["snap_vertex_id"])
        cur_lat = float(cities[graz]["lat"])
        cur_lon = float(cities[graz]["lon"])
        failed_at = None
        bridges = 0
        total_bridge_m = 0.0
        t_load = 0.0
        t_walk_only = 0.0
        t_total = time.time()
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            t_a = time.time()
            trunk = _load_trunk(db, a, b)
            t_load += time.time() - t_a
            t_b = time.time()
            leg, status, bridge_m = _walk_trunk(
                trunk, entry_vid, bridge_target=(cur_lat, cur_lon),
            )
            t_walk_only += time.time() - t_b
            if bridge_m > 0:
                bridges += 1
                total_bridge_m += bridge_m
            if not status.startswith("ok") or not leg:
                failed_at = i
                break
            full_path.extend(leg)
            entry_vid = leg[-1][0]
            cur_lat = leg[-1][1]; cur_lon = leg[-1][2]
        elapsed = time.time() - t_total
        print(f"[test] {label}: total={elapsed*1000:.1f} ms  "
              f"trunk-load={t_load*1000:.1f} ms  walk={t_walk_only*1000:.1f} ms  "
              f"bridges={bridges} ({total_bridge_m/1000:.1f} km)  "
              f"vertices={len(full_path):,}  failed_at={failed_at}")
        return full_path, t_load, t_walk_only, bridges, total_bridge_m

    print()
    full_path, _, _, bridges, total_bridge_m = run_chain("cold ")
    run_chain("warm1")
    run_chain("warm2")

    # Production hot-cache path: pre-load all chain trunks once
    # (corridor cache; ~36 MB for a 71-leg chain). Subsequent walks
    # are pure dict lookups, no SQL.
    print()
    t_pre = time.time()
    trunks_cache = [
        _load_trunk(db, chain[i], chain[i + 1])
        for i in range(len(chain) - 1)
    ]
    pre_ms = (time.time() - t_pre) * 1000
    cache_bytes = sum(len(t) for t in trunks_cache) * 64  # rough estimate
    print(f"[test] pre-loaded {len(trunks_cache)} trunks "
          f"(~{cache_bytes/1024:.0f} KB) in {pre_ms:.1f} ms")

    def run_chain_cached(label: str) -> None:
        t_total = time.time()
        full_path: list[tuple[int, float, float]] = []
        entry_vid = int(cities[graz]["snap_vertex_id"])
        cur_lat = float(cities[graz]["lat"])
        cur_lon = float(cities[graz]["lon"])
        for i in range(len(chain) - 1):
            leg, status, bridge_m = _walk_trunk(
                trunks_cache[i], entry_vid, bridge_target=(cur_lat, cur_lon),
            )
            if not status.startswith("ok") or not leg:
                break
            full_path.extend(leg)
            entry_vid = leg[-1][0]
            cur_lat = leg[-1][1]; cur_lon = leg[-1][2]
        elapsed_ms = (time.time() - t_total) * 1000
        print(f"[test] {label}: walk-only={elapsed_ms:.1f} ms  "
              f"vertices={len(full_path):,}")

    run_chain_cached("hot1 ")
    run_chain_cached("hot2 ")
    run_chain_cached("hot3 ")

    # ------------------------------------------------------------------
    # Variant B: BLOB-schema DB (paired_trunks_blob.db). Each pair is
    # one row with a packed numpy structured array. np.frombuffer is
    # zero-copy; walk uses precomputed next_idx (numpy int array).
    # ------------------------------------------------------------------
    blob_path = profile_dir / "paired_trunks_blob.db"
    if blob_path.exists():
        print()
        print(f"[test] === BLOB backend ({blob_path.name}) ===")
        db_blob = sqlite3.connect(blob_path)
        db_blob.execute("PRAGMA query_only = 1")
        db_blob.execute("PRAGMA cache_size = -262144")
        db_blob.execute("PRAGMA mmap_size  =  1073741824")

        def run_chain_blob(label: str, prefetched=None) -> list:
            full_path: list[tuple[int, float, float]] = []
            entry_vid = int(cities[graz]["snap_vertex_id"])
            cur_lat = float(cities[graz]["lat"])
            cur_lon = float(cities[graz]["lon"])
            t_load = 0.0
            t_walk_only = 0.0
            bridges = 0; total_bridge_m = 0.0
            t_total = time.time()
            for i in range(len(chain) - 1):
                if prefetched is not None:
                    arr_pair = prefetched[i]
                else:
                    t_a = time.time()
                    arr_pair = _load_trunk_blob(
                        db_blob, chain[i], chain[i + 1],
                    )
                    t_load += time.time() - t_a
                if arr_pair is None:
                    break
                arr, next_idx = arr_pair
                t_b = time.time()
                leg, status, bridge_m = _walk_trunk_blob(
                    arr, next_idx, entry_vid,
                    bridge_target=(cur_lat, cur_lon),
                )
                t_walk_only += time.time() - t_b
                if bridge_m > 0:
                    bridges += 1; total_bridge_m += bridge_m
                if not status.startswith("ok") or not leg:
                    break
                full_path.extend(leg)
                entry_vid = leg[-1][0]
                cur_lat = leg[-1][1]; cur_lon = leg[-1][2]
            elapsed = time.time() - t_total
            print(f"[test] {label}: total={elapsed*1000:.1f} ms  "
                  f"load={t_load*1000:.1f} ms  walk={t_walk_only*1000:.1f} ms  "
                  f"bridges={bridges} ({total_bridge_m/1000:.1f} km)  "
                  f"vertices={len(full_path):,}")
            return full_path

        run_chain_blob("blob_cold ")
        run_chain_blob("blob_warm1")
        run_chain_blob("blob_warm2")

        # Pre-fetch once, then walk: simulates a server with a corridor
        # cache holding all chain blobs as numpy arrays.
        t_pre = time.time()
        prefetched = [
            _load_trunk_blob(db_blob, chain[i], chain[i + 1])
            for i in range(len(chain) - 1)
        ]
        bytes_resident = sum(
            (p[0].nbytes + p[1].nbytes) for p in prefetched if p is not None
        )
        print(f"[test] blob pre-fetch: {(time.time()-t_pre)*1000:.1f} ms  "
              f"resident≈{bytes_resident/1024:.0f} KB")
        run_chain_blob("blob_hot1 ", prefetched=prefetched)
        run_chain_blob("blob_hot2 ", prefetched=prefetched)
        run_chain_blob("blob_hot3 ", prefetched=prefetched)
        db_blob.close()
    else:
        print(f"[test] (no blob DB at {blob_path}; skipping blob variant)")

    failed_at = None
    db.close()

    # Stats.
    if failed_at is not None:
        print(f"[test] !! failed at leg {failed_at}: chain handoff broke "
              f"(entry vertex not in next trunk)")
    print(f"[test] walked {len(full_path):,} trunk vertices in "
          f"{t_walk*1000:.0f} ms")

    if full_path:
        # Crude path length via consecutive haversine sums (note: many
        # successor hops are >1 graph edge apart since pruning thinned
        # the trunk; this is an upper bound on geographic spread, not
        # actual road distance).
        gross_m = 0.0
        for j in range(1, len(full_path)):
            _, lat1, lon1 = full_path[j-1]
            _, lat2, lon2 = full_path[j]
            gross_m += _haversine_m(lon1, lat1, lon2, lat2)
        bbox_lat = (min(p[1] for p in full_path),
                    max(p[1] for p in full_path))
        bbox_lon = (min(p[2] for p in full_path),
                    max(p[2] for p in full_path))
        print(f"[test] gross consecutive-haversine: {gross_m/1000:.1f} km")
        print(f"[test] bbox lat: {bbox_lat[0]:.3f} → {bbox_lat[1]:.3f}")
        print(f"[test] bbox lon: {bbox_lon[0]:.3f} → {bbox_lon[1]:.3f}")
        print(f"[test] start: ({full_path[0][1]:.4f}, {full_path[0][2]:.4f})")
        print(f"[test] end:   ({full_path[-1][1]:.4f}, {full_path[-1][2]:.4f})")

        # Write GeoJSON LineString for quick visual inspection.
        out_geojson = config.SPT_DIR / "lht" / "test_graz_cph.geojson"
        gj = {
            "type": "Feature",
            "properties": {
                "name": "Graz → Cph trunk-walk smoke test",
                "anchor_count": len(chain),
                "trunk_vertex_count": len(full_path),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat] for _, lat, lon in full_path],
            },
        }
        out_geojson.write_text(json.dumps(gj))
        print(f"[test] wrote {out_geojson}  ({out_geojson.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
