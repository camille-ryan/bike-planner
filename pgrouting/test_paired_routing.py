"""Routing experiment: walk through paired SPTs leg-by-leg and compare
timing against the production /spt/route endpoint.

Algorithm (paired-SPT walk):
  1. Snap start/end coords to nearest road vertices via Postgres.
  2. Plan chain via city_graph Dijkstra (same as production).
  3. For the first leg only: walk chain[0].SPT.parent_local from
     s_vid toward a chain[0]-seed. (Usually 0-few hundred steps —
     s_vid is typically inside chain[0]'s seed bbox already.)
  4. For each chain edge (chain[i], chain[i+1]):
       - Find current vertex's position in paired_SPT(i,i+1).
       - Walk parent_local until cost==0 (a chain[i+1]-seed).
  5. Final leg: walk chain[-1].SPT.parent_local from e_vid back to its
     tree root, append in reverse (same as production).

For comparison we also call /spt/route over HTTP and capture its time
and route length. Both should produce equivalent geometry.
"""
from __future__ import annotations
import heapq
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

import config


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def _load_spt(spt_dir: Path, city_idx: int) -> dict[str, np.ndarray]:
    with np.load(spt_dir / f"{city_idx}.npz") as d:
        return {
            "node_global":  np.asarray(d["node_global"]),
            "parent_local": np.asarray(d["parent_local"]),
            "cost":         np.asarray(d["cost"]),
        }


def _load_paired(paired_dir: Path, a: int, b: int) -> dict[str, np.ndarray]:
    with np.load(paired_dir / f"{a}_{b}.npz") as d:
        return {
            "node_global":  np.asarray(d["node_global"]),
            "parent_local": np.asarray(d["parent_local"]),
            "cost":         np.asarray(d["cost"]),
        }


def _load_meta(out_dir: Path):
    cities = json.load(open(out_dir / "cities.json"))
    cg = json.load(open(out_dir / "city_graph.json"))
    chain_adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for fa, tb, w in zip(cg["from_city"], cg["to_city"], cg["weight"]):
        chain_adj[int(tb)].append((int(fa), float(w)))
    return cities, chain_adj


# ---------------------------------------------------------------------
# Chain dijkstra (city level)
# ---------------------------------------------------------------------

def _chain_dijkstra(adj, src, dst):
    if src == dst: return [src]
    dist = {src: 0.0}; parent = {}
    heap = [(0.0, src)]
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
    return None


# ---------------------------------------------------------------------
# Postgres helpers
# ---------------------------------------------------------------------

def _snap(conn, lon, lat):
    expand_deg = 0.25
    with conn.cursor() as cur:
        cur.execute("""
            SELECT v.id FROM ways_vertices_pgr v
            WHERE v.the_geom && ST_Expand(ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s)
            ORDER BY v.the_geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326) LIMIT 1
        """, (lon, lat, expand_deg, lon, lat))
        row = cur.fetchone()
    return int(row[0])


def _coords_for(conn, vids):
    if not vids: return []
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ST_X(v.the_geom), ST_Y(v.the_geom)
            FROM ways_vertices_pgr v
            JOIN unnest(%s::bigint[]) WITH ORDINALITY AS u(vid, ord) ON v.id = u.vid
            ORDER BY u.ord
        """, (list(map(int, vids)),))
        return [(float(r[0]), float(r[1])) for r in cur.fetchall()]


# ---------------------------------------------------------------------
# Walk helpers
# ---------------------------------------------------------------------

def _local_idx(node_global: np.ndarray, vid: int) -> int:
    pos = int(np.searchsorted(node_global, vid))
    if pos >= len(node_global) or int(node_global[pos]) != vid:
        return -1
    return pos


def _walk_to_seed(spt: dict, start_local: int, max_steps: int = 200_000):
    """Walk parent_local from start_local until cost==0 or root.
    Returns list of GLOBAL vids visited (start ... seed)."""
    node_global = spt["node_global"]
    parent = spt["parent_local"]
    cost = spt["cost"]
    out = [int(node_global[start_local])]
    cur = start_local
    for _ in range(max_steps):
        if cost[cur] == 0.0:
            break
        nxt = int(parent[cur])
        if nxt < 0 or nxt == cur:
            break
        cur = nxt
        out.append(int(node_global[cur]))
    return out, cur


# ---------------------------------------------------------------------
# Paired-SPT route
# ---------------------------------------------------------------------

def route_via_paired(
    start: tuple[float, float], end: tuple[float, float], chain_names: list[str],
    profile: str = "lht",
):
    out_dir = config.SPT_DIR / profile
    spt_dir = out_dir / "spt"
    paired_dir = out_dir / "paired"

    cities, chain_adj = _load_meta(out_dir)

    # Resolve chain by name pairs (same as builder).
    name_to_idx = {c["name"]: c["city_idx"] for c in cities}
    waypoints = [name_to_idx[n] for n in chain_names]
    chain: list[int] = []
    for i in range(len(waypoints) - 1):
        leg = _chain_dijkstra(chain_adj, waypoints[i], waypoints[i + 1])
        chain.extend(leg if i == 0 else leg[1:])

    timings = {}
    t0 = time.time()
    with psycopg.connect(config.PG_DSN) as conn:
        s_vid = _snap(conn, start[0], start[1])
        e_vid = _snap(conn, end[0], end[1])
        timings["snap"] = time.time() - t0

        # Load every SPT we'll touch (chain[0] for approach, chain[-1] for
        # final, plus all paired SPTs in between).
        t1 = time.time()
        chain0_spt = _load_spt(spt_dir, chain[0])
        chainN_spt = _load_spt(spt_dir, chain[-1])
        paired = [
            _load_paired(paired_dir, chain[i], chain[i + 1])
            for i in range(len(chain) - 1)
        ]
        timings["load"] = time.time() - t1

        # Approach: walk chain[0].SPT from s_vid to its nearest seed.
        t2 = time.time()
        s_local = _local_idx(chain0_spt["node_global"], s_vid)
        if s_local < 0:
            raise RuntimeError(f"s_vid {s_vid} not in chain[0].SPT (Graz)")
        approach_path, last_local = _walk_to_seed(chain0_spt, s_local)
        current = approach_path[-1]
        full_path = list(approach_path)
        timings["approach"] = time.time() - t2
        approach_steps = len(approach_path)

        # Walk through each paired SPT.
        t3 = time.time()
        leg_steps = []
        for i, p in enumerate(paired):
            cur_local = _local_idx(p["node_global"], current)
            if cur_local < 0:
                raise RuntimeError(
                    f"current vertex {current} not in paired_SPT "
                    f"({cities[chain[i]]['name']} → {cities[chain[i + 1]]['name']}); "
                    f"approach landed off the trunk."
                )
            leg, _ = _walk_to_seed(p, cur_local)
            leg_steps.append(len(leg))
            full_path.extend(leg[1:])  # skip first = current (already in path)
            current = leg[-1]
        timings["paired_walk"] = time.time() - t3

        # Final: walk from e_vid in chain[-1].SPT to its root, append in reverse.
        t4 = time.time()
        e_local = _local_idx(chainN_spt["node_global"], e_vid)
        if e_local < 0:
            raise RuntimeError(f"e_vid {e_vid} not in chain[-1].SPT (Wien)")
        e_chain = []
        cur = e_local
        for _ in range(200_000):
            e_chain.append(cur)
            if chainN_spt["cost"][cur] == 0.0: break
            nxt = int(chainN_spt["parent_local"][cur])
            if nxt < 0 or nxt == cur: break
            cur = nxt
        for li in reversed(e_chain):
            full_path.append(int(chainN_spt["node_global"][li]))
        timings["final"] = time.time() - t4

        # Materialize coords.
        t5 = time.time()
        coords = _coords_for(conn, full_path)
        timings["coords"] = time.time() - t5

    timings["total"] = time.time() - t0

    # Track length.
    if len(coords) >= 2:
        lons = np.array([c[0] for c in coords]); lats = np.array([c[1] for c in coords])
        dlat = np.deg2rad(np.diff(lats)); dlon = np.deg2rad(np.diff(lons))
        a = (np.sin(dlat/2)**2 + np.cos(np.deg2rad(lats[:-1])) *
             np.cos(np.deg2rad(lats[1:])) * np.sin(dlon/2)**2)
        km = float(2 * 6371000 * np.arcsin(np.sqrt(a)).sum()) / 1000
    else:
        km = 0

    return {
        "vertex_count": len(full_path),
        "track_km": km,
        "approach_steps": approach_steps,
        "leg_steps": leg_steps,
        "timings": timings,
        "chain_names": [cities[c]["name"] for c in chain],
    }


# ---------------------------------------------------------------------
# Compare against production /spt/route
# ---------------------------------------------------------------------

def hit_production(start, end, profile="lht"):
    import os, urllib.request, urllib.parse
    base = os.environ.get("BIKE_API_BASE", "http://api:8000")
    qs = urllib.parse.urlencode({
        "from": f"{start[0]},{start[1]}",
        "to":   f"{end[0]},{end[1]}",
        "profile": profile,
    })
    t0 = time.time()
    with urllib.request.urlopen(f"{base}/spt/route?{qs}", timeout=120) as r:
        body = r.read()
    dt = time.time() - t0
    j = json.loads(body)
    coords = j["route"]["geometry"]["coordinates"]
    props = j["route"]["properties"]
    return {
        "vertex_count": len(coords),
        "track_km": (props.get("track-length", 0) or 0) / 1000,
        "total_s": dt,
        "chain": props.get("cities", []),
    }


if __name__ == "__main__":
    GRAZ = (15.4395, 47.0707)
    WIEN = (16.3725, 48.2082)

    # Hold globals once, then run 3 times in the same process so we see
    # cold-vs-warm behaviour — same as production's lru_cache pattern.
    print("PAIRED-SPT walk (3 runs, same process):")
    print("=" * 64)
    out_dir = config.SPT_DIR / "lht"
    spt_dir = out_dir / "spt"
    paired_dir = out_dir / "paired"
    cities, chain_adj = _load_meta(out_dir)

    # Resolve chain once.
    chain_names = ["Graz", "Wien"]
    name_to_idx = {c["name"]: c["city_idx"] for c in cities}
    waypoints = [name_to_idx[n] for n in chain_names]
    chain: list[int] = []
    for i in range(len(waypoints) - 1):
        leg = _chain_dijkstra(chain_adj, waypoints[i], waypoints[i + 1])
        chain.extend(leg if i == 0 else leg[1:])

    # Pre-load SPTs and paired files once (mimics what a server-side
    # lru_cache would do across requests).
    _t = time.time()
    chain0_spt = _load_spt(spt_dir, chain[0])
    chainN_spt = _load_spt(spt_dir, chain[-1])
    paired = [
        _load_paired(paired_dir, chain[i], chain[i + 1])
        for i in range(len(chain) - 1)
    ]
    print(f"  pre-load (Graz + Wien SPTs + 14 paired): {(time.time()-_t)*1000:.1f} ms")
    print()

    for run in range(3):
        t0 = time.time()
        with psycopg.connect(config.PG_DSN) as conn:
            t_open = time.time()
            s_vid = _snap(conn, GRAZ[0], GRAZ[1])
            e_vid = _snap(conn, WIEN[0], WIEN[1])
            t_snap = time.time()

            s_local = _local_idx(chain0_spt["node_global"], s_vid)
            approach_path, _ = _walk_to_seed(chain0_spt, s_local)
            current = approach_path[-1]
            full_path = list(approach_path)
            t_app = time.time()

            for p in paired:
                cur_local = _local_idx(p["node_global"], current)
                if cur_local < 0:
                    raise RuntimeError(f"current {current} off the trunk")
                leg, _ = _walk_to_seed(p, cur_local)
                full_path.extend(leg[1:])
                current = leg[-1]
            t_paired = time.time()

            e_local = _local_idx(chainN_spt["node_global"], e_vid)
            e_chain = []
            cur = e_local
            for _ in range(200_000):
                e_chain.append(cur)
                if chainN_spt["cost"][cur] == 0.0: break
                nxt = int(chainN_spt["parent_local"][cur])
                if nxt < 0 or nxt == cur: break
                cur = nxt
            for li in reversed(e_chain):
                full_path.append(int(chainN_spt["node_global"][li]))
            t_final = time.time()

            coords = _coords_for(conn, full_path)
            t_coords = time.time()

        total = t_coords - t0
        print(
            f"  run {run+1}:  total={total*1000:>7.1f} ms  "
            f"connect={ (t_open-t0)*1000:>5.1f}  snap={(t_snap-t_open)*1000:>5.1f}  "
            f"approach={(t_app-t_snap)*1000:>5.1f}  paired={(t_paired-t_app)*1000:>5.1f}  "
            f"final={(t_final-t_paired)*1000:>5.1f}  coords={(t_coords-t_final)*1000:>5.1f}  "
            f"|  pts={len(full_path):,}"
        )

    print()
    print("PRODUCTION /spt/route (3 runs):")
    print("=" * 64)
    for run in range(3):
        try:
            p = hit_production(GRAZ, WIEN)
            print(f"  run {run+1}:  total={p['total_s']*1000:>7.1f} ms  pts={p['vertex_count']:,}")
        except Exception as e:
            print(f"  run {run+1}:  ERROR  {e}")
