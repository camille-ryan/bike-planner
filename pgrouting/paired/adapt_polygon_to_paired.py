"""Adapt polygon-SPT output to the API's expected paired-step format.

The polygon SPT pipeline (compute_spts_polygon.py + cells) produces:
  - data/way_city_anchors.geojson  (1,943 anchors, 4 countries)
  - data/way_city_graph.json       (4,281 chain edges, cost_m = distance)
  - data/spt/<profile>_polygon/<ci>.npz  (one per anchor)
    NPZ schema: node_global, parent, cost, coords_lonlat

The existing API (api/app/cells_api.py, spt_router.py) expects:
  - data/spt/<profile>/cities.json  (list of city dicts)
  - data/spt/<profile>/city_graph.json  (from_city/to_city/weight cols)

This script bridges them:
  1. Read way_city_anchors.geojson — derive cities.json.
     anchor's snap_vertex_id = the node_global at argmin(cost) in its
     polygon NPZ (the seed vertex where Dijkstra started).
  2. For each (ci_from, ci_to) anchor pair, check if ci_from's snap
     vertex appears in ci_to's NPZ node_global array. If yes, the cost
     there is the directed edge weight ci_from → ci_to.
  3. Write cities.json + city_graph.json to data/spt/<profile>/.
  4. Symlink data/spt/<profile>/spt → data/spt/<profile>_polygon
     so the API can find the per-anchor NPZ files at the expected path.

After this:
  - API endpoints (/trunk/route, /spt/*) can serve <profile>.
  - The downstream paired-trunks.db build (build_paired_corridor.py)
    can also consume these — though its NPZ schema expectations differ;
    a verify route can be done via direct Dijkstra over city_graph
    without paired-trunks.db (see verify_polygon_route.py).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    import psycopg
    import config as _cfg
    HAVE_PG = True
except Exception:
    HAVE_PG = False


PROFILE = os.environ.get("SPT_PROFILE", "views")
MIN_FERRY_LENGTH_M = float(os.environ.get("MIN_FERRY_LENGTH_M", "100"))
# User policy 2026-06-30: penalize ferries with a fixed 20km cost
# regardless of physical length. Short ferries (river crossings, ~100m)
# stop being false shortcuts; long Baltic crossings (~60km) become
# cheaper than the multi-day detour around.
FERRY_FIXED_COST = float(os.environ.get("FERRY_FIXED_COST_M", "20000"))
if not re.fullmatch(r"[a-z_][a-z0-9_]*", PROFILE):
    raise SystemExit(f"unsafe profile name: {PROFILE!r}")

DATA_DIR    = Path(os.environ.get("DATA_DIR", "/data"))
ANCHORS_IN  = DATA_DIR / "way_city_anchors.geojson"
POLY_SPT_IN = DATA_DIR / "spt" / f"{PROFILE}_polygon"
PAIRED_OUT  = DATA_DIR / "spt" / PROFILE


def _load_anchors() -> list[dict]:
    """Read way_city_anchors.geojson, return list of dicts in
    compute_spts_multi cities.json schema."""
    print(f"[adapt] reading {ANCHORS_IN} ...", flush=True)
    with open(ANCHORS_IN) as fh:
        gj = json.load(fh)
    feats = gj.get("features", [])
    anchors = []
    for ci, f in enumerate(feats):
        props = f.get("properties", {})
        lon, lat = f["geometry"]["coordinates"][:2]
        anchors.append({
            "city_idx":   ci,
            "anchor_id":  ci,           # no separate id in geojson; reuse
            "ref":        props.get("ref"),
            "name":       props.get("name", f"city_{ci}"),
            "place":      props.get("place"),
            "population": props.get("population"),
            "country":    props.get("country"),
            "lon":        float(lon),
            "lat":        float(lat),
            "has_polygon": True,
            # snap_vertex_id filled in from NPZ below
        })
    print(f"[adapt]   {len(anchors):,} anchors loaded", flush=True)
    return anchors


def _annotate_with_seeds(anchors: list[dict]) -> None:
    """For each anchor, store the FULL set of cost=0 seed vids from its
    polygon NPZ (the 1km bbox multi-seed Dijkstra had many roots).
    Also store snap_vertex_id = first seed (for legacy compatibility)."""
    print(f"[adapt] reading {len(anchors):,} polygon NPZs for seed sets ...",
          flush=True)
    t0 = time.time()
    missing = 0
    total_seeds = 0
    for a in anchors:
        ci = a["city_idx"]
        path = POLY_SPT_IN / f"{ci}.npz"
        if not path.exists():
            a["snap_vertex_id"] = None
            a["snap_vids"] = []
            missing += 1
            continue
        with np.load(path) as z:
            cost = np.asarray(z["cost"])
            ng = np.asarray(z["node_global"])
            if len(cost) == 0:
                a["snap_vertex_id"] = None
                a["snap_vids"] = []
                missing += 1
                continue
            zero_idxs = np.where(cost == 0)[0]
            seeds = ng[zero_idxs].astype(np.int64).tolist()
            a["snap_vids"] = sorted(seeds)
            a["snap_vertex_id"] = int(a["snap_vids"][0]) if a["snap_vids"] else None
            total_seeds += len(seeds)
    print(f"[adapt]   done in {time.time()-t0:.1f}s ({total_seeds:,} total "
          f"seed vids across {len(anchors):,} anchors, "
          f"avg {total_seeds/max(1,len(anchors)):.1f}/anchor; "
          f"{missing:,} missing NPZ)", flush=True)


def _write_cities(anchors: list[dict]) -> None:
    PAIRED_OUT.mkdir(parents=True, exist_ok=True)
    cities = [{
        "city_idx":       a["city_idx"],
        "anchor_id":      a["anchor_id"],
        "ref":            a.get("ref"),
        "name":           a["name"],
        "place":          a.get("place"),
        "population":     a.get("population"),
        "country":        a.get("country"),
        "lon":            a["lon"],
        "lat":            a["lat"],
        "snap_vertex_id": a["snap_vertex_id"],
        "snap_vids":      a.get("snap_vids", []),
        "has_polygon":    True,
    } for a in anchors]
    path = PAIRED_OUT / "cities.json"
    with open(path, "w") as fh:
        json.dump(cities, fh, ensure_ascii=False, indent=1)
    print(f"[adapt] wrote {path} ({len(cities):,} entries)", flush=True)


def _fetch_ferry_edges(profile: str) -> list[tuple[int, int, float, float, float]]:
    """Read postgres ways for is_ferry edges with profile cost.
    Returns [(src_vid, dst_vid, fwd_cost, rev_cost, length_m), ...]."""
    if not HAVE_PG:
        print("[ferry] psycopg not available, skipping ferry edges", flush=True)
        return []
    cost_col = f"cost_{profile}"
    rev_col  = f"reverse_cost_{profile}"
    with psycopg.connect(_cfg.PG_DSN) as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT source, target,
                   COALESCE({cost_col}, 1e15),
                   COALESCE({rev_col}, 1e15),
                   length_m
            FROM ways
            WHERE is_ferry AND length_m >= %s
        """, (MIN_FERRY_LENGTH_M,))
        ferries = [(int(r[0]), int(r[1]), float(r[2]), float(r[3]), float(r[4]))
                   for r in cur.fetchall()]
    print(f"[ferry] {len(ferries):,} long ferries (≥{int(MIN_FERRY_LENGTH_M)} m)",
          flush=True)
    return ferries


def _find_ferry_owners(
    ferries: list[tuple[int, int, float, float, float]],
    anchors: list[dict],
) -> list[tuple[int, int, float]]:
    """For each ferry endpoint, find the anchor whose polygon SPT
    contains it with min cost. Endpoints not claimed by any SPT
    (typical for ferry terminals beyond the 1.5-hop hull of any
    coastal anchor) fall back to nearest-anchor by lat/lon.

    Then emit city-graph edges (anchor_of_src → anchor_of_dst,
    ferry_cost) for each ferry. Negative ferry costs (one-way edges
    represented as cost=-1) are skipped — Dijkstra requires positive
    weights."""
    if not ferries:
        return []
    endpoints: set[int] = set()
    for s, t, *_ in ferries:
        endpoints.add(s); endpoints.add(t)
    print(f"[ferry] finding owners for {len(endpoints):,} ferry endpoints "
          f"across {len(anchors):,} polygon SPTs ...", flush=True)
    t0 = time.time()
    owners: dict[int, tuple[int, float]] = {}
    ep_arr = np.array(sorted(endpoints), dtype=np.int64)
    last_log = t0
    for n_done, a in enumerate(anchors, 1):
        ci = a["city_idx"]
        path = POLY_SPT_IN / f"{ci}.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            ng = np.asarray(z["node_global"], dtype=np.int64)
            cost = np.asarray(z["cost"], dtype=np.float32)
        if len(ng) == 0:
            continue
        if len(ng) > 1 and ng[1] < ng[0]:
            order = np.argsort(ng, kind="stable")
            ng = ng[order]; cost = cost[order]
        idx = np.searchsorted(ng, ep_arr)
        in_range = idx < len(ng)
        matched = np.zeros_like(in_range)
        matched[in_range] = ng[idx[in_range]] == ep_arr[in_range]
        for pos in np.flatnonzero(matched):
            vid = int(ep_arr[pos])
            cv = float(cost[idx[pos]])
            if vid not in owners or cv < owners[vid][1]:
                owners[vid] = (ci, cv)
        if time.time() - last_log >= 15:
            print(f"[ferry]   scanned {n_done:,}/{len(anchors):,} SPTs, "
                  f"{len(owners):,} endpoints owned, "
                  f"{time.time()-t0:.0f}s", flush=True)
            last_log = time.time()
    print(f"[ferry] SPT-claim done: {len(owners):,}/{len(endpoints):,} "
          f"endpoints owned in {time.time()-t0:.0f}s", flush=True)
    # Nearest-anchor fallback for endpoints not claimed by any polygon
    # SPT — needed to keep legit Baltic ferries (Rostock harbor etc.
    # is beyond any coastal anchor's 1.5-hop hull). To avoid the bogus
    # river-ferry shortcuts from earlier, the city-graph cost will be
    # FIXED_FERRY + haversine(A_center, B_center) — putting realistic
    # last-mile distance back in the edge weight (see _find_ferry_owners
    # caller below for the cost computation).
    unclaimed = [vid for vid in endpoints if vid not in owners]
    if unclaimed:
        print(f"[ferry] {len(unclaimed):,} endpoints unclaimed — "
              f"assigning by nearest anchor (lat/lon) ...", flush=True)
        with psycopg.connect(_cfg.PG_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, ST_X(the_geom), ST_Y(the_geom) "
                "FROM ways_vertices_pgr WHERE id = ANY(%s)",
                (unclaimed,),
            )
            ep_coords = {int(r[0]): (float(r[1]), float(r[2]))
                         for r in cur.fetchall()}
        anchor_lonlat = np.array([(a["lon"], a["lat"]) for a in anchors],
                                 dtype=np.float64)
        anchor_ci = [a["city_idx"] for a in anchors]
        for vid in unclaimed:
            if vid not in ep_coords:
                continue
            elon, elat = ep_coords[vid]
            d2 = ((anchor_lonlat[:, 0] - elon) ** 2
                  + (anchor_lonlat[:, 1] - elat) ** 2)
            nearest_pos = int(np.argmin(d2))
            owners[vid] = (anchor_ci[nearest_pos], 0.0)
        print(f"[ferry]   post-fallback: {len(owners):,}/{len(endpoints):,} "
              f"endpoints owned", flush=True)

    # City-graph ferry edge cost = FERRY_FIXED_COST + haversine(A_center,
    # B_center). The haversine reinstates the last-mile cost we'd have
    # paid going A→ferry_terminal and ferry_terminal→B (otherwise short
    # river ferries become cheap shortcuts when the city centers are
    # actually 25 km apart). Long Baltic ferries (~150 km centers) stay
    # cheaper than the multi-day land detour.
    R_M = 6_371_000.0
    def _hav_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        from math import radians, sin, cos, asin, sqrt
        phi1, phi2 = radians(lat1), radians(lat2)
        dphi = radians(lat2 - lat1); dlam = radians(lon2 - lon1)
        a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlam/2)**2
        return 2 * R_M * asin(sqrt(a))
    centers = {a["city_idx"]: (a["lon"], a["lat"]) for a in anchors}

    out: list[tuple[int, int, float]] = []
    n_oneway_skipped = 0
    for s, t, c_st, c_ts, _ln in ferries:
        a = owners.get(s, (None,))[0]
        b = owners.get(t, (None,))[0]
        if a is None or b is None or a == b:
            continue
        lon_a, lat_a = centers[a]
        lon_b, lat_b = centers[b]
        cost = FERRY_FIXED_COST + _hav_m(lon_a, lat_a, lon_b, lat_b)
        if 0 <= c_st < 1e14:
            out.append((a, b, cost))
        else:
            n_oneway_skipped += 1
        if 0 <= c_ts < 1e14:
            out.append((b, a, cost))
        else:
            n_oneway_skipped += 1
    print(f"[ferry] emitted {len(out):,} ferry chain edges "
          f"(cost = {FERRY_FIXED_COST:.0f} + haversine(A,B); "
          f"{n_oneway_skipped:,} directions dropped as one-way/sentinel)",
          flush=True)
    return out


def _build_city_graph(anchors: list[dict]) -> None:
    """Translate the bidir-filtered chain graph into per-directed-edge
    metadata for build_paired.

    For each undirected chain edge (a, b) in way_city_graph.json, emit
    TWO directed metadata rows: (a→b) and (b→a). Weight is looked up
    from the destination anchor's polygon SPT NPZ:
      - a→b weight: look up a's snap vid in b's NPZ node_global; that
        cost is the road distance b→a walking outward from b's seed.
      - b→a weight: symmetric.

    No polygon-overlap search — trunks are precisely the chain graph.
    Router uses the same file (way_city_graph.json) at query time to
    know which trunks exist, so trunks and chain graph stay 1:1.

    Also emits ferry chain edges from the same file (they were already
    added by augment_way_city_graph_with_ferries.py).
    """
    print(f"[adapt] translating chain graph → city_graph…", flush=True)
    t0 = time.time()

    # Index anchors by ref for lookups.
    ref_to_anchor = {a["ref"]: a for a in anchors}

    # Load the bidir-filtered chain graph.
    chain_path = Path("/data/way_city_graph.json")
    chain_edges = json.loads(chain_path.read_text())
    print(f"[adapt]   {len(chain_edges):,} undirected chain edges "
          f"→ up to {2*len(chain_edges):,} directed", flush=True)

    from_city: list[int] = []
    to_city:   list[int] = []
    weight:    list[float] = []

    def _lookup_cost(from_anchor: dict, to_anchor: dict,
                     to_ng_cache: dict) -> float | None:
        """Look up from_anchor's snap_vids in to_anchor's SPT NPZ.
        Return min cost, or None if none of from's seeds are reachable
        in to's SPT."""
        ci_to = to_anchor["city_idx"]
        cache = to_ng_cache.get(ci_to)
        if cache is None:
            path = POLY_SPT_IN / f"{ci_to}.npz"
            if not path.exists():
                to_ng_cache[ci_to] = ("missing", None)
                return None
            with np.load(path) as z:
                ng = np.asarray(z["node_global"], dtype=np.int64)
                cost = np.asarray(z["cost"], dtype=np.float32)
            if len(ng) > 1 and ng[1] < ng[0]:
                order = np.argsort(ng, kind="stable")
                ng = ng[order]; cost = cost[order]
            to_ng_cache[ci_to] = (ng, cost)
            cache = to_ng_cache[ci_to]
        if isinstance(cache[0], str) and cache[0] == "missing":
            return None
        ng, cost = cache
        seeds = np.asarray(from_anchor.get("snap_vids") or [], dtype=np.int64)
        if seeds.size == 0:
            return None
        idx = np.searchsorted(ng, seeds)
        in_range = idx < len(ng)
        matched = in_range & (ng[np.clip(idx, 0, len(ng) - 1)] == seeds)
        if not matched.any():
            return None
        return float(cost[idx[matched]].min())

    # LRU-ish NPZ cache — keep the last 64 anchors' SPT arrays loaded.
    # 64 × ~1 M vertices × 12 bytes = ~750 MB, fits comfortably.
    to_ng_cache: dict[int, tuple] = {}
    from collections import OrderedDict as _OD
    to_ng_cache = _OD()  # type: ignore
    def _cache_bounded_lookup(from_anchor, to_anchor):
        ci_to = to_anchor["city_idx"]
        if ci_to in to_ng_cache:
            to_ng_cache.move_to_end(ci_to)
            return _lookup_cost(from_anchor, to_anchor, {ci_to: to_ng_cache[ci_to]})
        w = _lookup_cost(from_anchor, to_anchor, to_ng_cache)
        while len(to_ng_cache) > 64:
            to_ng_cache.popitem(last=False)
        return w

    n_missing = 0
    n_directed = 0
    for i, e in enumerate(chain_edges, 1):
        a = ref_to_anchor.get(e["a"])
        b = ref_to_anchor.get(e["b"])
        if a is None or b is None:
            n_missing += 1
            continue
        w_ab = _cache_bounded_lookup(a, b)
        w_ba = _cache_bounded_lookup(b, a)
        # Use MIN of SPT-lookup cost and chain graph's cost_m. Rationale:
        # bidir already computed a real-road weighted-Dijkstra cost that
        # isn't bounded by a polygon boundary. adapt's SPT lookup is
        # bounded by the DESTINATION anchor's polygon SPT — so if the
        # shortest road path leaves the polygon and re-enters, the SPT
        # only sees the detour cost (much larger than the true road
        # cost). Taking the min lets us keep the more accurate of the
        # two per direction. Falls back to chain graph cost if SPT
        # lookup returned None (NPZ missing, out of polygon, etc.).
        cost_chain = float(e.get("cost_m") or 0.0)
        if w_ab is None:
            w_ab = cost_chain
        elif cost_chain > 0:
            w_ab = min(w_ab, cost_chain)
        if w_ba is None:
            w_ba = cost_chain
        elif cost_chain > 0:
            w_ba = min(w_ba, cost_chain)
        from_city.append(a["city_idx"])
        to_city.append(b["city_idx"])
        weight.append(w_ab)
        from_city.append(b["city_idx"])
        to_city.append(a["city_idx"])
        weight.append(w_ba)
        n_directed += 2
        if i % 500 == 0:
            print(f"[adapt]   {i:,}/{len(chain_edges):,} chain edges → "
                  f"{n_directed:,} directed  ({time.time()-t0:.0f}s)",
                  flush=True)

    print(f"[adapt]   translated in {time.time()-t0:.0f}s: "
          f"{n_directed:,} directed edges "
          f"({n_missing} chain edges skipped for missing refs)",
          flush=True)

    cg = {"from_city": from_city, "to_city": to_city, "weight": weight}
    path = PAIRED_OUT / "city_graph.json"
    with open(path, "w") as fh:
        json.dump(cg, fh)
    print(f"[adapt] wrote {path}", flush=True)


def _ensure_spt_symlink() -> None:
    """Make data/spt/<profile>/spt -> ../<profile>_polygon so API
    finds per-anchor NPZ at the expected path."""
    spt_link = PAIRED_OUT / "spt"
    target = Path(f"../{PROFILE}_polygon")
    if spt_link.is_symlink() or spt_link.exists():
        if spt_link.is_symlink() and os.readlink(spt_link) == str(target):
            print(f"[adapt] {spt_link} already symlinked correctly", flush=True)
            return
        print(f"[adapt] WARNING: {spt_link} exists — leaving alone "
              f"(may shadow polygon NPZ files)", flush=True)
        return
    spt_link.symlink_to(target)
    print(f"[adapt] symlinked {spt_link} -> {target}", flush=True)


def main() -> None:
    t_start = time.time()
    print(f"[adapt] profile={PROFILE}", flush=True)
    print(f"[adapt]   polygon SPT input: {POLY_SPT_IN}", flush=True)
    print(f"[adapt]   paired-format out: {PAIRED_OUT}", flush=True)
    anchors = _load_anchors()
    _annotate_with_seeds(anchors)
    _write_cities(anchors)
    _build_city_graph(anchors)
    _ensure_spt_symlink()
    print(f"[adapt] DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
