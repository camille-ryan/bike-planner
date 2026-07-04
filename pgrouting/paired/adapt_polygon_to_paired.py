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
    """For each (ci_from, ci_to) pair, look up ci_from's snap vertex in
    ci_to's NPZ node_global array. If found, that's the cost of routing
    ci_from → ci_to (directed; both directions emit their own edges).

    OUTER loop: ci_to (load ci_to.npz once).
    INNER loop: ci_from (cheap searchsorted lookup).

    Then add ferry chain edges from postgres.
    """
    print(f"[adapt] deriving city_graph via multi-seed SPT overlap ...",
          flush=True)
    t0 = time.time()

    # MULTI-SEED: each anchor has a SET of zero-cost vids (the entire 1km
    # bbox of seed vertices). An (a → b) edge exists if ANY of a's seeds
    # appears in b's NPZ; cost = min cost over the matched seeds.
    valid_anchors = [a for a in anchors if a.get("snap_vids")]
    ci_list = [a["city_idx"] for a in valid_anchors]
    # Concatenate all seeds; remember which seed belongs to which ci.
    seed_concat: list[int] = []
    seed_owner_ci: list[int] = []
    for a in valid_anchors:
        for v in a["snap_vids"]:
            seed_concat.append(int(v))
            seed_owner_ci.append(int(a["city_idx"]))
    seed_arr = np.array(seed_concat, dtype=np.int64)
    owner_arr = np.array(seed_owner_ci, dtype=np.int32)
    print(f"[adapt]   {len(valid_anchors):,} anchors / {len(seed_arr):,} "
          f"total seed vids to probe in each NPZ", flush=True)

    from_city: list[int] = []
    to_city:   list[int] = []
    weight:    list[float] = []

    last_log = t0
    for n_done, ci_to in enumerate(ci_list, 1):
        path = POLY_SPT_IN / f"{ci_to}.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            to_ng = np.asarray(z["node_global"], dtype=np.int64)
            to_cost = np.asarray(z["cost"], dtype=np.float32)
        if len(to_ng) > 1 and to_ng[1] < to_ng[0]:
            order = np.argsort(to_ng, kind="stable")
            to_ng = to_ng[order]
            to_cost = to_cost[order]

        # Vectorized: searchsorted all seeds against this NPZ.
        idx = np.searchsorted(to_ng, seed_arr)
        in_range = idx < len(to_ng)
        matched_mask = np.zeros_like(in_range)
        matched_mask[in_range] = to_ng[idx[in_range]] == seed_arr[in_range]
        # For each owner_ci that has at least one matched seed in this
        # NPZ, take min cost over its matched seeds.
        matched_owners = owner_arr[matched_mask]
        matched_costs = to_cost[idx[matched_mask]]
        if len(matched_owners) == 0:
            continue
        # group-by min via sort
        order = np.argsort(matched_owners, kind="stable")
        mo_s = matched_owners[order]
        mc_s = matched_costs[order]
        # Find group boundaries
        change = np.concatenate([[True], mo_s[1:] != mo_s[:-1]])
        starts = np.flatnonzero(change)
        ends = np.concatenate([starts[1:], [len(mo_s)]])
        for s, e in zip(starts, ends):
            owner = int(mo_s[s])
            if owner == ci_to:
                continue   # no self-loop
            cost = float(mc_s[s:e].min())
            from_city.append(owner)
            to_city.append(ci_to)
            weight.append(cost)

        if time.time() - last_log >= 10:
            pct = 100.0 * n_done / len(ci_list)
            print(f"[adapt]   {n_done:,}/{len(ci_list):,} ({pct:.0f}%) "
                  f"to-cities done, {len(from_city):,} edges so far, "
                  f"{time.time()-t0:.0f}s", flush=True)
            last_log = time.time()

    print(f"[adapt]   {len(from_city):,} directed edges from SPT overlap in "
          f"{time.time()-t0:.0f}s", flush=True)

    # NOTE (2026-07-01): synthetic ferry chain edges (cost = 20 km +
    # haversine between anchor centers) removed. Ferry connectivity is
    # now handled by the polygon compute step, which emits a disjoint
    # 5 km disc around each ferry-neighbor anchor. Since ferry `ways`
    # are already bike-routable edges in the cell graph, the polygon
    # SPT walks across the ferry naturally and chain edges emerge from
    # real SPT overlap in the loop above.
    print(f"[adapt]   total {len(from_city):,} directed edges "
          f"(overlap-only; ferries handled via polygon SPT)",
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
