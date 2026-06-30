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


def _annotate_with_snap_vertex(anchors: list[dict]) -> None:
    """For each anchor, find its snap vertex by argmin(cost) over its
    polygon NPZ. The seed had cost=0 (or near-0 — multi-seed averages)."""
    print(f"[adapt] reading {len(anchors):,} polygon NPZs for snap ids ...",
          flush=True)
    t0 = time.time()
    missing = 0
    for a in anchors:
        ci = a["city_idx"]
        path = POLY_SPT_IN / f"{ci}.npz"
        if not path.exists():
            a["snap_vertex_id"] = None
            missing += 1
            continue
        with np.load(path) as z:
            cost = z["cost"]
            ng = z["node_global"]
            if len(cost) == 0:
                a["snap_vertex_id"] = None
                missing += 1
                continue
            min_idx = int(np.argmin(cost))
            a["snap_vertex_id"] = int(ng[min_idx])
    print(f"[adapt]   done in {time.time()-t0:.1f}s "
          f"({missing:,} anchors missing NPZ)", flush=True)


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

    # Fallback: for endpoints not claimed by any SPT, assign nearest
    # anchor by lat/lon. Needed because ferry terminals often sit
    # beyond the 1.5-hop hull of the nearest coastal anchor.
    unclaimed = [vid for vid in endpoints if vid not in owners]
    if unclaimed:
        print(f"[ferry] {len(unclaimed):,} endpoints unclaimed — assigning "
              f"by nearest anchor (lat/lon)", flush=True)
        # Need vertex coords. Fetch from postgres in one shot.
        with psycopg.connect(_cfg.PG_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, ST_X(the_geom), ST_Y(the_geom) "
                "FROM ways_vertices_pgr WHERE id = ANY(%s)",
                (unclaimed,),
            )
            ep_coords = {int(r[0]): (float(r[1]), float(r[2])) for r in cur.fetchall()}
        anchor_lonlat = np.array([(a["lon"], a["lat"]) for a in anchors],
                                 dtype=np.float64)
        anchor_ci = [a["city_idx"] for a in anchors]
        for vid in unclaimed:
            if vid not in ep_coords:
                continue
            elon, elat = ep_coords[vid]
            # Squared euclidean (degrees) — fine for nearest selection at
            # this scale; coastal anchors are < 50 km from their ferry
            # slips so the lat-cos approximation isn't critical.
            d2 = (anchor_lonlat[:, 0] - elon)**2 + (anchor_lonlat[:, 1] - elat)**2
            nearest_pos = int(np.argmin(d2))
            owners[vid] = (anchor_ci[nearest_pos], 0.0)
        print(f"[ferry] post-fallback: {len(owners):,}/{len(endpoints):,} "
              f"endpoints owned", flush=True)

    out: list[tuple[int, int, float]] = []
    n_dropped_neg = 0
    for s, t, c_st, c_ts, _ln in ferries:
        a = owners.get(s, (None,))[0]
        b = owners.get(t, (None,))[0]
        if a is None or b is None or a == b:
            continue
        # Filter both sentinel-large AND negative (one-way sentinel).
        if 0 <= c_st < 1e14:
            out.append((a, b, c_st))
        else:
            n_dropped_neg += 1
        if 0 <= c_ts < 1e14:
            out.append((b, a, c_ts))
        else:
            n_dropped_neg += 1
    print(f"[ferry] emitted {len(out):,} ferry chain edges "
          f"({n_dropped_neg:,} dropped as one-way/sentinel)", flush=True)
    return out


def _build_city_graph(anchors: list[dict]) -> None:
    """For each (ci_from, ci_to) pair, look up ci_from's snap vertex in
    ci_to's NPZ node_global array. If found, that's the cost of routing
    ci_from → ci_to (directed; both directions emit their own edges).

    OUTER loop: ci_to (load ci_to.npz once).
    INNER loop: ci_from (cheap searchsorted lookup).

    Then add ferry chain edges from postgres.
    """
    print(f"[adapt] deriving city_graph via SPT overlap ...", flush=True)
    t0 = time.time()

    snap_by_ci = {a["city_idx"]: a["snap_vertex_id"] for a in anchors
                  if a["snap_vertex_id"] is not None}
    ci_list = sorted(snap_by_ci.keys())
    # Pre-build an array of snap vertices in ci order (for vectorized lookup)
    snap_arr = np.array([snap_by_ci[ci] for ci in ci_list], dtype=np.int64)
    print(f"[adapt]   {len(ci_list):,} anchors with valid snap vertices",
          flush=True)

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
        # node_global is sorted ASC (from Dijkstra slice semantics)?
        # Polygon NPZ may not be sorted — sort if needed.
        if len(to_ng) > 1 and to_ng[1] < to_ng[0]:
            order = np.argsort(to_ng, kind="stable")
            to_ng = to_ng[order]
            to_cost = to_cost[order]

        # Vectorized: searchsorted all snap_arr at once.
        idx = np.searchsorted(to_ng, snap_arr)
        in_range = idx < len(to_ng)
        matched = np.zeros_like(in_range)
        matched[in_range] = to_ng[idx[in_range]] == snap_arr[in_range]
        # Don't emit self-edges
        self_pos = ci_list.index(ci_to) if ci_to in ci_list else -1
        if self_pos >= 0:
            matched[self_pos] = False
        for pos in np.flatnonzero(matched):
            from_city.append(ci_list[pos])
            to_city.append(ci_to)
            weight.append(float(to_cost[idx[pos]]))

        if time.time() - last_log >= 10:
            pct = 100.0 * n_done / len(ci_list)
            print(f"[adapt]   {n_done:,}/{len(ci_list):,} ({pct:.0f}%) "
                  f"to-cities done, {len(from_city):,} edges so far, "
                  f"{time.time()-t0:.0f}s", flush=True)
            last_log = time.time()

    print(f"[adapt]   {len(from_city):,} directed edges from SPT overlap in "
          f"{time.time()-t0:.0f}s", flush=True)

    # Add ferry chain edges so disconnected island clusters (e.g. Cph
    # on Sealand) bridge into the mainland network via ferries.
    ferries = _fetch_ferry_edges(PROFILE)
    ferry_edges = _find_ferry_owners(ferries, anchors)
    for a, b, w in ferry_edges:
        from_city.append(a); to_city.append(b); weight.append(w)
    print(f"[adapt]   total {len(from_city):,} directed edges "
          f"(overlap + ferry)", flush=True)

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
    _annotate_with_snap_vertex(anchors)
    _write_cities(anchors)
    _build_city_graph(anchors)
    _ensure_spt_symlink()
    print(f"[adapt] DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
