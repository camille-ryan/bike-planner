"""FastAPI service for the Graz → Copenhagen bike-tour planner.

Endpoints:
  GET /health
  GET /trunk/route  — paired-trunk routing (blob DB, ~25 ms hot path)
  GET /pois         — bbox + category lookup against the SpatiaLite POI DB
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import json
from pathlib import Path

from . import chat, db, pois, trunk_router
from .settings import DEFAULT_PROFILE, SPT_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the trunk router for the default profile so the first
    # /trunk/route request doesn't pay the ~30 s cold-load cost.
    try:
        trunk_router.preload(DEFAULT_PROFILE)
    except FileNotFoundError as exc:
        print(f"[startup] trunk_router preload skipped: {exc}", flush=True)
    yield


app = FastAPI(title="Bike Routing API", version="0.3.0", lifespan=lifespan)

# Open CORS — this server is on a private Tailscale network, accessed by a
# laptop running the static web UI from any local origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _parse_lonlat(s: str, name: str) -> tuple[float, float]:
    try:
        lon, lat = (float(x) for x in s.split(","))
    except Exception:
        raise HTTPException(400, f"{name} must be 'lon,lat' (got '{s}')")
    return (lon, lat)


app.include_router(chat.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/trunk/route")
async def trunk_route(
    from_: str | None = Query(None, alias="from", description="lon,lat"),
    to: str | None = Query(None, description="lon,lat"),
    from_ref: str | None = Query(
        None,
        description="Anchor ref (e.g. 'db:2', 'ferry:12345'). Alternative "
                    "to `from` for city-name tour planning — skips postgres "
                    "snap and starts routing from the anchor itself.",
    ),
    to_ref: str | None = Query(None, description="Anchor ref alternative to `to`"),
    via_refs: list[str] | None = Query(
        None,
        description="Intermediate stops as anchor refs, one query param "
                    "per stop (repeat the param for multi-stop tours). "
                    "e.g. `via_refs=db:2&via_refs=db:5`. Any mix of "
                    "via_refs and via_lonlats is accepted; positional "
                    "ordering follows the request (all via_refs first, "
                    "then via_lonlats). For explicit ordering use the "
                    "`stops` param.",
    ),
    via_lonlats: list[str] | None = Query(
        None,
        description="Intermediate stops as lon,lat strings, one per param.",
    ),
    stops: list[str] | None = Query(
        None,
        description="Ordered mixed list of stops. Each item is either "
                    "a ref ('db:2', 'ferry:34988792') or a lon,lat "
                    "string. The endpoints (from/to) are not in this "
                    "list — only the intermediate waypoints.",
    ),
    profile: str = DEFAULT_PROFILE,
    simplify_m: float = Query(
        100.0,
        description="Decimate polyline: drop points closer than this many "
                    "meters apart (default 100 m, visually equivalent at "
                    "any zoom; pass 0 to disable).",
    ),
) -> dict:
    """Paired-trunk routing: snap endpoints, run city_graph Dijkstra,
    then walk in-memory trunk arrays leg by leg. Trunks are preloaded
    at server startup; steady-state latency is dominated by snap +
    Dijkstra (~10-30 ms total for a Graz→Cph-scale chain).

    Each endpoint accepts EITHER `from`/`to` (lon,lat string) OR
    `from_ref`/`to_ref` (anchor ref string). Mixing is fine — e.g.
    from an anchor to a lat/lon destination.

    Multi-stop tours: pass intermediate waypoints via `stops` (ordered
    mixed list), `via_refs` (all refs), or `via_lonlats` (all coords).
    Pairwise chain-Dijkstra runs start→stop1→stop2→…→end and the
    concatenated chain is walked end-to-end — a single first-mile
    stitch at start, a single last-mile stitch at end, no per-waypoint
    bridges (task #48).
    """
    if from_ is None and from_ref is None:
        raise HTTPException(400, "must provide `from` or `from_ref`")
    if to is None and to_ref is None:
        raise HTTPException(400, "must provide `to` or `to_ref`")
    a = _parse_lonlat(from_, "from") if from_ is not None else None
    b = _parse_lonlat(to,    "to")   if to    is not None else None

    # Normalize the various via inputs to a single ordered list of
    # (ref, coord) tuples. Precedence: `stops` first (explicit
    # ordering), else via_refs + via_lonlats concatenated.
    via: list[tuple[str | None, tuple[float, float] | None]] = []
    if stops:
        for s in stops:
            if "," in s and _looks_like_lonlat(s):
                via.append((None, _parse_lonlat(s, "stops")))
            else:
                via.append((s, None))
    else:
        for r in (via_refs or []):
            via.append((r, None))
        for c in (via_lonlats or []):
            via.append((None, _parse_lonlat(c, "via_lonlats")))
    try:
        feat = trunk_router.route(
            a, b, profile,
            simplify_m=simplify_m,
            start_ref=from_ref, end_ref=to_ref,
            via=via or None,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"trunk route: {exc}")
    return {"profile": profile, "route": feat}


def _looks_like_lonlat(s: str) -> bool:
    parts = s.split(",")
    if len(parts) != 2:
        return False
    try:
        float(parts[0]); float(parts[1])
        return True
    except ValueError:
        return False


@app.get("/anchors/search")
def anchors_search(
    q: str = Query("", description="Prefix or substring to match against "
                                    "anchor names (case-insensitive)."),
    profile: str = DEFAULT_PROFILE,
    limit: int = Query(20, ge=1, le=100),
) -> dict:
    """Fuzzy-lookup autocomplete for city / anchor names. Feeds the web
    UI's waypoint typeahead. Returns `{results: [{ref, name, country,
    population}]}` ordered by prefix-match-first then population desc.
    Backed by the in-memory cities.json loaded at API startup.
    """
    prof = trunk_router._load_profile(profile)
    qn = (q or "").strip().lower()
    if not qn:
        # Empty query → top-N by population, useful for "recent" style pickers.
        ranked = sorted(
            prof.cities,
            key=lambda c: -(c.get("population") or 0),
        )[:limit]
    else:
        prefix, contains = [], []
        for c in prof.cities:
            nm = (c.get("name") or "").lower()
            if not nm:
                continue
            if nm.startswith(qn):
                prefix.append(c)
            elif qn in nm:
                contains.append(c)
        # Prefix hits first (sorted by population), then substring hits.
        prefix.sort(key=lambda c: -(c.get("population") or 0))
        contains.sort(key=lambda c: -(c.get("population") or 0))
        ranked = (prefix + contains)[:limit]
    return {
        "results": [
            {
                "ref":        c.get("ref"),
                "name":       c.get("name"),
                "country":    c.get("country"),
                "population": c.get("population"),
                "lon":        c.get("lon"),
                "lat":        c.get("lat"),
            }
            for c in ranked
        ],
        "query": q,
    }


@app.get("/way-graph/spt/{city_idx}")
def way_graph_spt(
    city_idx: int,
    profile: str = "views_polygon",
    max_cost: float | None = Query(None, description="Filter to edges with cost-from-anchor < this"),
    max_features: int = Query(300_000, ge=1_000, le=2_000_000,
                              description="Cap on returned edges; keeps the cheapest N if total exceeds it"),
) -> dict:
    """Polygon-bounded SPT visualization for a single anchor.

    Reads /data/spt/<profile>/<city_idx>.npz. The polygon-bounded npz
    carries coords_lonlat inline so each file is self-contained.

    Returns a GeoJSON FeatureCollection of LineString features, one
    per non-seed kept vertex (vertex -> its parent), with `cost`
    properties for color ramping in MapLibre.
    """
    import numpy as np
    npz_path = SPT_DIR / profile / f"{city_idx}.npz"
    if not npz_path.exists():
        raise HTTPException(404, f"SPT not built yet for city_idx={city_idx} (profile '{profile}')")
    with np.load(npz_path) as f:
        node_global = np.asarray(f["node_global"])
        parent = np.asarray(f["parent"])
        cost = np.asarray(f["cost"])
        coords = np.asarray(f["coords_lonlat"])

    valid_mask = parent >= 0
    if max_cost is not None:
        valid_mask = valid_mask & (cost < float(max_cost))
    valid_idx = np.where(valid_mask)[0]
    n_valid = int(len(valid_idx))
    if n_valid == 0:
        return {
            "type": "FeatureCollection", "features": [],
            "city_idx": city_idx, "total_visited": int(len(node_global)),
            "cost_min": 0.0, "cost_max": 0.0,
        }

    if n_valid > max_features and max_cost is None:
        valid_costs = cost[valid_idx]
        cheapest = np.argpartition(valid_costs, max_features)[:max_features]
        valid_idx = valid_idx[cheapest]

    child_lonlat  = coords[valid_idx]
    parent_lonlat = coords[parent[valid_idx]]
    edge_costs    = cost[valid_idx].astype("float32")

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(child_lonlat[i, 0]), float(child_lonlat[i, 1])],
                    [float(parent_lonlat[i, 0]), float(parent_lonlat[i, 1])],
                ],
            },
            "properties": {"cost": float(edge_costs[i])},
        }
        for i in range(len(valid_idx))
    ]
    return {
        "type": "FeatureCollection",
        "features": features,
        "city_idx": city_idx,
        "total_visited": int(len(node_global)),
        "kept_count": len(features),
        "cost_min": float(cost[valid_mask].min()) if valid_mask.any() else 0.0,
        "cost_max": float(cost.max()),
        "filtered_max_cost": float(max_cost) if max_cost is not None else None,
    }


@app.get("/way-graph/paired-spt/{a}/{b}")
def way_graph_paired_spt(
    a: int,
    b: int,
    profile: str = "views_polygon",
) -> dict:
    """Paired-SPT corridor visualization for a chain pair (A, B).

    A "paired SPT" is the corridor of vertices visited by BOTH A's
    polygon SPT and B's polygon SPT — i.e., the area where A's reach
    overlaps B's frontier when routing from A to B. Each returned edge
    is (vertex -> A.parent[vertex]) for vertices in the intersection,
    colored by cost from A (so green near A, red near B).

    Returns FeatureCollection of LineStrings with `cost`, `cost_a`,
    `cost_b` properties. The lens/crescent shape between the two
    anchors is the methodology screenshot.
    """
    import numpy as np
    a_path = SPT_DIR / profile / f"{a}.npz"
    b_path = SPT_DIR / profile / f"{b}.npz"
    if not a_path.exists() or not b_path.exists():
        raise HTTPException(404, f"SPT not built for one of city_idx {a}, {b} (profile '{profile}')")
    with np.load(a_path) as fa:
        a_vids = np.asarray(fa["node_global"])
        a_parent = np.asarray(fa["parent"])
        a_cost = np.asarray(fa["cost"])
        a_coords = np.asarray(fa["coords_lonlat"])
    with np.load(b_path) as fb:
        b_vids = np.asarray(fb["node_global"])
        b_cost = np.asarray(fb["cost"])

    # Map B's vid -> B's local cost, then look up cost_b for each A vertex.
    b_order = np.argsort(b_vids, kind="stable")
    b_vids_sorted = b_vids[b_order]
    b_cost_sorted = b_cost[b_order]
    pos = np.searchsorted(b_vids_sorted, a_vids)
    pos = np.clip(pos, 0, len(b_vids_sorted) - 1)
    in_b = b_vids_sorted[pos] == a_vids
    cost_b_for_a = np.where(in_b, b_cost_sorted[pos], np.inf)

    # Paired-SPT corridor: vertices in BOTH A and B, with a valid parent
    # in A (so we have an edge to draw).
    keep_mask = in_b & (a_parent >= 0)
    valid_idx = np.where(keep_mask)[0]
    n_valid = int(len(valid_idx))
    if n_valid == 0:
        return {
            "type": "FeatureCollection", "features": [],
            "a": a, "b": b, "kept_count": 0,
            "a_visited": int(len(a_vids)), "b_visited": int(len(b_vids)),
            "intersection_size": int(in_b.sum()),
        }

    child_lonlat  = a_coords[valid_idx]
    parent_lonlat = a_coords[a_parent[valid_idx]]
    ca = a_cost[valid_idx].astype("float32")
    cb = cost_b_for_a[valid_idx].astype("float32")
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(child_lonlat[i, 0]), float(child_lonlat[i, 1])],
                    [float(parent_lonlat[i, 0]), float(parent_lonlat[i, 1])],
                ],
            },
            "properties": {
                "cost":   float(ca[i]),    # cost from A — primary color
                "cost_b": float(cb[i]),    # cost from B — alt color
            },
        }
        for i in range(n_valid)
    ]
    return {
        "type": "FeatureCollection",
        "features": features,
        "a": a, "b": b,
        "kept_count": n_valid,
        "a_visited": int(len(a_vids)),
        "b_visited": int(len(b_vids)),
        "intersection_size": int(in_b.sum()),
        "cost_min": float(ca.min()), "cost_max": float(ca.max()),
    }


@app.get("/trunk/blob/{a}/{b}")
def trunk_blob(
    a: int,
    b: int,
    profile: str = "views",
    db: str = Query(
        "paired_trunks.db",
        description="Which trunk DB file under /data/spt/<profile>/ to "
                    "read. Use 'paired_trunks.db' for the currently-served "
                    "build, or 'paired_trunks_v1.db' / 'paired_trunks_v2.db' "
                    "for side-by-side comparison.",
    ),
) -> dict:
    """Return the (A, B) trunk blob as a GeoJSON FeatureCollection.

    Each row of the blob is (vid, succ, lat, lon). Emits:
      * a Point Feature per vertex, colored by whether its succ is
        NULL (frontier / terminus) or valid (interior);
      * a LineString Feature per (vertex, succ_vertex) edge — the
        arrow the router walks. Endpoints missing from the trunk are
        represented as bare Points (no edge drawn).

    Reads directly from the file — bypasses the in-memory preload —
    so you can visualize v1 or v2 without swapping which DB the API
    serves for routing.
    """
    import sqlite3
    import numpy as np
    if "/" in db or ".." in db:
        raise HTTPException(400, "`db` must be a bare filename, no path")
    db_path = SPT_DIR / profile / db
    if not db_path.exists():
        raise HTTPException(404, f"no trunk db at {db_path}")
    TRUNK_DTYPE = np.dtype([
        ("vid",  "<i8"),
        ("succ", "<i8"),
        ("lat",  "<f4"),
        ("lon",  "<f4"),
    ])
    # `immutable=1` skips WAL/SHM sidecar creation — the RO mount blocks
    # those, same reason trunk_router.py's preload uses this flag.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    try:
        row = conn.execute(
            "SELECT n_rows, blob FROM trunk_blobs "
            "WHERE src_city = ? AND dst_city = ?",
            (a, b),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(404, f"no trunk for ({a}, {b}) in {db}")
    _n_rows, blob = row
    arr = np.frombuffer(blob, dtype=TRUNK_DTYPE)

    # Point features for every vertex; color hint via `is_frontier`
    # (True when succ == -1, i.e., trunk terminates here).
    features: list[dict] = []
    for i in range(len(arr)):
        vid = int(arr["vid"][i])
        succ = int(arr["succ"][i])
        lat = float(arr["lat"][i])
        lon = float(arr["lon"][i])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "kind":  "vertex",
                "vid":   vid,
                "succ":  succ,
                "is_frontier": succ == -1,
            },
        })

    # Edge features: for each vertex, if succ is valid AND present in
    # the trunk, draw a LineString from vertex → succ vertex.
    vid_sorted = arr["vid"]  # v1 and v2 both sort by vid
    for i in range(len(arr)):
        succ = int(arr["succ"][i])
        if succ == -1:
            continue
        pos = int(np.searchsorted(vid_sorted, succ))
        if pos >= len(arr) or int(arr["vid"][pos]) != succ:
            continue   # succ points outside trunk (e.g., v1 SYNTH edge)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(arr["lon"][i]),   float(arr["lat"][i])],
                    [float(arr["lon"][pos]), float(arr["lat"][pos])],
                ],
            },
            "properties": {
                "kind":     "edge",
                "from_vid": int(arr["vid"][i]),
                "to_vid":   succ,
            },
        })

    n_frontier = int((arr["succ"] == -1).sum())
    return {
        "type": "FeatureCollection",
        "features": features,
        "a": a, "b": b,
        "db": db,
        "n_vertices": int(len(arr)),
        "n_frontier": n_frontier,
    }


@app.get("/trunk/chains/{a}/{b}")
def trunk_chains(
    a: int,
    b: int,
    profile: str = "views",
    db: str = Query(
        "paired_trunks.db",
        description="Which trunk DB file under /data/spt/<profile>/.",
    ),
) -> dict:
    """Return the (A, B) trunk as chained polylines — one LineString
    per succ-chain rooted at a LEAF (vertex with no incoming succ
    pointer). Each chain walks succ parents toward A-seed until it
    joins another already-emitted chain or hits succ=-1.

    Feature count ≈ number of leaves (tens to hundreds per trunk)
    instead of the ~2 × n_vertices returned by /trunk/blob. Used by
    the 'paired trunks along route' UI to visualize a full route
    without loading millions of little segments.
    """
    import sqlite3
    import numpy as np
    if "/" in db or ".." in db:
        raise HTTPException(400, "`db` must be a bare filename, no path")
    db_path = SPT_DIR / profile / db
    if not db_path.exists():
        raise HTTPException(404, f"no trunk db at {db_path}")
    TRUNK_DTYPE = np.dtype([
        ("vid",  "<i8"),
        ("succ", "<i8"),
        ("lat",  "<f4"),
        ("lon",  "<f4"),
    ])
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    try:
        row = conn.execute(
            "SELECT n_rows, blob FROM trunk_blobs "
            "WHERE src_city = ? AND dst_city = ?",
            (a, b),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(404, f"no trunk for ({a}, {b}) in {db}")
    _n_rows, blob = row
    arr = np.frombuffer(blob, dtype=TRUNK_DTYPE)
    n = len(arr)
    if n == 0:
        return {"type": "FeatureCollection", "features": [],
                "a": a, "b": b, "db": db, "n_chains": 0, "n_vertices": 0}
    vid = arr["vid"]
    succ = arr["succ"]
    pos = np.searchsorted(vid, succ)
    pos_clip = np.clip(pos, 0, n - 1)
    in_range = pos < n
    matched = in_range & (vid[pos_clip] == succ)
    next_pos = np.where(matched & (succ != -1), pos, -1).astype(np.int64)
    # LEAF: no other vertex points to it via succ.
    has_child = np.zeros(n, dtype=bool)
    valid_next = next_pos[next_pos >= 0]
    if len(valid_next):
        has_child[valid_next] = True
    leaves = np.flatnonzero(~has_child)

    features: list[dict] = []
    visited = np.zeros(n, dtype=bool)
    lat_arr = arr["lat"]
    lon_arr = arr["lon"]
    for leaf_i in leaves:
        coords: list[list[float]] = []
        cur = int(leaf_i)
        while cur >= 0 and not visited[cur]:
            visited[cur] = True
            coords.append([float(lon_arr[cur]), float(lat_arr[cur])])
            cur = int(next_pos[cur])
        # Touch the merge vertex so chains join visually.
        if cur >= 0 and len(coords) >= 1:
            coords.append([float(lon_arr[cur]), float(lat_arr[cur])])
        if len(coords) >= 2:
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"n_hops": len(coords) - 1},
            })
    return {
        "type": "FeatureCollection",
        "features": features,
        "a": a, "b": b, "db": db,
        "n_vertices": int(n),
        "n_chains": len(features),
    }


@app.get("/way-graph/spt-status")
def way_graph_spt_status(profile: str = "views_polygon") -> dict:
    """Which polygon-bounded SPTs have been written so far. Lets the
    web UI light up anchors green as the (slow) overnight build
    progresses, and grey-out the ones still pending.

    The polygon-bounded SPT compute writes <city_idx>.npz files to
    /data/spt/<profile>/. Sister cities.json carries the city_idx →
    ref/name/lon/lat mapping.
    """
    spt_dir = SPT_DIR / profile
    cities_path = spt_dir / "cities.json"
    if not cities_path.exists():
        raise HTTPException(
            404,
            f"no cities.json at {cities_path} — profile '{profile}' "
            f"hasn't been started or has a different layout",
        )
    cities = json.loads(cities_path.read_text())
    done: list[int] = []
    for f in spt_dir.glob("*.npz"):
        try:
            done.append(int(f.stem))
        except ValueError:
            continue
    done.sort()
    return {
        "profile": profile,
        "total": len(cities),
        "done": len(done),
        "done_indices": done,
        "cities": cities,
    }


@app.get("/pois")
def pois_endpoint(
    bbox: str = Query(..., description="minlon,minlat,maxlon,maxlat"),
    category: str | None = Query(
        None,
        description="Comma-separated: viewpoint,lodging,food,bike_service,water",
    ),
    limit: int = Query(500, ge=1, le=5000),
):
    parts = bbox.split(",")
    if len(parts) != 4:
        raise HTTPException(400, "bbox must be minlon,minlat,maxlon,maxlat")
    try:
        coords = tuple(float(p) for p in parts)
    except ValueError:
        raise HTTPException(400, "bbox values must be numeric")
    cats = [c.strip() for c in category.split(",")] if category else None
    items = pois.query_bbox(coords, cats, limit)
    return {"count": len(items), "items": items}
