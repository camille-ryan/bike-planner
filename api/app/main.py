"""FastAPI service for the Graz → Copenhagen bike-tour planner.

Endpoints:
  GET /health
  GET /trunk/route  — paired-trunk routing (blob DB, ~25 ms hot path)
  GET /pois         — bbox + category lookup against the SpatiaLite POI DB
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import json
from pathlib import Path

from . import db, pois, trunk_router
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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/trunk/route")
async def trunk_route(
    from_: str = Query(..., alias="from", description="lon,lat"),
    to: str = Query(..., description="lon,lat"),
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
    Dijkstra (~10-30 ms total for a Graz→Cph-scale chain)."""
    a = _parse_lonlat(from_, "from")
    b = _parse_lonlat(to, "to")
    try:
        feat = trunk_router.route(a, b, profile, simplify_m=simplify_m)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"trunk route: {exc}")
    return {"profile": profile, "route": feat}


@app.get("/way-graph/spt/{city_idx}")
def way_graph_spt(
    city_idx: int,
    profile: str = "direct_polygon",
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

    # Subsample if huge. Keep the cheapest `max_features` edges by
    # cost-from-anchor so the visible SPT is the core reach (densest
    # near the anchor) rather than a sparse scatter. argpartition is O(n).
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


@app.get("/way-graph/spt-status")
def way_graph_spt_status(profile: str = "direct_polygon") -> dict:
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
