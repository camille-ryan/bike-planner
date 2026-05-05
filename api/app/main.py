"""FastAPI service for the Graz → Copenhagen bike-tour planner.

Endpoints:
  GET /health
  GET /route   — proxy to BRouter, optionally with re-ranked alternatives
  GET /pois    — bbox + category lookup against the SpatiaLite POI DB
  GET /stages  — split a route into ~100km legs with lodging clusters
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import anchors, brouter, leg_cache, pois, scoring, stages
from .settings import DEFAULT_PROFILE

app = FastAPI(title="Bike Routing API", version="0.1.0")

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


@app.get("/route")
async def route(
    from_: str | None = Query(None, alias="from", description="lon,lat"),
    to: str | None = Query(None, description="lon,lat"),
    lonlats: str | None = Query(
        None,
        description="lon,lat|lon,lat|... — a multi-waypoint route. "
                    "Overrides from+to when set. Use this for long corridors "
                    "(BRouter scales much better with intermediate via-points).",
    ),
    profile: str = DEFAULT_PROFILE,
    alternatives: int = Query(0, ge=0, le=3),
    rerank: bool = Query(False, description="Compute scenic + curvature scoring and reorder"),
    auto_waypoint: bool = Query(
        True,
        description="For long 2-point routes, insert place=city|town waypoints "
                    "along the corridor to keep BRouter's per-leg search bounded. "
                    "Set false to force the engine to plan end-to-end.",
    ),
):
    if lonlats:
        try:
            points = []
            for pair in lonlats.split("|"):
                lon, lat = (float(x) for x in pair.split(","))
                points.append((lon, lat))
            if len(points) < 2:
                raise ValueError("need at least two points")
        except Exception as exc:
            raise HTTPException(400, f"lonlats must be 'lon,lat|lon,lat|...': {exc}")
    elif from_ and to:
        points = [_parse_lonlat(from_, "from"), _parse_lonlat(to, "to")]
    else:
        raise HTTPException(400, "provide either lonlats=... or from=&to=")
    # Auto-waypoint only when caller gave us exactly two points; multi-waypoint
    # callers have already expressed an opinion about the corridor and we
    # shouldn't second-guess it.
    inserted_anchors: list[dict] = []
    if auto_waypoint and len(points) == 2:
        picks = anchors.auto_waypoints(points[0], points[1])
        if picks:
            points = [points[0]] + [(lon, lat) for lon, lat, _ in picks] + [points[1]]
            inserted_anchors = [
                {"lon": lon, "lat": lat, "name": name} for lon, lat, name in picks
            ]
    try:
        # Primary route: cached per-leg. Alternatives (if any): full BRouter
        # call end-to-end uncached, since alt-idx>0 only makes sense over the
        # whole search and wouldn't compose across cached legs.
        primary = await brouter.fetch_split_route(points, profile)
        routes = [primary]
        if alternatives > 0:
            for idx in range(1, alternatives + 1):
                try:
                    routes.append(await brouter.fetch_route(points, profile, idx))
                except RuntimeError:
                    break
    except RuntimeError as exc:
        raise HTTPException(502, f"BRouter: {exc}")
    if rerank and len(routes) > 1:
        routes = scoring.rerank(routes)
    elif rerank:
        # Single route — still annotate with scoring so clients see consistent shape.
        scoring.score_route(routes[0])
    return {
        "profile": profile,
        "rerank": rerank,
        "count": len(routes),
        "auto_waypoints": inserted_anchors,
        "routes": routes,
    }


@app.get("/cache/stats")
def cache_stats() -> dict:
    return leg_cache.stats()


@app.post("/cache/clear")
def cache_clear(profile: str | None = Query(None)) -> dict:
    deleted = leg_cache.clear(profile)
    return {"deleted": deleted, "profile": profile}


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


@app.get("/stages")
async def stages_endpoint(
    from_: str = Query(..., alias="from"),
    to: str = Query(...),
    profile: str = DEFAULT_PROFILE,
    target_km: float = Query(100.0, gt=10.0, le=300.0),
    lodging_radius_m: int = Query(3000, ge=200, le=20000),
):
    a = _parse_lonlat(from_, "from")
    b = _parse_lonlat(to, "to")
    try:
        return await stages.plan(a, b, profile, target_km, lodging_radius_m)
    except RuntimeError as exc:
        raise HTTPException(502, f"BRouter: {exc}")
