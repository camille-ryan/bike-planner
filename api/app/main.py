"""FastAPI service for the Graz → Copenhagen bike-tour planner.

Endpoints:
  GET /health
  GET /route   — proxy to BRouter, optionally with re-ranked alternatives
  GET /pois    — bbox + category lookup against the SpatiaLite POI DB
  GET /stages  — split a route into ~100km legs with lodging clusters
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import brouter, pois, scoring, stages
from .settings import DEFAULT_PROFILE

app = FastAPI(title="Bike Routing API", version="0.1.0")

# Open CORS — this server is on a private Tailscale network, accessed by a
# laptop running the static web UI from any local origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
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
    try:
        routes = await brouter.fetch_alternatives(points, profile, alternatives)
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
        "routes": routes,
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
