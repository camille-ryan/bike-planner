"""FastAPI service for the Graz → Copenhagen bike-tour planner.

Endpoints:
  GET /health
  GET /spt/route   — SPT-based routing using the pgrouting preprocess output
  GET /cells/*     — Voronoi cell polygons + neighbor lists for the UI
  GET /pois        — bbox + category lookup against the SpatiaLite POI DB
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import cells_api, pois, spt_router
from .settings import DEFAULT_PROFILE

app = FastAPI(title="Bike Routing API", version="0.2.0")

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


@app.get("/spt/route")
async def spt_route(
    from_: str = Query(..., alias="from", description="lon,lat"),
    to: str = Query(..., description="lon,lat"),
    profile: str = DEFAULT_PROFILE,
) -> dict:
    """SPT-based routing: snap endpoints, plan a city sequence on the
    city graph, walk per-city SPT parent pointers to assemble the path."""
    a = _parse_lonlat(from_, "from")
    b = _parse_lonlat(to, "to")
    try:
        feat = spt_router.route(a, b, profile)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"SPT route: {exc}")
    return {"profile": profile, "route": feat}


@app.get("/cells/cities")
def cells_cities(profile: str = DEFAULT_PROFILE) -> dict:
    """List the cities for which we have a precomputed Voronoi cell."""
    items = cells_api.list_cities(profile)
    if items is None:
        raise HTTPException(404, f"no SPT data for profile '{profile}' — run preprocess first")
    return {"profile": profile, "count": len(items), "cities": items}


@app.get("/cells/{city_idx}")
def cells_one(city_idx: int, profile: str = DEFAULT_PROFILE) -> dict:
    """Return the Voronoi cell polygon for a city, plus its neighbor list."""
    out = cells_api.cell_for(profile, city_idx)
    if out is None:
        raise HTTPException(404, f"no cell for city_idx={city_idx} (profile '{profile}')")
    return out


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
