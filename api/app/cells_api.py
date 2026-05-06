"""Serve per-city Voronoi cell polygons computed by the preprocess pipeline.

We load the small JSON files (cities.json, cells.geojson, city_graph.json)
into memory at first use and keep them there — they're tiny (~500 KB
total for Austria, will be ~2-5 MB at corridor scale). The big numpy
SPT arrays are NOT loaded here; those are reserved for routing in
phase 3b.

The data is profile-keyed: `data/spt/<profile>/cells.geojson`. If the
preprocess hasn't been run for a given profile, the endpoints return 404.
"""
import json
from functools import lru_cache

from .settings import SPT_DIR


@lru_cache(maxsize=8)
def _load(profile: str) -> dict:
    """Load and merge the profile's cell artifacts into one in-memory blob."""
    base = SPT_DIR / profile
    if not (base / "cells.geojson").exists():
        return {}
    with open(base / "cells.geojson") as fh:
        cells = json.load(fh)
    with open(base / "cities.json") as fh:
        cities = json.load(fh)
    with open(base / "city_graph.json") as fh:
        city_graph = json.load(fh)

    # Index cells by city_idx for O(1) lookup.
    by_idx: dict[int, dict] = {}
    for feat in cells["features"]:
        ci = int(feat["properties"]["city_idx"])
        by_idx[ci] = feat

    # Index city neighbors for the click-to-show side panel.
    neighbors: dict[int, list[dict]] = {}
    for fa, tb, w in zip(city_graph["from_city"], city_graph["to_city"], city_graph["weight"]):
        neighbors.setdefault(int(fa), []).append({"city_idx": int(tb), "weight": float(w)})
    for ci, nbrs in neighbors.items():
        nbrs.sort(key=lambda x: x["weight"])

    return {
        "cities": cities,
        "cells_by_idx": by_idx,
        "neighbors": neighbors,
    }


def list_cities(profile: str) -> list[dict] | None:
    blob = _load(profile)
    if not blob:
        return None
    # Strip the dense node_idx / osm_id fields the frontend doesn't need.
    return [
        {
            "city_idx": i,
            "name": c["name"],
            "place": c["place"],
            "country": c.get("country"),
            "lon": c["lon"],
            "lat": c["lat"],
        }
        for i, c in enumerate(blob["cities"])
    ]


def cell_for(profile: str, city_idx: int) -> dict | None:
    blob = _load(profile)
    if not blob:
        return None
    feat = blob["cells_by_idx"].get(int(city_idx))
    if feat is None:
        return None
    nbrs = blob["neighbors"].get(int(city_idx), [])
    enriched_nbrs = []
    for n in nbrs:
        c = blob["cities"][n["city_idx"]]
        enriched_nbrs.append({
            "city_idx": n["city_idx"],
            "name": c["name"],
            "lon": c["lon"], "lat": c["lat"],
            "weight": n["weight"],
        })
    return {
        "city": {
            "city_idx": int(city_idx),
            "name": blob["cities"][int(city_idx)]["name"],
            "lon": blob["cities"][int(city_idx)]["lon"],
            "lat": blob["cities"][int(city_idx)]["lat"],
        },
        "polygon": feat,
        "neighbors": enriched_nbrs,
    }
