"""Per-edge cost function — Python port of the lht.brf semantics needed
to make Voronoi cells and gradient walks behave sensibly.

v1 deliberately drops two BRouter terms:
  - elevation (uphill quadratic) — requires SRTM join, deferred to v2.
  - scenic biases (estimated_traffic/forest/river/town/noise classes) —
    BRouter computes these from neighborhood spatial heuristics during
    its own preprocess. Approximating them is a project of its own; v1
    skips them and accepts that cells reflect cyclable-shortest-path,
    not scenic-tour-shortest-path.

What we *do* keep, sourced from `brouter/profiles/lht.brf`:
  - Highway-type cost factors (cycleway/residential cheap, primary
    expensive unless bike-equipped, motorway excluded).
  - Surface filter for 40 mm tires (sand/mud/grade5 excluded).
  - Track-type penalties (grade1 cheap, grade5 effectively excluded).
  - Cycleway/bicycle access discount.

Returns a per-meter cost factor (multiplied by length to get edge cost)
or `None` if the edge should be excluded entirely.
"""
EXCLUDE = {
    "motorway", "motorway_link",
    "proposed", "abandoned", "construction",
    "raceway", "elevator", "platform",
}

EXCLUDE_SURFACES = {"sand", "mud", "grass", "earth", "dirt"}

# Highway-type costfactor when the way *is* bike-accessible (has cycleway,
# bicycle=designated, etc.) vs not. Mirrors the LHT profile's branching
# at lht.brf:285-289 and the cycleway/residential cases above.
_HIGHWAY_COST_BIKE = {
    "cycleway":      1.0,
    "residential":   1.0,
    "living_street": 1.0,
    "service":       1.3,
    "track":         1.1,
    "path":          1.1,
    "footway":       1.1,
    "pedestrian":    2.2,
    "bridleway":     5.0,
    "unclassified":  1.0,
    "tertiary":      1.0,
    "secondary":     1.1,
    "primary":       1.3,
    "trunk":         1.5,
    "trunk_link":    1.5,
    "primary_link":  1.3,
    "secondary_link": 1.1,
    "tertiary_link": 1.0,
}
_HIGHWAY_COST_NOBIKE = {
    "cycleway":      1.0,   # cycleway implies bike-accessible
    "residential":   1.0,
    "living_street": 1.0,
    "service":       1.3,
    "track":         5.0,   # track without bike infra is rough
    "path":          5.0,
    "footway":       5.0,
    "pedestrian":    3.0,
    "bridleway":     5.0,
    "unclassified":  1.3,
    "tertiary":      1.4,
    "secondary":     1.8,
    "primary":       4.0,
    "trunk":         10.0,
    "trunk_link":    10.0,
    "primary_link":  4.0,
    "secondary_link": 1.8,
    "tertiary_link": 1.4,
}

# Per-tracktype multiplier on top of base highway cost. From lht.brf:275-279.
_TRACKTYPE = {
    "grade1": 1.0,
    "grade2": 1.4,
    "grade3": 2.0,
    "grade4": 4.0,
    "grade5": 100.0,   # effectively excluded
}


def _has_bike_infra(bicycle: str, cycleway: str, bicycle_road: str) -> bool:
    if bicycle_road == "yes":
        return True
    if bicycle in ("designated", "yes", "permissive"):
        return True
    if cycleway and cycleway not in ("no", "none", ""):
        return True
    return False


def bike_edge_cost(
    *,
    highway: str,
    surface: str,
    tracktype: str,
    bicycle: str,
    cycleway: str,
    access: str,
    bicycle_road: str,
    is_ferry: bool = False,
) -> float | None:
    """Return a unitless per-meter costfactor, or None to exclude the edge."""
    # Ferries are tagged `route=ferry` in OSM and typically don't carry
    # a `highway=*` tag. They're the only way across the Baltic for
    # Graz->Copenhagen, so we accept them with a bumped per-meter cost.
    # BRouter's lht.brf uses ~5.7 plus a 10000 initial cost penalty for
    # boarding; we don't model the initial cost, so we bump the per-meter
    # factor a bit higher (~8) to compensate. Bicycles explicitly excluded
    # from a particular ferry are still rejected.
    if is_ferry:
        if bicycle in ("no", "private"):
            return None
        return 8.0

    if not highway or highway in EXCLUDE:
        return None
    if access in ("private", "no") and bicycle not in ("yes", "designated", "permissive"):
        return None
    if surface in EXCLUDE_SURFACES:
        return None
    if bicycle in ("no", "private", "dismount"):
        return None

    has_bike = _has_bike_infra(bicycle, cycleway, bicycle_road)
    table = _HIGHWAY_COST_BIKE if has_bike else _HIGHWAY_COST_NOBIKE
    base = table.get(highway)
    if base is None:
        # Unknown highway tag — give it a moderate non-zero cost rather than
        # excluding outright; the routing graph has many odd ones.
        base = 2.0

    # Tracktype is only meaningful on track-like highways.
    if tracktype in _TRACKTYPE and highway in ("track", "path", "footway", "road"):
        base *= _TRACKTYPE[tracktype]

    return float(base)
