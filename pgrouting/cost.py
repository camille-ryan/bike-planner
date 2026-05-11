"""Per-edge cost function — V2 (see /V2.md §1).

V2 changes vs V1:
  - `highway=cycleway` base lowered to 0.9 (sub-1.0 floor) — the router
    will detour up to ~11% of length to use a designated cycle path,
    correcting the V1 problem where cycleway tied at 1.0 with residential
    and tertiary roads.
  - `highway=track` bike-infra distinction dropped (only 3.4% of tracks
    carry an explicit bike-infra tag in central Europe; the V1 NOBIKE
    value of 5.0 was an artifact of tagging diligence, not rideability).
    Base 1.1 in both tables; tracktype + surface multipliers carry the
    quality signal.
  - `highway=path` without bike infra: 5.0 → 2.0. Forest paths become
    routable when they save substantial distance. Failure mode is
    "annoying push-the-bike," not "illegal sidewalk shortcut."
  - `highway=footway` without bike infra: unchanged at 5.0 (footways
    are designed for pedestrians; riding bikes there is often illegal).
  - Surface filter replaced: V1 hard-excluded sand/mud/grass/earth/dirt;
    V2 grades every surface multiplicatively. Sand/mud/ice/snow at 10×
    (walking-equivalent, including misery premium), grass/woodchips 3×,
    dirt/earth/ground/unpaved 1.5×, gravel/sett/pebblestone 1.3×,
    cobblestone 1.7×, fine_gravel 1.1×, asphalt/paved/compacted 1.0×.
    No surface is hard-excluded — only access is.
  - New: `bicycle_road=yes` applies a 0.9× multiplicative bonus on top
    of the final cost. Captures Fahrradstraße / bicycle-priority road
    designations; stacks with the cycleway sub-1.0 (correct: a cycleway
    that's also a Fahrradstraße is a stronger signal than either alone).

V2 Phase A (this commit): elevation-aware grade + curvy-descent multipliers
(V2.md §1.4). Take effect only when `grade_pct` and `sinuosity` are
supplied; defaults (0.0 and 1.0) are no-ops so the existing per-tag
ingest path keeps working until DEM ingest lands.
  - Uphill: `1 + 0.028 × grade²` — pure quadratic on positive grade.
    Calibration: 6%→2.0 (user's "trouble to sustain"), 9%→3.3, 12%→5.0.
  - Straight downhill: `1 - 0.05g + 0.005g²` — peaked bonus, min 0.875
    at g=5% (the "fun zone"), back to 1.0 at g=10%, mild penalty above.
  - Curvy descent: `× (1 + 0.1 × (sinuosity - 1) × g)` — applied only
    on descents. Per-way sinuosity = actual_length / endpoint_distance.

Known gaps deferred to V3 (V2.md §1.2):
  - spatial scenic signals (landcover, water/camp POI density along
    corridor — see GAP+C&O calibration in design notes)
  - `access:conditional` / `seasonal=yes` parsing (mountain pass
    seasonal closures)
  - route-relation enrichment (`route=bicycle` with
    `network=icn|ncn|rcn|lcn`)
  - turn-aware routing (Phase B — fewer turns, dangerous turns)

Returns a per-meter cost factor (multiplied by length to get edge cost)
or `None` if the edge should be excluded entirely.
"""
EXCLUDE = {
    "motorway", "motorway_link",
    "proposed", "abandoned", "construction",
    "raceway", "elevator", "platform",
}

# Highway-type costfactor when the way *is* bike-accessible (has cycleway,
# bicycle=designated, etc.) vs not. Mirrors the LHT profile's branching
# at lht.brf:285-289 and the cycleway/residential cases above.
_HIGHWAY_COST_BIKE = {
    "cycleway":      0.9,    # V2: sub-1.0 floor — prefer cycleway
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
    "cycleway":      0.9,    # cycleway implies bike-accessible
    "residential":   1.0,
    "living_street": 1.0,
    "service":       1.3,
    "track":         1.1,    # V2: drop bike-infra distinction on tracks
    "path":          2.0,    # V2: was 5.0; soften to allow forest paths
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
# OSM tracktype=* describes the firmness/quality of unpaved tracks (not
# elevation grade). Only applied on track-like highways (see below).
_TRACKTYPE = {
    "grade1": 1.0,
    "grade2": 1.4,
    "grade3": 2.0,
    "grade4": 4.0,
    "grade5": 100.0,   # effectively excluded
}

# V2: graded surface multiplier (replaces V1 EXCLUDE_SURFACES blacklist).
# Anchored on a walking-equivalent reasoning: riding 10 m on sand is
# about as bad as riding 100 m on asphalt, including the misery premium
# over the pure walk-vs-ride speed ratio (~5×). A surface not in this
# table (including untagged) gets factor 1.0 — no penalty, no bonus.
_SURFACE = {
    "asphalt":       1.0,
    "paved":         1.0,
    "concrete":      1.0,
    "paving_stones": 1.0,
    "compacted":     1.0,
    "fine_gravel":   1.1,
    "gravel":        1.3,
    "pebblestone":   1.3,
    "sett":          1.3,
    "cobblestone":   1.7,
    "dirt":          1.5,
    "earth":         1.5,
    "ground":        1.5,
    "unpaved":       1.5,
    "grass":         3.0,
    "woodchips":     3.0,
    "sand":          10.0,
    "mud":           10.0,
    "ice":           10.0,
    "snow":          10.0,
}

# V2: legal bike-priority road designation (German Fahrradstraße and
# equivalents). A regular road repurposed for bike priority — cars are
# guests. Applied multiplicatively on the final cost so it stacks with
# the cycleway sub-1.0 floor.
_BICYCLE_ROAD_BONUS = 0.9

# V2 Phase A: elevation-based grade multipliers (require DEM ingest to
# populate `grade_pct` at the call site). Coefficients calibrated to
# match the user's cycling tolerance — see docstring for examples.
_UPHILL_COEFF = 0.028          # uphill:   1 + k·g²
_DOWNHILL_LINEAR = 0.05        # downhill: 1 - a·g + b·g², min at g = a/(2b) = 5%
_DOWNHILL_QUADRATIC = 0.005    #   so bonus peaks in the "fun zone," neutral by 10%, penalty beyond
_SINUOSITY_COEFF = 0.1         # curvy descent extra: 1 + c·(sinuosity-1)·g


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
    grade_pct: float = 0.0,     # positive = uphill, negative = downhill
    sinuosity: float = 1.0,     # per-way: actual_length / endpoint_distance
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
    if bicycle in ("no", "private", "dismount"):
        return None

    has_bike = _has_bike_infra(bicycle, cycleway, bicycle_road)
    table = _HIGHWAY_COST_BIKE if has_bike else _HIGHWAY_COST_NOBIKE
    cost = table.get(highway)
    if cost is None:
        # Unknown highway tag — give it a moderate non-zero cost rather than
        # excluding outright; the routing graph has many odd ones.
        cost = 2.0

    # Tracktype is only meaningful on track-like highways.
    if tracktype in _TRACKTYPE and highway in ("track", "path", "footway", "road"):
        cost *= _TRACKTYPE[tracktype]

    # V2: graded surface multiplier (was a blacklist exclude in V1).
    # Untagged or unknown surfaces fall through with no change.
    if surface in _SURFACE:
        cost *= _SURFACE[surface]

    # V2: Fahrradstraße / bicycle-priority road bonus.
    if bicycle_road == "yes":
        cost *= _BICYCLE_ROAD_BONUS

    # V2 Phase A: grade + curvy-descent multipliers. No-op when
    # grade_pct == 0.0 and sinuosity == 1.0 (the defaults), so callers
    # that haven't been updated to provide elevation data still work.
    if grade_pct > 0.0:
        cost *= 1.0 + _UPHILL_COEFF * grade_pct * grade_pct
    elif grade_pct < 0.0:
        g = -grade_pct
        cost *= 1.0 - _DOWNHILL_LINEAR * g + _DOWNHILL_QUADRATIC * g * g
        if sinuosity > 1.0:
            cost *= 1.0 + _SINUOSITY_COEFF * (sinuosity - 1.0) * g

    return float(cost)
