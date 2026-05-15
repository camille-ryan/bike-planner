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
(V2.md §1.4). Take effect only when `grade_pct` and `curv` are
supplied; defaults (0.0 and 0.0) are no-ops so the existing per-tag
ingest path keeps working until DEM ingest lands.
  - Uphill: `1 + 0.028 × grade²` — pure quadratic on positive grade.
    Calibration: 6%→2.0 (user's "trouble to sustain"), 9%→3.3, 12%→5.0.
  - Straight downhill: `1 - 0.05g + 0.005g²` — peaked bonus, min 0.875
    at g=5% (the "fun zone"), back to 1.0 at g=10%, mild penalty above.
  - Curvy descent: `× (1 + 0.0005 × curv × g)` — applied only on
    descents. `curv` is the total bend angle (degrees) the rider will
    encounter looking *ahead* in their direction of travel, summed over
    a ~300 m polyline window. Captures "steep descent into a curve":
    the descent segment looks ahead, sees the upcoming bend, gets
    penalized even if the descent itself is on a straight stretch.
    Forward and reverse edges carry separate `curv_fwd` / `curv_rev`
    in the database; the caller passes whichever applies for the
    direction it's pricing.

V2 Phase A.3: tree-cover overhead (universal modifier).
  - `canopy_frac` (∈ [0, 1]) is the fraction of an edge whose
    centerline falls inside a `landcover=forest` polygon. Populated
    by `compute_canopy_frac.py` from OSM `landuse=forest` /
    `natural=wood` areas.
  - Multiplier: `× (1 - 0.1 × canopy_frac)`. A fully-canopied edge
    gets a 10% bonus — the rider's "trees overhead = shade" value.
  - Applies to every profile (direct, balanced, scenic eventually).
    The "forests visible nearby" scenic-only signal lives elsewhere
    (`nearby_forest_frac`, deferred to scenic-profile work).

Known gaps deferred to V3 (V2.md §1.2):
  - additional spatial scenic signals (water proximity, low-traffic
    feel, camp/lodging density along corridor — see GAP+C&O
    calibration in design notes)
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
# Curvy-descent coefficient. Per-edge final factor at descent g% and
# `curv` degrees of upcoming bend is:
#     (1 - 0.05g + 0.005g²)   ×   (1 + k·curv·g)
#     └── straight downhill ──┘   └── curvy ────┘
# Calibration examples at k=0.0005, taken as total factor:
#   90° ahead at -10%:               1.45
#   90° ahead at -5%:                1.07
#   switchback group 540° at -8%:    2.91
#   switchback group 540° at -10%:   3.70
_CURV_COEFF = 0.0005

# V2 Phase A.3: tree-cover overhead bonus. A fully-canopied edge gets a
# 10% cost reduction (multiplier 0.9). Applied universally — direct,
# balanced, and (future) scenic all get the same bonus, because shade
# is objectively a better ride regardless of how scenic-leaning the
# profile is.
_CANOPY_BONUS = 0.1


# ---------------------------------------------------------------------
# V2 Phase A.3b: scenic profile weights.
#
# `direct`    — fastest/most-direct (default). Scenic signals ignored.
# `balanced`  — ~20% detour budget. Modest scenic preference.
# `scenic`    — ~50% detour budget. Strong scenic preference.
#
# Cost factor is `× (1 - k * tanh(scenic_score))`. `tanh` keeps the
# factor in (0, 1] no matter how large scenic_score gets, so no edge
# ever becomes negative-cost. `k` controls the maximum discount: 0.25
# for balanced (perfectly-scenic edges cost up to 25% less, breakeven
# at +20% detour) and 0.40 for scenic (breakeven at +50% detour).
#
# `scenic_score` is a weighted sum of two kinds of terms:
#
#   "local" signals add directly — the rider IS at the scenic thing
#   (forest, lake, river path, wetland, gorge).
#
#   "vista" signals multiply by `view_dominance / 200` (clipped >= 0)
#   — credit for seeing distant things ONLY if elevated relative to
#   the local terrain. A `_wide` signal alone is "this region has
#   lots of water/forest"; with view_dominance it becomes "you can
#   see it from here". Valley roads get no vista bonus.
#
# `regional_relief` is an always-on context term — being IN mountain
# country at all is scenic, even on a flat plain inside a basin.
#
# `waterway_along_edge` (the Mur-cycle-path detector) is modulated by
# local-terrain quality: a stream in a gorge or forest counts more
# than a stream in flat farmland. Multiplier:
#   wctx = 0.5 + 0.5 * local_relief_norm + 0.5 * forest_local
# ranges 0.5..1.5.
_SCENIC_W = {
    "forest_local":      0.5,
    "vineyard_local":    0.5,   # vineyards / wine country: pleasant cultivated land
    "water_local":       0.7,
    "wetland_local":     0.3,
    "waterway":          0.8,
    "relief":            0.6,
    "regional":          0.3,
    "vista_forest":      0.5,
    "vista_water":       0.7,
    "vista_sea":         1.5,   # ocean vistas are the strongest single-signal
    "vista_drama":       0.8,
    # Viewpoint POIs. Values are tiny (1 / disc-area-cells) so weights
    # have to be much larger than the polygon-fraction weights above.
    # The first experiment used 30 / 100 and got zero route change because
    # viewpoint density along forest-corridor routes is fairly uniform —
    # the contribution canceled out across alternatives. Bumped to
    # 200 / 500 to force viewpoint-rich detours.
    "viewpoint_local":     200.0,
    "viewpoint_regional":  500.0,
}

# Per-profile knob configuration. Each entry is the complete set of
# tunables for a profile so adding a new profile (e.g. experiments) is
# one dict literal and recompute-cost reads it via _PROFILES.get(name).
#
#   k_scenic            max scenic discount / penalty (tanh ceiling).
#                       0 disables the scenic multiplier entirely.
#   scenic_weight_mult  scalar multiplier applied to every term of
#                       scenic_score before tanh. 1.0 = nominal,
#                       higher = more aggressive scenic ranking.
#   surface_softening   compress (factor - 1.0) for tracktype/surface
#                       penalties. 1.0 = no softening, 0.5 = halve the
#                       penalty above 1.0. Bonuses (factor < 1.0) pass
#                       through unchanged.
#   uphill_threshold    grade%; below this the uphill multiplier is
#                       exactly 1.0 (no penalty). Discontinuous step
#                       to the normal formula above this grade.
#   uphill_offset       grade% to subtract before squaring in the
#                       uphill formula: 1 + COEFF * max(0, g - offset)^2.
#                       Continuous shift; offset=3 means a 3% climb
#                       feels flat.
_PROFILES: dict[str, dict] = {
    # Production profiles
    "direct": {
        "k_scenic":           0.0,
        "scenic_weight_mult": 0.0,
        "surface_softening":  1.0,
        "uphill_threshold":   0.0,
        "uphill_offset":      0.0,
        "softening_threshold": 0.0,
    },
    "balanced": {
        "k_scenic":           0.25,
        "scenic_weight_mult": 1.0,
        "surface_softening":  0.7,
        "uphill_threshold":   0.0,
        "uphill_offset":      0.0,
        "softening_threshold": 0.0,
    },
    "scenic": {
        "k_scenic":           0.40,
        "scenic_weight_mult": 1.0,
        "surface_softening":  0.5,
        "uphill_threshold":   0.0,
        "uphill_offset":      0.0,
        "softening_threshold": 0.0,
    },
    # Experiment 1 — grade variants on direct base.
    "direct_thresh5": {
        "k_scenic":           0.0,
        "scenic_weight_mult": 0.0,
        "surface_softening":  1.0,
        "uphill_threshold":   5.0,   # no uphill cost below 5% grade
        "uphill_offset":      0.0,
        "softening_threshold": 0.0,
    },
    "direct_minus3": {
        "k_scenic":           0.0,
        "scenic_weight_mult": 0.0,
        "surface_softening":  1.0,
        "uphill_threshold":   0.0,
        "uphill_offset":      3.0,   # treat a 3% climb as flat, shift quadratic
    },
    # Experiment 2 — scenic weight multipliers on direct base (no surface
    # softening, no grade tweaks; only the scenic contribution differs).
    "direct_scenic_2x": {
        "k_scenic":           0.40,
        "scenic_weight_mult": 2.0,
        "surface_softening":  1.0,
        "uphill_threshold":   0.0,
        "uphill_offset":      0.0,
        "softening_threshold": 0.0,
    },
    "direct_scenic_5x": {
        "k_scenic":           0.40,
        "scenic_weight_mult": 5.0,
        "surface_softening":  1.0,
        "uphill_threshold":   0.0,
        "uphill_offset":      0.0,
        "softening_threshold": 0.0,
    },
    "direct_scenic_10x": {
        "k_scenic":           0.40,
        "scenic_weight_mult": 10.0,
        "surface_softening":  1.0,
        "uphill_threshold":   0.0,
        "uphill_offset":      0.0,
        "softening_threshold": 0.0,
    },
    # Experiment 3 — interaction-softening: surface penalties are only
    # softened when the edge's scenic_score exceeds a threshold. The
    # idea is that a dirt road through forest deserves a discount but a
    # dirt road through farmland does not, so we don't pick up ugly
    # shortcuts the way plain `scenic` does. Pairs with a moderate
    # scenic_weight_mult so we still detour toward scenic features.
    "direct_interactive": {
        "k_scenic":            0.30,
        "scenic_weight_mult":  3.0,
        "surface_softening":   0.5,    # the "active" softening level
        "uphill_threshold":    0.0,
        "uphill_offset":       0.0,
        "softening_threshold": 0.5,    # softening kicks in above this score
    },
    # Experiment 3.b — interaction-softening + viewpoint signals.
    # Same shape as direct_interactive but with viewpoint_local and
    # viewpoint_regional contributing to scenic_score (weights live in
    # _SCENIC_W; the columns get populated by scenicness-bake).
    "direct_interactive_vp": {
        "k_scenic":            0.30,
        "scenic_weight_mult":  3.0,
        "surface_softening":   0.5,
        "uphill_threshold":    0.0,
        "uphill_offset":       0.0,
        "softening_threshold": 0.5,
    },
    # Experiment 4 — per-signal multiplicative scenic factor. Each
    # signal contributes its own (1 - k_i · tanh(weighted_i)) factor
    # which combine multiplicatively. Compared to the tanh-of-sum
    # formulation, no single signal saturates the discount, so
    # accumulating multiple scenic features (forest + viewpoint +
    # waterway) compounds the discount instead of being capped. Also
    # restores interaction-softening for surfaces.
    # ----------------------------------------------------------------
    # V2 Phase A.3e — multi-axis scenic profiles. Each profile
    # emphasizes ONE scenic dimension; the user picks which "kind of
    # scenic" they want rather than dialing a single direct↔scenic
    # slider. Common shape: direct base, no surface softening, uphill
    # tolerance to release the hilly-terrain features, multiplicative
    # per-signal scenic. The per_signal_k weights are aggressive on
    # the profile's chosen dimensions and zero elsewhere.
    # ----------------------------------------------------------------
    "vineyard_lover": {
        "k_scenic":            0.0,
        "scenic_weight_mult":  3.0,
        "surface_softening":   1.0,        # keep direct's surface costs
        "uphill_threshold":    0.0,
        "uphill_offset":       3.0,        # vineyards sit on hillsides
        "softening_threshold": 0.0,
        "per_signal_k": {
            "vineyard_local":     0.40,
            "relief":             0.20,
            "regional":           0.10,
            "forest_local":       0.05,
            "waterway":           0.10,
        },
    },
    "forest_lover": {
        "k_scenic":            0.0,
        "scenic_weight_mult":  3.0,
        "surface_softening":   1.0,
        "uphill_threshold":    0.0,
        "uphill_offset":       3.0,
        "softening_threshold": 0.0,
        "per_signal_k": {
            "forest_local":       0.30,
            "vista_forest":       0.30,
            "relief":             0.10,
            "waterway":           0.10,
            "wetland_local":      0.10,
        },
    },
    "views": {
        "k_scenic":            0.0,
        "scenic_weight_mult":  3.0,
        "surface_softening":   1.0,
        "uphill_threshold":    0.0,
        "uphill_offset":       3.0,        # views from elevation
        "softening_threshold": 0.0,
        "per_signal_k": {
            "viewpoint_local":    0.40,
            "viewpoint_regional": 0.20,
            "vista_drama":        0.30,
            "vista_forest":       0.15,
            "vista_water":        0.15,
            "relief":             0.20,
        },
    },
    "water": {
        "k_scenic":            0.0,
        "scenic_weight_mult":  3.0,
        "surface_softening":   1.0,
        "uphill_threshold":    0.0,
        "uphill_offset":       3.0,        # bridges + scenic banks
        "softening_threshold": 0.0,
        "per_signal_k": {
            "waterway":           0.50,    # waterway_along_edge × wctx
            "water_local":        0.30,
            "waterway_local":     0.20,    # 200m blur of stream buffers
            "vista_water":        0.20,
            "forest_local":       0.10,    # streams in forest > streams in cities
            "wetland_local":      0.05,
        },
    },
    # ----------------------------------------------------------------
    "direct_minus3_multi": {
        "k_scenic":            0.0,
        "scenic_weight_mult":  3.0,
        "surface_softening":   0.5,
        "uphill_threshold":    0.0,
        "uphill_offset":       3.0,        # the direct_minus3 hill-acceptance
        "softening_threshold": 0.5,
        "per_signal_k": {
            "forest_local":       0.25,
            "vineyard_local":     0.25,
            "water_local":        0.25,
            "wetland_local":      0.15,
            "waterway":           0.35,
            "relief":             0.25,
            "regional":           0.15,
            "vista_forest":       0.25,
            "vista_water":        0.25,
            "vista_sea":          0.35,
            "vista_drama":        0.25,
            "viewpoint_local":    0.25,
            "viewpoint_regional": 0.15,
            "waterway_local":     0.15,
        },
    },
    "direct_multi": {
        # tanh-of-sum knobs unused in multi-discount math but still
        # consulted for interaction-softening (which uses the legacy
        # _scenic_score). scenic_weight_mult=3.0 matches direct_interactive
        # so the softening threshold actually engages, AND the per-
        # signal tanh inputs land in the productive range instead of
        # stuck at low contribution.
        "k_scenic":            0.0,
        "scenic_weight_mult":  3.0,
        "surface_softening":   0.5,
        "uphill_threshold":    0.0,
        "uphill_offset":       0.0,
        "softening_threshold": 0.5,
        # Per-signal max discount. Sum of all k_i bounds the total max
        # discount (additively when small, multiplicatively when large).
        # With these 13 entries summing to ~1.30, full-saturation factor
        # ≈ ∏ (1 - k_i) ≈ 0.27 (73% max discount). Most edges hit just
        # a few signals so realistic factors land 0.7-0.95.
        "per_signal_k": {
            # v2.2: bumped 2.5x from 0.10/0.05/0.15 baseline. Sum is 3.3
            # so an all-signals-saturated edge can hit ~98% discount, but
            # realistic 3-4-active-signal edges land 30-50% which actually
            # competes with the cycleway base bonus and produces detours.
            "forest_local":       0.25,
            "vineyard_local":     0.25,
            "water_local":        0.25,
            "wetland_local":      0.15,
            "waterway":           0.35,
            "relief":             0.25,
            "regional":           0.15,
            "vista_forest":       0.25,
            "vista_water":        0.25,
            "vista_sea":          0.35,
            "vista_drama":        0.25,
            "viewpoint_local":    0.25,
            "viewpoint_regional": 0.15,
            "waterway_local":     0.15,   # the previous 0.3-secondary term
        },
    },
}
# Midpoint shift: scenic_factor = 1 - k * tanh(score - midpoint).
# At score = midpoint, factor = 1.0 (neutral edge). Below midpoint the
# factor exceeds 1.0 (penalty for low-scenic edges); above midpoint
# the factor drops below 1.0 (discount for scenic edges). Without this
# shift the cost function could only DISCOUNT scenic edges, never
# penalize their alternatives — and at Austrian latitudes the
# direct-optimal route is already scenic enough that no alternative could
# undercut it. Setting midpoint=0.5 puts roughly the corridor's
# typical scenic_score at neutral so detours toward better-than-
# average edges win meaningfully.
_SCENIC_MIDPOINT = 0.5

def _soften(factor: float, softening: float) -> float:
    """Compress how far a penalty multiplier sits above 1.0. A factor
    of 1.5 with softening 0.5 becomes 1.25. Factors <= 1.0 (bonuses
    or neutral) pass through unchanged so cycleway/Fahrradstrasse
    discounts are not affected."""
    if factor <= 1.0:
        return factor
    return 1.0 + (factor - 1.0) * softening
# Normalization constants (meters) for the unbounded signals.
_RELIEF_NORM       = 120.0     # local_relief saturates ~120 m σ
_REGIONAL_NORM     = 250.0     # regional_relief saturates ~250 m σ
_VIEW_NORM         = 200.0     # view_dominance ±200 m saturates
_DRAMA_SCALE_M     = 5000.0    # distance_to_drama: exp falloff over 5 km


def _per_signal_values(*, forest_local, forest_wide, vineyard_local,
                       water_local, water_wide, sea_local, sea_wide,
                       waterway_along_edge, waterway_local, wetland_local,
                       view_dominance, local_relief, regional_relief,
                       distance_to_drama,
                       viewpoint_local=0.0, viewpoint_regional=0.0,
                       ) -> dict[str, float]:
    """Build the dict of per-signal (already-normalized, composite-
    multiplied) values that gets fed into either the tanh-of-sum
    formulation OR the multiplicative per-signal formulation. Keys
    match _SCENIC_W / per_signal_k entries."""
    import math
    view     = max(view_dominance, 0.0) / _VIEW_NORM
    relief_n = max(0.0, min(1.0, local_relief / _RELIEF_NORM))
    region_n = max(0.0, min(1.0, regional_relief / _REGIONAL_NORM))
    drama_w  = math.exp(-distance_to_drama / _DRAMA_SCALE_M)
    wctx     = 0.5 + 0.5 * relief_n + 0.5 * forest_local
    return {
        "forest_local":       forest_local,
        "vineyard_local":     vineyard_local,
        "water_local":        water_local,
        "wetland_local":      wetland_local,
        "waterway":           waterway_along_edge * wctx,
        "relief":             relief_n,
        "regional":           region_n,
        "vista_forest":       view * forest_wide,
        "vista_water":        view * water_wide,
        "vista_sea":          view * sea_wide,
        "vista_drama":        view * drama_w,
        "viewpoint_local":    viewpoint_local,
        "viewpoint_regional": viewpoint_regional,
        "waterway_local":     waterway_local,
    }


def _scenic_score(prof: dict, *,
                  forest_local: float, forest_wide: float,
                  vineyard_local: float,
                  water_local: float, water_wide: float,
                  sea_local: float, sea_wide: float,
                  waterway_along_edge: float, waterway_local: float,
                  wetland_local: float,
                  view_dominance: float,
                  local_relief: float, regional_relief: float,
                  distance_to_drama: float,
                  viewpoint_local: float = 0.0,
                  viewpoint_regional: float = 0.0,
                  ) -> float:
    """Weighted scenic score (post-multiplier, pre-tanh). Returns >= 0.

    Used both by `_scenic_factor` (for the cost discount) and by the
    interaction-softening ramp (so surface penalties relax only on
    scenic edges). Caller passes the resolved _PROFILES entry to avoid
    a second dict lookup."""
    mult = prof["scenic_weight_mult"]
    if mult <= 0.0:
        return 0.0
    import math
    # Normalize unbounded signals to roughly [0, 1].
    view     = max(view_dominance, 0.0) / _VIEW_NORM
    relief_n = max(0.0, min(1.0, local_relief / _RELIEF_NORM))
    region_n = max(0.0, min(1.0, regional_relief / _REGIONAL_NORM))
    drama_w  = math.exp(-distance_to_drama / _DRAMA_SCALE_M)
    # Stream context: water along the edge counts more in interesting
    # local terrain (gorge, forest) than in flat farmland.
    wctx = 0.5 + 0.5 * relief_n + 0.5 * forest_local
    W = _SCENIC_W
    score = (
        W["forest_local"]      * forest_local
      + W["vineyard_local"]    * vineyard_local
      + W["water_local"]       * water_local
      + W["wetland_local"]     * wetland_local
      + W["waterway"]          * waterway_along_edge * wctx
      + W["relief"]            * relief_n
      + W["regional"]          * region_n
      + W["vista_forest"]      * view * forest_wide
      + W["vista_water"]       * view * water_wide
      + W["vista_sea"]         * view * sea_wide
      + W["vista_drama"]       * view * drama_w
      + W["viewpoint_local"]    * viewpoint_local
      + W["viewpoint_regional"] * viewpoint_regional
    )
    # Light secondary credit for "near a stream" even if not on it,
    # without inflating the main waterway term.
    score += 0.3 * waterway_local
    return score * mult


def _scenic_factor(prof: dict, score: float) -> float:
    """Cost factor for the tanh-of-sum profiles. Returns 1.0 when
    k_scenic <= 0; otherwise a number in (1 - k, 1 + k) — penalty below
    midpoint, discount above."""
    k = prof["k_scenic"]
    if k <= 0.0:
        return 1.0
    import math
    return 1.0 - k * math.tanh(score - _SCENIC_MIDPOINT)


def _scenic_factor_multi(prof: dict, signal_values: dict) -> float:
    """Multiplicative per-signal scenic factor (Experiment 4).
    factor = ∏ (1 - k_i · tanh(W_i · value_i · mult)) for each signal
    with per_signal_k[name] > 0. Each tanh is bounded, so no single
    signal can saturate the whole discount — multiple scenic features
    compound instead of capping out. Compared to the tanh-of-sum
    formulation, viewpoints + forest + waterway all contribute their
    own discount instead of getting drowned by tanh saturation.
    Returns 1.0 if per_signal_k is missing or empty."""
    per_k = prof.get("per_signal_k") or {}
    if not per_k:
        return 1.0
    import math
    mult = prof["scenic_weight_mult"]
    W = _SCENIC_W
    factor = 1.0
    for sig_name, value in signal_values.items():
        k_i = per_k.get(sig_name, 0.0)
        if k_i <= 0.0:
            continue
        w_i = W.get(sig_name, 1.0)
        # waterway_local uses the "secondary" 0.3 weight in the sum
        # formulation; preserve that magnitude in multi mode by treating
        # 0.3 as its effective weight.
        if sig_name == "waterway_local":
            w_i = 0.3
        weighted = w_i * value * mult
        factor *= 1.0 - k_i * math.tanh(weighted)
    return factor


def _interaction_softening(prof: dict, score: float) -> float:
    """Resolve the actual surface_softening for this edge under
    interaction-softening profiles. softening_threshold > 0 means
    softening ramps in linearly from `1.0` (no softening) at score==
    threshold up to `surface_softening` at score==2·threshold (saturates
    beyond). Profiles with threshold==0 just use `surface_softening`
    flat for all edges (the old behavior)."""
    threshold = prof.get("softening_threshold", 0.0)
    base      = prof["surface_softening"]
    if threshold <= 0.0:
        return base
    # Linear ramp from 0..1 between score==threshold and score==2*threshold
    curve = max(0.0, min(1.0, (score - threshold) / threshold))
    return 1.0 - (1.0 - base) * curve


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
    curv: float = 0.0,          # total bend angle (degrees) in the *forward*
                                # polyline window for the direction being priced
    canopy_frac: float = 0.0,   # ∈ [0,1] fraction of edge under forest canopy
    profile: str = "direct",    # 'direct' | 'balanced' | 'scenic' | …
    # Scenic signals — only consulted when profile != 'direct'. All
    # default to 0.0 (= "we don't have this signal" = no contribution).
    forest_local: float = 0.0,
    forest_wide: float = 0.0,
    vineyard_local: float = 0.0,
    water_local: float = 0.0,
    water_wide: float = 0.0,
    sea_local: float = 0.0,
    sea_wide: float = 0.0,
    waterway_along_edge: float = 0.0,
    waterway_local: float = 0.0,
    wetland_local: float = 0.0,
    view_dominance: float = 0.0,
    local_relief: float = 0.0,
    regional_relief: float = 0.0,
    distance_to_drama: float = 0.0,
    viewpoint_local: float = 0.0,
    viewpoint_regional: float = 0.0,
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

    # Compute per-signal values + the legacy summed scenic_score once
    # up front. The score drives the interaction-softening ramp for
    # surface penalties; the dict feeds either the tanh-of-sum factor
    # or the per-signal multiplicative factor (whichever the profile
    # config selects).
    _prof_cfg = _PROFILES.get(profile, _PROFILES["direct"])
    _signal_vals = _per_signal_values(
        forest_local=forest_local, forest_wide=forest_wide,
        vineyard_local=vineyard_local,
        water_local=water_local, water_wide=water_wide,
        sea_local=sea_local, sea_wide=sea_wide,
        waterway_along_edge=waterway_along_edge,
        waterway_local=waterway_local,
        wetland_local=wetland_local,
        view_dominance=view_dominance,
        local_relief=local_relief, regional_relief=regional_relief,
        distance_to_drama=distance_to_drama,
        viewpoint_local=viewpoint_local,
        viewpoint_regional=viewpoint_regional,
    )
    _scenic_s = _scenic_score(
        _prof_cfg,
        forest_local=forest_local, forest_wide=forest_wide,
        vineyard_local=vineyard_local,
        water_local=water_local, water_wide=water_wide,
        sea_local=sea_local, sea_wide=sea_wide,
        waterway_along_edge=waterway_along_edge,
        waterway_local=waterway_local,
        wetland_local=wetland_local,
        view_dominance=view_dominance,
        local_relief=local_relief, regional_relief=regional_relief,
        distance_to_drama=distance_to_drama,
        viewpoint_local=viewpoint_local,
        viewpoint_regional=viewpoint_regional,
    )

    # Tracktype is only meaningful on track-like highways. Under
    # scenic/balanced profiles the deviation from 1.0 is softened so
    # forest tracks become more attractive. Interaction-softening
    # profiles (softening_threshold > 0) only soften where the scenic
    # score is high — so a dirt road through forest gets cheaper but a
    # dirt road through farmland stays expensive.
    softening = _interaction_softening(_prof_cfg, _scenic_s)
    if tracktype in _TRACKTYPE and highway in ("track", "path", "footway", "road"):
        cost *= _soften(_TRACKTYPE[tracktype], softening)

    # V2: graded surface multiplier (was a blacklist exclude in V1).
    # Untagged or unknown surfaces fall through with no change. Same
    # per-profile softening as tracktype — sand/mud/snow stay punitive
    # because they're genuinely unrideable, but dirt/gravel get a
    # smaller penalty for scenic-leaning profiles.
    if surface in _SURFACE:
        cost *= _soften(_SURFACE[surface], softening)

    # V2: Fahrradstraße / bicycle-priority road bonus.
    if bicycle_road == "yes":
        cost *= _BICYCLE_ROAD_BONUS

    # V2 Phase A: grade + curvy-descent multipliers. No-op when
    # grade_pct == 0.0 and curv == 0.0 (the defaults), so callers
    # that haven't been updated to provide elevation/curvature data
    # still work.
    #
    # Per-profile knobs (Experiment 1):
    #   uphill_threshold > 0  → no uphill penalty below that grade.
    #                           Discontinuous step to the normal formula
    #                           at the threshold.
    #   uphill_offset   > 0   → shift the quadratic: a 3% climb feels
    #                           flat, an 8% climb feels like 5%, etc.
    #                           Continuous, but the very-steep regime
    #                           still hurts.
    if grade_pct > 0.0:
        threshold = _prof_cfg["uphill_threshold"]
        offset    = _prof_cfg["uphill_offset"]
        if threshold > 0.0 and grade_pct <= threshold:
            pass   # no uphill penalty in the tolerated band
        else:
            g_eff = max(0.0, grade_pct - offset)
            if g_eff > 0:
                cost *= 1.0 + _UPHILL_COEFF * g_eff * g_eff
    elif grade_pct < 0.0:
        g = -grade_pct
        cost *= 1.0 - _DOWNHILL_LINEAR * g + _DOWNHILL_QUADRATIC * g * g
        if curv > 0.0:
            cost *= 1.0 + _CURV_COEFF * curv * g

    # V2 Phase A.3: universal tree-cover-overhead bonus. Edge fraction
    # inside a forest polygon gets a proportional discount, max 10%.
    if canopy_frac > 0.0:
        cost *= 1.0 - _CANOPY_BONUS * canopy_frac

    # V2 Phase A.3b: scenic discount/penalty multiplier. Dispatch:
    # profiles with `per_signal_k` use the multiplicative per-signal
    # factor (Experiment 4); others use the legacy tanh-of-sum factor.
    if _prof_cfg.get("per_signal_k"):
        cost *= _scenic_factor_multi(_prof_cfg, _signal_vals)
    else:
        cost *= _scenic_factor(_prof_cfg, _scenic_s)

    return float(cost)
