"""Per-signal config registry.

Each entry describes a scenicness signal: where its source raster
comes from, what kernel to apply, how to sample per edge, and which
`ways` column to write. The bake orchestrator (`bake.py`) walks this
registry, dedupes shared rasters across signals, and produces one
column per signal in a single edge-streaming pass.

Adding a signal is one entry here plus a schema-level
`ALTER TABLE ways ADD COLUMN IF NOT EXISTS <column> real NOT NULL DEFAULT 0.0;`
in schema.sql.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """How to build the base raster.

    kind = 'polygons':  rasterize landcover.geom WHERE class = landcover_class
    kind = 'dem':       stitch Copernicus DEM tiles
    kind = 'points':    rasterize POI points from data/pois/pois.sqlite where
                        category = poi_category (one pixel per POI)
    """
    kind: str
    landcover_class: str | None = None
    poi_category:    str | None = None


@dataclass(frozen=True)
class Kernel:
    """Post-processing kernel applied to the source raster.

    kind = 'uniform_blur':              param_m = radius in meters
    kind = 'subtract_gaussian_blur':    param_m = sigma in meters
    kind = 'stddev_filter':             param_m = window radius in meters
    kind = 'distance_to_high_relief':   param_m = stddev threshold (m of
                                        terrain variation that counts as
                                        "dramatic"; cells above that are
                                        the distance-transform targets)
    kind = None:                        pass through unchanged
    """
    kind: str | None
    param_m: float | None = None


@dataclass(frozen=True)
class Signal:
    """One scenicness signal.

    sample_mode = 'at_midpoint':  sample the raster once at the edge's
                                  geometric midpoint. Cheap; right for
                                  signals that describe a *region*
                                  (forest density, view dominance,
                                  urban-ness).
    sample_mode = 'along_edge':   sample N points along the edge line
                                  (N = clamp(length/sample_spacing_m,
                                  3, 50)), return mean. Right for
                                  signals where the rider's experience
                                  *along the route* is the point
                                  (canopy_frac).
    """
    name: str
    column: str
    source: Source
    kernel: Kernel | None
    sample_mode: str = "at_midpoint"
    sample_spacing_m: float = 10.0   # only used when sample_mode == 'along_edge'
    description: str = ""


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

SIGNALS: dict[str, Signal] = {
    # Local "forest density right here." 200m uniform blur of the
    # forest mask, sampled at edge midpoint. Captures roads that are
    # *immediately* surrounded by forest even if the edge centerline
    # is right on a boundary.
    "forest_local": Signal(
        name="forest_local",
        column="forest_local",
        source=Source(kind="polygons", landcover_class="forest"),
        kernel=Kernel(kind="uniform_blur", param_m=200.0),
        sample_mode="at_midpoint",
        description="Fraction of a 200 m disc around the edge that's forest.",
    ),
    # Wide "forested landscape." 2 km blur. Pair with view_dominance
    # in cost.py for the vista mode: forest_wide * view_dominance
    # = "you can see forests because you're elevated."
    "forest_wide": Signal(
        name="forest_wide",
        column="forest_wide",
        source=Source(kind="polygons", landcover_class="forest"),
        kernel=Kernel(kind="uniform_blur", param_m=2000.0),
        sample_mode="at_midpoint",
        description="Fraction of a 2 km disc around the edge that's forest.",
    ),
    # Local terrain ruggedness: standard deviation of DEM in a 500 m
    # window. High wherever the surrounding ground is steep —
    # mountainsides, ravines, gorge walls. ~0 on plains, even at
    # high absolute elevation. The natural companion to
    # `view_dominance`: together they distinguish "ridge with a
    # view" (high view_dominance + medium relief) from "dramatic
    # gorge floor" (negative view_dominance + high relief) — the
    # gorge case `view_dominance` alone gets wrong.
    "local_relief": Signal(
        name="local_relief",
        column="local_relief",
        source=Source(kind="dem"),
        kernel=Kernel(kind="stddev_filter", param_m=500.0),
        sample_mode="at_midpoint",
        description="Std-dev of elevation in 500 m window (meters of "
                    "terrain variation — high in gorges and on mountainsides).",
    ),
    # Wider-scale terrain ruggedness. 10 km radius stddev — captures
    # "you're in mountainous country" vs "you're on plains." Useful
    # context for the Alps-on-the-horizon case where local_relief
    # and view_dominance are both ~0 on the plain itself but the
    # surrounding region is dramatic.
    "regional_relief": Signal(
        name="regional_relief",
        column="regional_relief",
        source=Source(kind="dem"),
        kernel=Kernel(kind="stddev_filter", param_m=10000.0),
        sample_mode="at_midpoint",
        description="Std-dev of elevation in 10 km window — regional "
                    "mountainous-ness, even where local terrain is flat.",
    ),
    # Distance (in meters) to the nearest cell where local_relief
    # exceeds an 80 m threshold. "Drama proximity": small values mean
    # mountains are right here; values up to ~15 km still register
    # "you can probably see them." For the Salzburg-with-the-Alps-
    # on-the-horizon case where neither local_relief nor
    # view_dominance light up but there's drama on the horizon.
    "distance_to_drama": Signal(
        name="distance_to_drama",
        column="distance_to_drama",
        source=Source(kind="dem"),
        kernel=Kernel(kind="distance_to_high_relief", param_m=80.0),
        sample_mode="at_midpoint",
        description="Meters to nearest cell with local_relief ≥ 80 m. "
                    "Small = visible mountains nearby. ~0 inside dramatic "
                    "terrain.",
    ),
    # Elevation prominence: DEM minus 2 km Gaussian blur of DEM.
    # Positive on ridges/summits, negative in valleys. The natural
    # weight for "what can this rider see?" — multiply against any
    # _wide signal in the scenic profile to credit elevated points
    # for distant features.
    "view_dominance": Signal(
        name="view_dominance",
        column="view_dominance",
        source=Source(kind="dem"),
        kernel=Kernel(kind="subtract_gaussian_blur", param_m=2000.0),
        sample_mode="at_midpoint",
        description="DEM elevation minus 2 km Gaussian blur — meters above local mean.",
    ),
    # --- Water / wetland family (V2 Phase A.3 second batch) ----------
    # Three landcover classes feeding these signals: 'water' (lakes,
    # ponds, wide-river polygons), 'sea' (pre-built coastline polygons),
    # 'waterway' (buffered river/canal/stream centerlines), 'wetland'
    # (`natural=wetland` polygons). Sea is split from water because
    # ocean/sea vistas materially exceed lake vistas; waterway is split
    # because riding *along* a flowing river ("Mur cycle path") is a
    # distinct experience from "lake nearby".
    "water_local": Signal(
        name="water_local",
        column="water_local",
        source=Source(kind="polygons", landcover_class="water"),
        kernel=Kernel(kind="uniform_blur", param_m=200.0),
        sample_mode="at_midpoint",
        description="Fraction of a 200 m disc around the edge that's "
                    "lake / pond / wide-river polygon.",
    ),
    "water_wide": Signal(
        name="water_wide",
        column="water_wide",
        source=Source(kind="polygons", landcover_class="water"),
        kernel=Kernel(kind="uniform_blur", param_m=2000.0),
        sample_mode="at_midpoint",
        description="Fraction of a 2 km disc around the edge that's "
                    "lake / pond / wide-river. Water-rich landscape.",
    ),
    "sea_local": Signal(
        name="sea_local",
        column="sea_local",
        source=Source(kind="polygons", landcover_class="sea"),
        kernel=Kernel(kind="uniform_blur", param_m=200.0),
        sample_mode="at_midpoint",
        description="Fraction of a 200 m disc around the edge that's "
                    "ocean / sea (pre-built coastline polygons).",
    ),
    "sea_wide": Signal(
        name="sea_wide",
        column="sea_wide",
        source=Source(kind="polygons", landcover_class="sea"),
        kernel=Kernel(kind="uniform_blur", param_m=2000.0),
        sample_mode="at_midpoint",
        description="Fraction of a 2 km disc around the edge that's "
                    "ocean / sea — coastal proximity.",
    ),
    # Sample the *raw* (un-blurred) buffered-waterway mask along the
    # edge centerline. Returns the fraction of N points along this
    # edge that fall inside a buffered river/canal/stream polygon —
    # i.e. how much of the edge literally runs along a waterway. This
    # is the "Mur cycle path" signal: it lights up when the rider is
    # *on* the river path, not just near it.
    "waterway_along_edge": Signal(
        name="waterway_along_edge",
        column="waterway_along_edge",
        source=Source(kind="polygons", landcover_class="waterway"),
        kernel=None,  # use the raw binary mask
        sample_mode="along_edge",
        description="Fraction of the edge centerline that runs inside "
                    "a 10 m-buffered river/canal/stream polygon.",
    ),
    # Forgiving "stream visible from this road" — 200 m blur of the
    # waterway mask, sampled at midpoint. Lights up for roads that
    # run *near* (but not on) a stream too. Pair with along_edge in
    # cost to distinguish "alongside the river" (along_edge ~ 1) from
    # "across the valley from a creek" (local high, along_edge low).
    "waterway_local": Signal(
        name="waterway_local",
        column="waterway_local",
        source=Source(kind="polygons", landcover_class="waterway"),
        kernel=Kernel(kind="uniform_blur", param_m=200.0),
        sample_mode="at_midpoint",
        description="Fraction of a 200 m disc that contains buffered "
                    "river/canal/stream polygons — waterway proximity.",
    ),
    "wetland_local": Signal(
        name="wetland_local",
        column="wetland_local",
        source=Source(kind="polygons", landcover_class="wetland"),
        kernel=Kernel(kind="uniform_blur", param_m=200.0),
        sample_mode="at_midpoint",
        description="Fraction of a 200 m disc that's wetland (marsh, "
                    "reedbed, bog, saltmarsh, etc.).",
    ),
    # V2 Phase A.3c: vineyards. Cultivated wine country — pleasant and
    # typically hilly. We don't blanket-positive-score farmland because
    # it's too generic; vineyards are singled out.
    "vineyard_local": Signal(
        name="vineyard_local",
        column="vineyard_local",
        source=Source(kind="polygons", landcover_class="vineyard"),
        kernel=Kernel(kind="uniform_blur", param_m=200.0),
        sample_mode="at_midpoint",
        description="Fraction of a 200 m disc that's vineyard (landuse=vineyard).",
    ),
    # V2 Phase A.3d: viewpoint POIs. EDA showed 77% of Austrian
    # corridor viewpoints sit within 10 m of a way (they're typically
    # tagged at road pullouts or trail lookouts), so a tight local
    # radius captures "you're at the viewpoint". The 2 km regional
    # version captures viewpoint-rich territory (Wienerwald, hills)
    # even when the immediate edge has no viewpoint.
    "viewpoint_local": Signal(
        name="viewpoint_local",
        column="viewpoint_local",
        source=Source(kind="points", poi_category="viewpoint"),
        kernel=Kernel(kind="uniform_blur", param_m=100.0),
        sample_mode="at_midpoint",
        description="Viewpoint POI density in 100 m disc (binary mask × uniform blur).",
    ),
    "viewpoint_regional": Signal(
        name="viewpoint_regional",
        column="viewpoint_regional",
        source=Source(kind="points", poi_category="viewpoint"),
        kernel=Kernel(kind="uniform_blur", param_m=2000.0),
        sample_mode="at_midpoint",
        description="Viewpoint POI density in 2 km disc — viewpoint-rich territory.",
    ),
}


def all_columns() -> list[str]:
    """Column names produced by every registered signal."""
    return [s.column for s in SIGNALS.values()]
