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
    kind = 'dem':       stitch Copernicus DEM tiles (landcover_class ignored)
    """
    kind: str
    landcover_class: str | None = None


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
}


def all_columns() -> list[str]:
    """Column names produced by every registered signal."""
    return [s.column for s in SIGNALS.values()]
