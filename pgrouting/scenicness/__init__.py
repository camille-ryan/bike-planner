"""Scenicness signal-rasters framework.

Each scenicness signal is computed by:
  1. Building a source raster (from landcover polygons, POIs, or DEM)
  2. Optionally applying a kernel (uniform/Gaussian blur, distance transform)
  3. Sampling the result per bike edge (along-edge, at-midpoint, or
     vista-weighted by view_dominance)
  4. COPYing the per-edge value into a column on `ways`

Adding a signal is a few-line registration in `signals.py`; the rest
of the pipeline (rasters.py + bake.py) is generic.

Universal vs. scenic-only: any signal can be wired into `cost.py`
either as a universal multiplier (applies to every profile) or as
part of `scenic_score` (only applies on the scenic profile). The
framework only writes columns; cost.py decides how to combine them.
"""
