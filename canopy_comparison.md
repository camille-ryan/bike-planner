# Canopy compute: polygon vs raster — overnight comparison

Run on `bike_v2_test`, Graz↔Wien bbox `14.5,46.8,17.0,48.5`, ~8.78 M
corridor edges, 72,842 forest polygons (`landuse=forest | natural=wood`).

## Timing (wall-clock, end-to-end)

| Stage                                  | Polygon path                       | Raster path     |
|----------------------------------------|------------------------------------|-----------------|
| Forest data loaded                     | (already in PG)                    | 1.6 s           |
| Pre-pass (build temp / rasterize)      | 80 s corridor + 10 s simplify      | 9.3 s rasterize |
| Spatial work (sample / intersect)      | 4 workers × ~50 min ≈ 50 min wall  | 6.6 min stream + sample |
| Apply to `ways.canopy_frac`            | within UPDATE                      | 7.3 min UPDATE  |
| **Total**                              | **~62 min**                        | **~15 min**     |

**Speedup: ~4×.** And crucially, the raster pipeline scales 1:1 to
future signals — `nearby_forest_frac`, water proximity, low-traffic
feel, etc. each just add another ~7 min sampling pass against a
convolved version of the same forest raster. No new polygon-style
ST_Intersection bake per signal.

## Coverage of `canopy_frac > 0` (per-edge agreement)

| Metric                               | Count             |
|--------------------------------------|-------------------|
| Total corridor edges                 | 8,777,165         |
| Raster nonzero                       | 3,902,119 (44.5 %)|
| Polygon nonzero                      | 3,583,395 (40.8 %)|
| Both agree there's canopy            | 3,514,706         |
| Raster sees canopy, polygon doesn't  | 387,413 (4.4 %)   |
| Polygon sees canopy, raster doesn't  | 68,689 (0.8 %)    |

Asymmetric: raster is **more permissive** than polygon — it picks up
~6× more "lone hits" than polygon does (387 K vs 69 K). Likely cause:
20 m raster pixels include some land in a forest pixel that the
(5 m-simplified) polygon doesn't strictly contain the line inside.
Tighter resolution (10 m) would close some of this at 4× memory cost.

## Per-edge difference `|raster − polygon|`

| Stat         | Value      |
|--------------|------------|
| Mean         | 0.0398     |
| Median       | 0.0000     |
| p90          | 0.0000     |
| p99          | 1.0000     |
| Max          | 1.0000     |
| Correlation  | **0.945**  |

**Median and p90 are both 0** — for 90 %+ of edges the two approaches
return literally the same value. The mean is dragged up by the ~3 %
of edges that disagree heavily.

## Disagreement buckets

| `|raster − polygon|`  | Edges     | Share  |
|-----------------------|-----------|--------|
| < 0.05 (close enough) | 8,040,500 | 91.6 % |
| 0.05–0.20 (small)     | 117,598   |  1.3 % |
| 0.20–0.50 (noticeable)| 353,629   |  4.0 % |
| > 0.50 (major)        | 265,438   |  3.0 % |

91.6 % of edges are "essentially identical," and the 3 % with major
disagreement is the expected failure mode: edges in fragmented forest
patches at the ~20 m raster resolution where one method classifies
the boundary one way and the other classifies it the other way.

## Verdict

- **For canopy alone:** polygon is the more literally-correct answer
  (no quantization, true length-fraction). But raster is **4× cheaper**
  for ~94 % correlation, ~92 % near-identical edges, and a route-level
  cost effect that will be almost indistinguishable.
- **For the multi-signal expansion (`nearby_forest_frac`, water,
  low-traffic, etc.):** raster wins decisively. Each new signal is a
  cheap convolution + the same ~7 min sampling pass, vs. each
  polygon-based signal needing its own multi-hour bake. Raster is the
  right substrate for the scenicness expansion in
  `feature_profiles_and_scenicness`.
- **Validated:** the polygon-based `with_canopy` Graz→Wien route is
  in `web/public/data/graz_wien_compare.geojson` (load the toggle in
  the web app to see it).

## Run notes / lessons for the framework

- **Memory:** holding all 8.78 M edge tuples in Python crashed WSL.
  Fix: stream via server-side cursor; memory bounded to one fetch
  chunk + the 88 MB raster.
- **Postgres parallel hash join + /dev/shm:** any heavy query that
  joins `ways × ways_vertices_pgr` twice has to set
  `max_parallel_workers_per_gather = 0` — the container's 1 GB
  `/dev/shm` can't fit the parallel hash table.
- **Server-cursor lifetime:** never `conn.commit()` inside the
  iteration over a server-side cursor; the cursor dies with
  `InvalidCursorName`. COPY into a temp table within the same
  transaction is fine.
- **Bloat from snapshot UPDATEs:** snapshotting
  `canopy_frac → canopy_frac_polygon` rewrote all 22 M rows; the
  next pgr_dijkstra became ~20× slower until `VACUUM ANALYZE ways`
  cleared the bloat. Bake a vacuum step into the chain after large
  UPDATEs.
- **Watch.sh contention:** the original watch query I wrote had a
  `WHERE source IN (SELECT id FROM ways_vertices_pgr ...)` join that
  acquired row-level locks; running it every 5 s deadlocked the
  canopy UPDATE. The current `watch.sh` uses a cheap single
  seq-scan instead — keep it that way.

## Files

- `pgrouting/compute_canopy_frac.py` — polygon path
  (corridor temp + ST_SimplifyPreserveTopology + 4-worker MP)
- `pgrouting/compute_canopy_frac_raster.py` — raster path
  (rasterize forests + server-cursor stream + per-edge sample)
- `pgrouting/compare_canopy.py` — produced the diff numbers above
- DB columns:
  - `ways.canopy_frac` — currently the **raster** result (last write wins)
  - `ways.canopy_frac_polygon` — snapshot of the polygon result
- Web: `web/public/data/graz_wien_compare.geojson` has `no_canopy`
  and `with_canopy` (computed under polygon canopy, before the
  raster overwrite).
