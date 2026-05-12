"""Compare the polygon-based and raster-based canopy_frac results
that have been preserved into separate columns on `ways`.

Assumes both columns exist:
  - ways.canopy_frac_polygon   (snapshot of polygon-based result)
  - ways.canopy_frac           (current raster-based result)

Run after:
  1. polygon canopy-compute completes
  2. SQL snapshot: UPDATE ways SET canopy_frac_polygon = canopy_frac;
  3. raster canopy-compute-raster runs (overwriting canopy_frac).

Writes a markdown summary to /data/canopy_comparison.md (and stdout).
"""
import argparse
import time
from pathlib import Path

import psycopg

import config


def compare(conn: psycopg.Connection,
            bbox: tuple[float, float, float, float] | None = None,
            out_path: Path | None = None) -> str:
    bbox_clause = ""
    bbox_params: tuple = ()
    if bbox:
        bbox_clause = (
            " AND w.source IN (SELECT id FROM ways_vertices_pgr "
            "  WHERE lon BETWEEN %s AND %s AND lat BETWEEN %s AND %s) "
        )
        bbox_params = (bbox[0], bbox[2], bbox[1], bbox[3])

    print("[compare] computing summary stats...", flush=True)
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        # Sanity: ensure both columns exist.
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'ways' AND "
            "      column_name IN ('canopy_frac', 'canopy_frac_polygon')"
        )
        cols = {r[0] for r in cur.fetchall()}
        if "canopy_frac_polygon" not in cols:
            raise SystemExit(
                "[compare] ways.canopy_frac_polygon missing — run the polygon "
                "bake + snapshot first (see module docstring)"
            )

        # Filter to corridor edges (both endpoints in bbox) — only
        # those received any compute attention.
        base = (
            "FROM ways w "
            "JOIN ways_vertices_pgr vs ON vs.id = w.source "
            "JOIN ways_vertices_pgr vt ON vt.id = w.target "
            "WHERE 1=1 "
            + bbox_clause
        )

        # 1) Counts in each column.
        cur.execute(
            "SELECT COUNT(*), "
            "  COUNT(*) FILTER (WHERE w.canopy_frac > 0), "
            "  COUNT(*) FILTER (WHERE w.canopy_frac_polygon > 0), "
            "  COUNT(*) FILTER (WHERE w.canopy_frac > 0 AND w.canopy_frac_polygon > 0), "
            "  COUNT(*) FILTER (WHERE w.canopy_frac > 0 AND w.canopy_frac_polygon = 0), "
            "  COUNT(*) FILTER (WHERE w.canopy_frac = 0 AND w.canopy_frac_polygon > 0) "
            + base,
            bbox_params,
        )
        (n_total, n_raster, n_poly, n_both,
         n_raster_only, n_poly_only) = cur.fetchone()

        # 2) Diff stats (abs diff per edge among edges in corridor).
        cur.execute(
            "SELECT "
            "  AVG(ABS(w.canopy_frac - w.canopy_frac_polygon))::numeric(10,6), "
            "  STDDEV(w.canopy_frac - w.canopy_frac_polygon)::numeric(10,6), "
            "  MAX(ABS(w.canopy_frac - w.canopy_frac_polygon))::numeric(10,6), "
            "  CORR(w.canopy_frac, w.canopy_frac_polygon)::numeric(10,6) "
            + base,
            bbox_params,
        )
        (mean_abs, sd_diff, max_abs, corr) = cur.fetchone()

        # 3) Diff percentiles.
        cur.execute(
            "SELECT "
            "  PERCENTILE_CONT(0.50) WITHIN GROUP "
            "    (ORDER BY ABS(w.canopy_frac - w.canopy_frac_polygon))::numeric(10,6), "
            "  PERCENTILE_CONT(0.90) WITHIN GROUP "
            "    (ORDER BY ABS(w.canopy_frac - w.canopy_frac_polygon))::numeric(10,6), "
            "  PERCENTILE_CONT(0.99) WITHIN GROUP "
            "    (ORDER BY ABS(w.canopy_frac - w.canopy_frac_polygon))::numeric(10,6) "
            + base,
            bbox_params,
        )
        (p50, p90, p99) = cur.fetchone()

        # 4) Disagreement buckets (where do the two disagree the most).
        cur.execute(
            "SELECT "
            "  COUNT(*) FILTER (WHERE ABS(w.canopy_frac - w.canopy_frac_polygon) < 0.05) AS w005, "
            "  COUNT(*) FILTER (WHERE ABS(w.canopy_frac - w.canopy_frac_polygon) BETWEEN 0.05 AND 0.20) AS w020, "
            "  COUNT(*) FILTER (WHERE ABS(w.canopy_frac - w.canopy_frac_polygon) BETWEEN 0.20 AND 0.50) AS w050, "
            "  COUNT(*) FILTER (WHERE ABS(w.canopy_frac - w.canopy_frac_polygon) > 0.50) AS w_big "
            + base,
            bbox_params,
        )
        (w005, w020, w050, w_big) = cur.fetchone()
    print(f"[compare]   stats computed in {time.time()-t0:.1f}s", flush=True)

    lines = []
    p = lines.append
    p("# Canopy-frac comparison: polygon vs raster")
    p("")
    p(f"Bbox: `{bbox}`" if bbox else "Bbox: (full graph)")
    p("")
    p("## Coverage")
    p("")
    p(f"- Corridor edges considered: **{n_total:,}**")
    p(f"- Edges with canopy > 0 (raster): **{n_raster:,}** "
      f"({100.0*n_raster/max(n_total,1):.1f}%)")
    p(f"- Edges with canopy > 0 (polygon): **{n_poly:,}** "
      f"({100.0*n_poly/max(n_total,1):.1f}%)")
    p(f"- Both agree there's canopy: **{n_both:,}**")
    p(f"- Raster sees canopy, polygon doesn't: **{n_raster_only:,}**")
    p(f"- Polygon sees canopy, raster doesn't: **{n_poly_only:,}**")
    p("")
    p("## Per-edge difference (|raster − polygon|)")
    p("")
    p(f"- Mean: **{mean_abs}**")
    p(f"- Stddev of signed diff: **{sd_diff}**")
    p(f"- Median: **{p50}**")
    p(f"- p90: **{p90}**")
    p(f"- p99: **{p99}**")
    p(f"- Max: **{max_abs}**")
    p(f"- Correlation: **{corr}**")
    p("")
    p("## Disagreement buckets (count of corridor edges)")
    p("")
    p(f"- diff < 0.05  (close enough): **{w005:,}** "
      f"({100.0*w005/max(n_total,1):.1f}%)")
    p(f"- 0.05–0.20 (small): **{w020:,}**")
    p(f"- 0.20–0.50 (noticeable): **{w050:,}**")
    p(f"- > 0.50    (large): **{w_big:,}**")
    p("")
    p("## Interpretation guide")
    p("")
    p("- Correlation close to **1.0** and mean diff close to **0**: the "
      "raster approach is producing essentially the same answer as polygon.")
    p("- Mean diff around 0.02-0.05 and high correlation: raster is doing "
      "what we want, with quantization noise from the sample / pixel grid.")
    p("- Big mean diff or low correlation: bug — investigate.")
    p("- Coverage discrepancy (large `raster_only` or `polygon_only`): the "
      "two paths disagree about whether some edges touch any forest at all. "
      "Usually points to sample-spacing being too coarse or pixel-size too "
      "coarse for narrow gaps.")
    p("")
    report = "\n".join(lines)
    print(report)
    if out_path is not None:
        out_path.write_text(report)
        print(f"\n[compare] report written to {out_path}", flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bbox", default=None,
                   help="min_lon,min_lat,max_lon,max_lat — restrict the "
                        "comparison to a corridor (recommended).")
    p.add_argument("--out", default="/data/canopy_comparison.md",
                   help="Output path for the markdown report.")
    args = p.parse_args()
    bbox = None
    if args.bbox:
        parts = tuple(float(x) for x in args.bbox.split(","))
        if len(parts) != 4:
            raise SystemExit("--bbox must be 4 comma-separated floats")
        bbox = parts
    with psycopg.connect(config.PG_DSN) as conn:
        compare(conn, bbox=bbox, out_path=Path(args.out) if args.out else None)
