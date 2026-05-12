"""Compute per-edge canopy_frac: fraction of edge length inside a
landcover forest polygon. Run after `ingest_landcover` populates
the `landcover` table.

Strategy:
  1. Master builds two UNLOGGED regular tables once:
       - `_corridor_edges` (gid, geom, edge_len) — the 2-point edge
         lines for the bbox, GiST + PK indexed.
       - `_forest_simplified` — landcover forests with
         ST_SimplifyPreserveTopology(geom, 5e-5) applied (~5 m
         tolerance), GiST indexed. Halves the average vertex count
         and consequently the spatial-op cost.
  2. Master zeros canopy_frac for the corridor once.
  3. Master partitions the corridor's gid range across N worker
     processes; each worker opens its own DB connection and pulls
     its share of gid-range batches in sequence.
       - Postgres handles N concurrent UPDATEs on disjoint row ranges
         without lock conflict — this is faster than relying on
         PG's per-query parallel workers because the UPDATE path
         itself is leader-only.
  4. Drop temp tables at end.

For a Graz->Wien-sized corridor (~9 M edges, ~70 K forest polygons)
this lands in tens of minutes vs. multi-hour single-process.

Numerically we use ST_Length on the 4326 geometry directly — the
ratio is unit-free, so the planar lon/lat approximation cancels out.
The error from anisotropy along a single sub-100 m edge is well
below the precision we care about.

`recompute-cost` is still required afterwards to fold the new
canopy_frac into `ways.cost` / `reverse_cost`.

bbox flag mirrors `recompute_cost.recompute` — handy for
validation runs on a corridor like Graz->Wien without pricing the
whole graph.
"""
import argparse
import multiprocessing as mp
import time

import psycopg

import config


_BATCH = 500_000

# Number of Python worker processes used for the per-batch spatial
# join. Each worker holds its own DB connection and processes a
# disjoint gid range; Postgres handles the concurrency natively (no
# PG parallel-worker config involved). 4 is a sweet spot for the
# 6 GB postgres container — each worker is bounded by work_mem so
# 4 × 256 MB stays well under the container cap.
_NUM_WORKERS = 4


def _split_gid_range(gid_min: int, gid_max: int,
                     n: int) -> list[tuple[int, int]]:
    """Split (gid_min, gid_max] into n half-open chunks suitable for
    workers — each chunk is processed as `WHERE gid > lo AND gid <= hi`,
    which makes them naturally disjoint at the boundaries.
    """
    if gid_max <= gid_min:
        return [(gid_min - 1, gid_max)]
    span = gid_max - gid_min + 1
    step = span // n + (1 if span % n else 0)
    chunks: list[tuple[int, int]] = []
    lo = gid_min - 1
    for _ in range(n):
        hi = min(lo + step, gid_max)
        chunks.append((lo, hi))
        lo = hi
        if lo >= gid_max:
            break
    return chunks


def _worker_main(worker_id: int,
                 gid_lo: int,
                 gid_hi: int) -> tuple[int, int, int]:
    """Process all _BATCH-sized sub-chunks of (gid_lo, gid_hi] in one
    worker process. Returns (worker_id, processed, nonzero).

    Each worker holds its own psycopg connection and commits per
    batch — failures in one worker don't roll back the others.
    """
    processed = 0
    nonzero = 0
    last_gid = gid_lo
    t_start = time.time()

    with psycopg.connect(config.PG_DSN) as conn:
        with conn.cursor() as cur:
            # Single-process inside each worker — we get our
            # parallelism from N worker connections, not PG's own
            # parallel-query machinery.
            cur.execute("SET max_parallel_workers_per_gather = 0")
            cur.execute("SET work_mem = '256MB'")

        while last_gid < gid_hi:
            batch_hi = min(last_gid + _BATCH, gid_hi)
            t0 = time.time()
            with conn.cursor() as cur:
                cur.execute(
                    "WITH overlap AS ( "
                    "  SELECT ce.gid, "
                    "         SUM(ST_Length(ST_Intersection(ce.geom, f.geom))) AS overlap_len, "
                    "         ce.edge_len "
                    "  FROM _corridor_edges ce "
                    "  JOIN _forest_simplified f "
                    "       ON ST_Intersects(f.geom, ce.geom) "
                    "  WHERE ce.gid > %s AND ce.gid <= %s "
                    "  GROUP BY ce.gid, ce.edge_len "
                    ") "
                    "UPDATE ways w "
                    "SET canopy_frac = LEAST(1.0::real, GREATEST(0.0::real, "
                    "    (o.overlap_len / o.edge_len)::real)) "
                    "FROM overlap o "
                    "WHERE w.gid = o.gid AND o.edge_len > 0",
                    (last_gid, batch_hi),
                )
                nonzero_this_batch = cur.rowcount

                cur.execute(
                    "SELECT COUNT(*) FROM _corridor_edges "
                    "WHERE gid > %s AND gid <= %s",
                    (last_gid, batch_hi),
                )
                processed_this_batch = int(cur.fetchone()[0])
            conn.commit()

            processed += processed_this_batch
            nonzero += nonzero_this_batch
            last_gid = batch_hi
            print(f"[w{worker_id}]   gid<= {batch_hi:,}: "
                  f"+{processed_this_batch:,} edges "
                  f"(+{nonzero_this_batch:,} canopied) in "
                  f"{time.time()-t0:.1f}s "
                  f"(worker total: {processed:,} in "
                  f"{(time.time()-t_start)/60:.1f} min)",
                  flush=True)

    return (worker_id, processed, nonzero)


def compute(conn: psycopg.Connection,
            bbox: tuple[float, float, float, float] | None = None) -> None:
    """Populate ways.canopy_frac over the whole graph or a bbox subset.

    bbox = (min_lon, min_lat, max_lon, max_lat) — both endpoints of an
    edge must fall inside the box for the edge to be included in the
    corridor working set.
    """
    with conn.cursor() as cur:
        # Build phase: parallel hash join of 22M-vertex tables can blow
        # past /dev/shm, so keep this single-process.
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute("SET work_mem = '512MB'")

        cur.execute("SELECT COUNT(*) FROM landcover WHERE class = 'forest'")
        n_forest = int(cur.fetchone()[0])
        if n_forest == 0:
            raise SystemExit(
                "[canopy] no forest polygons in `landcover` — run "
                "`landcover-ingest` first"
            )
        print(f"[canopy] {n_forest:,} forest polygons in landcover", flush=True)

        # ----------------------------------------------------------------
        # 1) Build the corridor working set once: gid + geom + length.
        #    UNLOGGED so the WAL stays small; regular (not TEMP) so the
        #    spatial-join batches in step 3 can run with parallel
        #    workers (TEMP tables are parallel-restricted in PG).
        # ----------------------------------------------------------------
        t0 = time.time()
        cur.execute("DROP TABLE IF EXISTS _corridor_edges")
        if bbox:
            cur.execute(
                "CREATE UNLOGGED TABLE _corridor_edges AS "
                "SELECT w.gid, "
                "       ST_MakeLine(vs.the_geom, vt.the_geom) AS geom, "
                "       ST_Length(ST_MakeLine(vs.the_geom, vt.the_geom)) AS edge_len "
                "FROM ways w "
                "JOIN ways_vertices_pgr vs ON vs.id = w.source "
                "JOIN ways_vertices_pgr vt ON vt.id = w.target "
                "WHERE vs.lon BETWEEN %s AND %s AND vs.lat BETWEEN %s AND %s "
                "  AND vt.lon BETWEEN %s AND %s AND vt.lat BETWEEN %s AND %s",
                (bbox[0], bbox[2], bbox[1], bbox[3]) * 2,
            )
        else:
            cur.execute(
                "CREATE UNLOGGED TABLE _corridor_edges AS "
                "SELECT w.gid, "
                "       ST_MakeLine(vs.the_geom, vt.the_geom) AS geom, "
                "       ST_Length(ST_MakeLine(vs.the_geom, vt.the_geom)) AS edge_len "
                "FROM ways w "
                "JOIN ways_vertices_pgr vs ON vs.id = w.source "
                "JOIN ways_vertices_pgr vt ON vt.id = w.target"
            )
        cur.execute("ALTER TABLE _corridor_edges ADD PRIMARY KEY (gid)")
        cur.execute("CREATE INDEX ON _corridor_edges USING gist(geom)")
        cur.execute("ANALYZE _corridor_edges")

        # 1b) Build a simplified copy of the forest polygons. The hot
        # path of the canopy compute is ST_Intersection / ST_Intersects
        # against forest geometries, which scales with vertex count.
        # OSM forest outlines tend to be over-detailed for our 5-25 m
        # canopy use case; smoothing them to ~5 m tolerance with
        # ST_SimplifyPreserveTopology gets a 3-5x speedup on the
        # spatial join without meaningfully changing canopy_frac.
        t_simp = time.time()
        cur.execute("DROP TABLE IF EXISTS _forest_simplified")
        cur.execute(
            "CREATE UNLOGGED TABLE _forest_simplified AS "
            "SELECT id, "
            "       ST_Multi(ST_SimplifyPreserveTopology(geom, 5e-5))"
            "         ::geometry(MultiPolygon, 4326) AS geom "
            "FROM landcover WHERE class = 'forest' AND geom IS NOT NULL"
        )
        cur.execute("CREATE INDEX ON _forest_simplified USING gist(geom)")
        cur.execute("ANALYZE _forest_simplified")
        cur.execute("SELECT COUNT(*), "
                    "       SUM(ST_NPoints(geom)) AS pts_simp, "
                    "       (SELECT SUM(ST_NPoints(geom)) FROM landcover "
                    "        WHERE class='forest') AS pts_orig "
                    "FROM _forest_simplified")
        n_simp, pts_simp, pts_orig = cur.fetchone()
        n_simp = int(n_simp); pts_simp = int(pts_simp); pts_orig = int(pts_orig or 0)
        print(f"[canopy] simplified forests: {n_simp:,} polys, "
              f"{pts_simp:,} vertices "
              f"(was {pts_orig:,}, {100.0*pts_simp/max(pts_orig,1):.1f}% kept) "
              f"in {time.time()-t_simp:.1f}s", flush=True)
        cur.execute("SELECT COUNT(*), MIN(gid), MAX(gid) FROM _corridor_edges")
        n_corridor, gid_min, gid_max = cur.fetchone()
        n_corridor = int(n_corridor)
        if n_corridor == 0:
            print("[canopy] corridor working set is empty", flush=True)
            return
        gid_min, gid_max = int(gid_min), int(gid_max)
        print(f"[canopy] corridor working set: {n_corridor:,} edges, "
              f"gid [{gid_min:,}..{gid_max:,}], built in {time.time()-t0:.1f}s",
              flush=True)

        # ----------------------------------------------------------------
        # 2) Pre-zero canopy_frac across the corridor in a single UPDATE.
        # ----------------------------------------------------------------
        t0 = time.time()
        cur.execute(
            "UPDATE ways w SET canopy_frac = 0 "
            "FROM _corridor_edges ce WHERE w.gid = ce.gid "
            "  AND w.canopy_frac <> 0"
        )
        print(f"[canopy] pre-zero: {cur.rowcount:,} edges reset in "
              f"{time.time()-t0:.1f}s", flush=True)
    conn.commit()

    # ----------------------------------------------------------------
    # 3) Fan out to N worker processes. Each worker opens its own DB
    #    connection and processes a disjoint chunk of the gid range.
    # ----------------------------------------------------------------
    chunks = _split_gid_range(gid_min, gid_max, _NUM_WORKERS)
    print(f"[canopy] dispatching {len(chunks)} workers across "
          f"gid {gid_min:,}..{gid_max:,}", flush=True)
    for i, (lo, hi) in enumerate(chunks):
        print(f"[canopy]   w{i}: gid ({lo:,}, {hi:,}]", flush=True)

    t_start = time.time()
    with mp.get_context("spawn").Pool(_NUM_WORKERS) as pool:
        results = pool.starmap(
            _worker_main,
            [(i, lo, hi) for i, (lo, hi) in enumerate(chunks)],
        )
    elapsed = time.time() - t_start
    total_processed = sum(r[1] for r in results)
    total_nonzero = sum(r[2] for r in results)
    print(f"[canopy] all workers done in {elapsed/60:.1f} min "
          f"({total_processed:,} processed, {total_nonzero:,} nonzero)",
          flush=True)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT AVG(canopy_frac), MAX(canopy_frac), "
            "  COUNT(*) FILTER (WHERE canopy_frac > 0), "
            "  COUNT(*) FILTER (WHERE canopy_frac > 0.5), "
            "  COUNT(*) FILTER (WHERE canopy_frac >= 0.99) "
            "FROM ways"
        )
        avg, mx, gt0, gt50, full = cur.fetchone()
        cur.execute("DROP TABLE IF EXISTS _corridor_edges")
        cur.execute("DROP TABLE IF EXISTS _forest_simplified")
    conn.commit()
    print(f"[canopy] done. avg={avg:.4f} max={mx:.4f} "
          f"nonzero={gt0:,} >50%={gt50:,} ~full={full:,}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bbox", type=str, default=None,
                   help="min_lon,min_lat,max_lon,max_lat — restrict to edges "
                        "whose both endpoints fall inside this box.")
    args = p.parse_args()
    bbox: tuple[float, float, float, float] | None = None
    if args.bbox:
        parts = tuple(float(x) for x in args.bbox.split(","))
        if len(parts) != 4:
            raise SystemExit("--bbox must be 4 comma-separated floats")
        bbox = parts
    with psycopg.connect(config.PG_DSN) as conn:
        compute(conn, bbox=bbox)
