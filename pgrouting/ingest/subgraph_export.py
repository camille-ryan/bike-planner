"""One-time export of the chain-graph subgraph as tiled flat files.

The chain-graph builder used to query the entire `seed + is_ferry +
named-promotable` subgraph directly from postgres on every rebuild.
Under WSL2 disk I/O that meant hours of cold-cache page reads per
subgraph load — a per-rebuild penalty that dominated wall clock.

This script does the postgres pull **once** and persists the result as
1° × 1° `.npz` cell files under `/data/subgraph_cells/`. Downstream
`build_way_graph.py` reads the cells relevant to its bbox and skips
postgres entirely. Same pattern as `spt/export_cells.py` did for stage
7 — the polygon-SPT compute used to hit postgres per anchor and now
reads pre-exported cells; after this change stage 4 works the same way.

Edge assignment: each edge is placed in the cell containing its
**source vertex's** `(floor(lon), floor(lat))`. A bbox query at read
time expands by 1 cell in each direction to catch edges whose target
is outside the source cell — enough for OSM edges, which are almost
always < 1 km end-to-end.

Cell npz fields (all numpy, no pickle):

    source, target     : int64        — global vertex IDs
    length_m           : float64
    osm_way_id         : int64
    hw_idx             : int8         — index into `highway_table` (in manifest)
    is_ferry           : bool_
    src_lon, src_lat   : float64
    dst_lon, dst_lat   : float64
    name_table         : array[str]   — per-cell string table
    name_idx           : int32        — index into name_table
    ref_table          : array[str]
    ref_idx            : int32

Interning name/ref per cell keeps the format numpy-native (no pickle)
and cuts storage substantially — most edges in a cell share the same
few road names.

Env vars:
  PGDATABASE            (default from container: bike)
  SUBGRAPH_CELLS_DIR    (default /data/subgraph_cells)
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

import config


SEED_HIGHWAYS = ("motorway", "trunk", "primary", "secondary")
PROMOTABLE_HIGHWAYS = (
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street", "road",
    "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link",
)
HIGHWAY_TABLE = list(PROMOTABLE_HIGHWAYS)
HW_TO_IDX = {h: i for i, h in enumerate(HIGHWAY_TABLE)}

OUT_DIR = Path(os.environ.get("SUBGRAPH_CELLS_DIR", "/data/subgraph_cells"))
CELL_DEG = 1.0


def _cell_of(lon: float, lat: float) -> tuple[int, int]:
    return int(math.floor(lon / CELL_DEG)), int(math.floor(lat / CELL_DEG))


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_ph    = ",".join(["%s"] * len(SEED_HIGHWAYS))
    promote_ph = ",".join(["%s"] * len(PROMOTABLE_HIGHWAYS))
    sql = f"""
        SELECT w.source, w.target, w.length_m, w.osm_way_id, w.highway,
               COALESCE(t.name, ''), COALESCE(t.ref, ''),
               ST_X(vs.the_geom), ST_Y(vs.the_geom),
               ST_X(vt.the_geom), ST_Y(vt.the_geom),
               w.is_ferry
        FROM ways w
        JOIN ways_vertices_pgr vs ON vs.id = w.source
        JOIN ways_vertices_pgr vt ON vt.id = w.target
        LEFT JOIN way_tags t ON t.osm_way_id = w.osm_way_id
        WHERE w.length_m > 0.0
          AND (
              w.highway IN ({seed_ph})
              OR w.is_ferry
              OR (
                  w.highway IN ({promote_ph})
                  AND (COALESCE(t.name, '') <> '' OR COALESCE(t.ref, '') <> '')
              )
          )
    """
    params = list(SEED_HIGHWAYS) + list(PROMOTABLE_HIGHWAYS)

    print(f"[subgraph-export] output: {OUT_DIR}", flush=True)
    print(f"[subgraph-export] cell: {CELL_DEG}° × {CELL_DEG}°", flush=True)
    print(f"[subgraph-export] streaming from postgres…", flush=True)

    # Bucket rows per cell as we read. Buckets are lists of tuples in
    # the natural column order; we convert to numpy arrays per cell
    # when writing.
    cell_rows: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    n_read = 0
    n_bytes_est = 0
    t_query = time.time()
    last_log = t_query

    with psycopg.connect(config.PG_DSN) as conn:
        with conn.cursor(name="subgraph_export_cursor") as cur:
            cur.itersize = 200_000
            cur.execute(sql, params)
            while True:
                batch = cur.fetchmany(200_000)
                if not batch:
                    break
                n_read += len(batch)
                for r in batch:
                    (sv, tv, lm, oid, hw, name, ref,
                     sx, sy, tx, ty, is_ferry) = r
                    cell = _cell_of(float(sx), float(sy))
                    cell_rows[cell].append((
                        int(sv), int(tv), float(lm), int(oid),
                        HW_TO_IDX.get(hw, -1),
                        name or "", ref or "",
                        float(sx), float(sy), float(tx), float(ty),
                        bool(is_ferry),
                    ))
                    n_bytes_est += 120
                if time.time() - last_log > 30:
                    dt = time.time() - t_query
                    print(f"[subgraph-export]   {n_read:,} rows fetched "
                          f"({len(cell_rows)} cells, ~{n_bytes_est / (1 << 30):.1f} GB) "
                          f"in {dt:.0f}s", flush=True)
                    last_log = time.time()

    print(f"[subgraph-export] fetch DONE: {n_read:,} rows into "
          f"{len(cell_rows)} cells in {time.time()-t_query:.1f}s",
          flush=True)

    # Write per-cell npz.
    t_write = time.time()
    total_bytes = 0
    for i, (cell, rows) in enumerate(sorted(cell_rows.items()), 1):
        clon, clat = cell
        path = OUT_DIR / f"{clon:+04d}_{clat:+04d}.npz"
        n = len(rows)
        source     = np.empty(n, dtype=np.int64)
        target     = np.empty(n, dtype=np.int64)
        length_m   = np.empty(n, dtype=np.float64)
        osm_way_id = np.empty(n, dtype=np.int64)
        hw_idx     = np.empty(n, dtype=np.int8)
        src_lon    = np.empty(n, dtype=np.float64)
        src_lat    = np.empty(n, dtype=np.float64)
        dst_lon    = np.empty(n, dtype=np.float64)
        dst_lat    = np.empty(n, dtype=np.float64)
        is_ferry_a = np.empty(n, dtype=np.bool_)
        names: list[str] = []
        refs:  list[str] = []
        for k, r in enumerate(rows):
            (source[k], target[k], length_m[k], osm_way_id[k], hw_idx[k],
             name, ref,
             src_lon[k], src_lat[k], dst_lon[k], dst_lat[k],
             is_ferry_a[k]) = r
            names.append(name)
            refs.append(ref)
        # Per-cell interning: most edges share the same handful of road
        # names/refs so the tables are tiny.
        name_table, name_idx = np.unique(names, return_inverse=True)
        ref_table,  ref_idx  = np.unique(refs,  return_inverse=True)
        np.savez_compressed(
            path,
            source=source, target=target, length_m=length_m,
            osm_way_id=osm_way_id, hw_idx=hw_idx, is_ferry=is_ferry_a,
            src_lon=src_lon, src_lat=src_lat,
            dst_lon=dst_lon, dst_lat=dst_lat,
            name_table=name_table, name_idx=name_idx.astype(np.int32),
            ref_table=ref_table,   ref_idx=ref_idx.astype(np.int32),
        )
        total_bytes += path.stat().st_size
        if i % 20 == 0 or i == len(cell_rows):
            print(f"[subgraph-export]   wrote {i}/{len(cell_rows)} cells, "
                  f"{total_bytes / (1 << 20):.0f} MB so far", flush=True)

    manifest = {
        "cell_deg":       CELL_DEG,
        "n_cells":        len(cell_rows),
        "n_rows_total":   n_read,
        "highway_table":  HIGHWAY_TABLE,
        "on_disk_bytes":  total_bytes,
        "cells":          [
            {"lon": c[0], "lat": c[1], "n_rows": len(rows)}
            for c, rows in sorted(cell_rows.items())
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[subgraph-export] wrote {len(cell_rows)} cells, "
          f"{total_bytes / (1 << 20):.0f} MB total, "
          f"in {time.time()-t_write:.1f}s", flush=True)
    print(f"[subgraph-export] DONE in {time.time()-t0:.1f}s "
          f"({n_read:,} edges → {len(cell_rows)} cells)", flush=True)


if __name__ == "__main__":
    main()
