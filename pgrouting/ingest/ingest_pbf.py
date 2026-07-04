"""Stream OSM PBFs into Postgres tables `ways` and `ways_vertices_pgr`.

Pipeline:
  1. Per country, pyosmium streams the PBF and emits (node, edge) rows
     into temp staging tables `tmp_nodes` and `tmp_edges` via COPY.
  2. After all PBFs are streamed, deduplicate nodes by OSM id and
     insert into `ways_vertices_pgr` (assigns the bigserial vertex id).
  3. Resolve each tmp_edges row's source_osm/target_osm to vertex ids
     and insert into `ways`.

Why the staging step: pgRouting's vertex `id` is bigserial, but each
edge row knows only the OSM node id of its endpoints. We can't fill
in `source`/`target` until vertex IDs are assigned. Staging keeps the
streaming write simple (no per-row lookups).

Memory: pyosmium streams the PBF; per-batch Python objects are flushed
to Postgres every BATCH_SIZE rows. Process RSS stays under ~1 GB
regardless of PBF size. Postgres's working memory is governed by the
service's shared_buffers / work_mem (set in docker-compose).

V2 Phase A.2 changes:
  - Raw OSM tag columns (highway, surface, tracktype, oneway, bicycle,
    cycleway, bicycle_road, access) are persisted on `ways` so cost
    can be recomputed without re-parsing PBFs.
  - Per-way sinuosity (actual_length / endpoint_distance) is computed
    inline and stamped on every edge of the way. Clamped to [1.0, 5.0].
  - Edge `cost`/`reverse_cost` are still computed inline using
    `bike_edge_cost` with the V2 defaults (grade_pct=0, sinuosity=1).
    The recompute_cost stage updates these after ingest_dem populates
    vertex elevations, so the graph is immediately usable for routing
    even before elevation is available.
"""
import os
import time
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from pathlib import Path
from typing import Iterable

import osmium
import psycopg

from cost.cost import bike_edge_cost, EXCLUDE


_EARTH_R = 6_371_000.0
_BATCH_SIZE = 100_000

# Highway classes that aren't real road network — never useful, drivable
# or otherwise. Subset of cost.EXCLUDE that we still want to drop here.
_NON_REAL_HW = {
    "proposed", "abandoned", "construction",
    "raceway", "elevator", "platform",
}
# Drivable-but-possibly-not-bikeable highway classes. Used when
# bike_edge_cost rejects a way — we still keep it for the chain-graph
# topology (motorways, bike-banned tunnels, etc.) marked bike_excluded=true.
# Matches PAVED_HIGHWAYS in connect_anchors_pairs.py plus motorway*.
_DRIVABLE_HW = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified", "residential", "living_street",
    "road", "service",
}

# Window over which directional curvature is accumulated. Each segment's
# `curv_fwd` is the total absolute bend angle at nodes within the next
# ~_CURV_WINDOW_M polyline meters (forward direction of travel); `curv_rev`
# is the same looking the other way. This captures the physics of "steep
# descent into a curve" — the descending segment sees the upcoming bend in
# its forward window even though the segment itself is straight.
#
# 300 m matches typical alpine-switchback / valley-bottom-curve scales.
_CURV_WINDOW_M = 300.0


def _haversine(lon1, lat1, lon2, lat2):
    rl1, rl2 = radians(lat1), radians(lat2)
    dl = radians(lat2 - lat1)
    dn = radians(lon2 - lon1)
    a = sin(dl / 2) ** 2 + cos(rl1) * cos(rl2) * sin(dn / 2) ** 2
    return 2 * _EARTH_R * asin(sqrt(a))


def _segment_curvatures(nodes: list[tuple[int, float, float]],
                        window_m: float = _CURV_WINDOW_M
                        ) -> list[tuple[float, float]]:
    """Per-segment directional bend totals over a polyline window.

    For each of the N-1 segments, returns `(curv_fwd, curv_rev)`:
      curv_fwd: sum of absolute bend angles (degrees) at nodes within
                the next `window_m` of polyline (forward in way order).
                "Forward" includes the bend at the segment's own target
                node — that's the first bend you encounter as you exit.
      curv_rev: same looking the other way.

    O(N) per way via prefix-sum + two-pointer window advancement.
    """
    n = len(nodes)
    if n < 2:
        return []

    # Cumulative polyline length per node.
    cumlen = [0.0] * n
    for i in range(n - 1):
        _, ln1, lt1 = nodes[i]
        _, ln2, lt2 = nodes[i + 1]
        cumlen[i + 1] = cumlen[i] + _haversine(ln1, lt1, ln2, lt2)

    # Per-node absolute bend angle (degrees). Endpoints get 0 since
    # they have no incoming-or-outgoing pair. Latitude-scaled local
    # frame: lon × cos(mean_lat) so angles aren't distorted at higher
    # latitudes. Relative scale only matters; absolute units cancel.
    bends = [0.0] * n
    if n >= 3:
        lat_mid = nodes[n // 2][2]
        lon_scale = cos(radians(lat_mid))
        for i in range(1, n - 1):
            ax = (nodes[i][1]     - nodes[i - 1][1]) * lon_scale
            ay =  nodes[i][2]     - nodes[i - 1][2]
            bx = (nodes[i + 1][1] - nodes[i][1])     * lon_scale
            by =  nodes[i + 1][2] - nodes[i][2]
            # Signed angle from incoming to outgoing direction.
            cross = ax * by - ay * bx
            dot   = ax * bx + ay * by
            bends[i] = abs(degrees(atan2(cross, dot)))

    # Prefix sums of bends so window queries are O(1) given indices.
    # bend_prefix[k] = sum of bends[0..k-1].
    bend_prefix = [0.0] * (n + 1)
    for i in range(n):
        bend_prefix[i + 1] = bend_prefix[i] + bends[i]

    out: list[tuple[float, float]] = []

    # Forward window for segment k = (node[k], node[k+1]):
    # bends at indices m where k+1 <= m and cumlen[m] - cumlen[k+1] <= window_m.
    # m is monotonic non-decreasing as k advances.
    m_fwd = 0
    for k in range(n - 1):
        if m_fwd < k + 1:
            m_fwd = k + 1
        while m_fwd + 1 < n and cumlen[m_fwd + 1] - cumlen[k + 1] <= window_m:
            m_fwd += 1
        curv_fwd = bend_prefix[m_fwd + 1] - bend_prefix[k + 1]
        out.append((curv_fwd, 0.0))   # curv_rev filled in next loop

    # Reverse window for segment k:
    # bends at indices m where m <= k and cumlen[k] - cumlen[m] <= window_m.
    # As k advances, the lower bound m_min is monotonic non-decreasing.
    m_min = 0
    for k in range(n - 1):
        while m_min < k and cumlen[m_min] < cumlen[k] - window_m:
            m_min += 1
        curv_rev = bend_prefix[k + 1] - bend_prefix[m_min]
        out[k] = (out[k][0], curv_rev)

    return out


def _create_staging(cur):
    """Drop+recreate the staging tables.

    LOGGED (not UNLOGGED): WAL overhead during streaming is the price
    we pay to keep staging data alive across a postgres restart / WSL
    VM pause. UNLOGGED tables are TRUNCATED on crash recovery, which
    forces a re-stream of the entire PBF and wastes the prior run's
    work. The 4-country corridor's resolve has been killed mid-INSERT
    multiple times by WSL idling out, so crash-survivable staging plus
    chunked resolve (see `_resolve_into_final`) is the path that
    actually finishes.

    The tmp_edges `_seq BIGSERIAL` column is the chunk key used by
    `_resolve_into_final` — assigned monotonically per row during COPY,
    contiguous, indexable, ideal for range-based batching.
    """
    cur.execute("DROP TABLE IF EXISTS tmp_nodes")
    cur.execute("DROP TABLE IF EXISTS tmp_edges")
    cur.execute("""
        CREATE TABLE tmp_nodes (
            osm_id bigint NOT NULL,
            lon    double precision NOT NULL,
            lat    double precision NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE tmp_edges (
            _seq          bigserial PRIMARY KEY,
            osm_way_id    bigint,
            source_osm    bigint NOT NULL,
            target_osm    bigint NOT NULL,
            cost          double precision NOT NULL,
            reverse_cost  double precision NOT NULL,
            length_m      double precision NOT NULL,
            is_ferry      boolean NOT NULL,
            bike_excluded boolean NOT NULL
        )
    """)


class _PgIngestHandler(osmium.SimpleHandler):
    """pyosmium handler that buffers nodes + edges and bulk-COPYs them.

    psycopg3 only allows ONE active COPY per connection at a time, so
    the obvious "keep two streams open" pattern doesn't work. Instead
    we buffer ~100k of each in Python and issue sequential COPYs when
    a batch fills. Memory stays at a few MB; throughput is excellent
    because each COPY amortizes the round-trip cost over 100k rows.
    """

    BATCH_SIZE = 100_000

    def __init__(self, conn: psycopg.Connection):
        super().__init__()
        self.conn = conn
        self._node_buf: list[tuple] = []
        self._edge_buf: list[tuple] = []
        self.skipped_no_highway       = 0
        self.skipped_excluded         = 0
        self.skipped_no_cost          = 0
        self.skipped_already_in_ways  = 0
        self.nodes_written            = 0
        self.edges_written            = 0

    def _flush_nodes(self):
        if not self._node_buf:
            return
        with self.conn.cursor() as cur:
            with cur.copy("COPY tmp_nodes (osm_id, lon, lat) FROM STDIN") as cp:
                for row in self._node_buf:
                    cp.write_row(row)
        self.nodes_written += len(self._node_buf)
        self._node_buf.clear()

    def _flush_edges(self):
        if not self._edge_buf:
            return
        with self.conn.cursor() as cur:
            with cur.copy(
                "COPY tmp_edges (osm_way_id, source_osm, target_osm, "
                "cost, reverse_cost, length_m, is_ferry, bike_excluded) FROM STDIN"
            ) as cp:
                for row in self._edge_buf:
                    cp.write_row(row)
        self.edges_written += len(self._edge_buf)
        self._edge_buf.clear()

    def way(self, w):
        tags = w.tags
        hw = tags.get("highway") or ""
        is_ferry = tags.get("route") == "ferry"
        if not hw and not is_ferry:
            self.skipped_no_highway += 1
            return
        # Non-real highway classes: never relevant, drivable or otherwise.
        if hw in _NON_REAL_HW:
            self.skipped_excluded += 1
            return
        surface      = tags.get("surface", "") or ""
        tracktype    = tags.get("tracktype", "") or ""
        bicycle      = tags.get("bicycle", "") or ""
        cycleway     = tags.get("cycleway", "") or ""
        access       = tags.get("access", "") or ""
        bicycle_road = tags.get("bicycle_road", "") or ""
        oneway_raw   = tags.get("oneway", "") or ""

        cost_factor = bike_edge_cost(
            highway=hw,
            surface=surface,
            tracktype=tracktype,
            bicycle=bicycle,
            cycleway=cycleway,
            access=access,
            bicycle_road=bicycle_road,
            is_ferry=is_ferry,
        )
        # SUPPLEMENT MODE (default): the existing ways table already
        # holds every bikeable row from a prior ingest. Skip bikeable
        # rows here so the resolve INSERT only deals with the
        # previously-dropped drivable-only rows (motorway, bicycle=no
        # tunnels, access=private through-routes). Cuts staging from
        # ~130M to ~1M rows, fits resolve's ON CONFLICT comfortably in
        # RAM, avoids the WSL OOM we hit on full re-insert.
        #
        # To DISABLE supplement mode (e.g., when extending to a new
        # country whose bikeable rows aren't in `ways` yet), set
        # INGEST_SUPPLEMENT_MODE=0 in the environment. Both bikeable
        # AND drivable-only rows then flow through.
        supplement_mode = os.environ.get("INGEST_SUPPLEMENT_MODE", "1") == "1"
        if cost_factor is not None:
            if supplement_mode:
                self.skipped_already_in_ways += 1
                return
            # cost_factor is the bikeable per-meter factor (computed by
            # bike_edge_cost above). Keep it and write the row as bikeable.
            bike_excluded = False
        else:
            # Not bikeable — include only if drivable (motorway etc.)
            # for chain-graph topology.
            if is_ferry or not hw or hw not in _DRIVABLE_HW:
                self.skipped_no_cost += 1
                return
            cost_factor   = 1.0
            bike_excluded = True

        ow = oneway_raw.lower()
        if ow in ("-1", "reverse"):
            oneway_dir = -1
        elif ow in ("yes", "true", "1"):
            oneway_dir = 1
        else:
            oneway_dir = 0

        # Pass 1: collect all resolvable node positions. Missing locations
        # (InvalidLocationError) just break the chain — same behavior as
        # the V1 code, kept for symmetry.
        nodes: list[tuple[int, float, float]] = []
        for node in w.nodes:
            try:
                lon = node.location.lon
                lat = node.location.lat
            except (osmium.InvalidLocationError, RuntimeError):
                continue
            nodes.append((int(node.ref), float(lon), float(lat)))
        if len(nodes) < 2:
            return

        # Pass 2: emit nodes and per-segment edges.
        prev_id = None
        prev_lon = prev_lat = 0.0
        for osm_id, lon, lat in nodes:
            self._node_buf.append((osm_id, lon, lat))
            if prev_id is not None:
                length_m = _haversine(prev_lon, prev_lat, lon, lat)
                if length_m > 0:
                    fwd_cost = float(length_m * cost_factor)
                    if oneway_dir == 0:           # both directions
                        cost, rev = fwd_cost, fwd_cost
                    elif oneway_dir == 1:         # forward only
                        cost, rev = fwd_cost, -1.0
                    else:                         # reverse only
                        cost, rev = -1.0, fwd_cost
                    self._edge_buf.append((
                        int(w.id), prev_id, osm_id,
                        cost, rev, float(length_m), bool(is_ferry),
                        bool(bike_excluded),
                    ))
            prev_id = osm_id
            prev_lon = lon
            prev_lat = lat

        if len(self._node_buf) >= self.BATCH_SIZE:
            self._flush_nodes()
        if len(self._edge_buf) >= self.BATCH_SIZE:
            self._flush_edges()

    def finalize(self):
        self._flush_nodes()
        self._flush_edges()


def _stream_pbf(conn: psycopg.Connection, pbf: Path) -> dict:
    print(f"[ingest] streaming {pbf.name}")
    h = _PgIngestHandler(conn)
    # Disk-backed node-location index: scales to corridor / continent-sized
    # PBFs without ballooning RAM. sparse_file_array spills to disk and
    # stays bounded.
    #
    # Per-PBF index path: an earlier version shared
    # `/tmp/osmium-locations.idx` across all four country PBFs. When
    # processing the corridor in May 2026, the last-processed country
    # (Denmark, after a 4.5 GB Germany PBF) ended up with ~12% of its
    # expected edge count and a road graph fragmented into thousands of
    # tiny disconnected components — Helsingør → Mørdrup wasn't routable
    # in our graph despite being trivially routable on osm.org. Root
    # cause: pyosmium silently swallows InvalidLocationError on missing
    # node locations, and reusing the index file across PBFs led to
    # nodes failing to resolve for the last country. Per-PBF idx files
    # eliminate the failure mode; we also unlink each file when its
    # PBF is done so /tmp doesn't grow unboundedly.
    idx_path = f"/tmp/osmium-locations-{pbf.stem}.idx"
    try:
        h.apply_file(str(pbf), locations=True,
                     idx=f"sparse_file_array,{idx_path}")
    finally:
        if os.path.exists(idx_path):
            os.unlink(idx_path)
    h.finalize()
    print(f"[ingest]   {pbf.name}: nodes_written={h.nodes_written:,} "
          f"edges_written={h.edges_written:,} "
          f"skipped(no_hw={h.skipped_no_highway:,}, "
          f"excluded={h.skipped_excluded:,}, "
          f"no_cost={h.skipped_no_cost:,}, "
          f"already_in_ways={h.skipped_already_in_ways:,})")
    return {
        "nodes": h.nodes_written, "edges": h.edges_written,
    }


_INSERT_COLUMNS = (
    "osm_way_id, source, target, cost, reverse_cost, length_m, is_ferry, "
    "bike_excluded"
)
_SELECT_COLUMNS = (
    "e.osm_way_id, v_src.id, v_dst.id, e.cost, e.reverse_cost, e.length_m, "
    "e.is_ferry, e.bike_excluded"
)


_VERTEX_BATCH = 20_000_000      # rows per vertex anti-join chunk
_EDGE_BATCH   = 2_000_000       # rows per ways anti-join chunk (small to keep memory bounded)
_HASH_WORK_MEM = "2GB"          # per-backend work_mem during the vertex insert
_EDGE_WORK_MEM = "128MB"        # smaller work_mem for ways: prevents the 3-way JOIN
                                # from blowing up postgres's memory (OOM-killed at 256MB)

# Strategy: dedup tmp_nodes once into tmp_nodes_unique, then INSERT
# via LEFT JOIN anti-join. PostgreSQL builds a HashJoin against
# ways_vertices_pgr.osm_id (one table scan), not per-row index probes
# — empirically 10-100x faster than ON CONFLICT for 100M-row inserts
# against a 30M+ row target. Trade-off: dedup pass costs ~5-10 min
# upfront. Worth it.


def _resolve_into_final(conn: psycopg.Connection) -> None:
    """Promote staging rows into final tables via anti-join INSERTs.

    Restart-survivable: each chunk runs in its own transaction so a
    WSL pause mid-resolve costs at most one chunk. Re-running picks
    up automatically — the LEFT JOIN anti-join naturally skips rows
    that already exist, and `tmp_nodes_unique` is preserved across
    runs (DROP TABLE IF EXISTS is idempotent on resume).

    Two chunked phases:
      1. Vertex insert via tmp_nodes_unique LEFT JOIN ways_vertices_pgr
      2. Ways insert via tmp_edges JOIN ways_vertices_pgr LEFT JOIN ways
    """
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE ways ADD COLUMN IF NOT EXISTS "
            "bike_excluded boolean NOT NULL DEFAULT false"
        )
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS ways_dedup_idx
            ON ways (osm_way_id, source, target)
        """)
    conn.commit()

    # ── one-shot dedup of tmp_nodes ──────────────────────────────────
    # Idempotent: drops + recreates each run. ~5-10 min for 100M rows.
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM tmp_nodes")
        n_tmp = cur.fetchone()[0]
    if n_tmp == 0:
        print("[ingest] tmp_nodes empty, skipping vertex insert", flush=True)
    else:
        # Resume optimization: if tmp_nodes_unique already exists from a
        # prior crashed run, reuse it (saves 5-10 min on re-dedup).
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_class WHERE relname='tmp_nodes_unique'"
            )
            exists = cur.fetchone() is not None
            if exists:
                cur.execute("SELECT count(*) FROM tmp_nodes_unique")
                n_unique = cur.fetchone()[0]
                if n_unique == 0:
                    cur.execute("DROP TABLE tmp_nodes_unique")
                    exists = False
        if exists:
            print(f"[ingest] reusing existing tmp_nodes_unique ({n_unique:,} rows)",
                  flush=True)
        else:
            t0 = time.time()
            print(f"[ingest] deduping {n_tmp:,} tmp_nodes → tmp_nodes_unique ...",
                  flush=True)
            with conn.cursor() as cur:
                cur.execute(f"SET work_mem = '{_HASH_WORK_MEM}'")
                cur.execute("SET max_parallel_workers_per_gather = 2")
                cur.execute("""
                    CREATE TABLE tmp_nodes_unique AS
                    SELECT osm_id, MIN(lon) AS lon, MIN(lat) AS lat
                      FROM tmp_nodes
                     GROUP BY osm_id
                """)
                cur.execute(
                    "ALTER TABLE tmp_nodes_unique ADD PRIMARY KEY (osm_id)"
                )
                cur.execute("ANALYZE tmp_nodes_unique")
                cur.execute("SELECT count(*) FROM tmp_nodes_unique")
                n_unique = cur.fetchone()[0]
            conn.commit()
            print(f"[ingest]   tmp_nodes_unique: {n_unique:,} unique osm_ids "
                  f"in {time.time()-t0:.1f}s", flush=True)

        # ── resume idempotency: skip vertex insert if previously done ──
        # We mark vertex-insert completion by ensuring both the GIST
        # and null-elev indexes are present (we drop them at start,
        # rebuild at end — so both-present means we finished cleanly).
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM pg_indexes
                 WHERE tablename = 'ways_vertices_pgr'
                   AND indexname IN ('ways_vertices_pgr_geom_idx',
                                     'ways_vertices_null_elev_idx')
            """)
            n_vertex_indexes = cur.fetchone()[0]
        if n_vertex_indexes == 2:
            print("[ingest] vertex insert previously completed "
                  "(both secondary indexes present), skipping",
                  flush=True)
            # Skip to ways insert (jump past the vertex insert block).
            # We do this by setting a sentinel; see end of vertex block.
            _skip_vertex_insert = True
        else:
            _skip_vertex_insert = False

        if not _skip_vertex_insert:
            # CRITICAL: drop the GIST index on ways_vertices_pgr.the_geom
            # before insert. Each row INSERT would otherwise update the
            # GIST tree (~1-2 ms each) — on 42M new rows that's 17+
            # hours just for index maintenance. Same for the partial
            # elev_m index. One-shot CREATE INDEX at the end is ~10-30
            # min for the whole table.
            print("[ingest] dropping ways_vertices_pgr secondary indexes "
                  "(geom GIST + null-elev) for fast insert ...", flush=True)
            t0 = time.time()
            with conn.cursor() as cur:
                cur.execute("DROP INDEX IF EXISTS ways_vertices_pgr_geom_idx")
                cur.execute("DROP INDEX IF EXISTS ways_vertices_null_elev_idx")
            conn.commit()
            print(f"[ingest]   indexes dropped in {time.time()-t0:.1f}s",
                  flush=True)

            print(f"[ingest] vertex anti-join INSERT (single-shot): "
                  f"{n_unique:,} rows ...", flush=True)
            t0 = time.time()
            with conn.cursor() as cur:
                cur.execute(f"SET work_mem = '{_HASH_WORK_MEM}'")
                cur.execute("SET max_parallel_workers_per_gather = 4")
                cur.execute("""
                    INSERT INTO ways_vertices_pgr (osm_id, lon, lat)
                    SELECT u.osm_id, u.lon, u.lat
                      FROM tmp_nodes_unique u
                      LEFT JOIN ways_vertices_pgr v ON v.osm_id = u.osm_id
                     WHERE v.osm_id IS NULL
                """)
                rc = cur.rowcount
            conn.commit()
            print(f"[ingest]   ways_vertices_pgr DONE: +{rc:,} new rows in "
                  f"{time.time()-t0:.1f}s", flush=True)

            print("[ingest] rebuilding ways_vertices_pgr_geom_idx (GIST) "
                  "and null-elev index ...", flush=True)
            t0 = time.time()
            with conn.cursor() as cur:
                cur.execute(f"SET maintenance_work_mem = '1GB'")
                cur.execute("""
                    CREATE INDEX ways_vertices_pgr_geom_idx
                    ON ways_vertices_pgr USING gist (the_geom)
                """)
                cur.execute("""
                    CREATE INDEX ways_vertices_null_elev_idx
                    ON ways_vertices_pgr (lat, lon) WHERE elev_m IS NULL
                """)
            conn.commit()
            print(f"[ingest]   indexes rebuilt in {time.time()-t0:.1f}s",
                  flush=True)

    # ── ways insert: chunked by tmp_edges._seq ───────────────────────
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM ways")
        incremental = int(cur.fetchone()[0]) > 0
        cur.execute("SELECT min(_seq), max(_seq), count(*) FROM tmp_edges")
        e_lo, e_hi, e_total = cur.fetchone()
    if e_total == 0:
        print("[ingest] tmp_edges empty, skipping ways insert", flush=True)
    elif incremental:
        # Drop source/target btree indexes; keep ways_dedup_idx (needed
        # for fast LEFT JOIN anti-join probes) and ways_pkey.
        print("[ingest] dropping ways source/target indexes for fast insert ...",
              flush=True)
        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS ways_source_idx")
            cur.execute("DROP INDEX IF EXISTS ways_target_idx")
        conn.commit()
        print(f"[ingest]   indexes dropped in {time.time()-t0:.1f}s", flush=True)

        # CHUNKED ways insert — single-shot triggered OOM on the 3-way
        # JOIN (tmp_edges × ways_vertices_pgr×2 × ways anti-join). Per
        # chunk: hash 2M tmp_edges (~150 MB) and probe via index on
        # the others. Memory bounded, ~1-3 min per chunk.
        n_chunks = max(1, (e_total + _EDGE_BATCH - 1) // _EDGE_BATCH)
        print(f"[ingest] ways anti-join INSERT (chunked): {e_total:,} "
              f"rows in {n_chunks} chunks of ~{_EDGE_BATCH:,}", flush=True)

        # Disable autovacuum on `ways` during insert: it'd race with
        # our INSERTs, hold locks, and eat memory we don't have.
        # Re-enabled at end.
        print("[ingest] disabling autovacuum on ways during insert ...",
              flush=True)
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE ways SET (autovacuum_enabled = false)")
        conn.commit()

        total_new = 0
        cur_lo = e_lo
        chunk = 0
        while cur_lo <= e_hi:
            cur_hi = cur_lo + _EDGE_BATCH
            t0 = time.time()
            with conn.cursor() as cur:
                cur.execute(f"SET work_mem = '{_EDGE_WORK_MEM}'")
                cur.execute("SET max_parallel_workers_per_gather = 0")
                cur.execute(f"""
                    INSERT INTO ways ({_INSERT_COLUMNS})
                    SELECT {_SELECT_COLUMNS}
                      FROM tmp_edges e
                      JOIN ways_vertices_pgr v_src ON v_src.osm_id = e.source_osm
                      JOIN ways_vertices_pgr v_dst ON v_dst.osm_id = e.target_osm
                      LEFT JOIN ways w ON w.osm_way_id = e.osm_way_id
                                      AND w.source     = v_src.id
                                      AND w.target     = v_dst.id
                     WHERE e._seq >= %s AND e._seq < %s
                       AND e.length_m > 0
                       AND w.gid IS NULL
                """, (cur_lo, cur_hi))
                rc = cur.rowcount
            conn.commit()
            total_new += rc
            chunk += 1
            print(f"[ingest]   echunk {chunk:>3}/{n_chunks} _seq<{cur_hi:>10}: "
                  f"+{rc:>9,} new (total +{total_new:,}) in "
                  f"{time.time()-t0:.1f}s", flush=True)
            cur_lo = cur_hi
        print(f"[ingest]   ways DONE: +{total_new:,} new rows total",
              flush=True)

        # Re-enable autovacuum on ways
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE ways RESET (autovacuum_enabled)")
        conn.commit()

        print("[ingest] rebuilding ways source/target indexes ...", flush=True)
        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute(f"SET maintenance_work_mem = '1GB'")
            cur.execute("CREATE INDEX ways_source_idx ON ways(source)")
            cur.execute("CREATE INDEX ways_target_idx ON ways(target)")
        conn.commit()
        print(f"[ingest]   indexes rebuilt in {time.time()-t0:.1f}s",
              flush=True)
    else:
        # Bulk path: empty target table, drop indexes for fast load.
        # NOT chunked — only used on truly fresh ingest, and the single
        # INSERT is faster without index maintenance overhead.
        print("[ingest] bulk ways insert (empty target, no chunking)",
              flush=True)
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS ways_source_idx")
            cur.execute("DROP INDEX IF EXISTS ways_target_idx")
            cur.execute(f"""
                INSERT INTO ways ({_INSERT_COLUMNS})
                SELECT {_SELECT_COLUMNS}
                  FROM tmp_edges e
                  JOIN ways_vertices_pgr v_src ON v_src.osm_id = e.source_osm
                  JOIN ways_vertices_pgr v_dst ON v_dst.osm_id = e.target_osm
                 WHERE e.length_m > 0
            """)
            print(f"[ingest]   ways: {cur.rowcount:,} new rows", flush=True)
            cur.execute("CREATE INDEX ways_source_idx ON ways(source)")
            cur.execute("CREATE INDEX ways_target_idx ON ways(target)")
        conn.commit()

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS tmp_nodes_unique")
        cur.execute("DROP TABLE tmp_nodes")
        cur.execute("DROP TABLE tmp_edges")
    conn.commit()


def _diag_tmp_state(conn: psycopg.Connection, label: str) -> None:
    """Print whether tmp_nodes/tmp_edges currently exist + row counts."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT relname, relpersistence FROM pg_class "
            "WHERE relname IN ('tmp_nodes','tmp_edges')"
        )
        rels = cur.fetchall()
        n_nodes = n_edges = None
        for r in rels:
            try:
                cur.execute(f"SELECT count(*) FROM {r[0]}")
                if r[0] == "tmp_nodes":
                    n_nodes = cur.fetchone()[0]
                else:
                    n_edges = cur.fetchone()[0]
            except Exception as e:
                print(f"[diag {label}] {r[0]} count failed: {e}", flush=True)
    print(f"[diag {label}] pg_class={rels}  tmp_nodes={n_nodes}  tmp_edges={n_edges}",
          flush=True)


def ingest(conn: psycopg.Connection, pbfs: Iterable[Path]) -> None:
    """Top-level: clear staging, stream each PBF, then resolve into
    the pgRouting-backed final tables."""
    pbfs = list(pbfs)
    _diag_tmp_state(conn, "before-create")

    # Resume detection: if tmp_nodes/tmp_edges already contain data from
    # a previous crashed run, skip _create_staging + streaming and go
    # straight to resolve. Saves ~30 min of re-streaming for the DE PBF
    # when WSL kills the resolve mid-chunk.
    has_staged_data = False
    n_nodes_existing = n_edges_existing = 0
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM tmp_nodes")
            n_nodes_existing = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM tmp_edges")
            n_edges_existing = cur.fetchone()[0]
        has_staged_data = n_nodes_existing > 0 and n_edges_existing > 0
    except psycopg.errors.UndefinedTable:
        conn.rollback()

    if has_staged_data:
        print(f"[ingest] RESUME: existing tmp_nodes={n_nodes_existing:,} "
              f"tmp_edges={n_edges_existing:,} — skipping create_staging "
              f"+ streaming, going straight to resolve", flush=True)
    else:
        with conn.cursor() as cur:
            _create_staging(cur)
        conn.commit()
        _diag_tmp_state(conn, "after-create-commit")

        for pbf in pbfs:
            if not pbf.exists():
                raise SystemExit(f"missing PBF: {pbf}")
            _stream_pbf(conn, pbf)
            conn.commit()
            _diag_tmp_state(conn, f"after-stream-{pbf.stem}-commit")

    _diag_tmp_state(conn, "before-resolve")
    _resolve_into_final(conn)
    # _resolve_into_final commits per chunk; this final commit is a no-op
    # but kept for symmetry with the rest of the pipeline.
    conn.commit()
