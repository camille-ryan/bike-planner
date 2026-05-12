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
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from pathlib import Path
from typing import Iterable

import osmium
import psycopg

from cost import bike_edge_cost, EXCLUDE


_EARTH_R = 6_371_000.0
_BATCH_SIZE = 100_000

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
    """Drop+recreate the staging tables. UNLOGGED skips WAL — fine
    because we either drop the data after merging into the final
    tables, or restart from scratch on failure."""
    cur.execute("DROP TABLE IF EXISTS tmp_nodes")
    cur.execute("DROP TABLE IF EXISTS tmp_edges")
    cur.execute("""
        CREATE UNLOGGED TABLE tmp_nodes (
            osm_id bigint NOT NULL,
            lon    double precision NOT NULL,
            lat    double precision NOT NULL
        )
    """)
    cur.execute("""
        CREATE UNLOGGED TABLE tmp_edges (
            osm_way_id   bigint,
            source_osm   bigint NOT NULL,
            target_osm   bigint NOT NULL,
            cost         double precision NOT NULL,
            reverse_cost double precision NOT NULL,
            length_m     double precision NOT NULL,
            is_ferry     boolean NOT NULL,
            highway      text NOT NULL,
            surface      text NOT NULL,
            tracktype    text NOT NULL,
            oneway       text NOT NULL,
            bicycle      text NOT NULL,
            cycleway     text NOT NULL,
            bicycle_road text NOT NULL,
            access       text NOT NULL,
            curv_fwd     real NOT NULL,
            curv_rev     real NOT NULL
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
        self.skipped_no_highway = 0
        self.skipped_excluded   = 0
        self.skipped_no_cost    = 0
        self.nodes_written      = 0
        self.edges_written      = 0

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
                "cost, reverse_cost, length_m, is_ferry, "
                "highway, surface, tracktype, oneway, bicycle, "
                "cycleway, bicycle_road, access, curv_fwd, curv_rev) FROM STDIN"
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
        if hw in EXCLUDE:
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
        if cost_factor is None:
            self.skipped_no_cost += 1
            return

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

        # Per-segment directional curvature (windowed; see _segment_curvatures).
        seg_curvs = _segment_curvatures(nodes)

        # Pass 2: emit nodes and per-segment edges. Each edge carries
        # its own (curv_fwd, curv_rev) so the cost recompute can
        # attribute upcoming curvature to whichever segment is
        # descending into it — not just to the curve itself.
        prev_id = None
        prev_lon = prev_lat = 0.0
        seg_idx = 0
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
                    cf, cr = (seg_curvs[seg_idx]
                              if seg_idx < len(seg_curvs)
                              else (0.0, 0.0))
                    self._edge_buf.append((
                        int(w.id), prev_id, osm_id,
                        cost, rev, float(length_m), bool(is_ferry),
                        hw, surface, tracktype, oneway_raw,
                        bicycle, cycleway, bicycle_road, access,
                        float(cf), float(cr),
                    ))
                seg_idx += 1
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
          f"no_cost={h.skipped_no_cost:,})")
    return {
        "nodes": h.nodes_written, "edges": h.edges_written,
    }


_INSERT_COLUMNS = (
    "osm_way_id, source, target, cost, reverse_cost, length_m, is_ferry, "
    "highway, surface, tracktype, oneway, bicycle, cycleway, bicycle_road, "
    "access, curv_fwd, curv_rev"
)
_SELECT_COLUMNS = (
    "e.osm_way_id, v_src.id, v_dst.id, e.cost, e.reverse_cost, e.length_m, "
    "e.is_ferry, e.highway, e.surface, e.tracktype, e.oneway, e.bicycle, "
    "e.cycleway, e.bicycle_road, e.access, e.curv_fwd, e.curv_rev"
)


def _resolve_into_final(cur):
    """Promote staging rows into the final pgRouting-backed tables.

    Performance levers (corridor scale: 240M edges):
      - No FK constraints on ways.source/target — those would force
        per-row index lookups during the INSERT and add ~10x runtime.
        Integrity is guaranteed by the JOINs below, which only emit
        rows whose endpoints exist in ways_vertices_pgr.
      - Drop ways_source_idx / ways_target_idx before INSERT, rebuild
        after. Postgres maintaining a btree during 240M inserts is
        far slower than a single CREATE INDEX from scratch.
    """
    print("[ingest] dedup + insert into ways_vertices_pgr...")
    cur.execute("""
        INSERT INTO ways_vertices_pgr (osm_id, lon, lat)
        SELECT DISTINCT ON (osm_id) osm_id, lon, lat
        FROM tmp_nodes
        ORDER BY osm_id
        ON CONFLICT (osm_id) DO NOTHING
    """)
    print(f"[ingest]   ways_vertices_pgr: {cur.rowcount:,} new rows")

    cur.execute("SELECT COUNT(*) FROM ways")
    incremental = int(cur.fetchone()[0]) > 0

    if incremental:
        # Corridor-extension path: existing graph already in `ways`.
        # Cross-border ways appear in multiple Geofabrik extracts, so
        # ON CONFLICT skips dupes. Keep source/target indexes in place;
        # the unique dedup index is what powers ON CONFLICT and a
        # 5-10% incremental insert isn't worth the index rebuild cost.
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS ways_dedup_idx
            ON ways (osm_way_id, source, target)
        """)
        print("[ingest] incremental insert into ways (dedup via ON CONFLICT)...")
        cur.execute(f"""
            INSERT INTO ways ({_INSERT_COLUMNS})
            SELECT {_SELECT_COLUMNS}
            FROM tmp_edges e
            JOIN ways_vertices_pgr v_src ON v_src.osm_id = e.source_osm
            JOIN ways_vertices_pgr v_dst ON v_dst.osm_id = e.target_osm
            WHERE e.length_m > 0
            ON CONFLICT (osm_way_id, source, target) DO NOTHING
        """)
        print(f"[ingest]   ways: {cur.rowcount:,} new rows (dupes skipped)")
    else:
        print("[ingest] dropping ways indexes for fast bulk insert...")
        cur.execute("DROP INDEX IF EXISTS ways_source_idx")
        cur.execute("DROP INDEX IF EXISTS ways_target_idx")

        print("[ingest] resolve ids + insert into ways (no indexes, no FKs)...")
        cur.execute(f"""
            INSERT INTO ways ({_INSERT_COLUMNS})
            SELECT {_SELECT_COLUMNS}
            FROM tmp_edges e
            JOIN ways_vertices_pgr v_src ON v_src.osm_id = e.source_osm
            JOIN ways_vertices_pgr v_dst ON v_dst.osm_id = e.target_osm
            WHERE e.length_m > 0
        """)
        print(f"[ingest]   ways: {cur.rowcount:,} new rows")

        print("[ingest] rebuilding ways indexes...")
        cur.execute("CREATE INDEX ways_source_idx ON ways(source)")
        cur.execute("CREATE INDEX ways_target_idx ON ways(target)")

    cur.execute("DROP TABLE tmp_nodes")
    cur.execute("DROP TABLE tmp_edges")


def ingest(conn: psycopg.Connection, pbfs: Iterable[Path]) -> None:
    """Top-level: clear staging, stream each PBF, then resolve into
    the pgRouting-backed final tables."""
    pbfs = list(pbfs)
    with conn.cursor() as cur:
        _create_staging(cur)
    conn.commit()

    for pbf in pbfs:
        if not pbf.exists():
            raise SystemExit(f"missing PBF: {pbf}")
        _stream_pbf(conn, pbf)
        conn.commit()

    with conn.cursor() as cur:
        _resolve_into_final(cur)
    conn.commit()
