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
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Iterable

import osmium
import psycopg

from cost import bike_edge_cost, EXCLUDE


_EARTH_R = 6_371_000.0
_BATCH_SIZE = 100_000

# Sinuosity clamp range. Endpoint-distance near zero (closed loops) or
# tiny ways with degenerate endpoints would otherwise blow up.
_SINUOSITY_MIN = 1.0
_SINUOSITY_MAX = 5.0

# Window over which per-segment sinuosity is measured. A pure per-way
# sinuosity would average a 1 km switchback section into a 10 km way's
# overall ~1.0 sinuosity, washing out the signal exactly where the
# cost function needs it. Computing sinuosity in a small polyline
# window around each segment localizes the metric. 300 m is roughly
# the scale of an alpine switchback group; smaller windows pick up
# noise from minor wiggles, larger ones re-introduce the averaging
# problem.
_SINUOSITY_WINDOW_M = 300.0


def _haversine(lon1, lat1, lon2, lat2):
    rl1, rl2 = radians(lat1), radians(lat2)
    dl = radians(lat2 - lat1)
    dn = radians(lon2 - lon1)
    a = sin(dl / 2) ** 2 + cos(rl1) * cos(rl2) * sin(dn / 2) ** 2
    return 2 * _EARTH_R * asin(sqrt(a))


def _segment_sinuosities(nodes: list[tuple[int, float, float]],
                         window_m: float = _SINUOSITY_WINDOW_M
                         ) -> list[float]:
    """Per-segment sinuosity over a polyline window centered on each segment.

    For each of the N-1 segments in `nodes`, returns
        actual_polyline_length / straight_line_distance
    computed over a window of ~`window_m` polyline meters centered on
    the segment's midpoint, clamped to [_SINUOSITY_MIN, _SINUOSITY_MAX].

    O(N) per way (two-pointer window advancement).
    """
    n = len(nodes)
    if n < 2:
        return []

    # Cumulative polyline length per node: cumlen[i] = polyline distance
    # from nodes[0] to nodes[i]. cumlen[0] == 0; cumlen[n-1] == total.
    cumlen = [0.0] * n
    for i in range(n - 1):
        _, ln1, lt1 = nodes[i]
        _, ln2, lt2 = nodes[i + 1]
        cumlen[i + 1] = cumlen[i] + _haversine(ln1, lt1, ln2, lt2)
    total_length = cumlen[-1]

    # Way is shorter than ~1.5×window — local-vs-global distinction
    # doesn't apply; fall back to whole-way sinuosity for every segment.
    if total_length <= window_m * 1.5:
        end_dist = _haversine(nodes[0][1], nodes[0][2],
                              nodes[-1][1], nodes[-1][2])
        if end_dist <= 0.0:
            sin_global = _SINUOSITY_MAX
        else:
            sin_global = max(_SINUOSITY_MIN,
                             min(_SINUOSITY_MAX, total_length / end_dist))
        return [sin_global] * (n - 1)

    half = window_m / 2.0
    a = 0           # window start node index
    b = 0           # window end node index
    out: list[float] = []
    for i in range(n - 1):
        mid = (cumlen[i] + cumlen[i + 1]) * 0.5
        lo = mid - half
        hi = mid + half
        # Advance window-start until cumlen[a] <= lo and cumlen[a+1] > lo.
        # i.e. a is the latest node at or before the window's lower edge.
        while a + 1 < n and cumlen[a + 1] <= lo:
            a += 1
        # Advance window-end until cumlen[b] >= hi (or hit the end).
        while b + 1 < n and cumlen[b] < hi:
            b += 1

        if b <= a:
            # Degenerate window (shouldn't really happen given the
            # short-way fallback above, but be defensive).
            out.append(_SINUOSITY_MIN)
            continue
        win_length = cumlen[b] - cumlen[a]
        win_endpoint = _haversine(
            nodes[a][1], nodes[a][2],
            nodes[b][1], nodes[b][2],
        )
        if win_endpoint <= 0.0:
            out.append(_SINUOSITY_MAX)
        else:
            out.append(max(_SINUOSITY_MIN,
                           min(_SINUOSITY_MAX, win_length / win_endpoint)))
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
            sinuosity    real NOT NULL
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
                "cycleway, bicycle_road, access, sinuosity) FROM STDIN"
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

        # Per-segment sinuosity (windowed; see _segment_sinuosities).
        seg_sinuosities = _segment_sinuosities(nodes)

        # Pass 2: emit nodes and per-segment edges, each stamped with
        # its own locally-windowed sinuosity so a steep curvy 1 km of
        # a longer way isn't averaged away.
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
                    sinuosity = (seg_sinuosities[seg_idx]
                                 if seg_idx < len(seg_sinuosities)
                                 else _SINUOSITY_MIN)
                    self._edge_buf.append((
                        int(w.id), prev_id, osm_id,
                        cost, rev, float(length_m), bool(is_ferry),
                        hw, surface, tracktype, oneway_raw,
                        bicycle, cycleway, bicycle_road, access,
                        float(sinuosity),
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
    "access, sinuosity"
)
_SELECT_COLUMNS = (
    "e.osm_way_id, v_src.id, v_dst.id, e.cost, e.reverse_cost, e.length_m, "
    "e.is_ferry, e.highway, e.surface, e.tracktype, e.oneway, e.bicycle, "
    "e.cycleway, e.bicycle_road, e.access, e.sinuosity"
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
