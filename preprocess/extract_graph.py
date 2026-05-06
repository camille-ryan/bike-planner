"""Extract a routable bike graph from an OSM PBF (streaming, numpy-buffered).

Uses pyosmium (libosmium Python bindings) to stream the PBF without
materializing pandas DataFrames. Per-way data is written directly into
preallocated numpy buffers that grow by doubling — no Python dict/list
overhead. Compared with the previous dict + list extractor, peak
memory drops from ~7 GB to ~1.5 GB on Austria, which is what lets us
scale to corridor without OOMing.

Pipeline:
  - apply_file streams the PBF; the way handler writes (osm_node_id,
    lon, lat) and (u_osm, v_osm, length, cost_factor, oneway_dir) to
    growable numpy arrays.
  - finalize: sort + dedupe node_ids (one node may appear in many
    ways), reindex edges via np.searchsorted, expand edges into
    forward + reverse pairs based on oneway tag.

We deliberately do not handle elevation in v1 — see the long-distance
optimization memory and PERF_NOTES. v1 cost function is in cost.py.
"""
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import osmium

from cost import bike_edge_cost, EXCLUDE


@dataclass
class Graph:
    """Densely-indexed routable bike graph.

    node_lon/lat/osm_id: arrays of length N indexed by dense node id.
    edge_src/dst: int32 arrays of length M (directed adjacency).
    edge_cost: per-edge cost (cost units; length × cost-factor).
    edge_length_m: metric length kept separately for path summaries.
    """
    node_lon:      np.ndarray  # float32, shape (N,)
    node_lat:      np.ndarray  # float32, shape (N,)
    node_osm_id:   np.ndarray  # int64,   shape (N,)
    edge_src:      np.ndarray  # int32,   shape (M,)
    edge_dst:      np.ndarray  # int32,   shape (M,)
    edge_cost:     np.ndarray  # float32, shape (M,)
    edge_length_m: np.ndarray  # float32, shape (M,)


_EARTH_R = 6_371_000.0


def _haversine(lon1, lat1, lon2, lat2):
    rl1, rl2 = radians(lat1), radians(lat2)
    dl = radians(lat2 - lat1)
    dn = radians(lon2 - lon1)
    a = sin(dl / 2) ** 2 + cos(rl1) * cos(rl2) * sin(dn / 2) ** 2
    return 2 * _EARTH_R * asin(sqrt(a))


class _Extractor(osmium.SimpleHandler):
    """Stream pass with growable numpy buffers.

    Initial sizes are educated guesses for Austria-scale (10M nodes,
    20M raw edges). Buffers double when full, so we pay an O(N) copy
    per doubling — total work is amortized O(N), peak memory is 2x
    final size during the doubling step.
    """

    _INIT_NODES = 10_000_000
    _INIT_EDGES = 20_000_000

    def __init__(self):
        super().__init__()
        self._osm_ids = np.empty(self._INIT_NODES, dtype=np.int64)
        self._lon     = np.empty(self._INIT_NODES, dtype=np.float32)
        self._lat     = np.empty(self._INIT_NODES, dtype=np.float32)
        self._n_nodes = 0

        self._edge_u   = np.empty(self._INIT_EDGES, dtype=np.int64)
        self._edge_v   = np.empty(self._INIT_EDGES, dtype=np.int64)
        self._edge_len = np.empty(self._INIT_EDGES, dtype=np.float32)
        self._edge_cf  = np.empty(self._INIT_EDGES, dtype=np.float32)
        self._edge_ow  = np.empty(self._INIT_EDGES, dtype=np.int8)
        self._n_edges  = 0

        self._skipped_no_highway = 0
        self._skipped_excluded   = 0
        self._skipped_no_cost    = 0

    def _grow_nodes(self) -> None:
        new = len(self._osm_ids) * 2
        n = self._n_nodes
        a = np.empty(new, dtype=np.int64);   a[:n] = self._osm_ids[:n]; self._osm_ids = a
        b = np.empty(new, dtype=np.float32); b[:n] = self._lon[:n];     self._lon = b
        c = np.empty(new, dtype=np.float32); c[:n] = self._lat[:n];     self._lat = c

    def _grow_edges(self) -> None:
        new = len(self._edge_u) * 2
        n = self._n_edges
        a = np.empty(new, dtype=np.int64);   a[:n] = self._edge_u[:n];   self._edge_u = a
        b = np.empty(new, dtype=np.int64);   b[:n] = self._edge_v[:n];   self._edge_v = b
        c = np.empty(new, dtype=np.float32); c[:n] = self._edge_len[:n]; self._edge_len = c
        d = np.empty(new, dtype=np.float32); d[:n] = self._edge_cf[:n];  self._edge_cf = d
        e = np.empty(new, dtype=np.int8);    e[:n] = self._edge_ow[:n];  self._edge_ow = e

    def way(self, w):
        tags = w.tags
        hw = tags.get("highway") or ""
        is_ferry = tags.get("route") == "ferry"
        # Accept either highway=* OR route=ferry. Without the ferry
        # branch the city graph splits at the Baltic and Denmark is
        # unreachable from Germany under the SPT engine.
        if not hw and not is_ferry:
            self._skipped_no_highway += 1
            return
        if hw in EXCLUDE:
            self._skipped_excluded += 1
            return
        cost_factor = bike_edge_cost(
            highway=hw,
            surface=tags.get("surface", "") or "",
            tracktype=tags.get("tracktype", "") or "",
            bicycle=tags.get("bicycle", "") or "",
            cycleway=tags.get("cycleway", "") or "",
            access=tags.get("access", "") or "",
            bicycle_road=tags.get("bicycle_road", "") or "",
            is_ferry=is_ferry,
        )
        if cost_factor is None:
            self._skipped_no_cost += 1
            return

        ow = (tags.get("oneway") or "").lower()
        if ow in ("-1", "reverse"):
            oneway_dir = -1
        elif ow in ("yes", "true", "1"):
            oneway_dir = 1
        else:
            oneway_dir = 0

        prev_id = -1
        prev_lon = prev_lat = 0.0
        for node in w.nodes:
            try:
                lon = node.location.lon
                lat = node.location.lat
            except (osmium.InvalidLocationError, RuntimeError):
                # Border-clipped node referenced by a way that crosses the
                # PBF boundary. Skip; the next iteration starts a new chain.
                prev_id = -1
                continue

            if self._n_nodes >= len(self._osm_ids):
                self._grow_nodes()
            i = self._n_nodes
            self._osm_ids[i] = node.ref
            self._lon[i]     = lon
            self._lat[i]     = lat
            self._n_nodes   += 1

            if prev_id >= 0:
                length_m = _haversine(prev_lon, prev_lat, lon, lat)
                if length_m > 0:
                    if self._n_edges >= len(self._edge_u):
                        self._grow_edges()
                    j = self._n_edges
                    self._edge_u[j]   = prev_id
                    self._edge_v[j]   = node.ref
                    self._edge_len[j] = length_m
                    self._edge_cf[j]  = cost_factor
                    self._edge_ow[j]  = oneway_dir
                    self._n_edges    += 1
            prev_id  = int(node.ref)
            prev_lon = lon
            prev_lat = lat


def extract(pbf: Path) -> Graph:
    print(f"[extract] streaming {pbf.name}")
    h = _Extractor()
    h.apply_file(str(pbf), locations=True, idx="flex_mem")
    print(f"[extract] raw: nodes_seen={h._n_nodes:,}  edges_seen={h._n_edges:,}  "
          f"skipped(no_hw={h._skipped_no_highway:,}, "
          f"excluded={h._skipped_excluded:,}, "
          f"no_cost={h._skipped_no_cost:,})")

    # Dedupe nodes: the same OSM id is appended for each way it appears
    # in. np.unique returns sorted unique values + the first index of
    # each, which doubles as our coord-source mapping.
    osm_ids_raw = h._osm_ids[: h._n_nodes]
    lon_raw     = h._lon[: h._n_nodes]
    lat_raw     = h._lat[: h._n_nodes]
    unique_osm_ids, first_idx = np.unique(osm_ids_raw, return_index=True)
    node_lon = lon_raw[first_idx]
    node_lat = lat_raw[first_idx]

    # Drop the streaming-time arrays before edge work — keeps peak
    # memory bounded while we build the larger output edge tables.
    h._osm_ids = h._lon = h._lat = None  # noqa: E501

    # Reindex edges: u_osm/v_osm -> dense u_idx/v_idx via searchsorted
    # on the sorted unique_osm_ids.
    u_osm = h._edge_u[: h._n_edges]
    v_osm = h._edge_v[: h._n_edges]
    u_idx = np.searchsorted(unique_osm_ids, u_osm).astype(np.int32)
    v_idx = np.searchsorted(unique_osm_ids, v_osm).astype(np.int32)
    n_unique = len(unique_osm_ids)
    # Some edge endpoints may reference border-clipped nodes that aren't
    # in our node table; drop those edges (rare in practice).
    valid = (u_idx < n_unique) & (v_idx < n_unique)
    valid[valid] = (
        unique_osm_ids[u_idx[valid]] == u_osm[valid]
    ) & (
        unique_osm_ids[v_idx[valid]] == v_osm[valid]
    )
    u_idx = u_idx[valid]
    v_idx = v_idx[valid]
    length = h._edge_len[: h._n_edges][valid]
    cost_factor = h._edge_cf[: h._n_edges][valid]
    oneway = h._edge_ow[: h._n_edges][valid]
    edge_cost = (length * cost_factor).astype(np.float32)

    # Drop the streaming-time edge arrays.
    h._edge_u = h._edge_v = h._edge_len = h._edge_cf = h._edge_ow = None

    # Expand into forward + reverse based on oneway. oneway==1 keeps
    # only u->v, oneway==-1 keeps only v->u, oneway==0 emits both.
    fwd_mask = oneway != -1
    rev_mask = oneway != 1

    src = np.concatenate([u_idx[fwd_mask], v_idx[rev_mask]]).astype(np.int32)
    dst = np.concatenate([v_idx[fwd_mask], u_idx[rev_mask]]).astype(np.int32)
    cost_out = np.concatenate([edge_cost[fwd_mask], edge_cost[rev_mask]]).astype(np.float32)
    len_out  = np.concatenate([length[fwd_mask],   length[rev_mask]]).astype(np.float32)

    print(f"[extract] graph: nodes={n_unique:,} directed_edges={len(src):,}")

    return Graph(
        node_lon=node_lon, node_lat=node_lat, node_osm_id=unique_osm_ids,
        edge_src=src, edge_dst=dst,
        edge_cost=cost_out, edge_length_m=len_out,
    )
