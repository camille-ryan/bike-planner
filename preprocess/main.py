"""SPT preprocess orchestrator.

  python3 main.py --profile lht --countries austria
  python3 main.py --profile lht --countries austria,czech-republic,germany,denmark

For v1 the cost function is hard-coded in cost.py — `--profile` is just
the output-directory name so we can later run multiple profiles in
parallel. Once we add elevation + scenic terms the profile selection
will actually pick a cost function.

Pipeline:
  1. extract_graph.extract(pbf) -> Graph (per country, then merged)
  2. load anchor cities from pois.sqlite, snap to graph nodes
  3. spt.compute_spt(graph, city_node_ids) -> (forward, reverse) SPT
  4. cells.build_polygons + cells.build_city_graph
  5. save.write_all
"""
import argparse
import gc
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np

import config
import extract_graph
import spt
import cells
import save
import per_city_spt


def _save_country_graph(g: "extract_graph.Graph", path: Path) -> None:
    """Serialize a country's Graph for later mmap-loading at merge time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        node_lon=g.node_lon, node_lat=g.node_lat, node_osm_id=g.node_osm_id,
        edge_src=g.edge_src, edge_dst=g.edge_dst,
        edge_cost=g.edge_cost, edge_length_m=g.edge_length_m,
    )


def merge_from_disk(country_files: list[Path]) -> "extract_graph.Graph":
    """Memory-frugal merge: mmap-load each country's extract instead of
    holding all country Graph objects in RAM at once.

    For a single-country build, this still costs one in-RAM copy at the
    end (the merged graph). For corridor-scale (4 countries, 115M total
    nodes), the saving is ~4 GB — which is the difference between
    fitting in 14 GB and OOMing during the dedupe step.
    """
    if len(country_files) == 1:
        f = np.load(country_files[0], mmap_mode="r")
        return extract_graph.Graph(
            node_lon=np.array(f["node_lon"]),
            node_lat=np.array(f["node_lat"]),
            node_osm_id=np.array(f["node_osm_id"]),
            edge_src=np.array(f["edge_src"]),
            edge_dst=np.array(f["edge_dst"]),
            edge_cost=np.array(f["edge_cost"]),
            edge_length_m=np.array(f["edge_length_m"]),
        )

    # Pass 1: mmap each country, sum sizes for the final allocation.
    handles = [np.load(p, mmap_mode="r") for p in country_files]
    n_per = [int(h["node_lon"].shape[0]) for h in handles]
    e_per = [int(h["edge_src"].shape[0])  for h in handles]
    n_offsets = np.cumsum([0] + n_per)
    e_offsets = np.cumsum([0] + e_per)
    total_n = int(n_offsets[-1])
    total_e = int(e_offsets[-1])
    print(f"[merge] streaming {len(country_files)} countries: "
          f"nodes={total_n:,} edges={total_e:,}")

    # Allocate the concatenated arrays once. Memory ~= 2.7 GB for the
    # corridor's node side, plus ~5 GB for the edge side. The mmaps are
    # disk-backed, so the inputs cost ~0 RAM on top.
    node_lon    = np.empty(total_n, dtype=np.float32)
    node_lat    = np.empty(total_n, dtype=np.float32)
    node_osm_id = np.empty(total_n, dtype=np.int64)
    edge_src    = np.empty(total_e, dtype=np.int32)
    edge_dst    = np.empty(total_e, dtype=np.int32)
    edge_cost   = np.empty(total_e, dtype=np.float32)
    edge_length = np.empty(total_e, dtype=np.float32)

    for i, h in enumerate(handles):
        sN, eN = int(n_offsets[i]), int(n_offsets[i + 1])
        sE, eE = int(e_offsets[i]), int(e_offsets[i + 1])
        node_lon[sN:eN]    = h["node_lon"]
        node_lat[sN:eN]    = h["node_lat"]
        node_osm_id[sN:eN] = h["node_osm_id"]
        # Edges need their endpoints reindexed by the country's offset
        # in the merged node array (pre-dedupe).
        edge_src[sE:eE]    = np.asarray(h["edge_src"]) + sN
        edge_dst[sE:eE]    = np.asarray(h["edge_dst"]) + sN
        edge_cost[sE:eE]   = h["edge_cost"]
        edge_length[sE:eE] = h["edge_length_m"]
        print(f"[merge]   {country_files[i].stem}: nodes={n_per[i]:,} "
              f"edges={e_per[i]:,}")

    # Drop mmap handles before the dedupe to free file handles.
    del handles
    gc.collect()

    # Cross-country dedupe by OSM id. np.unique allocates a sorted
    # copy + the inverse map (~1 GB each at corridor scale) so peak
    # ramps here briefly.
    unique_osm, inv = np.unique(node_osm_id, return_inverse=True)
    new_n = len(unique_osm)
    duplicates = total_n - new_n
    print(f"[merge] cross-country dedupe: {total_n:,} -> {new_n:,} "
          f"nodes ({duplicates:,} merged at borders)")

    _, first_idx = np.unique(node_osm_id, return_index=True)
    node_lon_d = node_lon[first_idx]
    node_lat_d = node_lat[first_idx]
    # We don't need the un-deduped node arrays anymore — drop them
    # before the final edge reindex which allocates int32 dst arrays.
    del node_lon, node_lat, node_osm_id, first_idx
    gc.collect()

    edge_src_d = inv[edge_src].astype(np.int32)
    edge_dst_d = inv[edge_dst].astype(np.int32)
    del inv, edge_src, edge_dst
    gc.collect()

    return extract_graph.Graph(
        node_lon=node_lon_d, node_lat=node_lat_d, node_osm_id=unique_osm,
        edge_src=edge_src_d, edge_dst=edge_dst_d,
        edge_cost=edge_cost, edge_length_m=edge_length,
    )


def merge_graphs(graphs: list) -> "extract_graph.Graph":
    """Concat per-country Graphs and stitch shared border nodes.

    Geofabrik PBFs clip ways at country borders. A way that crosses the
    AT/CZ frontier appears in both PBFs, with the same OSM node IDs at
    the shared boundary. After we concatenate per-country graphs the
    same OSM node ends up at two different dense indices — routing
    can't cross the border without explicit merging.

    The fix: dedupe by OSM id across all countries, then reindex every
    edge to the canonical dense index for its endpoints.
    """
    if len(graphs) == 1:
        return graphs[0]

    n_offsets = np.cumsum([0] + [len(g.node_lon) for g in graphs])
    total_n = int(n_offsets[-1])

    # Step 1: simple concatenation (with offsets), same as before.
    node_lon_concat    = np.empty(total_n, dtype=np.float32)
    node_lat_concat    = np.empty(total_n, dtype=np.float32)
    node_osm_id_concat = np.empty(total_n, dtype=np.int64)
    edge_src_list = []
    edge_dst_list = []
    edge_cost_list = []
    edge_len_list = []
    for i, g in enumerate(graphs):
        s = int(n_offsets[i]); e = int(n_offsets[i + 1])
        node_lon_concat[s:e]    = g.node_lon
        node_lat_concat[s:e]    = g.node_lat
        node_osm_id_concat[s:e] = g.node_osm_id
        edge_src_list.append(g.edge_src + s)
        edge_dst_list.append(g.edge_dst + s)
        edge_cost_list.append(g.edge_cost)
        edge_len_list.append(g.edge_length_m)
    edge_src  = np.concatenate(edge_src_list)
    edge_dst  = np.concatenate(edge_dst_list)
    edge_cost = np.concatenate(edge_cost_list)
    edge_len  = np.concatenate(edge_len_list)

    # Step 2: dedupe by OSM id. np.unique returns sorted-unique values
    # plus a mapping `inv` such that node_osm_id_concat == unique[inv].
    unique_osm, inv = np.unique(node_osm_id_concat, return_inverse=True)
    new_n = len(unique_osm)
    duplicates = total_n - new_n
    print(f"[merge] cross-country dedupe: {total_n:,} -> {new_n:,} "
          f"nodes ({duplicates:,} merged at borders)")

    # Step 3: pick coordinates from the *first* occurrence of each unique
    # id (np.unique with return_index=True). For border nodes that's
    # arbitrary but consistent — coords differ negligibly between PBFs.
    _, first_idx = np.unique(node_osm_id_concat, return_index=True)
    node_lon = node_lon_concat[first_idx]
    node_lat = node_lat_concat[first_idx]

    # Step 4: reindex edges. inv maps old dense index -> new dense index.
    edge_src_new = inv[edge_src].astype(np.int32)
    edge_dst_new = inv[edge_dst].astype(np.int32)

    return extract_graph.Graph(
        node_lon=node_lon, node_lat=node_lat, node_osm_id=unique_osm,
        edge_src=edge_src_new, edge_dst=edge_dst_new,
        edge_cost=edge_cost, edge_length_m=edge_len,
    )


def load_anchors_from_pois(db: Path, countries: list[str]) -> list[dict]:
    """Pull the anchor rows for the requested countries.

    Returns dicts shaped for the saved cities.json: name, place,
    population, country, lon, lat. The graph-node mapping is added
    after snapping.
    """
    placeholders = ",".join("?" for _ in countries)
    sql = (
        "SELECT name, place, population, country, X(geom) AS lon, Y(geom) AS lat, osm_id "
        f"FROM anchors WHERE country IN ({placeholders}) "
        "AND name IS NOT NULL"
    )
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")
    rows = conn.execute(sql, countries).fetchall()
    conn.close()
    out = []
    for name, place, pop, country, lon, lat, osm_id in rows:
        out.append({
            "name": name, "place": place, "population": pop,
            "country": country, "lon": float(lon), "lat": float(lat),
            "osm_id": osm_id,
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="lht",
                   help="Output dir name under data/spt/. Cost function is "
                        "currently hard-coded in cost.py irrespective of profile.")
    p.add_argument("--countries", default="austria",
                   help="Comma-separated country list; PBFs must already be in data/osm/.")
    args = p.parse_args()

    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    final_dir = config.SPT_DIR / args.profile
    # Stage output under <profile>.tmp/ and atomic-swap at the end.
    # If <profile>.tmp/ already exists from a previous run that died
    # mid-way, we keep its checkpointed files (per-country extracts,
    # merged graph, global SPT) and skip the phases that produced them.
    # Re-running picks up where the prior failure left off.
    staging_dir = config.SPT_DIR / (args.profile + ".tmp")
    staging_dir.mkdir(parents=True, exist_ok=True)
    out_dir = staging_dir
    if any(staging_dir.iterdir()):
        print(f"[preprocess] resuming from {staging_dir} (existing checkpoints preserved)")
    else:
        print(f"[preprocess] profile={args.profile} countries={countries} -> {final_dir} (via {staging_dir})")

    # Stage 1: per-country extracts (resumable: skip countries already on disk).
    extract_dir = out_dir / "_extracts"
    extract_dir.mkdir(parents=True, exist_ok=True)
    extract_files: list[Path] = []
    merged_path = out_dir / "_merged.npz"
    if not merged_path.exists():
        for c in countries:
            f = extract_dir / f"{c}.npz"
            extract_files.append(f)
            if f.exists():
                print(f"[extract] resume: {c} already extracted -> {f.name}")
                continue
            pbf = config.OSM_DIR / f"{c}-latest.osm.pbf"
            if not pbf.exists():
                raise SystemExit(f"missing PBF: {pbf}")
            g = extract_graph.extract(pbf)
            _save_country_graph(g, f)
            del g
            gc.collect()

    # Stage 2: cross-country merge (resumable: load `_merged.npz` if it
    # exists, otherwise build from per-country extracts). The merged
    # graph is the input to all later stages.
    if merged_path.exists():
        print(f"[merge] resume: loading prior merged graph from {merged_path.name}")
        m = np.load(str(merged_path))
        graph = extract_graph.Graph(
            node_lon=m["node_lon"], node_lat=m["node_lat"],
            node_osm_id=m["node_osm_id"],
            edge_src=m["edge_src"], edge_dst=m["edge_dst"],
            edge_cost=m["edge_cost"], edge_length_m=m["edge_length_m"],
        )
    else:
        graph = merge_from_disk(extract_files)
        np.savez(
            str(merged_path),
            node_lon=graph.node_lon, node_lat=graph.node_lat,
            node_osm_id=graph.node_osm_id,
            edge_src=graph.edge_src, edge_dst=graph.edge_dst,
            edge_cost=graph.edge_cost, edge_length_m=graph.edge_length_m,
        )
        # Per-country extracts are folded into the merged graph; drop them.
        shutil.rmtree(extract_dir, ignore_errors=True)
        print(f"[merge] spilled merged graph to {merged_path.name}")
    print(f"[preprocess] merged graph: nodes={len(graph.node_lon):,} edges={len(graph.edge_src):,}")

    anchors = load_anchors_from_pois(config.POIS_DB, countries)
    print(f"[preprocess] loaded {len(anchors)} anchors from {config.POIS_DB}")

    city_lons = np.array([a["lon"] for a in anchors], dtype=np.float32)
    city_lats = np.array([a["lat"] for a in anchors], dtype=np.float32)
    city_node_ids = spt.snap_cities_to_nodes(city_lons, city_lats,
                                             graph.node_lon, graph.node_lat)
    for i, a in enumerate(anchors):
        a["node_idx"] = int(city_node_ids[i])

    # Stage 3: global multi-source SPT (resumable: load if checkpoint exists).
    # This is the most likely failure point on corridor-scale runs since
    # scipy's Dijkstra peaks here. Save fwd outputs to disk so we don't
    # have to recompute on a restart.
    global_spt_path = out_dir / "_global_fwd.npz"
    if global_spt_path.exists():
        print(f"[spt] resume: loading prior global SPT from {global_spt_path.name}")
        with np.load(str(global_spt_path)) as f:
            # parent isn't read by anything downstream (per-city SPTs have
            # their own parents). Skipping the load saves ~456 MB.
            fwd = spt.SPTResult(
                cost=np.array(f["cost"]),
                parent=np.zeros(0, dtype=np.int32),
                city_idx=np.array(f["city_idx"]),
            )
        rev = None
        # `_merged.npz` was loaded with all 7 arrays, including
        # `edge_length_m` which nothing else uses. Drop it now to free
        # ~960 MB before the cells phase.
        graph.edge_length_m = None
        gc.collect()
    else:
        # Free edge arrays after CSR build to fit the corridor-scale Dijkstra
        # in 14 GB. The disk-spilled `_merged.npz` lets us reload below.
        fwd, rev = spt.compute_spt(
            graph, city_node_ids,
            directions=("forward",),
            free_edges_after_csr=True,
        )
        # Skip writing parent — nothing downstream reads it. Saves ~456 MB
        # disk and means the resume path doesn't need to load it either.
        np.savez(
            str(global_spt_path),
            cost=fwd.cost, city_idx=fwd.city_idx,
        )
        # Also drop in-memory parent immediately for the same reason.
        fwd.parent = np.zeros(0, dtype=np.int32)
        gc.collect()
        print(f"[spt] checkpointed global SPT -> {global_spt_path.name}")

    # Reload only the edge arrays we still need. cells.build_city_graph
    # uses src/dst/cost; per_city_spt uses the same three. node coords
    # were already loaded for snap and cells.build_polygons. We do NOT
    # reload edge_length_m (nothing downstream uses it). Closing `m`
    # explicitly drops the npz file handle + any internal buffers.
    if graph.edge_src is None:
        print(f"[preprocess] reloading edges from {merged_path.name} for per-city pass")
        with np.load(str(merged_path)) as m:
            graph.edge_src  = np.array(m["edge_src"])
            graph.edge_dst  = np.array(m["edge_dst"])
            graph.edge_cost = np.array(m["edge_cost"])
        gc.collect()

    city_names = [a["name"] for a in anchors]
    polygons = cells.build_polygons(graph, fwd, city_names)
    city_graph = cells.build_city_graph(graph, fwd)

    save.write_all(out_dir, graph, fwd, rev, anchors, city_graph, polygons)

    # Free node arrays (cells.build_polygons already consumed them) and
    # the parts of fwd we don't need anymore.
    graph.node_lon       = None
    graph.node_lat       = None
    graph.node_osm_id    = None
    graph.edge_length_m  = None
    fwd.cost   = None
    fwd.parent = None
    gc.collect()

    # Shard edges by source cell so per_city_spt can read just the
    # cells it needs each iteration instead of scanning the global
    # 240M-edge arrays. After the shard the in-memory edge arrays are
    # redundant (we read from disk). Free them too.
    per_city_spt.shard_edges_by_cell(graph, fwd.city_idx, out_dir)
    graph.edge_src  = None
    graph.edge_dst  = None
    graph.edge_cost = None
    gc.collect()

    # Per-city subgraph SPTs — the data structure routing actually uses
    # at query time. Each SPT covers a city's cell + adjacent cells, so
    # "in cell B, follow gradient toward C" reduces to a parent-pointer
    # walk in C's SPT.
    per_city_spt.build_per_city_spts(out_dir, graph, fwd, city_graph, anchors)

    # Per-cell edge shards are only needed during the per-city SPT
    # phase. Drop them now to keep the staged output dir clean.
    shard_dir = out_dir / "_edges_by_cell"
    if shard_dir.exists():
        shutil.rmtree(shard_dir, ignore_errors=True)

    # Atomic-ish swap: move existing <profile>/ aside, promote <profile>.tmp/
    # into place, then drop the old. Linux keeps mmap'd inodes alive after
    # rename so any in-flight API requests stay consistent.
    backup_dir = config.SPT_DIR / (args.profile + ".old")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if final_dir.exists():
        final_dir.rename(backup_dir)
    out_dir.rename(final_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)
    print(f"[preprocess] swapped {staging_dir.name} -> {final_dir.name}")
    print("[preprocess] done")


if __name__ == "__main__":
    main()
