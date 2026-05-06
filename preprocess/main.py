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
    out_dir = config.SPT_DIR / args.profile
    print(f"[preprocess] profile={args.profile} countries={countries} -> {out_dir}")

    graphs = []
    for c in countries:
        pbf = config.OSM_DIR / f"{c}-latest.osm.pbf"
        if not pbf.exists():
            raise SystemExit(f"missing PBF: {pbf}")
        graphs.append(extract_graph.extract(pbf))
    graph = merge_graphs(graphs)
    print(f"[preprocess] merged graph: nodes={len(graph.node_lon):,} edges={len(graph.edge_src):,}")

    anchors = load_anchors_from_pois(config.POIS_DB, countries)
    print(f"[preprocess] loaded {len(anchors)} anchors from {config.POIS_DB}")

    city_lons = np.array([a["lon"] for a in anchors], dtype=np.float32)
    city_lats = np.array([a["lat"] for a in anchors], dtype=np.float32)
    city_node_ids = spt.snap_cities_to_nodes(city_lons, city_lats,
                                             graph.node_lon, graph.node_lat)
    for i, a in enumerate(anchors):
        a["node_idx"] = int(city_node_ids[i])

    fwd, rev = spt.compute_spt(graph, city_node_ids)

    city_names = [a["name"] for a in anchors]
    polygons = cells.build_polygons(graph, fwd, city_names)
    city_graph = cells.build_city_graph(graph, fwd)

    save.write_all(out_dir, graph, fwd, rev, anchors, city_graph, polygons)

    # Per-city subgraph SPTs — the data structure routing actually uses
    # at query time. Each SPT covers a city's cell + adjacent cells, so
    # "in cell B, follow gradient toward C" reduces to a parent-pointer
    # walk in C's SPT.
    per_city_spt.build_per_city_spts(out_dir, graph, fwd, city_graph, anchors)

    print("[preprocess] done")


if __name__ == "__main__":
    main()
