"""pgRouting-backed preprocess orchestrator.

  python3 main.py ingest         --countries austria
  python3 main.py snap           --countries austria
  python3 main.py boundaries     --countries austria
  python3 main.py dem-download   --countries austria
  python3 main.py dem-ingest
  python3 main.py recompute-cost
  python3 main.py spts           --profile lht
  python3 main.py paired         --profile lht
  python3 main.py all            --countries austria --profile lht

Subcommands (run order):
  ingest         stream OSM PBFs into Postgres ways + ways_vertices_pgr
                 (V2: also persists raw OSM tags + per-way sinuosity)
  snap           load anchors from pois.sqlite, snap to nearest graph vertex
  boundaries     stream admin polygons into anchors.geom_boundary
  dem-download   fetch Copernicus DEM GLO-30 tiles covering the country bbox
                 union into data/dem/ (V2 Phase A.2)
  dem-ingest     bilinear-sample elevation at every vertex →
                 ways_vertices_pgr.elev_m
  recompute-cost re-apply bike_edge_cost over every edge with grade_pct +
                 sinuosity; writes ways.cost / reverse_cost in place
  spts           per-anchor 30 km multi-source Dijkstra → SPT npzs;
                 also writes road_topology/<id>.npz (lon/lat per vertex,
                 shared across profiles) and city_graph.json (with ferry
                 chain edges added for long sea/lake crossings)
  paired         pruned paired SPTs as a SQLite trunk DB plus optional
                 per-pair npzs (for trace-level inspection)
  all            ingest → snap → boundaries → dem-download → dem-ingest →
                 recompute-cost → spts → paired

`profile` is the output-directory name. cost.py is currently the only
profile but the pipeline is structured to support more (topology stays
shared; SPT and trunk DB are per-profile).
"""
import argparse
from pathlib import Path

import psycopg

import config
import ingest_pbf
import ingest_boundaries
import ingest_dem
import snap_anchors
import compute_spts
import download_dem
import recompute_cost


def cmd_ingest(args) -> None:
    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    pbfs = [config.OSM_DIR / f"{c}-latest.osm.pbf" for c in countries]
    print(f"[main] ingest countries={countries}")
    with psycopg.connect(config.PG_DSN) as conn:
        ingest_pbf.ingest(conn, pbfs)
    print("[main] ingest done")


def cmd_snap(args) -> None:
    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    print(f"[main] snap countries={countries}")
    anchors = snap_anchors.load_anchors_from_sqlite(str(config.POIS_DB), countries)
    with psycopg.connect(config.PG_DSN) as conn:
        snap_anchors.populate_anchors_table(conn, anchors)
    print("[main] snap done")


def cmd_boundaries(args) -> None:
    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    pbfs = [config.OSM_DIR / f"{c}-latest.osm.pbf" for c in countries]
    print(f"[main] boundaries countries={countries}")
    with psycopg.connect(config.PG_DSN) as conn:
        ingest_boundaries.ingest(conn, pbfs)
    print("[main] boundaries done")


def cmd_dem_download(args) -> None:
    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    bbox = download_dem.bbox_for_countries(countries)
    print(f"[main] dem-download countries={countries} bbox={bbox}")
    download_dem.download(bbox)
    print("[main] dem-download done")


def cmd_dem_ingest(args) -> None:
    print(f"[main] dem-ingest")
    with psycopg.connect(config.PG_DSN) as conn:
        ingest_dem.ingest(conn)
    print("[main] dem-ingest done")


def cmd_recompute_cost(args) -> None:
    print(f"[main] recompute-cost")
    with psycopg.connect(config.PG_DSN) as conn:
        recompute_cost.recompute(conn)
    print("[main] recompute-cost done")


def cmd_spts(args) -> None:
    out_dir = config.SPT_DIR / args.profile
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[main] spts profile={args.profile} -> {out_dir}")
    with psycopg.connect(config.PG_DSN) as conn:
        compute_spts.run(conn, out_dir)
    print("[main] spts done")


def cmd_paired(args) -> None:
    """Build pruned paired SPTs corridor-wide, output a trunk DB.

    The build script reads chain pairs from city_graph.json (now
    ferry-augmented), and writes:
      data/spt/<profile>/paired_trunks.db   (SQLite, indexed)
      data/spt/<profile>/paired/{a}_{b}.npz (optional, per-pair, if --keep-npzs)
    """
    import build_paired_corridor
    out_dir = config.SPT_DIR / args.profile
    print(f"[main] paired profile={args.profile} -> {out_dir}")
    build_paired_corridor.run(
        out_dir, polyline=args.polyline, max_km=args.max_km,
        prune=True, keep_npzs=args.keep_npzs,
    )
    print("[main] paired done")


def cmd_all(args) -> None:
    cmd_ingest(args)
    cmd_snap(args)
    cmd_boundaries(args)
    cmd_dem_download(args)
    cmd_dem_ingest(args)
    cmd_recompute_cost(args)
    cmd_spts(args)
    cmd_paired(args)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, func in (
        ("ingest", cmd_ingest),
        ("snap", cmd_snap),
        ("boundaries", cmd_boundaries),
        ("dem-download", cmd_dem_download),
        ("dem-ingest", cmd_dem_ingest),
        ("recompute-cost", cmd_recompute_cost),
        ("spts", cmd_spts),
        ("paired", cmd_paired),
        ("all", cmd_all),
    ):
        sp = sub.add_parser(name)
        sp.add_argument("--profile",   default="lht")
        sp.add_argument("--countries", default="austria")
        sp.add_argument("--polyline",  default=None,
            help="lon,lat,lon,lat,... — corridor polyline. paired-only.")
        sp.add_argument("--max-km",    type=float, default=80.0,
            help="Anchor inclusion radius around polyline. paired-only.")
        sp.add_argument("--keep-npzs", action="store_true",
            help="Also write per-pair npzs alongside the trunk DB. paired-only.")
        sp.set_defaults(func=func)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
