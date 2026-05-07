"""pgRouting-backed preprocess orchestrator.

  python3 main.py ingest     --countries austria
  python3 main.py snap       --countries austria
  python3 main.py boundaries --countries austria
  python3 main.py spts       --profile lht
  python3 main.py all        --countries austria --profile lht

Subcommands:
  ingest      stream OSM PBFs into Postgres tables ways + ways_vertices_pgr
  snap        load anchors from pois.sqlite + snap to nearest graph vertex
  boundaries  stream admin_level=8 polygons from PBFs into anchors.geom_boundary
  spts        multi-source Bellman-Ford + per-city Dijkstra -> data/spt/<profile>/
  all         ingest -> snap -> boundaries -> spts (orchestrate end-to-end)

`profile` is the output-directory name; cost.py is currently the only
profile so this is informational. Once we add elevation/scenic terms
the profile selection will pick a cost function.
"""
import argparse
from pathlib import Path

import psycopg

import config
import ingest_pbf
import ingest_boundaries
import snap_anchors
import compute_spts


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


def cmd_spts(args) -> None:
    out_dir = config.SPT_DIR / args.profile
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[main] spts profile={args.profile} -> {out_dir}")
    with psycopg.connect(config.PG_DSN) as conn:
        compute_spts.run(conn, out_dir)
    print("[main] spts done")


def cmd_all(args) -> None:
    cmd_ingest(args)
    cmd_snap(args)
    cmd_boundaries(args)
    cmd_spts(args)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, func in (("ingest", cmd_ingest), ("snap", cmd_snap),
                       ("boundaries", cmd_boundaries),
                       ("spts", cmd_spts), ("all", cmd_all)):
        sp = sub.add_parser(name)
        sp.add_argument("--profile",   default="lht")
        sp.add_argument("--countries", default="austria")
        sp.set_defaults(func=func)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
