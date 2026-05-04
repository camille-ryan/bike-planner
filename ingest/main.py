"""Run the ingest pipeline.

  python3 main.py --test     # Austria + 1 BRouter tile (smoke test, ~700 MB)
  python3 main.py            # full corridor (~7 GB)
"""
import argparse
import sqlite3
from pathlib import Path

import config
import download_osm
import download_brouter
import extract_pois
import build_db


def summarize(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT category, COUNT(*) FROM pois GROUP BY category").fetchall()
    total = sum(n for _, n in rows)
    print(f"\n[summary] {db_path}")
    print(f"  total POIs: {total}")
    for cat, n in sorted(rows):
        print(f"    {cat:14s} {n:>7d}")
    conn.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true",
                   help="Smoke test: Austria + Graz-area BRouter tile only.")
    p.add_argument("--skip-osm", action="store_true")
    p.add_argument("--skip-brouter", action="store_true")
    p.add_argument("--skip-pois", action="store_true")
    args = p.parse_args()

    countries = ["austria"] if args.test else config.COUNTRIES
    tiles     = ["E15_N45"] if args.test else config.BROUTER_TILES

    if not args.skip_osm:
        country_pbfs = download_osm.run(countries)
    else:
        country_pbfs = [config.DATA_DIR / "osm" / f"{c}-latest.osm.pbf"
                        for c in countries]

    if not args.skip_brouter:
        download_brouter.run(tiles)

    if not args.skip_pois:
        extracts = extract_pois.run(country_pbfs)
        build_db.run(extracts)
        summarize(build_db.DB_PATH)

    print("[ingest] done")


if __name__ == "__main__":
    main()
