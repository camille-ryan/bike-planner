"""Quick inspection of paired_trunks.db output by build_paired_corridor.py.

Usage: docker compose --profile preprocess run --rm --entrypoint python3 \
           pgrouting inspect_trunk_db.py
"""
from __future__ import annotations
import sqlite3
from pathlib import Path

import config


def main() -> None:
    db_path = config.SPT_DIR / "lht" / "paired_trunks.db"
    if not db_path.exists():
        print(f"[inspect] no DB at {db_path}")
        return

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    n_rows = db.execute("SELECT COUNT(*) FROM trunks").fetchone()[0]
    n_pairs = db.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT src_city, dst_city FROM trunks)"
    ).fetchone()[0]
    n_with_succ = db.execute(
        "SELECT COUNT(*) FROM trunks WHERE successor IS NOT NULL"
    ).fetchone()[0]
    n_roots = n_rows - n_with_succ

    print(f"[inspect] DB:    {db_path}")
    print(f"[inspect] size:  {db_path.stat().st_size / 1e6:.1f} MB")
    print(f"[inspect] rows:  {n_rows:,}")
    print(f"[inspect] pairs: {n_pairs:,}")
    print(f"[inspect] roots (B-frontier, successor IS NULL): {n_roots:,}")
    print(f"[inspect] interior (with successor): {n_with_succ:,}")

    print("\n[inspect] sample rows:")
    for r in db.execute(
        "SELECT * FROM trunks ORDER BY src_city, dst_city, vertex_id LIMIT 10"
    ).fetchall():
        print(f"  src={r['src_city']:>4} dst={r['dst_city']:>4} "
              f"vid={r['vertex_id']:>10} "
              f"succ={'NULL' if r['successor'] is None else r['successor']:>10} "
              f"({r['lat']:.4f}, {r['lon']:.4f})")

    print("\n[inspect] pair size distribution:")
    sizes = [r[0] for r in db.execute(
        "SELECT COUNT(*) AS cnt FROM trunks "
        "GROUP BY src_city, dst_city ORDER BY cnt"
    ).fetchall()]
    if sizes:
        print(f"  min:    {sizes[0]:,}")
        print(f"  median: {sorted(sizes)[len(sizes)//2]:,}")
        print(f"  max:    {sizes[-1]:,}")
        print(f"  avg:    {sum(sizes) // len(sizes):,}")

    # Sanity walk: pick the first pair, follow its successors from a
    # random trunk root.
    print("\n[inspect] sanity walk on first pair:")
    pair = db.execute(
        "SELECT src_city, dst_city FROM trunks LIMIT 1"
    ).fetchone()
    if pair:
        src, dst = pair["src_city"], pair["dst_city"]
        # Find a non-root vertex (one with a successor).
        start = db.execute(
            "SELECT vertex_id FROM trunks "
            "WHERE src_city = ? AND dst_city = ? AND successor IS NOT NULL "
            "LIMIT 1",
            (src, dst),
        ).fetchone()
        if start:
            cur = start["vertex_id"]
            steps = 0
            while steps < 5000:
                row = db.execute(
                    "SELECT successor, lat, lon FROM trunks "
                    "WHERE src_city = ? AND dst_city = ? AND vertex_id = ?",
                    (src, dst, cur),
                ).fetchone()
                if not row:
                    print(f"  step {steps}: vid={cur} NOT IN TRUNK (orphan?)")
                    break
                if row["successor"] is None:
                    print(f"  step {steps}: vid={cur} (lat={row['lat']:.4f}, "
                          f"lon={row['lon']:.4f}) — REACHED TRUNK ROOT")
                    break
                cur = row["successor"]
                steps += 1
            else:
                print(f"  walk exceeded 5000 steps without reaching root!")

    db.close()


if __name__ == "__main__":
    main()
