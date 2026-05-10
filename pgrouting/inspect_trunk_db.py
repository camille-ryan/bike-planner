"""Quick inspection of paired_trunks.db (blob schema) output by
build_paired_corridor.py.

Usage: docker compose --profile preprocess run --rm --entrypoint python3 \
           pgrouting inspect_trunk_db.py
"""
from __future__ import annotations
import sqlite3
from pathlib import Path

import numpy as np

import config


TRUNK_DTYPE = np.dtype([
    ("vid",  "<i8"),
    ("succ", "<i8"),
    ("lat",  "<f4"),
    ("lon",  "<f4"),
])
NULL_SENTINEL = np.int64(-1)


def main() -> None:
    db_path = config.SPT_DIR / "lht" / "paired_trunks.db"
    if not db_path.exists():
        print(f"[inspect] no DB at {db_path}")
        return

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    n_pairs = db.execute("SELECT COUNT(*) FROM trunk_blobs").fetchone()[0]
    total_rows = db.execute(
        "SELECT SUM(n_rows) FROM trunk_blobs"
    ).fetchone()[0] or 0
    total_blob = db.execute(
        "SELECT SUM(LENGTH(blob)) FROM trunk_blobs"
    ).fetchone()[0] or 0

    print(f"[inspect] DB:        {db_path}")
    print(f"[inspect] file size: {db_path.stat().st_size / 1e6:.1f} MB")
    print(f"[inspect] pairs:     {n_pairs:,}")
    print(f"[inspect] rows:      {total_rows:,}")
    print(f"[inspect] blob:      {total_blob/1e6:.1f} MB "
          f"(avg {total_blob/max(1,n_pairs)/1024:.1f} KB/pair)")

    # Pair size distribution.
    sizes = [
        r[0] for r in db.execute(
            "SELECT n_rows FROM trunk_blobs ORDER BY n_rows"
        )
    ]
    if sizes:
        print(f"\n[inspect] pair size distribution:")
        print(f"  min:    {sizes[0]:,}")
        print(f"  median: {sizes[len(sizes)//2]:,}")
        print(f"  max:    {sizes[-1]:,}")
        print(f"  avg:    {sum(sizes)//len(sizes):,}")

    # Pick the first pair, decode the blob, and walk one chain.
    print("\n[inspect] sanity walk on first pair:")
    row = db.execute(
        "SELECT src_city, dst_city, n_rows, blob FROM trunk_blobs LIMIT 1"
    ).fetchone()
    if row:
        src, dst, n, blob = row
        arr = np.frombuffer(blob, dtype=TRUNK_DTYPE)
        assert len(arr) == n, f"n_rows mismatch: header={n} arr={len(arr)}"
        print(f"  pair: ({src} → {dst})  n_rows={n}  "
              f"trunk_roots={int((arr['succ'] == NULL_SENTINEL).sum())}")
        # Walk from the first non-root vertex.
        non_root_mask = arr["succ"] != NULL_SENTINEL
        if non_root_mask.any():
            start_idx = int(np.argmax(non_root_mask))
            # Build next_idx via searchsorted.
            pos = np.searchsorted(arr["vid"], arr["succ"])
            in_range = pos < len(arr)
            pos_c = np.clip(pos, 0, len(arr) - 1)
            matched = in_range & (arr["vid"][pos_c] == arr["succ"])
            is_root = arr["succ"] == NULL_SENTINEL
            next_idx = np.where(matched & ~is_root, pos, -1).astype(np.int64)

            steps = 0
            i = start_idx
            while i >= 0 and steps < 5000:
                if next_idx[i] < 0:
                    print(f"  step {steps}: vid={int(arr['vid'][i])} "
                          f"({float(arr['lat'][i]):.4f}, "
                          f"{float(arr['lon'][i]):.4f}) — REACHED TRUNK ROOT")
                    break
                i = int(next_idx[i])
                steps += 1
            else:
                print(f"  walk exceeded 5000 steps without reaching root")

    db.close()


if __name__ == "__main__":
    main()
