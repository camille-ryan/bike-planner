"""Delete per-city SPT npz files whose reach is suspiciously small.

After fixing the seed function (e.g. switching polygonless anchors
from single-vertex snap to a 1 km bbox seed), existing trapped-anchor
npzs are stale: they were computed with the broken seeding. Rather
than restart the whole preprocess, delete any npz whose reach is
below a threshold so the next `spts` resume regenerates only those.

Usage:
    python cull_trapped.py            # threshold = 1000 vertices
    python cull_trapped.py 5000       # custom threshold
"""
import os
import sys
from pathlib import Path

import numpy as np


SPT_DIR = Path(os.environ.get("DATA_DIR", "/data")) / "spt" / "lht"


def main():
    threshold = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    spt_dir = SPT_DIR / "spt"
    npz_files = sorted(spt_dir.glob("*.npz"))
    print(f"[cull] scanning {len(npz_files):,} npz files, threshold={threshold:,}")

    to_delete = []
    for path in npz_files:
        try:
            with np.load(path, mmap_mode="r") as data:
                n = len(data["node_global"])
        except Exception as e:
            print(f"  ! could not read {path.name}: {e}")
            continue
        if n < threshold:
            to_delete.append((path, n))

    print(f"[cull] {len(to_delete):,} files reach < {threshold:,} vertices")
    if not to_delete:
        return

    # Show top offenders so user can sanity-check.
    to_delete.sort(key=lambda x: x[1])
    for path, n in to_delete[:20]:
        print(f"  {path.name:>15s}   reach={n:>6,d}")
    if len(to_delete) > 20:
        print(f"  ... ({len(to_delete) - 20} more)")

    if "--dry-run" in sys.argv:
        print("[cull] dry-run, no files deleted")
        return

    for path, _n in to_delete:
        path.unlink()
    print(f"[cull] deleted {len(to_delete):,} trapped npz files; "
          f"resume `spts` to regenerate with current seed function")


if __name__ == "__main__":
    main()
