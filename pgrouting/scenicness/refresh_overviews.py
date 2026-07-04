"""Rebuild z4–z11 XYZ overviews from whatever z12 base tiles currently
exist under /data/scenicness/xyz/. Run periodically alongside the bake
so the web app's wide-view picks up fresh data without waiting for
bake completion.

Safe to run concurrently with the bake: it's pure file I/O against the
XYZ tree, no postgres access. The bake writes z12 tiles per-bake-tile;
this script max-pools them into z4–z11 parents.
"""
from __future__ import annotations
from pathlib import Path

from scenicness import tiles


def main() -> None:
    xyz_root = Path("/data/scenicness/xyz")
    if not xyz_root.exists():
        print(f"[overviews] {xyz_root} does not exist yet — skipping", flush=True)
        return
    signals = sorted(d.name for d in xyz_root.iterdir() if d.is_dir())
    if not signals:
        print(f"[overviews] no signals in {xyz_root} yet — skipping", flush=True)
        return
    print(f"[overviews] rebuilding for {len(signals)} signals: {signals}",
          flush=True)
    summary = tiles.build_overviews(
        xyz_root, signals, zoom_min=4, zoom_max=12,
    )
    total = sum(summary.values())
    print(f"[overviews] done: {total:,} overview tiles across "
          f"{len(summary)} signals", flush=True)
    for sig, n in sorted(summary.items()):
        print(f"[overviews]   {sig}: {n:,}", flush=True)


if __name__ == "__main__":
    main()
