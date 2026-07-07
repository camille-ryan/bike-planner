"""Pipeline status inspector — reports which of the 12 rebuild stages
have current outputs, which are missing, and which look stale relative
to their upstream.

Meant to be called two ways:

  * Human-readable:  `python3 pipeline_status.py`
  * Machine-readable: `python3 pipeline_status.py --json`

Also drives the resumability check inside `run_full_rebuild.sh`:

  * `python3 pipeline_status.py --stage N --check-current` → exit 0
    if stage N's output is current (skip), non-zero otherwise (run).

The stage list mirrors the orchestrator's stage()-call sequence.
When a stage is added or reordered, update both here and in the
`.sh` at the same time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

try:
    import psycopg
except ImportError:                                # noqa: F401 — postgres check
    psycopg = None                                 # type: ignore[assignment]


DATA = Path(os.environ.get("DATA_DIR", "/mnt/e/proj/bike/data"))
SPT_PROFILE = os.environ.get("SPT_PROFILE", "views")
PG_DB = os.environ.get("PGDATABASE", os.environ.get("PG_DB", "bike_v2_test"))
# Prefer standard PG* env vars (PGHOST/PGUSER/PGPASSWORD/PGDATABASE)
# which are set inside the pgrouting container to reach the docker-
# network postgres host. Overridable via PG_DSN.
PG_DSN = os.environ.get("PG_DSN")


@dataclass
class StageStatus:
    n: int
    name: str
    ok: bool                    # [x] output current
    partial: bool = False       # [?] partial / older-than-upstream
    detail: str = ""            # human-readable summary
    output_path: str | None = None
    upstream_paths: list[str] = field(default_factory=list)

    @property
    def mark(self) -> str:
        if self.ok:      return "[x]"
        if self.partial: return "[?]"
        return "[ ]"


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _newest_child_mtime(dir_path: Path, pattern: str = "*") -> float | None:
    if not dir_path.exists():
        return None
    newest = None
    for p in dir_path.glob(pattern):
        m = p.stat().st_mtime
        if newest is None or m > newest:
            newest = m
    return newest


def _count_files(dir_path: Path, pattern: str = "*") -> int:
    if not dir_path.exists():
        return 0
    return sum(1 for _ in dir_path.glob(pattern))


def _ways_paved_status() -> StageStatus:
    """Stage 1: postgres.ways_paved. Requires psycopg — pipeline_status
    is meant to be invoked inside a pgrouting container (`docker
    compose run --rm --entrypoint python3 pgrouting /app/pipeline_status.py`),
    where psycopg is installed and `postgres:5432` is reachable via the
    docker network. From the host we skip the check."""
    if psycopg is None:
        return StageStatus(1, "build_paved", ok=False, partial=True,
                           detail="psycopg unavailable on host — "
                                  "run inside pgrouting container to check")
    try:
        conn_args = {"connect_timeout": 3}
        if PG_DSN:
            conn = psycopg.connect(PG_DSN, **conn_args)
        else:
            conn = psycopg.connect(**conn_args)
        with conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM ways_paved")
            (n,) = cur.fetchone()
            cur.execute("""
                SELECT COUNT(*) FROM pg_indexes
                WHERE tablename = 'ways_paved'
                  AND indexname = 'ways_paved_src_pt_idx'
            """)
            (n_idx,) = cur.fetchone()
        ok = n > 0 and n_idx > 0
        return StageStatus(1, "build_paved", ok=ok,
                           detail=f"{n:,} rows"
                                  + ("" if n_idx > 0 else ", MISSING GIST index"))
    except Exception as exc:                       # noqa: BLE001
        return StageStatus(1, "build_paved", ok=False,
                           detail=f"postgres.ways_paved: {type(exc).__name__}")


def _file_stage(n: int, name: str, out_path: Path,
                upstream: list[Path] = ()) -> StageStatus:
    """Generic file-output stage: current iff out exists AND is newer
    than every upstream file. Missing = [ ]; older-than-upstream = [?]."""
    m_out = _mtime(out_path)
    detail_parts = []
    if m_out is None:
        return StageStatus(
            n, name, ok=False, detail=f"MISSING {out_path}",
            output_path=str(out_path),
            upstream_paths=[str(u) for u in upstream],
        )
    size = out_path.stat().st_size
    detail_parts.append(_human_size(size))
    partial = False
    for u in upstream:
        m_u = _mtime(u)
        if m_u and m_out < m_u:
            partial = True
            detail_parts.append(f"STALE vs {u.name}")
            break
    detail_parts.append(_iso_short(m_out))
    return StageStatus(
        n, name, ok=not partial, partial=partial,
        detail=", ".join(detail_parts),
        output_path=str(out_path),
        upstream_paths=[str(u) for u in upstream],
    )


def _spt_dir_stage() -> StageStatus:
    """Stage 7 special: /data/spt/<profile>_polygon has one .npz per
    anchor. Current iff #npz matches #anchors in the source polygons
    geojson AND newest npz > polygons file."""
    npz_dir = DATA / "spt" / f"{SPT_PROFILE}_polygon"
    polys = DATA / "way_city_spt_polygons.geojson"
    if not npz_dir.exists():
        return StageStatus(
            7, "spt_polygon", ok=False,
            detail=f"MISSING {npz_dir}",
            output_path=str(npz_dir),
            upstream_paths=[str(polys)],
        )
    n_npz = _count_files(npz_dir, "*.npz")
    m_newest = _newest_child_mtime(npz_dir, "*.npz")
    m_poly = _mtime(polys)
    partial = False
    detail = f"{n_npz:,} npz"
    if m_poly and m_newest and m_newest < m_poly:
        partial = True
        detail += f", newer polygons ({_iso_short(m_poly)}) — STALE"
    elif m_newest:
        detail += f", newest {_iso_short(m_newest)}"
    return StageStatus(
        7, "spt_polygon", ok=(not partial and n_npz > 0),
        partial=partial, detail=detail,
        output_path=str(npz_dir),
        upstream_paths=[str(polys)],
    )


def _api_symlink_stage() -> StageStatus:
    """Stage 10: paired_trunks.db symlink → paired_trunks_v2d.db.
    Current iff symlink exists AND points to v2d.db AND v2d is newer
    than v2c."""
    link = DATA / "spt" / SPT_PROFILE / "paired_trunks.db"
    v2d = DATA / "spt" / SPT_PROFILE / "paired_trunks_v2d.db"
    v2c = DATA / "spt" / SPT_PROFILE / "paired_trunks_v2c.db"
    if not link.is_symlink():
        return StageStatus(12, "symlink", ok=False,
                           detail=f"not a symlink: {link}",
                           output_path=str(link))
    target = os.readlink(link)
    if target != "paired_trunks_v2d.db":
        return StageStatus(12, "symlink", ok=False,
                           detail=f"symlink points to {target}, expected paired_trunks_v2d.db",
                           output_path=str(link))
    m_v2d = _mtime(v2d)
    m_v2c = _mtime(v2c)
    if m_v2d and m_v2c and m_v2d < m_v2c:
        return StageStatus(12, "symlink", ok=False, partial=True,
                           detail="v2d.db older than v2c.db — pruner never ran on this v2c",
                           output_path=str(link))
    return StageStatus(12, "symlink", ok=True,
                       detail=f"→ {target}",
                       output_path=str(link))


def _api_running_stage() -> StageStatus:
    """Stage 11: API restart. Consider current iff bike-api container
    is healthy. This is a runtime check, not a build artifact."""
    import subprocess
    try:
        r = subprocess.run(
            ["docker", "ps", "--filter", "name=bike-api",
             "--filter", "status=running", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5,
        )
        ok = "bike-api" in r.stdout
        detail = "container running" if ok else "container NOT running"
    except Exception as exc:                     # noqa: BLE001 — best-effort
        ok = False
        detail = f"docker ps failed: {exc}"
    return StageStatus(13, "api_restart", ok=ok, detail=detail)


def _verify_stage() -> StageStatus:
    """Stage 12: verify. No lasting artifact — always report as
    'not current' so the orchestrator re-runs it every time."""
    return StageStatus(14, "verify", ok=False,
                       detail="runtime check — always re-runs")


def _stages() -> list[Callable[[], StageStatus]]:
    profile_dir = DATA / "spt" / SPT_PROFILE
    return [
        _ways_paved_status,                            # 1
        lambda: _file_stage(                           # 2
            2, "classify_piers",
            DATA / "sea_piers.geojsonseq",
            upstream=[DATA / "ferry_piers.geojsonseq"],
        ),
        lambda: _file_stage(                           # 3
            3, "anchors",
            DATA / "way_city_anchors.geojson",
            upstream=[DATA / "sea_piers.geojsonseq"],
        ),
        lambda: _file_stage(                           # 4
            4, "chain_land",
            DATA / "way_city_graph.json",
            upstream=[DATA / "way_city_anchors.geojson"],
        ),
        lambda: _file_stage(                           # 5
            5, "chain_ferry",
            DATA / "way_city_graph.geojson",  # updated by augment
            upstream=[DATA / "way_city_graph.json"],
        ),
        # Pair-scope Dijkstra reachability filter (task #54). Emits a
        # dropped-edges audit file — its presence proves bidir ran.
        lambda: _file_stage(                           # 6
            6, "bidir_reach",
            DATA / "way_city_graph_dropped.json",
            upstream=[DATA / "way_city_graph.json"],
        ),
        # Chain-triangle deduplication. Presence of the .pre_dedup.json
        # backup file (written on first dedup run) proves the step ran.
        lambda: _file_stage(                           # 7
            7, "dedup_chain",
            DATA / "way_city_graph.pre_dedup.json",
            upstream=[DATA / "way_city_graph_dropped.json"],
        ),
        lambda: _file_stage(                           # 8
            8, "anchor_polys",
            DATA / "way_city_spt_polygons.geojson",
            upstream=[DATA / "way_city_graph.json"],
        ),
        _spt_dir_stage,                                # 9
        lambda: _file_stage(                           # 10
            10, "adapt_paired",
            profile_dir / "city_graph.json",
            upstream=[DATA / "spt" / f"{SPT_PROFILE}_polygon"],
        ),
        lambda: _file_stage(                           # 11
            11, "build_paired",
            profile_dir / "paired_trunks_v2c.db",
            upstream=[profile_dir / "city_graph.json"],
        ),
        lambda: _file_stage(                           # 12
            12, "prune",
            profile_dir / "paired_trunks_v2d.db",
            upstream=[profile_dir / "paired_trunks_v2c.db"],
        ),
        _api_symlink_stage,                            # 13
        _api_running_stage,                            # 14
        _verify_stage,                                 # 15
    ]


def _iso_short(mtime: float) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--stage", type=int, help="report only stage N")
    p.add_argument("--check-current", action="store_true",
                   help="exit 0 if stage --stage is [x], non-zero otherwise")
    args = p.parse_args()

    reports = [fn() for fn in _stages()]

    if args.stage is not None:
        st = next((r for r in reports if r.n == args.stage), None)
        if st is None:
            print(f"unknown stage {args.stage}", file=sys.stderr)
            sys.exit(2)
        if args.check_current:
            sys.exit(0 if st.ok else 1)
        reports = [st]

    if args.json:
        print(json.dumps([
            {"n": r.n, "name": r.name, "ok": r.ok, "partial": r.partial,
             "detail": r.detail, "output_path": r.output_path,
             "upstream_paths": r.upstream_paths}
            for r in reports
        ], indent=2))
        return

    for r in reports:
        print(f"STAGE {r.n:>2} {r.name:<15} {r.mark}  {r.detail}")


if __name__ == "__main__":
    main()
