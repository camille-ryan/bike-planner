"""Derive cost_<profile> for the remaining multi-axis profiles by
multiplying out the scenic factor algebraically — much faster than
re-running recompute_cost for each profile.

Background:
  bike_edge_cost = base × surface_softening × grade_penalty × scenic_factor

  For the 4 multi-axis profiles, base × surface_softening × grade_penalty
  is IDENTICAL (they all share `surface_softening=1.0`, `uphill_offset=3`,
  `scenic_weight_mult=3`). Only the `per_signal_k` dict differs.

  So if we have cost_<ANCHOR> for any one multi-axis profile, then for
  any other multi-axis profile P:

      cost_P = cost_ANCHOR × scenic_factor_P(signals) ÷ scenic_factor_ANCHOR(signals)

  scenic_factor is a pure SQL expression of the per-edge signal columns
  + the profile's per_signal_k dict (and the global _SCENIC_W weights
  and normalization constants from cost.py).

Reverse-cost handling: where `reverse_cost_<ANCHOR> = -1` (one-way edge),
keep `reverse_cost_<P> = -1`. Otherwise apply the same scenic ratio.

Run inside the pgrouting container:
    docker compose --profile preprocess run --rm \\
      -e PGDATABASE=bike_v2_test \\
      --entrypoint python3 pgrouting \\
      /app/derive_multi_axis_costs.py <ANCHOR> <DERIVE>[,<DERIVE>...]

Example:
    /app/derive_multi_axis_costs.py vineyard_lover views,water
"""
from __future__ import annotations

import sys
import time

import psycopg

import config
from cost import (
    _PROFILES, _SCENIC_W,
    _RELIEF_NORM, _REGIONAL_NORM, _VIEW_NORM, _DRAMA_SCALE_M,
)


def _signal_value_sql(sig_name: str) -> str:
    """SQL expression for the normalized per-edge signal value, matching
    `cost._per_signal_values()`. Uses table alias `w` for the ways row."""
    relief_norm   = float(_RELIEF_NORM)
    regional_norm = float(_REGIONAL_NORM)
    view_norm     = float(_VIEW_NORM)
    drama_scale   = float(_DRAMA_SCALE_M)

    # view  = max(view_dominance, 0) / VIEW_NORM
    view_sql   = f"(GREATEST(w.view_dominance, 0) / {view_norm})"
    # relief_n = clamp(local_relief / RELIEF_NORM, 0..1)
    relief_sql = f"GREATEST(0, LEAST(1, w.local_relief / {relief_norm}))"
    # region_n = clamp(regional_relief / REGIONAL_NORM, 0..1)
    region_sql = f"GREATEST(0, LEAST(1, w.regional_relief / {regional_norm}))"
    # drama_w  = exp(-distance_to_drama / DRAMA_SCALE_M)
    drama_sql  = f"exp(-w.distance_to_drama / {drama_scale})"
    # wctx     = 0.5 + 0.5 * relief_n + 0.5 * forest_local
    wctx_sql   = f"(0.5 + 0.5 * {relief_sql} + 0.5 * w.forest_local)"

    if sig_name == "forest_local":        return "w.forest_local"
    if sig_name == "vineyard_local":      return "w.vineyard_local"
    if sig_name == "water_local":         return "w.water_local"
    if sig_name == "wetland_local":       return "w.wetland_local"
    if sig_name == "waterway":            return f"(w.waterway_along_edge * {wctx_sql})"
    if sig_name == "relief":              return relief_sql
    if sig_name == "regional":            return region_sql
    if sig_name == "vista_forest":        return f"({view_sql} * w.forest_wide)"
    if sig_name == "vista_water":         return f"({view_sql} * w.water_wide)"
    if sig_name == "vista_sea":           return f"({view_sql} * w.sea_wide)"
    if sig_name == "vista_drama":         return f"({view_sql} * {drama_sql})"
    if sig_name == "viewpoint_local":     return "w.viewpoint_local"
    if sig_name == "viewpoint_regional":  return "w.viewpoint_regional"
    if sig_name == "waterway_local":      return "w.waterway_local"
    raise ValueError(f"unknown signal {sig_name!r}")


def _scenic_factor_sql(prof: dict) -> str:
    """Build the SQL expression for scenic_factor_multi(prof, signals)
    matching `cost._scenic_factor_multi`."""
    per_k = prof.get("per_signal_k") or {}
    if not per_k:
        return "1.0"
    mult = float(prof["scenic_weight_mult"])
    terms = []
    for sig_name, k in per_k.items():
        if k <= 0:
            continue
        # waterway_local uses 0.3 as its effective weight in multi mode
        if sig_name == "waterway_local":
            w_i = 0.3
        else:
            w_i = float(_SCENIC_W.get(sig_name, 1.0))
        value_sql = _signal_value_sql(sig_name)
        terms.append(f"(1.0 - {float(k)} * tanh({w_i} * {value_sql} * {mult}))")
    return " * ".join(terms)


def derive(anchor: str, profiles: list[str]) -> None:
    if anchor not in _PROFILES:
        raise SystemExit(f"unknown anchor profile {anchor!r}")
    for p in profiles:
        if p not in _PROFILES:
            raise SystemExit(f"unknown profile {p!r}")
        if p == anchor:
            raise SystemExit(f"profile {p!r} is the anchor — skip it in DERIVE list")

    anchor_factor_sql = _scenic_factor_sql(_PROFILES[anchor])
    print(f"[derive] anchor={anchor}", flush=True)
    print(f"[derive]   scenic_factor SQL ({len(anchor_factor_sql)} chars):",
          flush=True)
    print(f"[derive]   {anchor_factor_sql[:200]}...", flush=True)

    with psycopg.connect(config.PG_DSN) as conn:
        # Ensure target columns exist
        with conn.cursor() as cur:
            for p in profiles:
                cur.execute(
                    f"ALTER TABLE ways ADD COLUMN IF NOT EXISTS cost_{p} real"
                )
                cur.execute(
                    f"ALTER TABLE ways ADD COLUMN IF NOT EXISTS reverse_cost_{p} real"
                )
        conn.commit()

        for p in profiles:
            t0 = time.time()
            derive_factor_sql = _scenic_factor_sql(_PROFILES[p])
            print(f"[derive] {p}: building cost_{p} from cost_{anchor} ...",
                  flush=True)

            # Forward cost. Ratio = factor_<p> / factor_<anchor>.
            # NULLIF guards against divide-by-zero in the rare case a row
            # has scenic_factor == 0 (k_i × tanh(...) == 1 for some signal).
            sql_fwd = f"""
                UPDATE ways w SET cost_{p} =
                    w.cost_{anchor} * ({derive_factor_sql})
                    / NULLIF({anchor_factor_sql}, 0)
                WHERE w.cost_{anchor} IS NOT NULL
            """
            with conn.cursor() as cur:
                cur.execute("SET max_parallel_workers_per_gather = 0")
                cur.execute("SET work_mem = '512MB'")
                cur.execute(sql_fwd)
                n_fwd = cur.rowcount
            conn.commit()

            # Reverse cost — preserve -1 for one-way edges.
            sql_rev = f"""
                UPDATE ways w SET reverse_cost_{p} =
                    CASE WHEN w.reverse_cost_{anchor} < 0 THEN -1.0
                         ELSE w.reverse_cost_{anchor} * ({derive_factor_sql})
                              / NULLIF({anchor_factor_sql}, 0)
                    END
                WHERE w.reverse_cost_{anchor} IS NOT NULL
            """
            with conn.cursor() as cur:
                cur.execute(sql_rev)
                n_rev = cur.rowcount
            conn.commit()
            print(f"[derive] {p}: cost_{p} {n_fwd:,} rows, "
                  f"reverse_cost_{p} {n_rev:,} rows "
                  f"in {time.time()-t0:.1f}s", flush=True)


def _cli() -> None:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    anchor = sys.argv[1].strip()
    profiles = [s.strip() for s in sys.argv[2].split(",") if s.strip()]
    derive(anchor, profiles)


if __name__ == "__main__":
    _cli()
