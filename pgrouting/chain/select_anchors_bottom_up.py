"""Bottom-up greedy anchor selection.

Algorithm:
  1. Pool = all OSM settlements (cities + towns + villages-with-or-without-pop).
  2. Iterate the pool in ascending population order.
  3. For each settlement, drop it iff any OTHER still-kept settlement
     lies within MIN_SPACING_M (15 km).
  4. The survivors are exactly the settlements that are the largest in
     their 15 km neighborhood — equivalent to greedy Poisson-disk
     sampling with population as priority.

Properties:
  * Uniform geographic coverage (no part of the country can have an
    anchor-less 15 km disc unless there is genuinely no settlement
    there).
  * Population priority: Vienna stays, Vienna suburbs drop. Tiny but
    isolated villages stay (they anchor their otherwise-empty region).
  * No clusters: any two anchors are ≥ 15 km apart by construction.

The downstream chain graph is then `compute_chain_graph` of the
survivors over the existing flood-filled + bridged subgraph.

Outputs:
  /data/way_city_graph.json
  /data/way_city_graph.geojson
  /data/way_city_anchors.geojson
  /data/way_city_anchors_orphans.geojson
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import psycopg
from scipy.spatial import cKDTree

import config
from deprecated.build_way_graph import (
    _load_db_anchors,
    _lonlat_to_xyz,
    _chord_for_arc,
)

ANCHORS_OUT = Path("/data/way_city_anchors.geojson")


MIN_SPACING_M  = 10_000.0
VILLAGE_FILES  = [
    Path("/data/osm/austria-villages.geojsonseq"),
    Path("/data/osm/czech-villages.geojsonseq"),
    Path("/data/osm/germany-villages.geojsonseq"),
    Path("/data/osm/denmark-villages.geojsonseq"),
]
# Ferry piers are promoted to first-class anchors so the chain graph
# connects ferry-linked coasts via real ferry polylines rather than
# synthetic straight-line edges between anchor centers. Piers are
# `_protected=True` — they survive the greedy dropout regardless of
# whether a bigger settlement lies within MIN_SPACING_M.
#
# Task #49: we now read `sea_piers.geojsonseq` (only piers on ferry
# systems > 20 km total length) instead of all piers. Short-hop
# river-crossing ferries stay in `ways` as is_ferry rows the bike
# Dijkstra can still take, but they don't get chain anchors or paired
# SPTs — chain-Dijkstra can't zip through a 5-hop Vltava pier sequence
# when the actual bike path should follow the riverside road.
#
# `classify_piers.py` (stage 2b in run_full_rebuild.sh) produces
# sea_piers.geojsonseq. If that file doesn't exist yet, fall back to
# the legacy `ferry_piers.geojsonseq` — same shape, all piers included.
FERRY_PIERS_FILE = Path("/data/sea_piers.geojsonseq")
FERRY_PIERS_FALLBACK = Path("/data/ferry_piers.geojsonseq")


def _load_all_places(conn: psycopg.Connection) -> list[dict]:
    db_anchors = _load_db_anchors(conn)
    for a in db_anchors:
        a["pop"] = int(a["population"]) if a["population"] else 0
    villages: list[dict] = []
    seq = 0
    for path in VILLAGE_FILES:
        if not path.exists():
            print(f"[bottom-up] missing villages file: {path}", flush=True)
            continue
        country_stem = path.stem.split("-")[0]  # e.g. "austria"
        n0 = len(villages)
        for line in open(path):
            line = line.strip().lstrip("\x1e").strip()
            if not line:
                continue
            f = json.loads(line)
            lon, lat = f["geometry"]["coordinates"]
            props = f.get("properties", {})
            try:
                pop = int(props["population"]) if props.get("population") else 0
            except (TypeError, ValueError):
                pop = 0
            osm_id = str(f.get("id") or props.get("@id") or "")
            ref = f"osm:{osm_id}" if osm_id else f"osm:seq-{seq}"
            seq += 1
            villages.append({
                "kind":       "village",
                "ref":        ref,
                "name":       props.get("name") or "?",
                "place":      "village",
                "population": pop if pop else None,
                "country":    country_stem,
                "lon":        float(lon),
                "lat":        float(lat),
                "pop":        pop,
            })
        print(f"[bottom-up]   loaded {len(villages)-n0:,} villages from "
              f"{path.name}", flush=True)

    ferry_piers: list[dict] = []
    piers_path = FERRY_PIERS_FILE if FERRY_PIERS_FILE.exists() else (
        FERRY_PIERS_FALLBACK if FERRY_PIERS_FALLBACK.exists() else None
    )
    if piers_path is not None:
        for line in open(piers_path):
            line = line.strip().lstrip("\x1e").strip()
            if not line:
                continue
            f = json.loads(line)
            lon, lat = f["geometry"]["coordinates"]
            props = f.get("properties", {})
            vid = int(props.get("vid") or 0)
            ferry_piers.append({
                "kind":        "ferry_pier",
                "ref":         f"ferry:{vid}",
                "name":        props.get("name") or f"Ferry pier {vid}",
                "place":       "ferry_pier",
                "population":  None,
                "country":     None,
                "lon":         float(lon),
                "lat":         float(lat),
                "pop":         0,
                "_protected": True,   # never dropped by _greedy_dropout
                "vid":         vid,
            })
        print(f"[bottom-up]   loaded {len(ferry_piers):,} ferry piers from "
              f"{piers_path.name}", flush=True)
    else:
        print(f"[bottom-up] missing ferry piers file: "
              f"{FERRY_PIERS_FILE} (and fallback {FERRY_PIERS_FALLBACK})",
              flush=True)

    return db_anchors + villages + ferry_piers


PIER_DEDUP_M = float(os.environ.get("PIER_DEDUP_M", "500"))
PIER_EATS_TOWN_M = float(os.environ.get("PIER_EATS_TOWN_M", "5000"))


def _dedup_and_absorb(places: list[dict]) -> list[dict]:
    """Two pier-focused passes before the main greedy dropout.

    1. Dedup piers within PIER_DEDUP_M (default 500 m). OSM often has
       multiple pier features per physical ferry landing (different
       ferry routes docking at the same terminal); we keep the first
       one seen per cluster.
    2. Piers "eat" nearby town anchors: any town/village whose centroid
       is within PIER_EATS_TOWN_M (default 5 km) of a surviving pier
       gets dropped. If the town has a name/population, splice its
       name onto the pier so the anchor retains a human-readable
       identity (e.g. "Gedser" instead of "Ferry pier 34988792").

    Rationale: user diagnostic on Graz→Copenhagen — the Rødby ferry
    pier and Gedser town were both anchors 570 m apart. Chain-Dijkstra
    picked hops through both, leading to a spurious near-zero-distance
    chain edge + a 49 km paired-SPT handoff gap. Merging them into one
    anchor eliminates the failing handoff.
    """
    if not places:
        return places
    is_pier = np.array([p["kind"] == "ferry_pier" for p in places])
    xyz = _lonlat_to_xyz(
        np.array([p["lon"] for p in places]),
        np.array([p["lat"] for p in places]),
    )
    tree = cKDTree(xyz)

    # Pass 1: pier dedup.
    keep = np.ones(len(places), dtype=bool)
    chord_dedup = _chord_for_arc(PIER_DEDUP_M)
    for i in np.flatnonzero(is_pier):
        if not keep[i]:
            continue
        nearby = tree.query_ball_point(xyz[i], r=chord_dedup)
        for j in nearby:
            if j != i and keep[j] and is_pier[j]:
                keep[j] = False
    n_dedup = int((~keep & is_pier).sum())

    # Pass 2: piers eat nearby towns. Note we splice the town's name
    # (and population, for future promotion logic) onto the pier so the
    # anchor retains its human-readable identity.
    chord_eat = _chord_for_arc(PIER_EATS_TOWN_M)
    n_eaten = 0
    for i in np.flatnonzero(is_pier & keep):
        nearby = tree.query_ball_point(xyz[i], r=chord_eat)
        best_j = -1
        best_pop = -1
        for j in nearby:
            if j == i or not keep[j] or is_pier[j]:
                continue
            # eat the highest-pop town in range so identity attaches to
            # the "real" town for the anchor
            pop = places[j].get("pop", 0) or 0
            if pop > best_pop:
                best_pop = pop; best_j = j
        # Then drop ALL non-pier neighbours in range (whether or not
        # they were the "identity" one).
        for j in nearby:
            if j == i or not keep[j] or is_pier[j]:
                continue
            keep[j] = False
            n_eaten += 1
        if best_j >= 0:
            t = places[best_j]
            places[i]["name"] = t.get("name") or places[i].get("name")
            # Preserve town's population so downstream can still favor
            # this anchor by importance.
            places[i]["pop"] = max(places[i].get("pop", 0) or 0,
                                   int(t.get("population") or 0))
            # Bookkeeping — makes it obvious in the anchors geojson.
            places[i]["_absorbed_ref"] = t.get("ref")

    print(f"[bottom-up]   pier dedup dropped {n_dedup}, "
          f"piers ate {n_eaten} nearby town anchors", flush=True)
    return [places[i] for i in range(len(places)) if keep[i]]


def _greedy_dropout(places: list[dict], min_spacing_m: float) -> list[dict]:
    """Iterate places in ascending-pop order. Drop a place if any OTHER
    still-kept, NON-protected place lies within min_spacing_m. Returns
    the survivors.

    Entries with `_protected=True` (ferry piers) are never dropped, and
    ALSO don't displace other places. Without that second half of the
    rule, coastal/riverside cities like Wien (Danube), Hamburg (Elbe),
    Praha (Vltava), København (Øresund) were being dropped in favor of
    a ferry pier within 10 km — silently deleting them from routing.
    """
    n = len(places)
    xyz = _lonlat_to_xyz(
        np.array([p["lon"] for p in places]),
        np.array([p["lat"] for p in places]),
    )
    tree = cKDTree(xyz)
    chord = _chord_for_arc(min_spacing_m)

    keep = np.ones(n, dtype=bool)
    protected = np.array([p.get("_protected", False) for p in places], dtype=bool)
    # Iterate by pop ASC; ties broken by ref for determinism.
    order = sorted(range(n), key=lambda i: (places[i]["pop"], places[i]["ref"]))
    n_dropped_by_bucket = {"city": 0, "town": 0, "village": 0, "ferry_pier": 0}
    for i in order:
        if not keep[i]:
            continue
        if protected[i]:
            continue          # never drop protected entries
        nearby = tree.query_ball_point(xyz[i], r=chord)
        # Drop if any OTHER still-kept NON-protected place lies within
        # chord. Protected entries (ferry piers) don't count as
        # displacers — they can coexist with real anchors.
        has_neighbor = any(
            j != i and keep[j] and not protected[j] for j in nearby
        )
        if has_neighbor:
            keep[i] = False
            kind = places[i]["place"] if places[i]["place"] in n_dropped_by_bucket else "village"
            n_dropped_by_bucket[kind] += 1

    survivors = [places[i] for i in range(n) if keep[i]]
    print(f"[bottom-up] dropped {n - len(survivors):,} of {n:,}  "
          f"(cities -{n_dropped_by_bucket['city']}, "
          f"towns -{n_dropped_by_bucket['town']}, "
          f"villages -{n_dropped_by_bucket['village']}, "
          f"ferry piers -{n_dropped_by_bucket['ferry_pier']})",
          flush=True)
    return survivors


def _write_anchors(survivors: list[dict]) -> None:
    features = []
    for s in survivors:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
            "properties": {
                "ref":        s["ref"],
                "name":       s["name"],
                "place":      s.get("place"),
                "population": s.get("population"),
                "country":    s.get("country"),
                "kind":       s.get("kind"),
                # Placeholder values overwritten by connect_anchors_pairs.py
                # after the chain build. Set true here so the web app shows
                # every anchor as fully opaque if pairs hasn't run yet.
                "in_graph":   True,
                "snap_dist_m": 0,
            },
        })
    ANCHORS_OUT.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": features,
    }))
    print(f"[bottom-up] wrote {len(features):,} anchors -> {ANCHORS_OUT}",
          flush=True)


def main() -> None:
    t0 = time.time()
    print(f"[bottom-up] min spacing {MIN_SPACING_M:.0f} m  "
          f"({MIN_SPACING_M/1000:.1f} km)", flush=True)

    with psycopg.connect(config.PG_DSN) as conn:
        print("[bottom-up] loading candidate places…", flush=True)
        places = _load_all_places(conn)
    print(f"[bottom-up] {len(places):,} candidate places "
          f"({sum(1 for p in places if p['kind']=='db'):,} db + "
          f"{sum(1 for p in places if p['kind']=='village'):,} villages + "
          f"{sum(1 for p in places if p['kind']=='ferry_pier'):,} ferry piers)",
          flush=True)

    print(f"[bottom-up] pier dedup + eats-town pass "
          f"(dedup {PIER_DEDUP_M:.0f} m, eat radius {PIER_EATS_TOWN_M:.0f} m)…",
          flush=True)
    places = _dedup_and_absorb(places)
    print(f"[bottom-up]   {len(places):,} places remain after pier passes",
          flush=True)

    print("[bottom-up] running greedy dropout…", flush=True)
    t = time.time()
    survivors = _greedy_dropout(places, MIN_SPACING_M)
    print(f"[bottom-up] {len(survivors):,} survivors in {time.time()-t:.1f}s",
          flush=True)

    pops = sorted((p["pop"] for p in survivors), reverse=True)
    if pops:
        print(f"[bottom-up] survivor pop: max={pops[0]:,} "
              f"p25={pops[len(pops)*3//4 if pops else 0]:,} "
              f"p50={pops[len(pops)//2 if pops else 0]:,} "
              f"min={pops[-1]:,}",
              flush=True)

    _write_anchors(survivors)
    print(f"[bottom-up] DONE in {time.time()-t0:.1f}s — run "
          f"connect_anchors_pairs.py next to build chain edges",
          flush=True)


if __name__ == "__main__":
    main()
