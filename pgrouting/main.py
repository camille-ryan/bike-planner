"""pgRouting-backed preprocess orchestrator.

  python3 main.py ingest             --countries austria
  python3 main.py snap               --countries austria
  python3 main.py boundaries         --countries austria
  python3 main.py dem-download       --countries austria
  python3 main.py dem-ingest
  python3 main.py landcover-ingest   --countries austria
  python3 main.py canopy-compute
  python3 main.py recompute-cost
  python3 main.py spts               --profile lht
  python3 main.py paired             --profile lht
  python3 main.py all                --countries austria --profile lht

Subcommands (run order):
  ingest            stream OSM PBFs into Postgres ways + ways_vertices_pgr
                    (V2: also persists raw OSM tags + per-edge curvature)
  snap              load anchors from pois.sqlite, snap to nearest graph vertex
  boundaries        stream admin polygons into anchors.geom_boundary
  dem-download      fetch Copernicus DEM GLO-30 tiles covering the country bbox
                    union into data/dem/ (V2 Phase A.2)
  dem-ingest        bilinear-sample elevation at every vertex →
                    ways_vertices_pgr.elev_m
  landcover-ingest  stream `*-landuse.osm.pbf` files into `landcover` table;
                    forest polygons only for now (V2 Phase A.3 — scenicness
                    tree-cover signal). Per-country, idempotent.
  canopy-compute    per-edge fraction inside a forest polygon →
                    ways.canopy_frac. Universal cost-bonus input.
  recompute-cost    re-apply bike_edge_cost over every edge with grade_pct +
                    curvature + canopy_frac; writes cost / reverse_cost
                    in place
  spts              per-anchor 30 km multi-source Dijkstra → SPT npzs;
                    also writes road_topology/<id>.npz (lon/lat per vertex,
                    shared across profiles) and city_graph.json (with ferry
                    chain edges added for long sea/lake crossings)
  paired            pruned paired SPTs as a SQLite trunk DB plus optional
                    per-pair npzs (for trace-level inspection)
  all               ingest → snap → boundaries → dem-download → dem-ingest →
                    landcover-ingest → canopy-compute → recompute-cost →
                    spts → paired

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
import ingest_landcover
import ingest_coastline
import ingest_waterways
import ingest_railways
import ingest_lodging
import download_gtfs
import route_city_pairs
import compute_canopy_frac
import compute_canopy_frac_raster
from scenicness import bake as scenicness_bake
from scenicness import signals as scenicness_signals
from scenicness import tiles as scenicness_tiles
import compare_canopy
import reannotate_canopy_km
import snap_anchors
import compute_spts
import download_dem
import recompute_cost
import export_route_compare


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


def cmd_landcover_ingest(args) -> None:
    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    # `*-landuse.osm.pbf` lives next to `*-latest.osm.pbf` (DATA_DIR/osm)
    # is too unrelated; the landuse extracts are stored under
    # DATA_DIR/landcover by the upstream extract pipeline
    # (extract_landuse_pbfs.sh — runs osmium tags-filter so multipolygon
    # relation members are preserved for downstream area assembly).
    landcover_dir = config.DATA_DIR / "landcover"
    pbfs = [landcover_dir / f"{c}-landuse.osm.pbf" for c in countries]
    bbox = _parse_bbox(args.bbox)
    print(f"[main] landcover-ingest countries={countries} bbox={bbox}")
    with psycopg.connect(config.PG_DSN) as conn:
        ingest_landcover.ingest(conn, pbfs, countries, bbox=bbox)
    print("[main] landcover-ingest done")


def cmd_coastline_ingest(args) -> None:
    """One-shot global ingest of pre-built sea polygons into landcover.

    Downloads simplified-water-polygons-split-3857.zip from
    osmdata.openstreetmap.de on first run (~23 MB) and inserts every
    polygon with class='sea', country='_coastline'. Re-runs are
    idempotent: the `_coastline` rows are cleared and reloaded.
    """
    print("[main] coastline-ingest")
    with psycopg.connect(config.PG_DSN) as conn:
        ingest_coastline.ingest(conn)
    print("[main] coastline-ingest done")


def cmd_gtfs_download(args) -> None:
    """Download national GTFS feeds for the railway ingest."""
    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    force = bool(getattr(args, "force", False))
    for c in countries:
        download_gtfs.download(c, force=force)


def cmd_railway_ingest(args) -> None:
    """GTFS stations + OSM lines + spatial filter → rail_stations / rail_lines.

    Per-country, idempotent. Requires the GTFS zip in `data/gtfs/` and
    the country PBF in `data/osm/`. Run `gtfs-download` first.
    """
    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    print(f"[main] railway-ingest countries={countries}")
    with psycopg.connect(config.PG_DSN) as conn:
        for c in countries:
            gtfs_zip = download_gtfs.feed_path(c)
            if not gtfs_zip.exists():
                raise SystemExit(
                    f"missing GTFS zip for {c}: {gtfs_zip}. "
                    f"Run gtfs-download first."
                )
            pbf = config.OSM_DIR / f"{c}-latest.osm.pbf"
            if not pbf.exists():
                raise SystemExit(f"missing PBF for {c}: {pbf}")
            ingest_railways.ingest(conn, c, gtfs_zip, pbf)
    print("[main] railway-ingest done")


def cmd_route_city_pairs(args) -> None:
    """Pre-compute all 15 city-pair routes for ONE cost profile against
    the currently-loaded ways.cost column.

    Appends one Feature per pair to <out> (creates the file or merges
    into an existing FeatureCollection). The outer pipeline calls
    `recompute-cost --profile X` before each invocation to refresh
    ways.cost for that profile.
    """
    import json
    from cost import _PROFILES
    profile = args.profile or "direct"
    if profile == "lht":
        profile = "direct"
    if profile not in _PROFILES:
        raise SystemExit(f"unknown profile {profile!r}; have {sorted(_PROFILES)}")
    out_path = Path(args.out) if args.out else (
        config.DATA_DIR / "web_overlays" / "city_routes.geojson"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Optional pair filter, e.g. "INN-WIE" or "INN-WIE,SAL-LIN" — useful
    # for smoke-testing on the historically OOM-prone Innsbruck-Vienna
    # pair before committing to all 15.
    if args.pairs:
        wanted = set()
        for token in args.pairs.split(","):
            a, b = token.strip().split("-")
            wanted.add(frozenset({a.upper(), b.upper()}))
        pairs = [(a, b) for (a, b) in route_city_pairs.all_pairs()
                 if frozenset({a, b}) in wanted]
    else:
        pairs = None  # all 15
    corridor_m = float(args.corridor_m) if args.corridor_m else None
    kwargs = {"corridor_m": corridor_m} if corridor_m is not None else {}
    print(f"[main] route-city-pairs profile={profile} "
          f"pairs={[(a,b) for a,b in (pairs or route_city_pairs.all_pairs())]} "
          f"corridor_m={corridor_m}")
    with psycopg.connect(config.PG_DSN) as conn:
        results = route_city_pairs.route_all_pairs(conn, profile, pairs=pairs, **kwargs)
    # Read-modify-write: append features to existing FeatureCollection.
    if out_path.exists():
        fc = json.loads(out_path.read_text())
        features = fc.get("features", [])
    else:
        features = []
    # Drop any prior features for the (a,b,profile) tuples we just ran
    # so re-runs replace rather than duplicate.
    redo_keys = {(r.a, r.b, r.profile) for r in results}
    features = [
        f for f in features
        if (f["properties"].get("a"),
            f["properties"].get("b"),
            f["properties"].get("profile")) not in redo_keys
    ]
    for r in results:
        features.append({
            "type": "Feature",
            "geometry": json.loads(r.geom_geojson),
            "properties": {
                "a": r.a, "b": r.b, "profile": r.profile,
                "length_km": r.length_km, "cost": r.cost,
                "n_edges": r.n_edges,
            },
        })
    out_path.write_text(json.dumps({
        "type": "FeatureCollection", "features": features,
    }))
    print(f"[main] route-city-pairs wrote {len(results)} routes → {out_path} "
          f"(file now contains {len(features)} features total)")


def cmd_lodging_ingest(args) -> None:
    """Materialize lodging POIs from pois.sqlite into postgres."""
    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    print(f"[main] lodging-ingest countries={countries}")
    with psycopg.connect(config.PG_DSN) as conn:
        ingest_lodging.ingest(conn, countries=countries)
    print("[main] lodging-ingest done")


def cmd_export_rails(args) -> None:
    """Dump rail_stations + rail_lines for one or more countries into
    GeoJSON files for the web overlay.

    Stations are filtered to those with at least one lodging point
    (hotel / guest_house / hostel / motel) within `--lodging-radius-m`
    of the station (default 3000 m). Pass `--lodging-radius-m 0` to
    disable the filter entirely.

    Output: <data-dir>/rail_stations.geojson, rail_lines.geojson
    """
    import json
    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    out_dir = Path(args.out) if args.out else (config.DATA_DIR / "web_overlays")
    out_dir.mkdir(parents=True, exist_ok=True)
    stations_path = out_dir / "rail_stations.geojson"
    lines_path = out_dir / "rail_lines.geojson"
    lodging_radius_m = float(args.lodging_radius_m) if args.lodging_radius_m else 3000.0
    # The "real" lodging types for the daily-meet use case — campsites
    # and wilderness huts are excluded from the station-filter signal.
    lodging_subtypes = ("hotel", "guest_house", "hostel", "motel")
    print(f"[main] export-rails countries={countries} → {out_dir} "
          f"lodging_radius_m={lodging_radius_m:.0f}")
    with psycopg.connect(config.PG_DSN) as conn, conn.cursor() as cur:
        # Stations + lodging count via LATERAL spatial join.
        if lodging_radius_m > 0:
            cur.execute(
                "SELECT s.gtfs_id, s.name, s.n_routes, s.country, "
                "       ST_X(s.geom), ST_Y(s.geom), nl.n_lodging "
                "FROM rail_stations s "
                "JOIN LATERAL (SELECT COUNT(*) AS n_lodging FROM lodging l "
                "  WHERE l.subtype = ANY(%s) "
                "    AND ST_DWithin(l.geom::geography, s.geom::geography, %s)"
                ") nl ON TRUE "
                "WHERE s.country = ANY(%s) AND nl.n_lodging >= 1 "
                "ORDER BY s.n_routes DESC",
                (list(lodging_subtypes), lodging_radius_m, countries),
            )
        else:
            cur.execute(
                "SELECT gtfs_id, name, n_routes, country, "
                "       ST_X(geom), ST_Y(geom), 0 "
                "FROM rail_stations WHERE country = ANY(%s) "
                "ORDER BY n_routes DESC",
                (countries,),
            )
        stations_features = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "gtfs_id": gtfs_id, "name": name,
                    "n_routes": n_routes, "country": country,
                    "n_lodging": int(n_lodging),
                },
            }
            for gtfs_id, name, n_routes, country, lon, lat, n_lodging in cur
        ]
        # Lines
        cur.execute(
            "SELECT osm_id, name, operator, usage, electrified, country, "
            "       ST_AsGeoJSON(geom) "
            "FROM rail_lines WHERE country = ANY(%s)",
            (countries,),
        )
        lines_features = [
            {
                "type": "Feature",
                "geometry": json.loads(geom_json),
                "properties": {
                    "osm_id": osm_id, "name": name,
                    "operator": operator, "usage": usage,
                    "electrified": electrified, "country": country,
                },
            }
            for osm_id, name, operator, usage, electrified, country, geom_json in cur
        ]
    stations_path.write_text(json.dumps({
        "type": "FeatureCollection", "features": stations_features,
    }))
    lines_path.write_text(json.dumps({
        "type": "FeatureCollection", "features": lines_features,
    }))
    print(f"[main] wrote {len(stations_features):,} stations → "
          f"{stations_path}")
    print(f"[main] wrote {len(lines_features):,} lines → {lines_path}")


def cmd_waterway_ingest(args) -> None:
    """Stream full country PBFs for waterway lines, buffer to ~10m
    polygons, insert as class='waterway'. Requires the landcover table
    to already exist (run landcover-ingest first).
    """
    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    # Full country PBFs (not the landuse extract — too aggressive a filter
    # upstream drops most waterway lines).
    pbfs = [config.OSM_DIR / f"{c}-latest.osm.pbf" for c in countries]
    bbox = _parse_bbox(args.bbox)
    print(f"[main] waterway-ingest countries={countries} bbox={bbox}")
    with psycopg.connect(config.PG_DSN) as conn:
        ingest_waterways.ingest(conn, pbfs, countries, bbox=bbox)
    print("[main] waterway-ingest done")


def cmd_canopy_compute(args) -> None:
    bbox = _parse_bbox(args.bbox)
    print(f"[main] canopy-compute bbox={bbox}")
    with psycopg.connect(config.PG_DSN) as conn:
        compute_canopy_frac.compute(conn, bbox=bbox)
    print("[main] canopy-compute done")


def cmd_restitch_overlays(args) -> None:
    """Re-stitch global PNG overlays from existing per-tile PNGs.

    Reads {export_rasters}/manifest.json + {export_rasters}/tiles/*.png
    and rewrites {export_rasters}/<col>.png with the current
    _stitch_signal logic. Useful after changing stitch/downsample
    behavior — avoids re-running the multi-hour bake when only the
    final overlay rendering changed.
    """
    import json
    import numpy as np
    if not args.export_rasters:
        raise SystemExit("restitch-overlays requires --export-rasters")
    export_dir = Path(args.export_rasters)
    tiles_dir = export_dir / "tiles"
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest: {manifest_path}")
    if not tiles_dir.is_dir():
        raise SystemExit(f"no tiles dir: {tiles_dir}")
    manifest = json.loads(manifest_path.read_text())
    extent = tuple(manifest["bbox"])
    res_m = float(manifest["res_m"])
    tile_size_deg = float(manifest.get("tile_size_deg", 1.0))
    cos_lat = float(np.cos(np.radians((extent[1] + extent[3]) / 2.0)))
    # Reconstruct per-signal tile entries from filenames.
    # Filename pattern: <col>_t<NNN>.png ; the index encodes the tile's
    # position in _iter_tiles() ordering.
    all_tiles = scenicness_bake._iter_tiles(extent, tile_size_deg)
    print(f"[restitch] extent={extent} res_m={res_m} "
          f"tile_size_deg={tile_size_deg} cos_lat={cos_lat:.4f}")
    print(f"[restitch] {len(all_tiles)} candidate tiles, "
          f"{len(manifest['signals'])} signals")
    for col in manifest["signals"]:
        tile_pngs = sorted(tiles_dir.glob(f"{col}_t*.png"))
        if not tile_pngs:
            print(f"[restitch]   {col}: no tile PNGs, skipping")
            continue
        entries = []
        for png_path in tile_pngs:
            stem = png_path.stem  # e.g. "forest_local_t007"
            ti = int(stem.rsplit("_t", 1)[1])
            tile_bbox = all_tiles[ti]
            entries.append((col, png_path, None, tile_bbox))
        out_path = export_dir / f"{col}.png"
        print(f"\n[restitch] {col}: {len(entries)} tiles -> {out_path}")
        scenicness_bake._stitch_signal(
            col, entries, extent, res_m, out_path, cos_lat,
        )
    print("[restitch] done")


def cmd_build_tiles(args) -> None:
    """Build a Web-Mercator XYZ raster tile pyramid from the per-tile PNG
    fragments — the scalable replacement for the single full-extent overlay
    PNG (which is ~34 GB at 4-country/20 m scale). MapLibre lazy-loads
    {z}/{x}/{y}.png per viewport. Reads bbox/res/signals from manifest.json;
    rewrites the manifest to point each signal at its tile template."""
    import json
    import numpy as np
    if not args.export_rasters:
        raise SystemExit("build-tiles requires --export-rasters")
    export_dir = Path(args.export_rasters)
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    extent = tuple(manifest["bbox"])
    res_m = float(manifest["res_m"])
    tile_size_deg = float(manifest.get("tile_size_deg", 1.0))
    cos_lat = float(np.cos(np.radians((extent[1] + extent[3]) / 2.0)))
    signals = list(manifest["signals"].keys())
    zmin = int(args.zoom_min) if args.zoom_min else 4
    zmax = int(args.zoom_max) if args.zoom_max else 12
    print(f"[build-tiles] extent={extent} res_m={res_m} "
          f"tile_size_deg={tile_size_deg} signals={len(signals)} z{zmin}-{zmax}")
    summary = scenicness_tiles.build_pyramid(
        export_dir, signals, extent, res_m, tile_size_deg, cos_lat,
        zoom_min=zmin, zoom_max=zmax,
    )
    # Point the manifest at the tile pyramid so the web uses a raster source.
    for col in manifest["signals"]:
        manifest["signals"][col]["tiles"] = f"xyz/{col}/{{z}}/{{x}}/{{y}}.png"
        manifest["signals"][col]["minzoom"] = zmin
        manifest["signals"][col]["maxzoom"] = zmax
    manifest["tile_layout"] = "xyz"
    manifest["minzoom"] = zmin
    manifest["maxzoom"] = zmax
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[build-tiles] done: {sum(summary.values())} tiles "
          f"across {len(summary)} signals")


def cmd_scenicness_bake(args) -> None:
    """Compute per-edge scenicness signals in one pass.

    --bbox is optional. When omitted, bakes the full ways-table extent
    via internal tiling (see bake.py)."""
    bbox = _parse_bbox(args.bbox)
    if args.signals:
        names = [s.strip() for s in args.signals.split(",") if s.strip()]
    else:
        names = list(scenicness_signals.SIGNALS.keys())
    unknown = [n for n in names if n not in scenicness_signals.SIGNALS]
    if unknown:
        raise SystemExit(f"unknown signal(s): {unknown}; "
                         f"available: {list(scenicness_signals.SIGNALS.keys())}")
    res_m = float(args.res_m) if args.res_m else 20.0
    tile_size_deg = float(args.tile_size_deg) if args.tile_size_deg else 1.0
    export_dir = Path(args.export_rasters) if args.export_rasters else None
    print(f"[main] scenicness-bake bbox={bbox} res_m={res_m} "
          f"tile_size_deg={tile_size_deg} signals={names} "
          f"export_rasters={export_dir}")
    with psycopg.connect(config.PG_DSN) as conn:
        scenicness_bake.bake(
            conn, names, bbox, res_m=res_m,
            dem_dir=config.DEM_DIR,
            export_rasters_dir=export_dir,
            tile_size_deg=tile_size_deg,
        )
    print("[main] scenicness-bake done")


def cmd_canopy_compute_raster(args) -> None:
    bbox = _parse_bbox(args.bbox)
    if bbox is None:
        raise SystemExit("canopy-compute-raster requires --bbox")
    res_m = float(args.res_m) if args.res_m else 20.0
    print(f"[main] canopy-compute-raster bbox={bbox} res_m={res_m}")
    with psycopg.connect(config.PG_DSN) as conn:
        compute_canopy_frac_raster.compute(conn, bbox, res_m=res_m)
    print("[main] canopy-compute-raster done")


def cmd_reannotate_canopy_km(args) -> None:
    """Re-annotate canopy_km on the existing compare GeoJSON without
    re-routing. See pgrouting/reannotate_canopy_km.py for details."""
    in_path = Path(args.out) if args.out else reannotate_canopy_km.DEFAULT_PATH
    if not in_path.exists():
        raise SystemExit(f"missing GeoJSON: {in_path}")
    print(f"[main] reannotate-canopy-km -> {in_path}")
    import json
    with in_path.open() as h:
        fc = json.load(h)
    with psycopg.connect(config.PG_DSN) as conn:
        for feat in fc.get("features", []):
            v = feat.get("properties", {}).get("variant", "?")
            print(f"[main]   variant={v}")
            reannotate_canopy_km._annotate_feature(conn, feat)
    with in_path.open("w") as h:
        json.dump(fc, h)
    for feat in fc["features"]:
        p = feat["properties"]
        print(f"[main]   variant={p.get('variant')}: "
              f"canopy_km={p.get('canopy_km')}")


def cmd_compare_canopy(args) -> None:
    bbox = _parse_bbox(args.bbox)
    out_path = Path(args.out) if args.out else None
    print(f"[main] compare-canopy bbox={bbox} -> {out_path}")
    with psycopg.connect(config.PG_DSN) as conn:
        compare_canopy.compare(conn, bbox=bbox, out_path=out_path)
    print("[main] compare-canopy done")


def cmd_recompute_cost(args) -> None:
    bbox = _parse_bbox(args.bbox)
    # Map legacy "lht" alias to "direct" for backward compat. The
    # shared --profile argparse default is "lht" (used by spts/paired
    # as an output-directory name), so existing recompute-cost callers
    # without --profile get "lht" → mapped to "direct" here.
    profile = args.profile or "direct"
    if profile == "lht":
        profile = "direct"
    # Import here to avoid circular import at module load.
    from cost import _PROFILES
    if profile not in _PROFILES:
        raise SystemExit(
            f"unknown --profile={profile!r}; "
            f"expected one of {sorted(_PROFILES.keys())}"
        )
    print(f"[main] recompute-cost bbox={bbox} profile={profile}")
    with psycopg.connect(config.PG_DSN) as conn:
        recompute_cost.recompute(conn, bbox=bbox, profile=profile)
    print("[main] recompute-cost done")


def cmd_export_route_compare(args) -> None:
    # In-container default writes to /data/ (mounted as ./data/ on host);
    # user copies the file out to web/public/data/. The standalone script
    # (run on host) has a different default that points directly at the
    # web tree.
    out = Path(args.out) if args.out else (config.DATA_DIR / "graz_wien_compare.geojson")
    if not args.variant:
        raise SystemExit("export-route-compare requires --variant (no_canopy | with_canopy)")

    # Optional OD override. Each must be "lon,lat". If omitted, build_feature
    # defaults to Graz<->Wien (the historical OD pair for this script).
    def _parse_pt(s: str | None) -> tuple[float, float] | None:
        if not s:
            return None
        parts = tuple(float(x) for x in s.split(","))
        if len(parts) != 2:
            raise SystemExit(f"expected lon,lat pair, got {s!r}")
        return parts
    start_pt = _parse_pt(args.start_lonlat)
    end_pt   = _parse_pt(args.end_lonlat)

    print(f"[main] export-route-compare variant={args.variant} -> {out}"
          + (f"  start={start_pt}" if start_pt else "")
          + (f"  end={end_pt}" if end_pt else ""))
    import json
    with psycopg.connect(config.PG_DSN) as conn:
        kw = {}
        if start_pt is not None: kw["start_lonlat"] = start_pt
        if end_pt is not None:   kw["end_lonlat"]   = end_pt
        feat = export_route_compare.build_feature(conn, args.variant, **kw)
    if out.exists():
        with out.open() as h:
            fc = json.load(h)
        fc["features"] = [
            f for f in fc.get("features", [])
            if f.get("properties", {}).get("variant") != args.variant
        ]
    else:
        fc = {"type": "FeatureCollection", "features": []}
    fc["features"].append(feat)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as h:
        json.dump(fc, h)
    p = feat["properties"]
    print(f"[main] export-route-compare done: edges={p['edges']} "
          f"length_km={p['length_km']} climb_m={p['climb_m']} "
          f"canopy_km={p['canopy_km']}")
    print(f"[main]   variants in file: "
          f"{[f['properties']['variant'] for f in fc['features']]}")


def _parse_bbox(arg: str | None) -> tuple[float, float, float, float] | None:
    if not arg:
        return None
    parts = tuple(float(x) for x in arg.split(","))
    if len(parts) != 4:
        raise SystemExit("--bbox must be 4 comma-separated floats")
    return parts  # type: ignore[return-value]


def cmd_spts(args) -> None:
    out_dir = config.SPT_DIR / args.profile
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[main] spts profile={args.profile} -> {out_dir}")
    with psycopg.connect(config.PG_DSN) as conn:
        compute_spts.run(conn, out_dir)
    print("[main] spts done")


def cmd_spts_multi(args) -> None:
    """Multi-profile per-anchor backward SPTs over a shared CSR.

    V3 replacement for cmd_spts. Loads the global edge graph once into
    an in-memory transposed CSR with all profile cost columns attached,
    then dispatches (anchor × profile) dijkstras against it. See
    compute_spts_multi.py module docstring for design notes.
    """
    import compute_spts_multi

    profiles = (
        tuple(s.strip() for s in args.profiles.split(","))
        if args.profiles else compute_spts_multi.PROFILES
    )
    cache_path = Path(args.cache_path) if args.cache_path else None
    compute_spts_multi.run(
        profiles=profiles,
        anchor_filter=args.anchor,
        max_radius_m=float(args.max_radius_m) if args.max_radius_m else
            compute_spts_multi.SPT_RADIUS_M_DEFAULT,
        force=args.force,
        cache_path=cache_path,
        force_cache=args.force_cache,
    )


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
    cmd_landcover_ingest(args)
    cmd_canopy_compute(args)
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
        ("landcover-ingest", cmd_landcover_ingest),
        ("coastline-ingest", cmd_coastline_ingest),
        ("waterway-ingest", cmd_waterway_ingest),
        ("gtfs-download", cmd_gtfs_download),
        ("railway-ingest", cmd_railway_ingest),
        ("lodging-ingest", cmd_lodging_ingest),
        ("export-rails", cmd_export_rails),
        ("route-city-pairs", cmd_route_city_pairs),
        ("canopy-compute", cmd_canopy_compute),
        ("canopy-compute-raster", cmd_canopy_compute_raster),
        ("scenicness-bake", cmd_scenicness_bake),
        ("restitch-overlays", cmd_restitch_overlays),
        ("build-tiles", cmd_build_tiles),
        ("compare-canopy", cmd_compare_canopy),
        ("reannotate-canopy-km", cmd_reannotate_canopy_km),
        ("recompute-cost", cmd_recompute_cost),
        ("export-route-compare", cmd_export_route_compare),
        ("spts", cmd_spts),
        ("spts-multi", cmd_spts_multi),
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
        sp.add_argument("--bbox", default=None,
            help="min_lon,min_lat,max_lon,max_lat — bbox-restrict canopy-compute "
                 "and recompute-cost (validation runs).")
        sp.add_argument("--variant", default=None,
            help="export-route-compare: variant tag (no_canopy | with_canopy)")
        sp.add_argument("--out", default=None,
            help="export-route-compare: output path. Default writes inside "
                 "the container to /data/graz_wien_compare.geojson — copy out "
                 "to web/public/data/ to view.")
        sp.add_argument("--start-lonlat", default=None,
            help="export-route-compare: start point as lon,lat. "
                 "Default = Graz (15.4395,47.0707).")
        sp.add_argument("--end-lonlat", default=None,
            help="export-route-compare: end point as lon,lat. "
                 "Default = Wien (16.3725,48.2082).")
        sp.add_argument("--res-m", default=None,
            help="canopy-compute-raster / scenicness-bake: raster "
                 "resolution in meters (default 20).")
        sp.add_argument("--signals", default=None,
            help="scenicness-bake: comma-separated signal names. "
                 "Default = every registered signal.")
        sp.add_argument("--export-rasters", default=None,
            help="scenicness-bake: write PNG overlays + manifest.json "
                 "to this directory (e.g. /data/web_overlays/scenicness).")
        sp.add_argument("--tile-size-deg", default=None,
            help="scenicness-bake: tile size in degrees (default 1.0). "
                 "Lower values reduce per-tile memory at higher tile-count cost.")
        sp.add_argument("--zoom-min", default=None,
            help="build-tiles: min XYZ zoom level (default 4).")
        sp.add_argument("--zoom-max", default=None,
            help="build-tiles: max XYZ zoom level (default 12, ~19 m/px).")
        sp.add_argument("--force", action="store_true",
            help="gtfs-download: redownload even if the local zip exists.")
        sp.add_argument("--lodging-radius-m", default=None,
            help="export-rails: only include stations with ≥1 lodging POI "
                 "(hotel / guest_house / hostel / motel) within this many "
                 "meters (default 3000). 0 disables the filter.")
        sp.add_argument("--pairs", default=None,
            help="route-city-pairs: comma-separated pair list "
                 "(e.g. 'INN-WIE,GRA-KLA'). Default = all 15.")
        sp.add_argument("--corridor-m", default=None,
            help="route-city-pairs: half-width of the line-buffer "
                 "corridor in meters (default 30000).")
        sp.add_argument("--anchor", default=None,
            help="spts-multi: anchor name ILIKE substring "
                 "(default: all anchors).")
        sp.add_argument("--profiles", default=None,
            help="spts-multi: comma-separated profile names "
                 "(default: all 5 V3 profiles). NOTE: --profile (singular) "
                 "is unused by spts-multi; use this plural form instead.")
        sp.add_argument("--max-radius-m", default=None,
            help="spts-multi: per-anchor SPT GEOGRAPHIC radius in meters "
                 "(default 30000). Spatial slice via postgres ST_DWithin.")
        sp.add_argument("--cache-path", default=None,
            help="spts-multi: override the global graph cache path.")
        sp.add_argument("--force-cache", action="store_true",
            help="spts-multi: rebuild the global graph cache.")
        sp.set_defaults(func=func)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
