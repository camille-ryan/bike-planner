-- Schema for the pgRouting-backed bike graph.
--
-- Chainless SPT preprocess: per-city multi-source Dijkstra from each
-- anchor's polygon, no global K=1 partition. Tables:
--
--   ways_vertices_pgr  road graph vertices
--   ways               directed edges with bike-cost weights
--   anchors            place=city|town POIs + their OSM admin polygons
--
-- compute_spts.py loads the graph into scipy CSR once, then runs a
-- bounded Dijkstra per anchor and writes <city_idx>.npz under
-- data/spt/<profile>/spt/. No `visited` or `city_adjacency` SQL tables
-- — adjacency is derived post-hoc from SPT overlaps and stored as
-- city_graph.json next to the npz files.
--
-- Run idempotently: every CREATE has IF NOT EXISTS so re-running
-- against an existing DB is a no-op for schema. Drop the tables
-- explicitly if you want to re-ingest from scratch.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgrouting;

-- Vertices: each unique OSM node referenced by an accepted way.
-- pgRouting's built-in functions look for `ways_vertices_pgr.id` by
-- convention; sticking to that name lets us call them without a
-- vertices_table override.
--
-- elev_m is sampled from Copernicus DEM GLO-30 by `ingest_dem.py`;
-- NULL until that stage runs. Used by the cost recompute step to
-- derive per-edge grade.
CREATE TABLE IF NOT EXISTS ways_vertices_pgr (
    id      bigserial PRIMARY KEY,
    osm_id  bigint UNIQUE NOT NULL,
    lon     double precision NOT NULL,
    lat     double precision NOT NULL,
    elev_m  real,
    -- Generated geometry column kept in sync with lon/lat. Spatial
    -- index lives on this column (pgRouting can use it for KNN snap).
    the_geom geometry(Point, 4326) GENERATED ALWAYS AS
             (ST_SetSRID(ST_MakePoint(lon, lat), 4326)) STORED
);
ALTER TABLE ways_vertices_pgr ADD COLUMN IF NOT EXISTS elev_m real;
CREATE INDEX IF NOT EXISTS ways_vertices_pgr_geom_idx
    ON ways_vertices_pgr USING gist(the_geom);

-- Edges: directed segments between adjacent OSM nodes within a way.
-- pgRouting treats `cost` as forward weight and `reverse_cost` as
-- backward weight. `reverse_cost = -1` means no reverse edge exists
-- (oneway), which is the convention pgr_dijkstra honours.
--
-- NB: source/target are NOT declared as FOREIGN KEY references. The
-- naive `REFERENCES ways_vertices_pgr(id)` triggers per-row FK index
-- lookups on every INSERT — for a 240M-edge corridor that adds ~10x
-- to ingest time. Our ingest pipeline JOINs against ways_vertices_pgr
-- to resolve the IDs, which is its own integrity check.
--
-- V2 Phase A.2: raw OSM tag columns + directional curvature + grade
-- are persisted at ingest time so cost can be recomputed (with
-- elevation + curvature) without re-parsing PBFs.
--   - tag columns (highway, surface, tracktype, oneway, bicycle,
--     cycleway, bicycle_road, access) are per-way but stored per-edge
--     for simplicity. Empty string when the tag is absent.
--   - curv_fwd / curv_rev are per-edge: the total absolute bend
--     angle (in degrees) the rider encounters in the next ~300 m of
--     polyline, looking *ahead* in the forward and reverse direction
--     of travel respectively. Default 0.0 = no bends.
--   - grade_pct is per-edge, derived from vertex elevations after
--     ingest_dem runs; defaults to 0.0 (flat) when uncomputed.
CREATE TABLE IF NOT EXISTS ways (
    gid           bigserial PRIMARY KEY,
    osm_way_id    bigint,
    source        bigint NOT NULL,
    target        bigint NOT NULL,
    cost          double precision NOT NULL,           -- length × cost-factor (forward)
    reverse_cost  double precision NOT NULL,           -- same backwards, or -1 if oneway
    length_m      double precision NOT NULL,
    is_ferry      boolean NOT NULL DEFAULT false,
    -- Raw OSM tag columns for cost recompute.
    highway       text NOT NULL DEFAULT '',
    surface       text NOT NULL DEFAULT '',
    tracktype     text NOT NULL DEFAULT '',
    oneway        text NOT NULL DEFAULT '',
    bicycle       text NOT NULL DEFAULT '',
    cycleway      text NOT NULL DEFAULT '',
    bicycle_road  text NOT NULL DEFAULT '',
    access        text NOT NULL DEFAULT '',
    -- Derived per-edge fields populated by ingest pipeline.
    curv_fwd      real NOT NULL DEFAULT 0.0,
    curv_rev      real NOT NULL DEFAULT 0.0,
    grade_pct     real NOT NULL DEFAULT 0.0,
    -- V2 scenicness: fraction of edge length whose centerline falls
    -- inside a `landcover.class='forest'` polygon. Populated by
    -- `compute_canopy_frac`; 0.0 when uncomputed. Used as a universal
    -- cost bonus ("trees overhead = shade & nicer ride"); applies to
    -- every profile, not just scenic.
    canopy_frac   real NOT NULL DEFAULT 0.0
);
ALTER TABLE ways ADD COLUMN IF NOT EXISTS highway      text NOT NULL DEFAULT '';
ALTER TABLE ways ADD COLUMN IF NOT EXISTS surface      text NOT NULL DEFAULT '';
ALTER TABLE ways ADD COLUMN IF NOT EXISTS tracktype    text NOT NULL DEFAULT '';
ALTER TABLE ways ADD COLUMN IF NOT EXISTS oneway       text NOT NULL DEFAULT '';
ALTER TABLE ways ADD COLUMN IF NOT EXISTS bicycle      text NOT NULL DEFAULT '';
ALTER TABLE ways ADD COLUMN IF NOT EXISTS cycleway     text NOT NULL DEFAULT '';
ALTER TABLE ways ADD COLUMN IF NOT EXISTS bicycle_road text NOT NULL DEFAULT '';
ALTER TABLE ways ADD COLUMN IF NOT EXISTS access       text NOT NULL DEFAULT '';
ALTER TABLE ways ADD COLUMN IF NOT EXISTS curv_fwd     real NOT NULL DEFAULT 0.0;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS curv_rev     real NOT NULL DEFAULT 0.0;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS grade_pct    real NOT NULL DEFAULT 0.0;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS canopy_frac  real NOT NULL DEFAULT 0.0;
-- V2 scenicness: per-edge raster-sampled signals. Populated by
-- pgrouting/scenicness/bake.py (see `signals.py` for the registry).
-- forest_local: forest fraction in 200 m disc, sampled at midpoint
-- forest_wide:  forest fraction in 2 km disc, sampled at midpoint
-- view_dominance: DEM minus 2 km Gaussian blur (meters), midpoint
-- local_relief:  stddev of DEM in 500 m window (meters), midpoint
ALTER TABLE ways ADD COLUMN IF NOT EXISTS forest_local   real NOT NULL DEFAULT 0.0;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS forest_wide    real NOT NULL DEFAULT 0.0;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS view_dominance real NOT NULL DEFAULT 0.0;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS local_relief   real NOT NULL DEFAULT 0.0;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS regional_relief   real NOT NULL DEFAULT 0.0;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS distance_to_drama real NOT NULL DEFAULT 0.0;
-- V2 scenicness water/wetland signals. Fed by 4 landcover classes:
--   water     lakes + wide-river polygons from `natural=water`
--   sea       pre-built coastline polygons (osmdata.openstreetmap.de)
--   waterway  buffered `waterway in (river,canal,stream)` lines
--   wetland   `natural=wetland` polygons
-- Sea is split from water because oceans/seas afford materially better
-- vistas than lakes. Waterway is split from water because riding along
-- the bank of a flowing river ("Mur cycle path") is a distinct
-- experience worth weighting separately from "there's a lake nearby".
ALTER TABLE ways ADD COLUMN IF NOT EXISTS water_local         real NOT NULL DEFAULT 0.0;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS water_wide          real NOT NULL DEFAULT 0.0;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS sea_local           real NOT NULL DEFAULT 0.0;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS sea_wide            real NOT NULL DEFAULT 0.0;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS waterway_along_edge real NOT NULL DEFAULT 0.0;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS waterway_local      real NOT NULL DEFAULT 0.0;

-- V3 per-profile cost columns. Each of the 5 production profiles gets
-- its own (cost, reverse_cost) pair so we can pre-compute all of them
-- without overwriting each other. Routing picks the right column via
-- SQL alias at query time. NULL means "not yet recomputed for this
-- profile" — pgr_dijkstra/pgr_bdAstar will reject NULL costs, so the
-- column must be fully populated before routing against it.
-- Stored as `real` (float32) — ~7 sig figs is plenty for routing
-- since the inputs (scenicness signals, kernel outputs) are already
-- coarse approximations. Saves ~50% on disk vs double precision.
ALTER TABLE ways ADD COLUMN IF NOT EXISTS cost_direct          real;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS reverse_cost_direct  real;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS cost_vineyard_lover  real;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS reverse_cost_vineyard_lover real;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS cost_forest_lover    real;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS reverse_cost_forest_lover   real;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS cost_views           real;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS reverse_cost_views   real;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS cost_water           real;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS reverse_cost_water   real;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS wetland_local       real NOT NULL DEFAULT 0.0;
-- V2 Phase A.3c: vineyard scenic signal — pleasant cultivated land,
-- often in hilly wine country. Other agricultural classes (farmland,
-- meadow) are too generic to score positively; we only single out
-- vineyards (and could add orchards similarly).
ALTER TABLE ways ADD COLUMN IF NOT EXISTS vineyard_local      real NOT NULL DEFAULT 0.0;
-- V2 Phase A.3d: viewpoint POI signals. Source is points (not polygons)
-- from pois.sqlite category='viewpoint'. EDA found 77% of Austrian
-- viewpoints are within 10 m of a way, so local radius is tight (100 m)
-- and regional is wider (2 km).
ALTER TABLE ways ADD COLUMN IF NOT EXISTS viewpoint_local     real NOT NULL DEFAULT 0.0;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS viewpoint_regional  real NOT NULL DEFAULT 0.0;
ALTER TABLE ways DROP COLUMN IF EXISTS sinuosity;
CREATE INDEX IF NOT EXISTS ways_source_idx ON ways(source);
CREATE INDEX IF NOT EXISTS ways_target_idx ON ways(target);

-- Routing anchors: place=city|town from the POI extract, snapped to
-- the nearest graph vertex. The snap step runs after edge ingest so
-- vertex IDs are stable.
CREATE TABLE IF NOT EXISTS anchors (
    id              bigserial PRIMARY KEY,
    osm_id          bigint NOT NULL,
    name            text,
    place           text NOT NULL,    -- 'city' | 'town'
    population      integer,
    country         text,
    snap_vertex_id  bigint REFERENCES ways_vertices_pgr(id),
    geom            geometry(Point, 4326) NOT NULL,
    -- City/municipality polygon (boundary=administrative, admin_level
    -- in {6,7,8}). Populated by `boundaries`. NULL means the SPT seed
    -- step falls back to single-vertex seeding at snap_vertex_id.
    geom_boundary   geometry(MultiPolygon, 4326)
);
CREATE INDEX IF NOT EXISTS anchors_geom_idx ON anchors USING gist(geom);
CREATE INDEX IF NOT EXISTS anchors_snap_idx ON anchors(snap_vertex_id);
CREATE INDEX IF NOT EXISTS anchors_geom_boundary_idx
    ON anchors USING gist(geom_boundary);

-- V2 scenicness: full-resolution landcover polygons sourced from
-- per-country `*-landuse.osm.pbf` extracts. Populated by
-- `ingest_landcover.py`, one row per OSM area. `class` is the rolled-up
-- category (currently always 'forest' for tree-cover signals; the
-- column is in place so other classes — water, agricultural — can be
-- added without schema churn). `country` tags the source PBF so a
-- country can be re-ingested in isolation without wiping the others.
CREATE TABLE IF NOT EXISTS landcover (
    id      bigserial PRIMARY KEY,
    osm_id  bigint,
    country text NOT NULL,
    class   text NOT NULL,
    geom    geometry(MultiPolygon, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS landcover_geom_idx
    ON landcover USING gist(geom);
CREATE INDEX IF NOT EXISTS landcover_class_idx ON landcover(class);
CREATE INDEX IF NOT EXISTS landcover_country_idx ON landcover(country);


-- V3 passenger rails: stations + lines for routing context and (later)
-- multi-modal routing. Stations come from a national GTFS feed (only
-- stops actually served by rail routes — definitive "active" list).
-- Lines come from OSM `railway=rail|light_rail` with usage=main|branch,
-- spatially filtered post-ingest to lines that pass within 200 m of
-- any served station (drops freight-only mainlines).
CREATE TABLE IF NOT EXISTS rail_stations (
    id          bigserial PRIMARY KEY,
    gtfs_id     text NOT NULL,
    name        text NOT NULL,
    n_routes    integer NOT NULL DEFAULT 0,   -- distinct rail routes serving the stop
    country     text NOT NULL,
    geom        geometry(Point, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS rail_stations_geom_idx
    ON rail_stations USING gist(geom);
CREATE INDEX IF NOT EXISTS rail_stations_country_idx
    ON rail_stations(country);
CREATE UNIQUE INDEX IF NOT EXISTS rail_stations_country_gtfs_idx
    ON rail_stations(country, gtfs_id);

CREATE TABLE IF NOT EXISTS rail_lines (
    id          bigserial PRIMARY KEY,
    osm_id      bigint NOT NULL,
    name        text,
    operator    text,
    usage       text,
    electrified text,
    country     text NOT NULL,
    geom        geometry(LineString, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS rail_lines_geom_idx
    ON rail_lines USING gist(geom);
CREATE INDEX IF NOT EXISTS rail_lines_country_idx
    ON rail_lines(country);


-- V3 lodging POIs (subset of `pois.sqlite` materialized for fast
-- spatial joins from postgres — e.g. "does this rail station have a
-- hotel within 3 km?"). Holds all `tourism in (hotel, guest_house,
-- hostel, motel, camp_site, wilderness_hut)` points. Idempotent per
-- country.
CREATE TABLE IF NOT EXISTS lodging (
    id        bigserial PRIMARY KEY,
    osm_id    text,
    name      text,
    subtype   text NOT NULL,
    country   text,
    geom      geometry(Point, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS lodging_geom_idx
    ON lodging USING gist(geom);
CREATE INDEX IF NOT EXISTS lodging_subtype_idx ON lodging(subtype);
CREATE INDEX IF NOT EXISTS lodging_country_idx ON lodging(country);
