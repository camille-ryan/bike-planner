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
-- V2 Phase A.2: raw OSM tag columns + sinuosity + grade_pct are
-- persisted at ingest time so cost can be recomputed (with elevation
-- + sinuosity) without re-parsing PBFs.
--   - tag columns (highway, surface, tracktype, oneway, bicycle,
--     cycleway, bicycle_road, access) are per-way but stored per-edge
--     for simplicity. Empty string when the tag is absent.
--   - sinuosity is per-way (actual_length / endpoint_distance) and
--     also denormalized per-edge for the same reason; defaults to 1.0
--     (straight) when uncomputed.
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
    -- Derived per-way / per-edge fields populated by ingest pipeline.
    sinuosity     real NOT NULL DEFAULT 1.0,
    grade_pct     real NOT NULL DEFAULT 0.0
);
ALTER TABLE ways ADD COLUMN IF NOT EXISTS highway      text NOT NULL DEFAULT '';
ALTER TABLE ways ADD COLUMN IF NOT EXISTS surface      text NOT NULL DEFAULT '';
ALTER TABLE ways ADD COLUMN IF NOT EXISTS tracktype    text NOT NULL DEFAULT '';
ALTER TABLE ways ADD COLUMN IF NOT EXISTS oneway       text NOT NULL DEFAULT '';
ALTER TABLE ways ADD COLUMN IF NOT EXISTS bicycle      text NOT NULL DEFAULT '';
ALTER TABLE ways ADD COLUMN IF NOT EXISTS cycleway     text NOT NULL DEFAULT '';
ALTER TABLE ways ADD COLUMN IF NOT EXISTS bicycle_road text NOT NULL DEFAULT '';
ALTER TABLE ways ADD COLUMN IF NOT EXISTS access       text NOT NULL DEFAULT '';
ALTER TABLE ways ADD COLUMN IF NOT EXISTS sinuosity    real NOT NULL DEFAULT 1.0;
ALTER TABLE ways ADD COLUMN IF NOT EXISTS grade_pct    real NOT NULL DEFAULT 0.0;
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
