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
CREATE TABLE IF NOT EXISTS ways_vertices_pgr (
    id      bigserial PRIMARY KEY,
    osm_id  bigint UNIQUE NOT NULL,
    lon     double precision NOT NULL,
    lat     double precision NOT NULL,
    -- Generated geometry column kept in sync with lon/lat. Spatial
    -- index lives on this column (pgRouting can use it for KNN snap).
    the_geom geometry(Point, 4326) GENERATED ALWAYS AS
             (ST_SetSRID(ST_MakePoint(lon, lat), 4326)) STORED
);
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
CREATE TABLE IF NOT EXISTS ways (
    gid           bigserial PRIMARY KEY,
    osm_way_id    bigint,
    source        bigint NOT NULL,
    target        bigint NOT NULL,
    cost          double precision NOT NULL,           -- length × cost-factor (forward)
    reverse_cost  double precision NOT NULL,           -- same backwards, or -1 if oneway
    length_m      double precision NOT NULL,
    is_ferry      boolean NOT NULL DEFAULT false
);
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
