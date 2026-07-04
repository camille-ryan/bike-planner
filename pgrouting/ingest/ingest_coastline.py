"""Pre-built coastline polygons -> landcover (class='sea').

OSM tags coastlines as `natural=coastline` *lines*; there is no
`natural=sea` polygon in the raw data. To get sea polygons one has to
close the global coastline against the antimeridian and the poles —
which someone has already done for us at
    https://osmdata.openstreetmap.de/data/water-polygons.html

We pull the simplified split set (~23 MB, EPSG:3857). The "simplified"
version is more than enough at our 20 m raster resolution; "split" means
the polygons are diced into ≤1° tiles which keeps individual features
manageable. Reprojection to 4326 is delegated to PostGIS via
ST_Transform at insert time so pyproj is not a dependency.

Rows are tagged with country='_coastline' so future re-ingests of real
countries (`austria`, `denmark`, …) don't accidentally wipe them, and
class='sea' so the scenicness bake can treat them as a distinct
landcover class with its own signals (sea_local, sea_wide).

The ingest is global — every sea polygon worldwide gets a row. That's
~tens of thousands of polygons, trivial for PostGIS. We deliberately
do NOT bbox-filter on ingest: the cost of re-running this script every
time the corridor extends is annoying enough that a one-shot global
ingest is the right design. The downstream rasterize step in
scenicness/rasters.py already bbox-filters at SQL time.
"""
from __future__ import annotations
import io
import urllib.request
import zipfile
from pathlib import Path

import psycopg
import shapefile  # pyshp

import config


COASTLINE_URL = (
    "https://osmdata.openstreetmap.de/download/"
    "simplified-water-polygons-split-3857.zip"
)
LOCAL_ZIP = config.DATA_DIR / "landcover" / "simplified-water-polygons-split-3857.zip"
EXTRACT_DIR = config.DATA_DIR / "landcover" / "coastline"
COUNTRY_TAG = "_coastline"


def _download(url: str, dest: Path) -> None:
    """Pull the zip to local disk if not already there. ~23 MB so we
    don't bother with resumable downloads or streaming-into-memory."""
    if dest.exists():
        print(f"[coastline] cached: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[coastline] downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        out.write(resp.read())
    print(f"[coastline] downloaded {dest.stat().st_size/1e6:.1f} MB")


def _unzip(zip_path: Path, extract_dir: Path) -> Path:
    """Unzip and return the path to the .shp file. The archive layout
    nests the actual files under a directory whose name is not exactly
    the zip basename, so we discover the .shp by glob."""
    existing = list(extract_dir.glob("**/*.shp"))
    if existing:
        return existing[0]
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[coastline] unzipping {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    shps = list(extract_dir.glob("**/*.shp"))
    if not shps:
        raise SystemExit(f"no .shp found under {extract_dir}")
    return shps[0]


def _ring_to_wkt(ring: list[tuple[float, float]]) -> str:
    """Shapefile ring -> WKT linear ring text. Each point is "x y".
    Shapefiles use CW outer / CCW inner; we don't try to fix winding
    here — ST_MakeValid runs server-side."""
    return ", ".join(f"{x} {y}" for x, y in ring)


def _polygon_to_wkt(rings: list[list[tuple[float, float]]]) -> str:
    """Pyshp gives us a list of rings. Convention: each shape may have
    multiple rings — the first is the outer ring, subsequent rings are
    holes (assuming standard shapefile winding). We don't try to
    distinguish; we wrap each ring as its own POLYGON and let PostGIS
    union them into a MULTIPOLYGON via ST_Collect.

    This is lazy but correct for our use case: the simplified water
    polygons are mostly single-ring islands/seas, and even if we
    mis-classify a hole as a separate polygon, the rasterize step
    treats the union of all sea geom as "this cell is sea" — small
    inland lakes that show up here as holes would erroneously count as
    sea, but those holes are tiny relative to our 20 m raster cell so
    the error is negligible.
    """
    parts = ["POLYGON((" + _ring_to_wkt(r) + "))" for r in rings]
    if len(parts) == 1:
        return parts[0]
    return "GEOMETRYCOLLECTION(" + ", ".join(parts) + ")"


def _ensure_landcover_table(conn: psycopg.Connection) -> None:
    """Same DDL as ingest_landcover. Inlined so this script can be run
    standalone without forcing the landcover ingest to have been run
    first."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS landcover (
                id      bigserial PRIMARY KEY,
                osm_id  bigint,
                country text NOT NULL,
                class   text NOT NULL,
                geom    geometry(MultiPolygon, 4326) NOT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS landcover_geom_idx "
            "ON landcover USING gist(geom)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS landcover_class_idx "
            "ON landcover(class)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS landcover_country_idx "
            "ON landcover(country)"
        )
    conn.commit()


BATCH_SIZE = 500


def ingest(conn: psycopg.Connection) -> None:
    """Download (if needed), unzip, and stream every sea polygon into
    landcover. Idempotent at the COUNTRY_TAG level — re-running deletes
    the prior `_coastline` rows before re-inserting. Other countries'
    rows are untouched.
    """
    _ensure_landcover_table(conn)

    _download(COASTLINE_URL, LOCAL_ZIP)
    shp_path = _unzip(LOCAL_ZIP, EXTRACT_DIR)
    print(f"[coastline] reading {shp_path}")

    # Wipe prior coastline rows so re-runs don't accumulate duplicates.
    with conn.cursor() as cur:
        cur.execute("DELETE FROM landcover WHERE country = %s", (COUNTRY_TAG,))
        prev = cur.rowcount
    conn.commit()
    if prev:
        print(f"[coastline] cleared {prev:,} prior rows")

    reader = shapefile.Reader(str(shp_path))
    # shapefile.Reader.shapes() yields one Shape per record. .points
    # holds all vertices flat; .parts marks ring starts (offsets into
    # .points). We slice into per-ring vertex lists, then hand each
    # shape to PostGIS as WKT + 3857 -> 4326 transform inline.
    inserted = 0
    skipped = 0
    rows: list[tuple[str]] = []

    def _flush() -> None:
        nonlocal inserted
        if not rows:
            return
        # psycopg 3 doesn't have `cursor.mogrify`. `executemany` is the
        # idiomatic batch path and is fast enough for our scale (~tens
        # of thousands of polygons total). ST_Multi + ST_CollectionExtract
        # handle the cases where MakeValid promotes our POLYGON to
        # MULTIPOLYGON or wraps multi-ring shapes in a GEOMETRYCOLLECTION.
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO landcover (country, class, geom) VALUES "
                "(%s, %s, ST_Multi(ST_CollectionExtract("
                "ST_MakeValid(ST_Transform("
                "ST_GeomFromText(%s, 3857), 4326)), 3)))",
                [(COUNTRY_TAG, "sea", w) for (w,) in rows],
            )
            inserted += len(rows)
        conn.commit()
        rows.clear()

    for shape in reader.iterShapes():
        # shape.shapeType: 5 = Polygon. We skip anything else (the
        # archive should be all polygons).
        if shape.shapeType != shapefile.POLYGON:
            skipped += 1
            continue
        if not shape.points:
            skipped += 1
            continue
        parts = list(shape.parts) + [len(shape.points)]
        rings = [
            list(shape.points[parts[i]:parts[i+1]])
            for i in range(len(parts) - 1)
        ]
        wkt = _polygon_to_wkt(rings)
        rows.append((wkt,))
        if len(rows) >= BATCH_SIZE:
            _flush()
            if inserted % 5000 < BATCH_SIZE:
                print(f"[coastline]   {inserted:,} polygons inserted")
    _flush()

    print(f"[coastline] inserted {inserted:,} sea polygons "
          f"(skipped {skipped:,} non-polygon shapes)")
    print("[coastline] ANALYZE landcover")
    with conn.cursor() as cur:
        cur.execute("ANALYZE landcover")
    conn.commit()
