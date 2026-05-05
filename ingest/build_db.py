"""Load filtered POI extracts into a SpatiaLite database.

Phase 1 loads point POIs only. Protected-area polygons and bike-route
multilinestrings are extracted to PBF but not yet imported — that comes in
Phase 2 alongside the routing engine, which is the consumer.
"""
import json
import subprocess
import sqlite3
from pathlib import Path

from config import DATA_DIR

DB_PATH = DATA_DIR / "pois" / "pois.sqlite"

# Schema is created in two phases: table + geometry column up front, and the
# R-Tree spatial index *after* bulk insert. Calling CreateSpatialIndex before
# data exists left the R-Tree empty in practice — the SpatiaLite triggers that
# are supposed to keep it in sync did not populate it during executemany loads
# in this stack. Building the index post-load is also faster (no per-row
# trigger overhead) and gives a fresh, complete R-Tree.
POI_SCHEMA = """
DROP TABLE IF EXISTS pois;
CREATE TABLE pois (
    id        INTEGER PRIMARY KEY,
    osm_type  TEXT,
    osm_id    TEXT,
    category  TEXT,
    subtype   TEXT,
    name      TEXT,
    country   TEXT,
    tags      TEXT
);
SELECT DisableSpatialIndex('pois', 'geom');  -- harmless if not present; clears stale RTree from prior runs
SELECT AddGeometryColumn('pois', 'geom', 4326, 'POINT', 'XY');
"""


def categorize(props: dict) -> tuple[str, str] | None:
    tourism = props.get("tourism")
    amenity = props.get("amenity")
    shop    = props.get("shop")
    if tourism == "viewpoint":
        return ("viewpoint", "viewpoint")
    if tourism in {"hotel", "hostel", "guest_house", "motel", "camp_site", "wilderness_hut"}:
        return ("lodging", tourism)
    if amenity in {"restaurant", "cafe", "fast_food", "pub", "bar", "biergarten"}:
        return ("food", amenity)
    if amenity == "drinking_water":
        return ("water", "drinking_water")
    if amenity == "bicycle_repair_station" or shop == "bicycle":
        return ("bike_service", amenity or shop)
    return None


def feature_point(feature: dict) -> tuple[float, float] | None:
    geom = feature.get("geometry") or {}
    if geom.get("type") != "Point":
        return None
    coords = geom.get("coordinates")
    if not coords or len(coords) < 2:
        return None
    return float(coords[0]), float(coords[1])


def export_geojsonseq(pbf: Path):
    """Yield features from a PBF via `osmium export -f geojsonseq`."""
    proc = subprocess.Popen(
        ["osmium", "export", "-f", "geojsonseq", str(pbf)],
        stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.lstrip("\x1e").strip()
        if line:
            yield json.loads(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"osmium export failed for {pbf}")


def open_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    fresh = not DB_PATH.exists()
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")
    conn.enable_load_extension(False)
    if fresh:
        conn.executescript("SELECT InitSpatialMetaData(1);")
    return conn


def load_country(conn: sqlite3.Connection, country: str, pois_pbf: Path) -> dict:
    counts: dict[str, int] = {}
    cur = conn.cursor()
    for feat in export_geojsonseq(pois_pbf):
        props = feat.get("properties") or {}
        cat = categorize(props)
        if not cat:
            continue
        category, subtype = cat
        pt = feature_point(feat)
        if not pt:
            continue
        lon, lat = pt
        osm_id   = str(props.get("@id") or feat.get("id") or "")
        osm_type = props.get("@type") or "node"
        tags_clean = {k: v for k, v in props.items() if not k.startswith("@")}
        cur.execute(
            "INSERT INTO pois (osm_type, osm_id, category, subtype, name, country, tags, geom)"
            " VALUES (?,?,?,?,?,?,?, MakePoint(?,?,4326))",
            (osm_type, osm_id, category, subtype, props.get("name"), country,
             json.dumps(tags_clean, ensure_ascii=False), lon, lat),
        )
        counts[category] = counts.get(category, 0) + 1
    conn.commit()
    return counts


def run(extracts: list[dict]) -> dict:
    conn = open_db()
    conn.executescript(POI_SCHEMA)
    totals: dict[str, int] = {}
    for ex in extracts:
        cs = load_country(conn, ex["country"], ex["pois"])
        for k, v in cs.items():
            totals[k] = totals.get(k, 0) + v
        print(f"[db] {ex['country']}: {cs}")
    # Build the R-Tree from fully-loaded data. CreateSpatialIndex on a
    # populated table is fast and produces a complete index; doing it before
    # inserts left the R-Tree empty in this setup.
    print("[db] building spatial index...", flush=True)
    conn.execute("SELECT CreateSpatialIndex('pois', 'geom')")
    conn.commit()
    conn.close()
    return totals
