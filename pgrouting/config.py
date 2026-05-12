"""Static config for the pgRouting-backed preprocess pipeline."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
OSM_DIR  = DATA_DIR / "osm"
POIS_DB  = DATA_DIR / "pois" / "pois.sqlite"
SPT_DIR  = DATA_DIR / "spt"      # per-profile subdirs created at write time
DEM_DIR  = DATA_DIR / "dem"      # Copernicus DEM GLO-30 GeoTIFF tiles

# Postgres connection — defaults pull from env vars set by the
# Dockerfile, override in compose for non-default deployments.
PG_DSN = (
    f"host={os.environ.get('PGHOST', 'postgres')} "
    f"user={os.environ.get('PGUSER', 'bike')} "
    f"password={os.environ.get('PGPASSWORD', 'bike')} "
    f"dbname={os.environ.get('PGDATABASE', 'bike')}"
)

# Smoke-test default: Austria-only. Full corridor: all four.
SMOKE_COUNTRIES = ["austria"]
FULL_COUNTRIES  = ["austria", "czech-republic", "germany", "denmark"]
