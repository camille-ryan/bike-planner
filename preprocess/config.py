"""Static config for the SPT preprocess pipeline."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
OSM_DIR  = DATA_DIR / "osm"
POIS_DB  = DATA_DIR / "pois" / "pois.sqlite"
SPT_DIR  = DATA_DIR / "spt"      # per-profile subdirs created at write time

# Country shortlist used for the v1 Austria smoke test (pyrosm reads
# one PBF per call). The full corridor is austria + czech-republic +
# germany + denmark and will need a country-merge step before SPT.
SMOKE_COUNTRIES = ["austria"]
FULL_COUNTRIES  = ["austria", "czech-republic", "germany", "denmark"]
