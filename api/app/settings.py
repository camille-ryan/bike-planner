"""Runtime configuration for the API container."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
POIS_DB = DATA_DIR / "pois" / "pois.sqlite"
BROUTER_URL = os.environ.get("BROUTER_URL", "http://brouter:17777")
DEFAULT_PROFILE = os.environ.get("DEFAULT_PROFILE", "lht")
