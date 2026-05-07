"""Runtime configuration for the API container."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
POIS_DB = DATA_DIR / "pois" / "pois.sqlite"
SPT_DIR = DATA_DIR / "spt"     # per-profile subdir is appended at use time
DEFAULT_PROFILE = os.environ.get("DEFAULT_PROFILE", "lht")
