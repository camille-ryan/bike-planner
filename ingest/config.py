"""Static config for the ingest pipeline.

Corridor: Graz (15.43°E, 47.07°N) → Copenhagen (12.57°E, 55.68°N).
"""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))

# --- Geofabrik country PBFs ---
GEOFABRIK_BASE = "https://download.geofabrik.de/europe"
COUNTRIES = ["austria", "czech-republic", "germany", "denmark"]

# --- POI extraction filters (osmium tags-filter syntax) ---
# Node-only filters keep POIs as point markers — sufficient for
# overlay rendering and proximity scoring.
POI_FILTERS = [
    "n/tourism=viewpoint,hotel,hostel,guest_house,motel,camp_site,wilderness_hut",
    "n/amenity=restaurant,cafe,fast_food,pub,bar,biergarten,bicycle_repair_station,drinking_water",
    "n/shop=bicycle",
]
# Areas: parks/reserves are usually relations or closed ways.
PROTECTED_AREA_FILTERS = [
    "a/boundary=national_park,protected_area",
    "a/leisure=nature_reserve",
]
# Cycle networks: tagged route relations.
BIKE_ROUTE_FILTERS = ["r/route=bicycle"]

# Routing anchors — `place=city|town` populated places. Consumed by the
# pgrouting preprocess to seed the multi-source SPT. Filtering on the
# `place` tag (rather than a population threshold) avoids dropping towns
# whose `population=*` tag is missing, which is common in OSM. Yields
# ~hundreds–low-thousands per country.
ANCHOR_FILTERS = ["n/place=city,town"]
