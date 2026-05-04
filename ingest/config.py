"""Static config for the ingest pipeline.

Corridor: Graz (15.43°E, 47.07°N) → Copenhagen (12.57°E, 55.68°N).
"""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))

# --- Geofabrik country PBFs ---
GEOFABRIK_BASE = "https://download.geofabrik.de/europe"
COUNTRIES = ["austria", "czech-republic", "germany", "denmark"]

# --- BRouter segment tiles (5°×5°, named by SW corner) ---
# Corridor span: lon 5°E–20°E × lat 45°N–60°N.
BROUTER_BASE = "https://brouter.de/brouter/segments4"
BROUTER_TILES = [
    f"E{lon}_N{lat}"
    for lon in (5, 10, 15)
    for lat in (45, 50, 55)
]

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
