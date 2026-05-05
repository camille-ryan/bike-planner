"""Per-leg routing cache.

A request like Graz->Copenhagen turns into 8 short legs after the
auto-waypoint pass. Each leg `(profile, from, to)` is a deterministic
function of BRouter + the profile, so we cache the resulting GeoJSON
feature on disk and skip the BRouter call on a hit.

Coordinates are rounded to 6 decimals (~11 cm) before keying so that
near-identical requests share a cache entry. The cache is keyed by
profile *name* — if the profile content changes (`lht.brf` edited),
clear the cache via `/cache/clear` or by deleting `legs.sqlite`.
"""
import json
import sqlite3
import time
from contextlib import contextmanager

from .settings import LEG_CACHE_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS legs (
    key        TEXT PRIMARY KEY,
    profile    TEXT NOT NULL,
    from_lon   REAL NOT NULL,
    from_lat   REAL NOT NULL,
    to_lon     REAL NOT NULL,
    to_lat     REAL NOT NULL,
    feature    TEXT NOT NULL,    -- JSON-serialized GeoJSON Feature
    created_at INTEGER NOT NULL  -- unix seconds
);
"""


def _key(profile: str, a: tuple[float, float], b: tuple[float, float]) -> str:
    return f"{profile}|{round(a[0], 6)},{round(a[1], 6)}|{round(b[0], 6)},{round(b[1], 6)}"


@contextmanager
def _conn():
    LEG_CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(LEG_CACHE_DB)
    try:
        conn.execute(SCHEMA)
        yield conn
    finally:
        conn.close()


def get(profile: str, a: tuple[float, float], b: tuple[float, float]) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT feature FROM legs WHERE key = ?", (_key(profile, a, b),),
        ).fetchone()
    if not row:
        return None
    return json.loads(row[0])


def put(profile: str, a: tuple[float, float], b: tuple[float, float], feature: dict) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO legs"
            " (key, profile, from_lon, from_lat, to_lon, to_lat, feature, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (_key(profile, a, b), profile, a[0], a[1], b[0], b[1],
             json.dumps(feature, separators=(",", ":")), int(time.time())),
        )
        conn.commit()


def stats() -> dict:
    with _conn() as conn:
        n, total_size = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(feature)), 0) FROM legs"
        ).fetchone()
        per_profile = conn.execute(
            "SELECT profile, COUNT(*) FROM legs GROUP BY profile"
        ).fetchall()
    return {
        "count": n,
        "approx_bytes": total_size,
        "by_profile": dict(per_profile),
    }


def clear(profile: str | None = None) -> int:
    with _conn() as conn:
        if profile:
            cur = conn.execute("DELETE FROM legs WHERE profile = ?", (profile,))
        else:
            cur = conn.execute("DELETE FROM legs")
        conn.commit()
        return cur.rowcount
