"""HTTP client for the BRouter routing server.

BRouter sometimes returns 200 with a plain-text body containing an error
message (e.g. profile parse errors, or "no route found"). We treat any
non-JSON-looking body as an error and surface it clearly.
"""
import httpx

from .settings import BROUTER_URL


def _format_lonlats(points: list[tuple[float, float]]) -> str:
    return "|".join(f"{lon},{lat}" for lon, lat in points)


async def fetch_route(
    points: list[tuple[float, float]],
    profile: str,
    alternativeidx: int = 0,
) -> dict:
    """Fetch a single route from BRouter; raises RuntimeError on failure."""
    params = {
        "lonlats": _format_lonlats(points),
        "profile": profile,
        "alternativeidx": str(alternativeidx),
        "format": "geojson",
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.get(f"{BROUTER_URL}/brouter", params=params)
    body = r.text
    if not body or not body.lstrip().startswith("{"):
        raise RuntimeError(f"brouter[{alternativeidx}]: {body[:300] or '(empty)'}")
    gj = r.json()
    feats = gj.get("features") or []
    if not feats:
        raise RuntimeError(f"brouter[{alternativeidx}]: no features")
    feat = feats[0]
    feat["properties"]["alternativeidx"] = alternativeidx
    return feat


async def fetch_alternatives(
    points: list[tuple[float, float]],
    profile: str,
    extra: int,
) -> list[dict]:
    """Fetch the primary route + up to `extra` alternatives.

    Stops at the first failure on alt index >= 1 (alts may not exist).
    Re-raises if the primary route itself fails.
    """
    out: list[dict] = []
    for idx in range(extra + 1):
        try:
            out.append(await fetch_route(points, profile, idx))
        except RuntimeError:
            if idx == 0:
                raise
            break
    return out
