"""Tool-call dispatcher for the MCP server.

Backends
--------
A ``Backend`` is anything with an ``async call(tool: str, args: dict)
-> dict`` method. Two ship today:

- ``LocalPython`` — imports ``api.app.tools`` and calls the function
  in-process. Lazy import so an MCP server that never routes to
  ``local`` doesn't pay the api/ package's import cost.
- ``HttpForward`` — POSTs ``/tools/<name>`` on a running api container
  (see ``api/app/main.py``). Networked, region-portable.

Routing
-------
For region-aware tools (``route``, ``stations_along_route``, etc.)
the dispatcher extracts a "target coord" from the args (via anchor
ref lookup or a raw ``lon,lat`` string) and picks the first backend
whose bbox contains it. Region-agnostic tools go straight to the
default backend.

If no region matches (e.g. the user asks for a route across a gap
we don't cover), the dispatcher returns a clean
``{"error": "no backend covers this region"}`` — the MCP client
surfaces that to the LLM which surfaces it to the user. When a
second region is added later, cross-region routes will decompose
into per-region legs stitched at border anchors, but that's Phase 4
future work.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# tomllib is stdlib on 3.11+.
try:
    import tomllib
except ImportError:  # pragma: no cover — only on <3.11
    import tomli as tomllib  # type: ignore

import httpx


# Tools that need a geographic backend chosen from args. Anything not
# listed here goes to the default backend regardless of args.
REGION_AWARE_TOOLS: frozenset[str] = frozenset({
    "route",
    "stations_along_route",
    "pois_along_route",
    "pois_near_anchor",
    "stations_near",
    "direct_rail_service",
})


@dataclass
class Bbox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def contains(self, lon: float, lat: float) -> bool:
        return (self.min_lon <= lon <= self.max_lon
                and self.min_lat <= lat <= self.max_lat)


class Backend(Protocol):
    """Minimum surface a backend implementation exposes."""
    async def call(self, tool: str, args: dict) -> dict: ...
    async def aclose(self) -> None: ...


class LocalPython:
    """Backend that imports and calls api.app.tools in-process. Lazy
    import means an MCP server that only forwards to HTTP never pays
    the api/ package's transitive dependency cost."""

    def __init__(self) -> None:
        self._tools = None  # populated on first call
        self._resolve_coord = None

    def _ensure_loaded(self) -> None:
        if self._tools is not None:
            return
        # Lazy import — pulls in numpy/scipy/psycopg/anthropic/etc.
        # The 30-60s "first tool call" cost the MCP server docs mention
        # comes from here: trunk_router preloads its ~5 GB paired-trunks
        # blob on first route lookup.
        from api.app import tools as _tools
        self._tools = _tools

    async def call(self, tool: str, args: dict) -> dict:
        self._ensure_loaded()
        # call_tool wraps exceptions into {"error": ...} — safe to await
        # even for tools that raise.
        return self._tools.call_tool(tool, args)  # type: ignore[union-attr]

    async def aclose(self) -> None:
        return None


class HttpForward:
    """Backend that POSTs /tools/<name> on a remote api container.
    Config: base_url (required), bbox (optional; only used for region
    matching — a backend without a bbox is only reachable via the
    default fallback)."""

    def __init__(self, base_url: str, bbox: Bbox | None = None,
                 timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.bbox = bbox
        self._client = httpx.AsyncClient(timeout=timeout)

    async def call(self, tool: str, args: dict) -> dict:
        url = f"{self.base_url}/tools/{tool}"
        try:
            r = await self._client.post(url, json={"args": args})
        except httpx.HTTPError as e:
            return {"error": f"http_forward transport: {type(e).__name__}: {e}"}
        if r.status_code == 404:
            return {"error": f"remote backend has no tool {tool!r}"}
        if r.status_code != 200:
            return {"error": f"http_forward {r.status_code}: {r.text[:200]}"}
        try:
            return r.json()
        except ValueError:
            return {"error": f"http_forward non-JSON response: {r.text[:200]}"}

    async def aclose(self) -> None:
        await self._client.aclose()


@dataclass
class RegionEntry:
    name: str
    backend: Backend
    bbox: Bbox | None = None


@dataclass
class Dispatcher:
    """Map (tool_name, args) → Backend. Owns backend lifecycles."""

    default: Backend
    regions: list[RegionEntry] = field(default_factory=list)
    # Optional callback (coord_resolver(args, tool_name) → (lon, lat) | None)
    # lets tests override anchor-ref → coord lookup without needing the
    # api/ package loaded. Prod uses the built-in resolver.
    coord_resolver: Any = None

    async def call(self, tool: str, args: dict) -> dict:
        args = args or {}
        backend = self._backend_for(tool, args)
        return await backend.call(tool, args)

    def _backend_for(self, tool: str, args: dict) -> Backend:
        if tool not in REGION_AWARE_TOOLS or not self.regions:
            return self.default
        coord = self._extract_coord(tool, args)
        if coord is None:
            return self.default
        lon, lat = coord
        for region in self.regions:
            if region.bbox and region.bbox.contains(lon, lat):
                return region.backend
        return self.default

    def _extract_coord(self, tool: str, args: dict) -> tuple[float, float] | None:
        """Pull a representative (lon, lat) out of a tool's args so
        we can pick a region. Anchor refs (`db:123`, `ferry:456`)
        need a lookup into the profile's cities.json — done via the
        `coord_resolver` callback, or via the default lookup which
        imports api.app.tools' trunk_router in-process.
        """
        # Explicit lon/lat fields.
        if "lon" in args and "lat" in args:
            try:
                return float(args["lon"]), float(args["lat"])
            except (TypeError, ValueError):
                pass
        for k in ("from_lonlat", "to_lonlat"):
            v = args.get(k)
            if isinstance(v, str):
                parts = v.split(",")
                if len(parts) == 2:
                    try:
                        return float(parts[0]), float(parts[1])
                    except ValueError:
                        pass
        # Ref-based args — need a lookup.
        for k in ("from_ref", "to_ref", "ref"):
            ref = args.get(k)
            if isinstance(ref, str):
                coord = self._resolve_ref(ref)
                if coord is not None:
                    return coord
        return None

    def _resolve_ref(self, ref: str) -> tuple[float, float] | None:
        if self.coord_resolver is not None:
            return self.coord_resolver(ref)
        # Default: use api.app.trunk_router's preloaded profile.
        # Imports lazily so the resolver only fires when a region-aware
        # tool with a ref-arg is dispatched AND at least one region is
        # configured.
        try:
            from api.app import trunk_router
            from api.app.settings import DEFAULT_PROFILE
            prof = trunk_router._load_profile(DEFAULT_PROFILE)
            ci = prof.city_idx_by_ref.get(ref)
            if ci is None:
                return None
            c = prof.cities[int(ci)]
            return float(c["lon"]), float(c["lat"])
        except Exception:
            return None

    async def aclose(self) -> None:
        await self.default.aclose()
        for r in self.regions:
            await r.backend.aclose()


# ---------------------------------------------------------------------
# Config loading

_BBOX_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def _load_backend(name: str, cfg: dict) -> Backend:
    kind = cfg.get("kind")
    if kind == "local_python":
        return LocalPython()
    if kind == "http_forward":
        url = cfg.get("url")
        if not url:
            raise ValueError(f"backend {name!r}: http_forward needs `url`")
        bbox_list = cfg.get("bbox")
        bbox = None
        if bbox_list:
            if (not isinstance(bbox_list, (list, tuple))
                    or len(bbox_list) != 4):
                raise ValueError(f"backend {name!r}: bbox must be [min_lon,min_lat,max_lon,max_lat]")
            bbox = Bbox(*map(float, bbox_list))
        return HttpForward(url, bbox=bbox)
    raise ValueError(f"backend {name!r}: unknown kind {kind!r}")


def load_dispatcher(config_path: Path | None = None) -> Dispatcher:
    """Read regions.toml (or a caller-provided path), instantiate each
    backend, and return a wired Dispatcher."""
    if config_path is None:
        config_path = Path(__file__).parent / "regions.toml"
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    backends_cfg = cfg.get("backends", {})
    if not backends_cfg:
        raise ValueError(f"{config_path}: no [backends.*] tables")

    default_name = cfg.get("default")
    if not default_name or default_name not in backends_cfg:
        raise ValueError(f"{config_path}: `default` must name a backend")

    backends: dict[str, Backend] = {
        name: _load_backend(name, subcfg)
        for name, subcfg in backends_cfg.items()
    }

    default = backends[default_name]
    # A "region" is any backend that has a bbox set. Preserve TOML order
    # so users can express matching priority.
    regions: list[RegionEntry] = []
    for name, subcfg in backends_cfg.items():
        if name == default_name:
            continue
        bbox_list = subcfg.get("bbox")
        if not bbox_list:
            continue
        bbox = Bbox(*map(float, bbox_list))
        regions.append(RegionEntry(name=name, backend=backends[name], bbox=bbox))
    return Dispatcher(default=default, regions=regions)
