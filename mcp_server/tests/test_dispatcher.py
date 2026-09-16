"""Unit tests for the MCP dispatcher's routing logic.

Scope: verify that `Dispatcher._backend_for` picks the right backend
for each (tool_name, args) combination — region-agnostic tools always
land on `default`; region-aware tools go to the region whose bbox
contains the args' target coord, or fall back to default when no
region matches.

Uses fake in-memory backends (no network, no api/ imports), and an
injected `coord_resolver` so ref-based args resolve without needing
the trunk_router preloaded.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from mcp_bike_planner.dispatcher import (
    Backend, Bbox, Dispatcher, RegionEntry,
)


@dataclass
class FakeBackend:
    """A backend that records every call for assertion."""
    name: str
    calls: list[tuple[str, dict]] = field(default_factory=list)

    async def call(self, tool: str, args: dict) -> dict:
        self.calls.append((tool, args))
        return {"ok": self.name}

    async def aclose(self) -> None:
        pass


def _mk_dispatcher(with_regions: bool = True, coord_resolver=None):
    default = FakeBackend("default")
    regions = []
    if with_regions:
        eu = FakeBackend("eu")
        us = FakeBackend("us")
        # EU bbox is the union of AT/DE/CZ/DK country bboxes (see
        # pgrouting/ingest/ingest_railways.py::_COUNTRY_BBOX).
        regions = [
            RegionEntry(name="eu", backend=eu,
                        bbox=Bbox(5.5, 46.3, 19.0, 57.9)),
            RegionEntry(name="us_sw", backend=us,
                        bbox=Bbox(-114.0, 31.3, -102.0, 42.0)),
        ]
    return Dispatcher(default=default, regions=regions,
                      coord_resolver=coord_resolver)


def _run(coro):
    # asyncio.run() creates + tears down a fresh event loop per call —
    # avoids the "There is no current event loop" DeprecationWarning
    # from get_event_loop() on Python 3.12+.
    return asyncio.run(coro)


# ---------------------------------------------------------------------
# region-agnostic tools


def test_search_anchors_goes_to_default_even_with_regions():
    d = _mk_dispatcher(with_regions=True)
    res = _run(d.call("search_anchors", {"query": "Vienna"}))
    assert res == {"ok": "default"}


def test_split_into_stages_goes_to_default():
    d = _mk_dispatcher(with_regions=True)
    res = _run(d.call("split_into_stages",
                      {"from_ref": "db:66", "to_ref": "db:224",
                       "target_km_per_day": 80}))
    assert res == {"ok": "default"}


# ---------------------------------------------------------------------
# region-aware tools with explicit lon/lat


def test_stations_near_uses_lonlat_to_pick_region_eu():
    d = _mk_dispatcher(with_regions=True)
    # Vienna area — inside EU bbox.
    _run(d.call("stations_near", {"lon": 16.37, "lat": 48.21, "radius_km": 5}))
    eu = next(r.backend for r in d.regions if r.name == "eu")
    assert len(eu.calls) == 1
    assert eu.calls[0][0] == "stations_near"


def test_stations_near_uses_lonlat_to_pick_region_us():
    d = _mk_dispatcher(with_regions=True)
    # Durango, CO — inside US-SW bbox.
    _run(d.call("stations_near", {"lon": -107.88, "lat": 37.27, "radius_km": 5}))
    us = next(r.backend for r in d.regions if r.name == "us_sw")
    assert len(us.calls) == 1


def test_stations_near_falls_back_to_default_when_no_region_matches():
    d = _mk_dispatcher(with_regions=True)
    # Middle of the Pacific — no region covers it.
    _run(d.call("stations_near", {"lon": -160.0, "lat": 0.0, "radius_km": 5}))
    assert d.default.calls == [("stations_near",
                                {"lon": -160.0, "lat": 0.0, "radius_km": 5})]


# ---------------------------------------------------------------------
# ref-based args: uses the coord_resolver hook


def test_route_with_ref_args_resolves_to_region():
    # Fake resolver: Graz + Wien both in EU.
    def resolver(ref):
        return {"db:66": (15.44, 47.07),   # Graz
                "db:224": (16.37, 48.21)   # Wien
                }.get(ref)
    d = _mk_dispatcher(with_regions=True, coord_resolver=resolver)
    _run(d.call("route", {"from_ref": "db:66", "to_ref": "db:224"}))
    eu = next(r.backend for r in d.regions if r.name == "eu")
    assert len(eu.calls) == 1


def test_route_with_unknown_ref_falls_back():
    def resolver(ref):
        return None  # unknown
    d = _mk_dispatcher(with_regions=True, coord_resolver=resolver)
    _run(d.call("route", {"from_ref": "unknown:foo", "to_ref": "unknown:bar"}))
    assert d.default.calls == [("route",
                                {"from_ref": "unknown:foo",
                                 "to_ref": "unknown:bar"})]


# ---------------------------------------------------------------------
# from_lonlat / to_lonlat string form


def test_from_lonlat_string_parses_correctly():
    d = _mk_dispatcher(with_regions=True)
    _run(d.call("route", {"from_lonlat": "16.37,48.21",
                          "to_lonlat":   "13.5,52.5"}))
    # Only from_lonlat is used (first hit wins) — Wien is EU.
    eu = next(r.backend for r in d.regions if r.name == "eu")
    assert len(eu.calls) == 1


# ---------------------------------------------------------------------
# no regions configured → everything to default


def test_no_regions_configured_everything_to_default():
    d = _mk_dispatcher(with_regions=False)
    _run(d.call("route", {"from_ref": "db:66", "to_ref": "db:224"}))
    _run(d.call("stations_near", {"lon": 16.37, "lat": 48.21}))
    _run(d.call("search_anchors", {"query": "Graz"}))
    assert len(d.default.calls) == 3
