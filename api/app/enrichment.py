"""Post-plan enrichment sub-agents: lodging + transit booking.

Runs AFTER `multiagent.run_supervisor_merge` finishes writing the
final plan. Extracts:
  - The list of base overnights (one per stage boundary except day
    zero) → fans out one LodgingAgent per overnight.
  - The list of consecutive overnight pairs → fans out one
    TransitAgent per pair, so the partner-by-train constraint gets
    a booking link.

Both sub-agent types are LLM-driven with tight prompts and small
tool surfaces:

  LodgingAgent tools:  search_lodging, submit_lodging (terminal)
  TransitAgent tools:  direct_rail_service, train_booking_links,
                       submit_transit (terminal)

Each uses `chat._run_chat_inner` with `agent_id="lodging[<i>]"` or
`transit[<i>]"`, so their SSE events route to their own bubble in
the frontend.

The whole enrichment stage is bounded by `asyncio.Semaphore` so we
don't blow through Anthropic RPM/TPM on a 10-overnight tour.

Failure mode: any sub-agent that fails is reported with
`status="failed"` in its `agent_end` event; the overall plan
continues without that enrichment (a missing lodging card is fine).
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, AsyncIterator

import anthropic
from pydantic import BaseModel, ValidationError

from .tracing import RequestTrace


# ---------------------------------------------------------------------
# Config

MAX_PARALLEL_ENRICHMENT = int(
    os.environ.get("MAX_PARALLEL_ENRICHMENT", "5")
)
ENRICHMENT_MAX_ROUNDS = int(
    os.environ.get("ENRICHMENT_MAX_ROUNDS", "4")
)
ENRICHMENT_TIMEOUT_S = float(
    os.environ.get("ENRICHMENT_TIMEOUT_S", "120")
)


# ---------------------------------------------------------------------
# Schemas — structured outputs for the terminal submit_* tools.

class LodgingHotel(BaseModel):
    name: str | None = None
    subtype: str | None = None
    distance_m: float | None = None
    website: str | None = None
    stars: str | int | None = None


class LodgingResult(BaseModel):
    overnight_ref: str
    overnight_name: str
    hotels: list[LodgingHotel] = []
    summary: str = ""
    status: str = "ok"
    error: str | None = None


class TransitBookingUrl(BaseModel):
    operator: str
    url: str


class TransitResult(BaseModel):
    from_ref: str
    to_ref: str
    from_name: str
    to_name: str
    direct_rail: bool | None = None
    booking_urls: list[TransitBookingUrl] = []
    summary: str = ""
    status: str = "ok"
    error: str | None = None


# ---------------------------------------------------------------------
# Prompts

LODGING_PROMPT = """You are a LODGING sub-agent in a bike-tour planner. Your job is narrow:

You will be told which overnight stop this is (city name + coordinates). Do this:

1. Call `search_lodging(lonlat=<lon,lat>, radius_km=1.5)` ONCE. It returns nearby OSM lodging (hotels/hostels/guest_houses).
2. Pick the top 3 options ranked by a bike-tourist's priorities:
   - Closer to the anchor beats farther.
   - Prefer variety: 1 mid-range hotel, 1 hostel/budget, 1 upscale/well-rated if the mix allows.
   - Skip entries with no name or no website unless nothing else is available.
3. Call `submit_lodging` with your top 3 (or fewer, if the search returned less) and a ONE-SENTENCE summary of what's nearby. Then STOP.

Do not call any other tool. Do not narrate outside `submit_lodging.summary`. Keep it tight — you have ~4 rounds max."""


TRANSIT_PROMPT = """You are a TRANSIT sub-agent in a bike-tour planner. Your job is narrow:

You will be told the (from_ref, to_ref) pair — the previous overnight and the current overnight. The user's partner needs a direct train from previous → current. Do this:

1. Call `direct_rail_service(from_ref, to_ref)` to confirm direct-train service exists.
2. Call `train_booking_links(from_ref, to_ref)` to get deep-link URLs for the partner to buy the ticket.
3. Call `submit_transit` with the results and a ONE-SENTENCE summary ("Direct trains available on <shared route count>; book via <operator>." or "No direct train — transfer required.").

STOP after submit. Keep it under 4 rounds."""


# ---------------------------------------------------------------------
# Terminal tools (per-agent scoped — passed as extra_tools to
# `_run_chat_inner` so N sub-agents running concurrently don't share
# a mutable global registry).

SUBMIT_LODGING_TOOL: dict = {
    "name": "submit_lodging",
    "description": (
        "TERMINAL ACTION for a LodgingAgent. Commit your final 3-lodging "
        "pick + a one-sentence summary. Do not call any other tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "overnight_ref":  {"type": "string"},
            "overnight_name": {"type": "string"},
            "hotels": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "name":       {"type": "string"},
                        "subtype":    {"type": "string"},
                        "distance_m": {"type": "number"},
                        "website":    {"type": "string"},
                        "stars":      {"type": ["string", "integer", "number"]},
                    },
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["overnight_ref", "overnight_name"],
    },
}


SUBMIT_TRANSIT_TOOL: dict = {
    "name": "submit_transit",
    "description": (
        "TERMINAL ACTION for a TransitAgent. Commit the direct-rail "
        "verdict + booking URLs + a one-sentence summary. Do not call "
        "any other tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "from_ref":    {"type": "string"},
            "to_ref":      {"type": "string"},
            "from_name":   {"type": "string"},
            "to_name":     {"type": "string"},
            "direct_rail": {"type": "boolean"},
            "booking_urls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "operator": {"type": "string"},
                        "url":      {"type": "string"},
                    },
                    "required": ["operator", "url"],
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["from_ref", "to_ref", "from_name", "to_name"],
    },
}


# ---------------------------------------------------------------------
# Coordinator

async def run_enrichment_stage(
    overnights: list[dict],
    client: anthropic.AsyncAnthropic,
    parent_trace: RequestTrace,
) -> AsyncIterator[bytes]:
    """Fan out per-overnight LodgingAgent + per-consecutive-pair
    TransitAgent. Merges all sub-agent SSE bytes onto the shared
    stream in whatever order they complete."""
    from .chat import _sse

    if not overnights:
        return

    # Announce all enrichment bubbles up front so the frontend renders
    # them in a stable order.
    for i, ov in enumerate(overnights):
        yield _sse("agent_start", {
            "role":           "lodging",
            "overnight_i":    i,
            "overnight_ref":  ov.get("ref"),
            "overnight_name": ov.get("name"),
        }, agent_id=_lodging_agent_id(i))
    # Transit: N-1 pairs (day K-1 end → day K end).
    for i in range(1, len(overnights)):
        prev = overnights[i - 1]
        cur  = overnights[i]
        yield _sse("agent_start", {
            "role":      "transit",
            "pair_i":    i - 1,
            "from_ref":  prev.get("ref"),
            "to_ref":    cur.get("ref"),
            "from_name": prev.get("name"),
            "to_name":   cur.get("name"),
        }, agent_id=_transit_agent_id(i - 1))

    sem = asyncio.Semaphore(MAX_PARALLEL_ENRICHMENT)
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def _run_lodging(i: int, ov: dict) -> None:
        agent_id = _lodging_agent_id(i)
        t0 = time.monotonic()
        status = "ok"
        error: str | None = None
        try:
            async with sem:
                await asyncio.wait_for(
                    _drain_lodging_agent(ov, client, parent_trace,
                                         agent_id, queue),
                    timeout=ENRICHMENT_TIMEOUT_S,
                )
        except asyncio.TimeoutError:
            status = "failed"
            error = f"lodging agent timed out after {ENRICHMENT_TIMEOUT_S:.0f} s"
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        await queue.put(_sse("agent_end", {
            "role":        "lodging",
            "overnight_i": i,
            "status":      status,
            "error":       error,
            "wall_ms":     int((time.monotonic() - t0) * 1000),
        }, agent_id=agent_id))
        await queue.put(None)

    async def _run_transit(pair_i: int, prev: dict, cur: dict) -> None:
        agent_id = _transit_agent_id(pair_i)
        t0 = time.monotonic()
        status = "ok"
        error: str | None = None
        try:
            async with sem:
                await asyncio.wait_for(
                    _drain_transit_agent(prev, cur, client, parent_trace,
                                         agent_id, queue),
                    timeout=ENRICHMENT_TIMEOUT_S,
                )
        except asyncio.TimeoutError:
            status = "failed"
            error = f"transit agent timed out after {ENRICHMENT_TIMEOUT_S:.0f} s"
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        await queue.put(_sse("agent_end", {
            "role":    "transit",
            "pair_i":  pair_i,
            "status":  status,
            "error":   error,
            "wall_ms": int((time.monotonic() - t0) * 1000),
        }, agent_id=agent_id))
        await queue.put(None)

    tasks: list[asyncio.Task] = []
    for i, ov in enumerate(overnights):
        tasks.append(asyncio.create_task(_run_lodging(i, ov)))
    for i in range(1, len(overnights)):
        tasks.append(
            asyncio.create_task(_run_transit(i - 1, overnights[i - 1],
                                              overnights[i]))
        )

    done_sentinels = 0
    total = len(tasks)
    while done_sentinels < total:
        chunk = await queue.get()
        if chunk is None:
            done_sentinels += 1
            continue
        yield chunk
    await asyncio.gather(*tasks, return_exceptions=True)


def _lodging_agent_id(i: int) -> str:
    return f"lodging[{i}]"


def _transit_agent_id(pair_i: int) -> str:
    return f"transit[{pair_i}]"


# ---------------------------------------------------------------------
# Per-agent runners

async def _drain_lodging_agent(
    ov: dict, client: anthropic.AsyncAnthropic,
    parent_trace: RequestTrace, agent_id: str,
    queue: asyncio.Queue[bytes | None],
) -> None:
    from .chat import _run_chat_inner

    # Only expose the tools this agent should ever call.
    tool_names = ["search_lodging"]

    prompt = LODGING_PROMPT
    user_msg = (
        f"Overnight #{ov.get('day', '?')}: "
        f"{ov.get('name', '?')} (ref={ov.get('ref', '?')}). "
        f"Coordinates: {ov.get('lonlat', '?')}"
    )
    messages = [{"role": "user", "content": user_msg}]

    with RequestTrace(
        model=parent_trace._model,
        prompt_head=f"[lodging {ov.get('name')}]",
        parent_request_id=parent_trace.request_id,
        agent_role=f"lodging[{ov.get('name')}]",
    ) as tr:
        async for chunk in _run_chat_inner(
            client, messages, tr,
            agent_id=agent_id,
            system_prompt=prompt,
            tool_names=tool_names,
            max_rounds=ENRICHMENT_MAX_ROUNDS,
            extra_tools=[SUBMIT_LODGING_TOOL],
            extra_impls={"submit_lodging": _capture_lodging_result},
        ):
            await queue.put(chunk)


async def _drain_transit_agent(
    prev: dict, cur: dict, client: anthropic.AsyncAnthropic,
    parent_trace: RequestTrace, agent_id: str,
    queue: asyncio.Queue[bytes | None],
) -> None:
    from .chat import _run_chat_inner

    tool_names = ["direct_rail_service", "train_booking_links"]

    prompt = TRANSIT_PROMPT
    user_msg = (
        f"Partner needs to ride from "
        f"{prev.get('name', '?')} (ref={prev.get('ref', '?')}) → "
        f"{cur.get('name', '?')} (ref={cur.get('ref', '?')})."
    )
    messages = [{"role": "user", "content": user_msg}]

    with RequestTrace(
        model=parent_trace._model,
        prompt_head=f"[transit {prev.get('name')}→{cur.get('name')}]",
        parent_request_id=parent_trace.request_id,
        agent_role=f"transit[{prev.get('name')}→{cur.get('name')}]",
    ) as tr:
        async for chunk in _run_chat_inner(
            client, messages, tr,
            agent_id=agent_id,
            system_prompt=prompt,
            tool_names=tool_names,
            max_rounds=ENRICHMENT_MAX_ROUNDS,
            extra_tools=[SUBMIT_TRANSIT_TOOL],
            extra_impls={"submit_transit": _capture_transit_result},
        ):
            await queue.put(chunk)


def _capture_lodging_result(inp: dict) -> dict:
    """Validate the submitted result. The Pydantic side is
    fire-and-forget — we don't need to keep the parsed object here
    because the SSE stream already carried the raw tool_call event
    to the frontend, and the frontend renders directly from that."""
    try:
        LodgingResult(**inp)
        return {"ok": True, "n_hotels": len(inp.get("hotels") or [])}
    except ValidationError as ve:
        return {"error": f"schema mismatch: {ve.errors()}"}


def _capture_transit_result(inp: dict) -> dict:
    try:
        TransitResult(**inp)
        return {"ok": True, "n_urls": len(inp.get("booking_urls") or [])}
    except ValidationError as ve:
        return {"error": f"schema mismatch: {ve.errors()}"}
