"""Multi-agent flagship planner: supervisor + fan-out segment agents.

Shape (from ~/.claude/plans/splendid-swinging-pizza.md):

    user prompt
        ↓
    SUPERVISOR  (one LLM, small tool surface)
        picks corridor hubs, decomposes into SegmentSpec[]
        ↓
    FAN-OUT     (asyncio.gather, semaphore-bounded)
        SEG 0 → SEG 1 → SEG N   in parallel
        each runs the standard route/stations/re-route/split/rail
        pipeline scoped to a single hub-to-hub segment
        each writes a SegmentResult via `submit_segment`
        ↓
    MERGE       (supervisor stage 2)
        writes intro + segment narratives (in order) + closing
        → tokens stream to the user

Each agent is a `RequestTrace` child of the coordinator trace, so
downstream log tooling can reconstruct the tree. Each agent's SSE
events carry an `agent_id` tag so the frontend routes them to the
right per-segment bubble.

Failure handling: segment failures are captured (not raised) —
merge sees a `status="failed"` result and produces a best-effort
plan. Supervisor failure falls back to the single-agent path
(handled in `chat._run_chat`, not here).
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, AsyncIterator

import anthropic
from pydantic import BaseModel, ValidationError

from .tracing import RequestTrace


# ---------------------------------------------------------------------
# Config

# Bound sub-agent concurrency. Kept modest so we stay well under
# Anthropic RPM/TPM for typical accounts. Tune with env var.
MAX_PARALLEL_AGENTS = int(os.environ.get("MAX_PARALLEL_AGENTS", "5"))
# Per-segment budget. Segment agents plan one leg — much tighter
# than the whole trip, so a low cap catches runaway loops early.
SEGMENT_MAX_ROUNDS = int(os.environ.get("SEGMENT_MAX_ROUNDS", "15"))
SUPERVISOR_MAX_ROUNDS = int(os.environ.get("SUPERVISOR_MAX_ROUNDS", "12"))
MERGE_MAX_ROUNDS = int(os.environ.get("MERGE_MAX_ROUNDS", "3"))
# Per-segment wall-clock cap. If a segment stalls past this, we
# cancel it and fall back to a failed placeholder in the merge.
SEGMENT_TIMEOUT_S = float(os.environ.get("SEGMENT_TIMEOUT_S", "300"))


# ---------------------------------------------------------------------
# Schemas — structured supervisor + segment output via Anthropic
# tool-use, so we never parse free-form JSON out of prose.

class SegmentSpec(BaseModel):
    """One hub-to-hub leg the supervisor hands to a segment agent."""
    segment_i: int
    from_ref: str
    to_ref: str
    from_name: str
    to_name: str
    # "hub"      — an anchor the user wants passed through on the bike
    # "detour"   — a hub the user wants to spend 2-3 days at
    role: str = "hub"


class CorridorPlan(BaseModel):
    """Supervisor's high-level output before the fan-out."""
    corridor_hubs: list[dict]  # [{ref, name}, ...] in order
    segments: list[SegmentSpec]
    rationale: str = ""


class SegmentOvernight(BaseModel):
    ref: str
    name: str
    day: int
    km_from_prev: float | None = None


class SegmentResult(BaseModel):
    segment_i: int
    from_ref: str
    to_ref: str
    from_name: str
    to_name: str
    overnights: list[SegmentOvernight] = []
    total_km: float | None = None
    n_days: int | None = None
    narrative_md: str = ""
    status: str = "ok"  # "ok" or "failed"
    error: str | None = None


# ---------------------------------------------------------------------
# Prompts

SUPERVISOR_PROMPT = """You are the SUPERVISOR of a multi-agent bike-tour planner. Your job is narrow and structured:

1. Read the user's request.
2. Resolve any named cities with `search_anchors`.
3. Decide the corridor's HUB SEQUENCE — the ordered list of major cities the tour passes through. For a Graz→Copenhagen tour this is typically 5-7 hubs (Graz, Wien, Brno, Praha, Dresden, Berlin, Hamburg, København). If the user names detour cities ("spend 2-3 days in each"), include them in the hub sequence at the natural corridor position.
4. If the user mentioned rail constraints (direct trains, partner meets by train), call `rail_path` between adjacent hub PAIRS to check the rail spine passes through them; if not, either add an intermediate rail hub or note the gap.
5. **When you are confident about the hub sequence, CALL `finalize_segment_plan` with the segments.** Each segment is one hub-to-hub leg. That's your terminal action — do not narrate the trip, do not compute routes, do not verify overnights. Sub-agents will do all of that in parallel.

Rules:
- Prefer 5-7 segments. Fewer means each segment is huge; more means overhead. The corridor's natural hubs (major cities) usually dictate this.
- Every segment's `from_ref` MUST equal the previous segment's `to_ref` (contiguous chain).
- The first segment's `from_ref` = user's start; the last segment's `to_ref` = user's end.
- Segment `role` is "detour" if the user asked to spend multiple days there, else "hub".
- Do NOT call `route`, `split_into_stages`, `stations_along_route` — those belong to segment agents. Your surface is small on purpose.

Style: be terse. One-line rationale is enough."""


SEGMENT_PROMPT_HEADER = """You are a SEGMENT PLANNER agent in a multi-agent bike-tour system. Your ONE JOB is to plan a single hub-to-hub leg — nothing else. The supervisor already picked the hubs; you take that as fixed. Other segment agents are planning other legs in parallel; do not touch anything outside your assigned segment.

Your segment:  {from_name} ({from_ref})  →  {to_name} ({to_ref})

Corridor context (all hubs, in order, for orientation): {corridor_context}

## Strict tool-call budget (this is the hard part — don't overspend it)

You have room for approximately these tool calls, in this order:

1. `rail_path({from_ref}, {to_ref})` — optional, only if the user's prompt has a rail constraint. If it returns `reachable: false`, DO NOT retry it — use `stations_along_route` fallback (see below).
2. `route({from_ref}, {to_ref}, via_refs=<rail spine>)` — once, at most twice. If your first call gives a bad km total, one revision is OK; more is not.
3. `split_into_stages({from_ref}, {to_ref}, via_refs=<same>, target_km_per_day=<user's target>)` — once.
4. `direct_rail_service_batch` on the CONSECUTIVE-PAIR list of overnights — once. **If ANY pair comes back `direct=false`, ACCEPT IT** and note the caveat in your narrative_md. DO NOT loop trying to find a perfect chain — a partial-rail segment with a noted transfer is FINE, an unsubmitted segment is a failure.
5. `submit_segment` — TERMINAL. Call this after the direct-rail check no matter what the results are.

Total: ≤6 tool calls in a healthy run. If you find yourself past your fourth `route` or third `direct_rail_service_batch` call, you're wasting the budget. SUBMIT what you have with a caveat and stop.

## Non-negotiables

- `submit_segment` is your TERMINAL action. Once called, stop. Do not call any other tool.
- Do not narrate outside `submit_segment.narrative_md`.
- Do not verify things the supervisor already verified (hub choice).
- If the direct-rail check returns partial results, THAT IS ACCEPTABLE. Note in narrative_md that leg K-1 → K needs a transfer; submit anyway.

## Style

Your `narrative_md` is a compact Markdown fragment: an H2 heading (## Segment K: X → Y), a per-day distance table, a one-line rail check summary. No preamble.

**Units**: internal tools always return km. In `narrative_md`, use whichever unit the user asked for — if they said "50 miles/day", convert to miles (1 km ≈ 0.6214 mi) and label the column "mi". Otherwise use km. Never mix."""


MERGE_PROMPT = """You are the MERGE stage of a multi-agent bike-tour planner. You have:

1. The user's original request.
2. The supervisor's chosen hub sequence (in order).
3. N SegmentResult narratives, each covering one hub-to-hub leg — already written as Markdown and **already renumbered globally** by the controller (Day 1 = trip's day 1; segment K's days follow segment K-1's without gap).

Your job: emit ONE cohesive final response. Structure:

- One-paragraph intro naming the corridor, total km, n days, and what the partner-by-train constraint yields.
- Concatenate the segment narratives IN ORDER (segment_i ascending) VERBATIM. Their H2 headings and day-column tables are already correct — DO NOT renumber, do not rewrite the tables. Just print each narrative back-to-back.
- One-paragraph closing summary: total km, n riding days, n rest/detour days, buffer days if any.
- **End with a one-line opt-in for follow-up help**: "Say **book lodging** or **book trains** if you'd like per-overnight hotel picks and train-ticket links for the partner." No em-dashes, keep it plain. This is the only trailing line — no other closer.
- If any segment failed (status != "ok"), note it clearly under a "## Segment [N]: FAILED" heading with the error message.

DO NOT re-verify anything. DO NOT call any tools. Just stream the merged Markdown."""


# ---------------------------------------------------------------------
# Structured-output tools — the supervisor and each segment agent call
# these as their terminal action. Args match the Pydantic schemas.

FINALIZE_SEGMENT_PLAN_TOOL: dict = {
    "name": "finalize_segment_plan",
    "description": (
        "TERMINAL ACTION for the supervisor. Commit the corridor's hub "
        "sequence + segment decomposition. Sub-agents will plan each "
        "segment in parallel. Do not call any other tool after this."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "corridor_hubs": {
                "type": "array",
                "description": "Every hub in order (start, intermediate hubs, end). Refs + names.",
                "items": {
                    "type": "object",
                    "properties": {
                        "ref":  {"type": "string"},
                        "name": {"type": "string"},
                    },
                    "required": ["ref", "name"],
                },
            },
            "segments": {
                "type": "array",
                "description": "Hub-to-hub legs (must chain: each segment's from_ref = prev segment's to_ref).",
                "items": {
                    "type": "object",
                    "properties": {
                        "segment_i":  {"type": "integer"},
                        "from_ref":   {"type": "string"},
                        "to_ref":     {"type": "string"},
                        "from_name":  {"type": "string"},
                        "to_name":    {"type": "string"},
                        "role":       {"type": "string", "enum": ["hub", "detour"]},
                    },
                    "required": ["segment_i", "from_ref", "to_ref",
                                 "from_name", "to_name"],
                },
            },
            "rationale": {"type": "string"},
        },
        "required": ["corridor_hubs", "segments"],
    },
}

SUBMIT_SEGMENT_TOOL: dict = {
    "name": "submit_segment",
    "description": (
        "TERMINAL ACTION for a segment agent. Commit this leg's plan: "
        "overnights, total km, n days, and a Markdown narrative fragment "
        "that the merger will splice into the final response. Do not "
        "call any other tool after this."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "segment_i":  {"type": "integer"},
            "from_ref":   {"type": "string"},
            "to_ref":     {"type": "string"},
            "from_name":  {"type": "string"},
            "to_name":    {"type": "string"},
            "overnights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ref":          {"type": "string"},
                        "name":         {"type": "string"},
                        "day":          {"type": "integer"},
                        "km_from_prev": {"type": "number"},
                    },
                    "required": ["ref", "name", "day"],
                },
            },
            "total_km":     {"type": "number"},
            "n_days":       {"type": "integer"},
            "narrative_md": {"type": "string"},
        },
        "required": ["segment_i", "from_ref", "to_ref",
                     "from_name", "to_name", "narrative_md"],
    },
}


# ---------------------------------------------------------------------
# Coordinator

async def run_multiagent_plan(
    user_messages: list[dict],
    client: anthropic.AsyncAnthropic,
    parent_trace: RequestTrace,
) -> AsyncIterator[bytes]:
    """Public entry point invoked by chat._run_chat when
    `mode="multiagent"`. Yields SSE bytes for the whole plan."""
    from .chat import _sse, _prompt_head

    # ---- Stage 1: supervisor ----
    supervisor_id = "supervisor"
    yield _sse("agent_start", {
        "role": "supervisor",
        "from_name": None,
        "to_name":   None,
    }, agent_id=supervisor_id)

    plan: CorridorPlan | None = None
    supervisor_error: str | None = None
    try:
        async for chunk in _run_supervisor_stage(
            user_messages, client, parent_trace, supervisor_id,
        ):
            if isinstance(chunk, CorridorPlan):
                plan = chunk
            else:
                yield chunk
    except Exception as exc:
        supervisor_error = f"{type(exc).__name__}: {exc}"

    yield _sse("agent_end", {
        "role": "supervisor",
        "status": "ok" if plan and not supervisor_error else "failed",
        "error": supervisor_error,
    }, agent_id=supervisor_id)

    if plan is None:
        yield _sse("error", {"message": (
            "Supervisor couldn't decompose the corridor into segments. "
            + (supervisor_error or "Model returned no plan.") + " "
            "Fall back to single-agent mode by resending without "
            "mode=multiagent."
        )})
        yield _sse("done", {"stop_reason": "supervisor_failed"})
        return

    # ---- Stage 2: fan out segment agents ----
    sem = asyncio.Semaphore(MAX_PARALLEL_AGENTS)
    # Announce all segment bubbles up front so the frontend can
    # render them in order (not completion order).
    for spec in plan.segments:
        yield _sse("agent_start", {
            "role": "segment",
            "segment_i": spec.segment_i,
            "from_name": spec.from_name,
            "to_name":   spec.to_name,
        }, agent_id=_seg_agent_id(spec.segment_i))

    corridor_context = " → ".join(h.get("name", "?") for h in plan.corridor_hubs)
    results: dict[int, SegmentResult] = {}
    # Queue that fans multiple sub-agent streams onto one output.
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def _run_one(spec: SegmentSpec) -> None:
        """Run one segment agent, pipe its SSE bytes to the queue,
        stash its result in `results`, and emit a `segment_committed`
        event with the segment's polyline as soon as it lands so the
        map lights up progressively (not all at once at the end)."""
        agent_id = _seg_agent_id(spec.segment_i)
        t0 = time.monotonic()
        status = "ok"
        error: str | None = None
        result: SegmentResult | None = None

        async def _drain() -> None:
            nonlocal result
            async for chunk in _iter_segment_agent(
                spec, user_messages, corridor_context,
                client, parent_trace, agent_id,
            ):
                if isinstance(chunk, SegmentResult):
                    result = chunk
                else:
                    await queue.put(chunk)

        try:
            async with sem:
                await asyncio.wait_for(_drain(), timeout=SEGMENT_TIMEOUT_S)
        except asyncio.TimeoutError:
            status = "failed"
            error = f"segment timed out after {SEGMENT_TIMEOUT_S:.0f} s"
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        wall_ms = int((time.monotonic() - t0) * 1000)
        if result is None:
            # Segment agent ran to its round cap without calling
            # `submit_segment`. Fill in a placeholder narrative and
            # mark the segment failed. Reporting status="ok" here
            # (as the old code did) would silently mislead the
            # merge stage / frontend.
            status = "failed"
            error = error or "segment agent hit its round cap without submitting a result"
            result = SegmentResult(
                segment_i=spec.segment_i,
                from_ref=spec.from_ref, to_ref=spec.to_ref,
                from_name=spec.from_name, to_name=spec.to_name,
                narrative_md=(
                    f"## Segment {spec.segment_i}: "
                    f"{spec.from_name} → {spec.to_name}\n\n"
                    f"*Planning failed: {error}.*"
                ),
                status="failed",
                error=error,
            )
        results[spec.segment_i] = result

        # Emit segment_committed as soon as THIS segment lands — before
        # the other segments finish — so the map draws progressively.
        # Day numbers on pins are LOCAL to the segment (Day 1..n_days)
        # at this point; the coordinator's post-gather pass fixes the
        # merged narrative_md text to use global numbering. If we want
        # global numbers on pins too, that's a later re-render pass
        # after all segments in.
        if status == "ok" and result is not None:
            from .tools import _ROUTE_CACHE
            polyline = None
            for key in reversed(list(_ROUTE_CACHE.keys())):
                fr, to, _via = key
                if fr == spec.from_ref and to == spec.to_ref:
                    polyline = _ROUTE_CACHE[key]
                    break
            overnights_out = []
            for i, ov in enumerate(result.overnights):
                overnights_out.append({
                    "ref":          ov.ref,
                    "name":         ov.name,
                    "day":          ov.day if ov.day is not None else i,
                    "km_from_prev": ov.km_from_prev,
                    "lonlat":       _coord_of_ref(ov.ref),
                })
            await queue.put(_sse("segment_committed", {
                "segment_i":  spec.segment_i,
                "from_ref":   spec.from_ref, "to_ref":   spec.to_ref,
                "from_name":  spec.from_name, "to_name":  spec.to_name,
                "n_days":     result.n_days,
                "overnights": overnights_out,
                "polyline":   polyline,
                "status":     status,
            }))

        await queue.put(_sse("agent_end", {
            "role":   "segment",
            "segment_i":     spec.segment_i,
            "status":        status,
            "error":         error,
            "wall_ms":       wall_ms,
        }, agent_id=agent_id))
        await queue.put(None)  # sentinel — one per task

    tasks = [asyncio.create_task(_run_one(s)) for s in plan.segments]
    done_sentinels = 0
    while done_sentinels < len(tasks):
        chunk = await queue.get()
        if chunk is None:
            done_sentinels += 1
            continue
        yield chunk
    # Ensure all tasks are collected (surfaces any uncaught exceptions).
    await asyncio.gather(*tasks, return_exceptions=True)

    # ---- Stage 2b: coordinator renumbers narratives for merge ----
    # Segment agents emitted their own `segment_committed` events as
    # they landed (see _run_one). Those carry SEGMENT-LOCAL day
    # numbers on the map pins — good enough for immediate map
    # rendering. What still needs a global pass: the merge stage's
    # input text. Rewrite each segment's `narrative_md` so day
    # numbers are cumulative across the whole tour (segment K's
    # local Day 1 becomes trip's Day (prior segments' days + 1)).
    ordered_results = [results[i] for i in sorted(results)]
    day_offset = 0
    for r in ordered_results:
        seg_n_days = r.n_days or max(0, len(r.overnights) - 1)
        r.narrative_md = _renumber_day_keywords(
            r.narrative_md, offset=day_offset, max_n=seg_n_days,
        )
        day_offset += seg_n_days

    # ---- Stage 3: merge ----
    merge_id = "merge"
    yield _sse("agent_start", {
        "role": "merge",
        "from_name": None,
        "to_name":   None,
    }, agent_id=merge_id)
    try:
        async for chunk in _run_merge_stage(
            user_messages, plan, ordered_results,
            client, parent_trace, merge_id,
        ):
            yield chunk
    except Exception as exc:
        yield _sse("error", {
            "message": f"Merge stage failed: {type(exc).__name__}: {exc}",
        }, agent_id=merge_id)
    yield _sse("agent_end", {
        "role": "merge", "status": "ok",
    }, agent_id=merge_id)

    # NOTE: post-plan enrichment (lodging + transit booking) is
    # OPT-IN. The merge prompt asks the user whether they'd like
    # help with lodging or booking; when they say yes on the next
    # turn, chat._run_chat routes to enrichment against the plan
    # still visible in the conversation history. Auto-running
    # enrichment on every plan spent minutes on sub-agents the user
    # may not want.

    yield _sse("done", {"stop_reason": "end_turn"})


def _seg_agent_id(segment_i: int) -> str:
    return f"seg[{segment_i}]"


def _coord_of_ref(ref: str | None) -> list[float] | None:
    """Anchor ref → [lon, lat] via the loaded profile. Used to attach
    coords to overnights in `segment_committed` events."""
    if not ref:
        return None
    from . import trunk_router
    from .settings import DEFAULT_PROFILE
    prof = trunk_router._load_profile(DEFAULT_PROFILE)
    ci = prof.city_idx_by_ref.get(ref)
    if ci is None:
        return None
    c = prof.cities[int(ci)]
    return [float(c["lon"]), float(c["lat"])]


import re as _re
_DAY_KW_RE = _re.compile(r"\bDay\s+(\d+)\b")
# Match table body rows whose FIRST cell is an integer (a day
# column). Avoids over-matching km values elsewhere in the row.
_TABLE_ROW_DAY_RE = _re.compile(r"^(\s*\|\s*)(\d+)(\s*\|)", flags=_re.MULTILINE)


def _renumber_day_keywords(md: str, offset: int, max_n: int) -> str:
    """Segment narratives use per-segment day numbering (Day 1..N).
    The controller renumbers them to global (Day 1+offset..N+offset).
    Two substitution passes:
      1. `Day N` keyword mentions in prose.
      2. Table body rows whose first cell is a bare integer 1..N.
    Skips `Day N` with N > max_n as a light guard against clobbering
    unrelated numbers ("Day 30 buffer") the model may reference."""
    if not md or offset == 0:
        return md

    def _sub_kw(m: _re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= max_n:
            return f"Day {n + offset}"
        return m.group(0)

    def _sub_row(m: _re.Match) -> str:
        n = int(m.group(2))
        if 1 <= n <= max_n:
            return f"{m.group(1)}{n + offset}{m.group(3)}"
        return m.group(0)

    md = _DAY_KW_RE.sub(_sub_kw, md)
    md = _TABLE_ROW_DAY_RE.sub(_sub_row, md)
    return md


# ---------------------------------------------------------------------
# Enrichment follow-up (opt-in second turn)

async def run_enrichment_followup(
    user_messages: list[dict],
    client: anthropic.AsyncAnthropic,
    parent_trace: RequestTrace,
) -> AsyncIterator[bytes]:
    """Called when the user's follow-up message asks for lodging /
    transit help after a prior plan turn. Runs a small extract-
    overnights agent to pull the overnight sequence from the prior
    assistant turn, then fans out lodging + transit sub-agents."""
    from .chat import _run_chat_inner, _sse
    from . import enrichment, trunk_router
    from .settings import DEFAULT_PROFILE

    extract_id = "extract-overnights"
    yield _sse("agent_start", {
        "role":      "extract",
        "from_name": None,
        "to_name":   None,
    }, agent_id=extract_id)

    captured: dict[str, list[dict]] = {"overnights": []}

    def _capture(inp: dict) -> dict:
        ovs = inp.get("overnights") or []
        captured["overnights"] = [
            {"ref": o.get("ref"), "name": o.get("name")}
            for o in ovs if o.get("ref")
        ]
        return {"ok": True, "n": len(captured["overnights"])}

    with RequestTrace(
        model=parent_trace._model,
        prompt_head="[extract-overnights]",
        parent_request_id=parent_trace.request_id,
        agent_role="extract-overnights",
    ) as tr:
        messages = [dict(m) for m in user_messages]
        async for chunk in _run_chat_inner(
            client, messages, tr,
            agent_id=extract_id,
            system_prompt=EXTRACT_OVERNIGHTS_PROMPT,
            tool_names=["search_anchors"],
            max_rounds=6,
            extra_tools=[EXTRACT_OVERNIGHTS_TOOL],
            extra_impls={"submit_overnights": _capture},
        ):
            yield chunk

    overnights = captured["overnights"]
    yield _sse("agent_end", {
        "role":   "extract",
        "status": "ok" if overnights else "failed",
        "error":  None if overnights else "no overnights found in prior turn",
    }, agent_id=extract_id)

    if not overnights:
        yield _sse("error", {"message": (
            "Couldn't extract the overnights from the plan I wrote. "
            "Try 'book trains and hotels for Graz, Wien, Praha, ...' "
            "with the city names spelled out."
        )})
        yield _sse("done", {"stop_reason": "extract_failed"})
        return

    # Look up lonlats for the extracted overnights.
    prof = trunk_router._load_profile(DEFAULT_PROFILE)
    enriched: list[dict] = []
    for i, ov in enumerate(overnights):
        ref = ov.get("ref")
        ci = prof.city_idx_by_ref.get(ref) if ref else None
        if ci is None:
            continue
        c = prof.cities[int(ci)]
        enriched.append({
            "ref":    ref,
            "name":   ov.get("name") or c.get("name"),
            "day":    i,
            "lonlat": f"{float(c['lon'])},{float(c['lat'])}",
        })

    async for chunk in enrichment.run_enrichment_stage(
        enriched, client, parent_trace,
    ):
        yield chunk

    yield _sse("done", {"stop_reason": "end_turn"})


EXTRACT_OVERNIGHTS_TOOL: dict = {
    "name": "submit_overnights",
    "description": (
        "TERMINAL. Emit the ordered list of overnights from the plan "
        "you found in the prior assistant turn. Each entry: `ref` "
        "(anchor ref like `db:66`) and `name` (city name). Order "
        "matches the plan's day sequence."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "overnights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ref":  {"type": "string"},
                        "name": {"type": "string"},
                    },
                    "required": ["ref", "name"],
                },
            },
        },
        "required": ["overnights"],
    },
}


EXTRACT_OVERNIGHTS_PROMPT = """You are an EXTRACTOR. The conversation history contains an assistant plan the user is now asking to enrich (book lodging or trains). Do one thing:

Read the last assistant turn (the plan). Pull out the ORDERED list of overnights (base stops). Each is a city name; you can use `search_anchors` to resolve names to refs when needed. Call `submit_overnights` with the ordered list and STOP.

Do not narrate. Do not answer any question. Just resolve overnight names to refs and submit."""


# ---------------------------------------------------------------------
# Supervisor stage

async def _run_supervisor_stage(
    user_messages: list[dict],
    client: anthropic.AsyncAnthropic,
    parent_trace: RequestTrace,
    agent_id: str,
) -> AsyncIterator[bytes | CorridorPlan]:
    """Run the supervisor. Yields SSE bytes as normal, plus ONE
    CorridorPlan sentinel at the end when finalize_segment_plan is
    invoked. Caller separates the two."""
    from .chat import _run_chat_inner, _sse

    # Supervisor sees a small tool subset. finalize_segment_plan is
    # injected as its terminal action.
    supervisor_tools = [
        "search_anchors", "rail_path",
        "direct_rail_service", "direct_rail_service_batch",
    ]

    plan_captured: dict[str, CorridorPlan | None] = {"plan": None}

    def _finalize_impl(inp: dict) -> dict:
        # Called from run_in_executor; validate + capture. Returns a
        # plain dict so _run_chat_inner's tool_call event serializes.
        try:
            cp = CorridorPlan(**inp)
            plan_captured["plan"] = cp
            return {"ok": True, "n_segments": len(cp.segments)}
        except ValidationError as ve:
            return {"error": f"schema mismatch: {ve.errors()}"}

    with RequestTrace(
        model=parent_trace._model,
        prompt_head="[supervisor]",
        parent_request_id=parent_trace.request_id,
        agent_role="supervisor",
    ) as tr:
        messages = [dict(m) for m in user_messages]
        async for chunk in _run_chat_inner(
            client, messages, tr,
            agent_id=agent_id,
            system_prompt=SUPERVISOR_PROMPT,
            tool_names=supervisor_tools,
            max_rounds=SUPERVISOR_MAX_ROUNDS,
            extra_tools=[FINALIZE_SEGMENT_PLAN_TOOL],
            extra_impls={"finalize_segment_plan": _finalize_impl},
        ):
            yield chunk

    if plan_captured["plan"] is not None:
        yield plan_captured["plan"]


# ---------------------------------------------------------------------
# Segment stage

async def _iter_segment_agent(
    spec: SegmentSpec,
    user_messages: list[dict],
    corridor_context: str,
    client: anthropic.AsyncAnthropic,
    parent_trace: RequestTrace,
    agent_id: str,
) -> AsyncIterator[bytes | SegmentResult]:
    """One segment agent. Yields SSE bytes, then one SegmentResult
    sentinel when submit_segment is invoked."""
    from .chat import _run_chat_inner

    segment_tools = [
        "search_anchors", "route", "stations_along_route",
        "rail_path", "direct_rail_service_batch", "split_into_stages",
    ]

    result_captured: dict[str, SegmentResult | None] = {"result": None}

    def _submit_impl(inp: dict) -> dict:
        try:
            sr = SegmentResult(**inp, status="ok")
            result_captured["result"] = sr
            return {"ok": True}
        except ValidationError as ve:
            return {"error": f"schema mismatch: {ve.errors()}"}

    prompt = SEGMENT_PROMPT_HEADER.format(
        from_ref=spec.from_ref, to_ref=spec.to_ref,
        from_name=spec.from_name, to_name=spec.to_name,
        corridor_context=corridor_context,
        max_rounds=SEGMENT_MAX_ROUNDS,
    )
    with RequestTrace(
        model=parent_trace._model,
        prompt_head=f"[segment {spec.segment_i}]",
        parent_request_id=parent_trace.request_id,
        agent_role=f"segment[{spec.segment_i}]",
    ) as tr:
        messages = [dict(m) for m in user_messages]
        async for chunk in _run_chat_inner(
            client, messages, tr,
            agent_id=agent_id,
            system_prompt=prompt,
            tool_names=segment_tools,
            max_rounds=SEGMENT_MAX_ROUNDS,
            extra_tools=[SUBMIT_SEGMENT_TOOL],
            extra_impls={"submit_segment": _submit_impl},
        ):
            yield chunk

    if result_captured["result"] is not None:
        yield result_captured["result"]


# ---------------------------------------------------------------------
# Merge stage

async def _run_merge_stage(
    user_messages: list[dict],
    plan: CorridorPlan,
    results: list[SegmentResult],
    client: anthropic.AsyncAnthropic,
    parent_trace: RequestTrace,
    agent_id: str,
) -> AsyncIterator[bytes]:
    """Text-only stage. No tools. Streams the merged Markdown."""
    from .chat import _sse

    corridor_txt = " → ".join(h.get("name", "?") for h in plan.corridor_hubs)
    segments_txt = "\n\n---\n\n".join(
        (
            f"[segment_i={r.segment_i} status={r.status}]\n\n"
            + (r.narrative_md or "*(no narrative)*")
        )
        for r in results
    )
    merge_user = (
        "Corridor hubs: " + corridor_txt + "\n\n"
        "Segment narratives (in order):\n\n" + segments_txt
    )
    messages = [dict(m) for m in user_messages] + [
        {"role": "user", "content": merge_user},
    ]

    with RequestTrace(
        model=parent_trace._model,
        prompt_head="[merge]",
        parent_request_id=parent_trace.request_id,
        agent_role="merge",
    ) as tr:
        tr.round_start(0, messages_len=len(messages))
        t_round = time.monotonic()
        async with client.messages.stream(
            model=parent_trace._model,
            max_tokens=16384,
            system=[{
                "type": "text",
                "text": MERGE_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=messages,
        ) as stream:
            async for event in stream:
                if event.type == "text":
                    yield _sse("text", {"delta": event.text},
                               agent_id=agent_id)
                else:
                    yield b": tick\n\n"
            final = await stream.get_final_message()
        round_ms = int((time.monotonic() - t_round) * 1000)
        usage = getattr(final, "usage", None)
        usage_dict = None
        if usage is not None:
            usage_dict = {
                "input_tokens":  getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
                "cache_read_input_tokens":     getattr(usage, "cache_read_input_tokens", None),
            }
        tr.round_end(
            round_i=0,
            stop_reason=str(final.stop_reason),
            n_text=sum(1 for b in final.content if b.type == "text"),
            n_tool=0,
            latency_ms=round_ms,
            usage=usage_dict,
        )
        tr.set_stop_reason(str(final.stop_reason))
