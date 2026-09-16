"""LangGraph fork of the /chat endpoint.

Selected at runtime by CHAT_BACKEND=langgraph. Same SSE contract as
the native `chat.py` so the web frontend serves either transparently.

Graph shape:

              ┌── planner ──┐
              │             │
              ▼             │  (revise)
           router           │
              │             │
              ▼             │
     ┌──── enricher_map ────┤
     │        │  (Send fan-out per stage)
     │        │
     ▼        ▼
  enrich    enrich    …    ← one enricher_node per stage in parallel
  stage 1   stage 2         (LangGraph's `Send` primitive)
     │        │
     └────┬───┘
          ▼
      composer
          │
          ▼
       critic ───(approve)────► END
          │
          └───(reject, iter<2)──▶ back to planner with feedback

The three model calls (planner, composer, critic) use
langchain-anthropic. Tool calls (route, split_into_stages,
pois_along_route, stations_along_route) are direct Python via
`api.app.tools.call_tool` — no LLM in the loop for tool invocation
itself. That's the langgraph shape: the graph is the control flow,
the LLM contributes at planner/composer/critic nodes only.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Annotated, Any, Iterator, Literal, Optional, TypedDict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from .tools import call_tool


LG_MODEL = os.environ.get("LANGGRAPH_MODEL", "claude-sonnet-5")
LG_MAX_REVISIONS = int(os.environ.get("LANGGRAPH_MAX_REVISIONS", "2"))

router = APIRouter()


# ---------------------------------------------------------------------------
# State

class StageEnrichment(TypedDict, total=False):
    day: int
    from_ref: str
    to_ref: str
    km: float
    stations: list[dict]        # from stations_along_route (nearby anchors)
    viewpoints: list[dict]      # from pois_along_route


def _merge_enrichments(a: dict[int, StageEnrichment],
                       b: dict[int, StageEnrichment]) -> dict[int, StageEnrichment]:
    """Reducer for parallel enricher writes. LangGraph fan-out writes
    each Send's result into state; we merge by `day` key."""
    out = dict(a or {})
    out.update(b or {})
    return out


def _append_events(a: list[dict], b: list[dict]) -> list[dict]:
    """Reducer for `stream_events`. Concurrent nodes (the parallel
    enrichers under Send fan-out) each contribute their event lists;
    without a reducer LangGraph raises InvalidUpdateError. Append
    preserves the order of node completion."""
    return (a or []) + (b or [])


class PlanState(TypedDict, total=False):
    # Immutable per turn.
    user_prompt: str

    # Planner output.
    from_ref: Optional[str]
    to_ref: Optional[str]
    via_refs: list[str]
    target_km_per_day: Optional[float]
    need_stages: bool
    need_enrichment: bool
    enrichment_categories: list[str]

    # Router output.
    route_result: Optional[dict]
    stages: list[dict]

    # Enricher output — fan-out reducer merges per-stage writes.
    enrichments: Annotated[dict[int, StageEnrichment], _merge_enrichments]

    # Composer output.
    itinerary_md: str

    # Critic output — drives the revision loop.
    approved: bool
    critic_feedback: str
    iteration: int

    # Streaming events. Each node appends its tool calls / status
    # messages; the SSE bridge drains this list between graph steps.
    # Annotated with the append reducer so concurrent nodes (the
    # parallel enrichers) don't conflict.
    stream_events: Annotated[list[dict], _append_events]

    # Per-worker payload fields. Only populated inside an enricher
    # worker; None elsewhere. LangGraph's Send delivers by merging
    # the send payload into the target node's state view, so these
    # keys must exist on the state schema for the node to see them.
    stage: Optional[dict]
    stage_categories: Optional[list[str]]
    stage_from_ref_root: Optional[str]
    stage_to_ref_root: Optional[str]
    stage_via_refs_root: Optional[list[str]]


# ---------------------------------------------------------------------------
# LLM helpers

def _chat_anthropic():
    """Lazily construct a ChatAnthropic client. Kept as a function so
    module import doesn't fail when ANTHROPIC_API_KEY is unset (e.g.
    in CI's import smoke test)."""
    from langchain_anthropic import ChatAnthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set")
    return ChatAnthropic(
        model=LG_MODEL,
        api_key=key,
        timeout=120.0,
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# Node: planner

PLANNER_SYSTEM = """You are the PLANNING NODE of an agentic bike-tour planner. Your job is to translate the user's request into a structured plan the deterministic router node can execute. You do NOT call tools directly; you emit a JSON plan.

Corridor coverage: Austria, Czechia, Germany, Denmark. Anchor refs look like `db:1234` (city) or `ferry:12345` (pier).

For the from_ref / to_ref, and any via_refs, use your knowledge of European geography to name PLAUSIBLE anchor refs based on the cities the user mentions. The router node will handle resolving names to refs via a search step if your guess is wrong.

Cues:
- "plan a X-day tour", "N km/day", "break into stages" → need_stages=true
- "rail-accessible overnights", "nearby viewpoints", "enrich with destinations" → need_enrichment=true, enrichment_categories filled

If the user asks you to REVISE (critic_feedback in the message), adjust `via_refs`, `target_km_per_day`, or other fields to address the feedback while staying faithful to the original request."""


class PlannerOutput(BaseModel):
    """Structured plan the router node executes."""
    from_hint: str = Field(description="Start city name or anchor ref")
    to_hint: str = Field(description="End city name or anchor ref")
    via_hints: list[str] = Field(default_factory=list,
                                 description="Ordered intermediate city names / refs")
    target_km_per_day: Optional[float] = Field(default=None)
    need_stages: bool = Field(default=False)
    need_enrichment: bool = Field(default=False)
    enrichment_categories: list[Literal["viewpoint", "food", "lodging", "water", "bike_service"]] = \
        Field(default_factory=list)


async def _planner_node(state: PlanState) -> PlanState:
    llm = _chat_anthropic().with_structured_output(PlannerOutput)
    msg = f"User request: {state['user_prompt']}"
    if state.get("critic_feedback"):
        msg += f"\n\nCritic feedback from previous iteration: {state['critic_feedback']}\n" \
               f"Adjust the plan to address this while staying faithful to the request."
    plan: PlannerOutput = await llm.ainvoke([
        ("system", PLANNER_SYSTEM),
        ("human", msg),
    ])

    # Resolve each hint to a ref via search_anchors — deterministic,
    # not an LLM call.
    def _resolve(hint: str) -> Optional[str]:
        if hint.startswith(("db:", "ferry:", "osm:")):
            return hint
        res = call_tool("search_anchors", {"query": hint, "limit": 1})
        results = res.get("results", [])
        return results[0]["ref"] if results else None

    # Emit only the delta — the append reducer merges with existing events.
    new_events = [{"event": "text", "delta":
        f"🧭 Planning route from {plan.from_hint} to {plan.to_hint}"
        + (f" via {', '.join(plan.via_hints)}" if plan.via_hints else "")
        + (f", {plan.target_km_per_day:.0f} km/day" if plan.target_km_per_day else "")
        + "…\n\n"}]

    from_ref = _resolve(plan.from_hint)
    to_ref   = _resolve(plan.to_hint)
    via_refs = [r for r in (_resolve(h) for h in plan.via_hints) if r]

    return {
        "from_ref": from_ref,
        "to_ref":   to_ref,
        "via_refs": via_refs,
        "target_km_per_day": plan.target_km_per_day,
        "need_stages": plan.need_stages,
        "need_enrichment": plan.need_enrichment,
        "enrichment_categories": plan.enrichment_categories,
        "stream_events": new_events,
    }


# ---------------------------------------------------------------------------
# Node: router (deterministic)

async def _router_node(state: PlanState) -> PlanState:
    new_events: list[dict] = []

    if not state.get("from_ref") or not state.get("to_ref"):
        new_events.append({"event": "error",
                           "message": "Planner could not resolve start/end refs."})
        return {"stream_events": new_events}

    route_args = {
        "from_ref": state["from_ref"],
        "to_ref":   state["to_ref"],
        "via_refs": state.get("via_refs") or None,
    }
    route_result = call_tool("route", route_args)
    new_events.append({"event": "tool_call",
                       "name": "route", "input": route_args, "output": route_result})

    stages = []
    if state.get("need_stages") and state.get("target_km_per_day"):
        stage_args = {
            "from_ref": state["from_ref"],
            "to_ref":   state["to_ref"],
            "via_refs": state.get("via_refs") or None,
            "target_km_per_day": state["target_km_per_day"],
        }
        stage_result = call_tool("split_into_stages", stage_args)
        new_events.append({"event": "tool_call",
                           "name": "split_into_stages",
                           "input": stage_args, "output": stage_result})
        stages = stage_result.get("stages", [])

    return {
        "route_result": route_result,
        "stages": stages,
        "stream_events": new_events,
    }


# ---------------------------------------------------------------------------
# Node: enricher fan-out
#
# When the plan calls for enrichment, we spawn one `Send` per stage.
# LangGraph runs them in parallel; each writes to state.enrichments[day].
# The reducer defined above merges by day.

def _enrich_map(state: PlanState) -> list[Send]:
    """Router → enricher_map: emit one Send per stage. Returns a list
    of Send objects, one per parallel worker. Fields ("stage",
    "stage_categories", ...) become state overrides in each Send's
    target-node invocation."""
    if not state.get("need_enrichment") or not state.get("stages"):
        return []
    cats = state.get("enrichment_categories") or ["viewpoint"]
    return [
        Send("enrich_stage", {
            "stage": stage,
            "stage_categories": cats,
            "stage_from_ref_root": state.get("from_ref"),
            "stage_to_ref_root":   state.get("to_ref"),
            "stage_via_refs_root": state.get("via_refs") or [],
        })
        for stage in state["stages"]
    ]


async def _enrich_stage_node(state: PlanState) -> PlanState:
    """Per-stage enrichment worker. Runs in parallel with siblings.
    Reads only the Send-delivered `stage*` keys; writes into
    `enrichments[day]` (reducer-merged) and `stream_events`."""
    stage = state.get("stage") or {}
    if not stage:
        return {}   # nothing to do — Send never fired
    day = stage.get("day", 0)

    enrichment: StageEnrichment = {
        "day": day,
        "from_ref": stage.get("from_ref"),
        "to_ref":   stage.get("to_ref"),
        "km":       stage.get("km"),
        "stations": [],
        "viewpoints": [],
    }
    events: list[dict] = []

    # Nearby rail stations at the overnight anchor.
    if stage.get("to_lonlat"):
        st_args = {
            "lon": stage["to_lonlat"][0],
            "lat": stage["to_lonlat"][1],
            "radius_km": 5, "limit": 3,
        }
        st = call_tool("stations_near", st_args)
        events.append({"event": "tool_call",
                       "name": f"stations_near[day{day}]",
                       "input": st_args, "output": st})
        enrichment["stations"] = st.get("stations", [])

    # POIs along the full route (cache-hits via the ROOT refs). Per-
    # stage slicing happens client-side; the model gets the whole
    # POI list and picks per-day.
    cats = state.get("stage_categories") or ["viewpoint"]
    for cat in cats:
        args = {
            "from_ref": state.get("stage_from_ref_root"),
            "to_ref":   state.get("stage_to_ref_root"),
            "via_refs": state.get("stage_via_refs_root") or None,
            "category": cat,
            "buffer_km": 2,
            "limit": 5,
        }
        res = call_tool("pois_along_route", args)
        events.append({"event": "tool_call",
                       "name": f"pois_along_route[{cat},day{day}]",
                       "input": args, "output": res})
        if cat == "viewpoint":
            enrichment["viewpoints"] = res.get("pois", [])

    return {
        "enrichments": {day: enrichment},
        "stream_events": events,
    }


# ---------------------------------------------------------------------------
# Node: composer

COMPOSER_SYSTEM_BASE = """You are the COMPOSER NODE. You DO NOT call tools. You receive the router's output and produce a markdown response proportional to what the user actually asked for.

The user's ORIGINAL prompt is included in the data. Match its scope:
- Simple "route X to Y" → ONE bullet: total km + chain waypoints. No table, no per-day breakdown.
- "plan a X-day tour" / "N km/day" → markdown table with columns: Day, Route, km, Overnight rail, Notable en-route. ONE paragraph rationale under the table.
- Reformat request ("as a table", "just the overnights", "shorter") → match the requested format from the data available.

Style: no preamble, no epilogue, no hedging."""


def _extract_text(content) -> str:
    """Anthropic extended-thinking responses come back as a list of
    typed content blocks ({type: thinking, ...}, {type: text, ...}).
    Pull just the human-visible text out."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


async def _composer_node(state: PlanState) -> PlanState:
    llm = _chat_anthropic()
    route = state.get("route_result") or {}
    stages = state.get("stages") or []
    enrich = state.get("enrichments") or {}

    payload = {
        "user_prompt":  state.get("user_prompt"),
        "total_km":     route.get("total_km"),
        "chain_stops":  route.get("chain_stops"),
        "n_days":       len(stages),
        "need_stages":  state.get("need_stages", False),
        "need_enrichment": state.get("need_enrichment", False),
    }
    if stages:
        payload["stages"] = [
            {
                "day":      s["day"],
                "from":     s.get("from_name"),
                "to":       s.get("to_name"),
                "km":       s.get("km"),
                "stations": (enrich.get(s["day"]) or {}).get("stations", [])[:2],
                "viewpoints": (enrich.get(s["day"]) or {}).get("viewpoints", [])[:3],
            }
            for s in stages
        ]

    reply = await llm.ainvoke([
        ("system", COMPOSER_SYSTEM_BASE),
        ("human", "Data to compose from:\n\n" + json.dumps(payload, indent=2)),
    ])
    text = _extract_text(reply.content).strip()

    return {"itinerary_md": text,
            "stream_events": [{"event": "text", "delta": text}]}


# ---------------------------------------------------------------------------
# Node: critic — validates constraints, decides whether to loop back

CRITIC_SYSTEM = """You are the CRITIC NODE. Given the composed itinerary and the original user request, validate:
- If the user specified a number of days, does the itinerary match?
- If the user specified km/day, are the actual per-day km within ±25% of the target?
- If the user asked for rail-accessible overnights, does each overnight have at least one rail station with n_routes_rail >= 2 within 5 km?
- If the user asked for enrichment, does each riding day have at least one POI listed?

Return `approved: true` if all applicable constraints pass. Otherwise `approved: false` with concise feedback (< 30 words) the planner can act on."""


class CriticVerdict(BaseModel):
    approved: bool
    feedback: str = ""


async def _critic_node(state: PlanState) -> PlanState:
    llm = _chat_anthropic().with_structured_output(CriticVerdict)
    verdict: CriticVerdict = await llm.ainvoke([
        ("system", CRITIC_SYSTEM),
        ("human", f"User request:\n{state.get('user_prompt')}\n\n"
                  f"Composed itinerary:\n{state.get('itinerary_md')}"),
    ])
    new_events = [{"event": "text",
                   "delta": ("\n\n_✓ critic approved_\n" if verdict.approved
                             else f"\n\n_⚠ critic revising: {verdict.feedback}_\n")}]
    return {
        "approved": verdict.approved,
        "critic_feedback": verdict.feedback,
        "iteration": state.get("iteration", 0) + 1,
        "stream_events": new_events,
    }


def _after_critic(state: PlanState) -> Literal["planner", "end"]:
    if state.get("approved") or state.get("iteration", 0) >= LG_MAX_REVISIONS:
        return "end"
    return "planner"


# ---------------------------------------------------------------------------
# Graph assembly

def _route_after_router(state: PlanState):
    """One conditional edge from router: return either a list of
    `Send` objects (fan-out to per-stage enrich_stage workers) OR
    the string name of the next node. Merging these two behaviours
    into one function is the LangGraph pattern for conditional
    fan-out — two separate `add_conditional_edges` from the same
    node lead to nondeterministic routing bugs."""
    if state.get("need_enrichment") and state.get("stages"):
        return _enrich_map(state)   # list[Send]
    return "composer"


def _build_graph():
    g: StateGraph = StateGraph(PlanState)
    g.add_node("planner", _planner_node)
    g.add_node("router",  _router_node)
    g.add_node("enrich_stage", _enrich_stage_node)
    g.add_node("composer", _composer_node)
    g.add_node("critic",   _critic_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "router")
    g.add_conditional_edges(
        "router", _route_after_router,
        ["enrich_stage", "composer"],
    )
    g.add_edge("enrich_stage", "composer")
    g.add_edge("composer", "critic")
    g.add_conditional_edges(
        "critic", _after_critic,
        {"planner": "planner", "end": END},
    )
    return g.compile()


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# ---------------------------------------------------------------------------
# SSE bridge — translate graph events → the same wire format the
# native `chat.py` emits, so the web client is unchanged.


def _sse(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


async def _run_graph_sse(req: ChatRequest):
    graph = _get_graph()
    # Extract just the latest user message for the planner. The
    # LangGraph fork is currently single-turn — persistent multi-turn
    # state is a Phase 4b extension (checkpointer).
    user_msg = next(
        (m.content for m in reversed(req.messages) if m.role == "user"),
        "",
    )
    if isinstance(user_msg, list):
        # Anthropic block-content shape; pluck the first text block.
        user_msg = next((b.get("text", "") for b in user_msg
                         if isinstance(b, dict) and b.get("type") == "text"),
                        "")

    state: PlanState = {
        "user_prompt": user_msg,
        "iteration": 0,
        "stream_events": [],
        "enrichments": {},
    }

    # graph.astream(stream_mode="updates") yields per-super-step
    # dicts of {node_name: node_return_delta}. Each node's delta
    # already contains only its OWN new events (nodes return only
    # deltas; the reducer merges into state). So we just drain the
    # events out of every yielded delta.
    try:
        async for step in graph.astream(state, {"recursion_limit": 25}):
            for _node, delta in step.items():
                for ev in (delta.get("stream_events") or []):
                    kind = ev.get("event", "text")
                    payload = {k: v for k, v in ev.items() if k != "event"}
                    yield _sse(kind, payload)
    except Exception as exc:
        yield _sse("error", {"message": f"graph failed: {type(exc).__name__}: {exc}"})
    finally:
        yield _sse("done", {"stop_reason": "end_turn"})


@router.post("/chat_langgraph")
async def chat_langgraph(req: ChatRequest):
    """LangGraph-backed variant of /chat. Same SSE wire format; the
    native backend at /chat is unchanged. Selected by the client (or
    the CHAT_BACKEND env-var switch on the api container) via the
    /chat route in main.py."""
    return StreamingResponse(_run_graph_sse(req),
                             media_type="text/event-stream")
