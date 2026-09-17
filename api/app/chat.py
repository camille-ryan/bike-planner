"""Chat-driven planner: /chat SSE endpoint backed by Claude + tool use.

The backend is stateless — the browser sends the full message history on
every turn. That keeps the API simple to reason about; it also means
switching sessions / editing history is a purely client-side concern.

Tool implementations live in `api.app.tools` — that module is shared
with the MCP server (`mcp_server.mcp_bike_planner`), so both the web
chat and Claude Desktop see the exact same tool set.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Iterator

import anthropic
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from .tools import TOOLS, TOOL_IMPLS
from .tracing import RequestTrace


CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-sonnet-5")
CHAT_MAX_TOOL_ROUNDS = int(os.environ.get("CHAT_MAX_TOOL_ROUNDS", "30"))

router = APIRouter()

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise HTTPException(500, "ANTHROPIC_API_KEY not set in the API container's env")
        # Per-request timeout so a stalled upstream can't hang the SSE
        # loop indefinitely. 120s covers even long extended-thinking
        # rounds; anything longer than that is a genuine failure the
        # client should see quickly.
        _client = anthropic.Anthropic(api_key=key, timeout=240.0)
    return _client


def _strip_bulk_for_llm(name: str, result: dict) -> dict:
    """Return a compact form of the tool result for the LLM's context.
    The frontend still receives the full result via the SSE frame."""
    if not isinstance(result, dict) or "error" in result:
        return result
    if name == "route":
        return {
            "total_km":    result.get("total_km"),
            "chain_stops": result.get("chain_stops", []),
            "n_bridges":   result.get("n_bridges"),
            "polyline_verts": len(result.get("polyline") or []),
        }
    if name == "split_into_stages":
        # Each stage now carries a `polyline` (its per-day route)
        # for the frontend to render. Strip it before the LLM sees
        # the stages — the LLM plans off `km` alone.
        return {
            "total_km": result.get("total_km"),
            "n_days":   result.get("n_days"),
            "stages": [
                {k: v for k, v in st.items() if k != "polyline"}
                for st in result.get("stages") or []
            ],
        }
    return result


# ---------------------------------------------------------------------------
# SSE endpoint.

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: Any  # string or list[block]


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


SYSTEM_PROMPT = """You are a bike-touring co-planner for a route spanning Austria, Czechia, Germany, and Denmark. The user is planning a Graz → Copenhagen tour (they can also plan sub-trips within that corridor).

Facts about the routing profile:
- Anchor references look like `db:1234` (city from population db) or `ferry:12345` (ferry pier). Prefer resolving names via `search_anchors` before calling `route`.
- The routing engine is bike-optimized (avoids highways, prefers bike lanes / dedicated paths).
- Ferry crossings are handled implicitly by the router — you don't need to plan them explicitly.
- Rail stations along the corridor matter because the user's partner meets them daily by train.
- Anchor names come from OSM's local-language `name` tag, so Copenhagen appears as "København", Vienna as "Wien", Prague as "Praha", Munich as "München", Cologne as "Köln", etc. `search_anchors` auto-tries a handful of English↔local pairs; if a search returns zero results, try the local-language name explicitly.

Workflow — do exactly what the user asked, no more. DO NOT stop mid-workflow for confirmation between steps that ARE needed:

- Always resolve every named place with `search_anchors` (English + local variants are auto-tried) before any other tool.
- Always call `route` between the resolved anchors if the user asked for a route.
- `route` is FAST by default (~50 ms, chain-graph + straight lines between anchors). That's the mode for exploration — cheap enough to try many candidate corridors. Pass `precise: true` ONLY on the final route you're handing to the user (adds ~1-3 s for real pathfinding + first/last-mile stitching). Do not sprinkle `precise: true` on intermediate calls.
- **Corridor pipeline is `route → stations → RE-ROUTE → split` — the "re-route" step is not optional.** After `stations_along_route` returns, YOU MUST:
    1. Look at the returned anchors and decide which cover the corridor at reasonable spacing (~1 day apart) AND lie on the same rail line.
    2. If the returned rail-served anchors are NOT on the current route's polyline (i.e. the initial route missed the rail spine — very common when the direct bike route diverges from the rail line, like Wien→Praha direct vs Wien→Brno→Praha rail), call `route` AGAIN with those anchors as `via_refs` before calling `split_into_stages`.
    3. Verify with `direct_rail_service_batch` on the CONSECUTIVE-PAIR chain of chosen overnights.
    4. THEN call `split_into_stages` with the revised via_refs.
  Do NOT skip step 2 and go straight from `stations_along_route` to `split_into_stages`. Split-first freezes the corridor at whatever the initial route picked, which is usually the bike-shortest path (not the rail-following one), stranding overnights off the rail line. If you find yourself splitting immediately after stations without a re-route in between, you're doing it wrong.
- Call `split_into_stages` ONLY if the user asked for a multi-day plan, daily stages, km/day, or overnights — cues like "plan a X-day tour", "80 km/day", "break into stages". Do NOT split just because a route is long.
- **`split_into_stages` already returns per-stage km, from/to anchor names+refs, and full polylines.** After it returns, narrate the days directly from that result. DO NOT call `route` on each consecutive stage pair to "get the km" — you already have it. DO NOT call `route` and `split_into_stages` on the same segment; pick one. Only re-`route` a stage if the user explicitly asks for an alternate routing on that specific stage.
- For a multi-leg tour (Graz→Wien→Praha→Berlin→Hamburg→CPH), one `split_into_stages` call covers the whole thing when you pass the intermediate hubs as `via_refs`. Don't call `split_into_stages` per-leg AND once for the whole trip — the whole-trip call is authoritative.
- Budget: your tool loop is capped. A 50-day plan burns budget fast if you route each day individually. Prefer one whole-trip `split_into_stages` + one `direct_rail_service_batch` + one `stations_along_route` (if rail matters). Leave ≥30% of the budget for the final narrative response.
- **Phase-stream your writeup, don't dump it at the end.** For a multi-hub plan (Graz→Wien→Praha→…), the user's UI renders your text tokens in real time. Write each phase's narrative in the SAME round that its tool results come back — heading, table, rail-check summary — THEN if you still need tool calls for later phases, emit them in the same response. The model API happily combines text output + tool_use in one round. Do NOT wait until every last tool call has returned to start writing. A plan that streams phase-by-phase feels dramatically faster than one that appears all at once at minute 11.
- **Before EACH batch of tool calls, write one short sentence naming what you're about to do and why.** The user sees a live stream of your text and tool bubbles; without a rationale, they see `🔧 stations_along_route`, `🔧 direct_rail_service_batch`, `🔧 route`, `🔧 split_into_stages` and have no idea what you're thinking. A one-liner like *"Checking rail-served anchors along Wien→Praha before I pick overnights…"* or *"No direct Wien↔Praha train — re-routing via Brno since that's the actual rail spine."* makes the trace legible. Keep it to a single sentence per batch; the phase-writeup is separate.
- Call `stations_near` ONLY if the user asked about rail, train, meeting the partner, or station-accessible overnights.
- When you need rail-accessible overnights across multiple candidate towns along a route, PREFER `stations_along_route` (one call, returns the ranked corridor) over N × `stations_near` calls.
- **For rail-constrained plans, call `stations_along_route` BEFORE `split_into_stages`, not after.** The natural pipeline is: (a) `stations_along_route` returns every rail-served anchor along the corridor with its `km_along_route` — pick anchors ~one day's ride apart (from the km column), (b) pass those refs as `via_refs` to `split_into_stages` so overnights land at real rail-served cities by construction, (c) `direct_rail_service_batch` verifies direct-train connectivity in one round. Doing split-first and stations-after strands you with overnights in villages that have no station and forces re-planning.
- **One `stations_along_route` call for the whole corridor, not one per segment.** Pass the full hub sequence in `via_refs` (e.g. `from_ref=Graz, to_ref=CPH, via_refs=[Wien, Brno, Praha, Dresden, Berlin, Hamburg]`) and you get every rail-served anchor for the whole trip in a single tool round. Splitting it into per-segment calls duplicates the same profile-load + polyline projection work per call.
- **"Detour to nearby major cities" = ROUTE VIA that city, not visit it by separate train.** When the user asks to "detour to X and spend 2-3 days" or "include X on the way," pass X's ref in `via_refs` so the bike corridor goes through X. Two consequences: (a) the shortest bike route between two hubs (A, B) may not be the best corridor; if the direct rail line A↔B routes via intermediate hub C (Wien↔Praha via Brno; Praha↔Berlin via Dresden), pass C as a `via_ref` so bike and rail stay aligned — otherwise overnights between A and B are on the wrong side of the rail spine and won't have direct trains. (b) A pure "sightsee by train" side trip is only appropriate when the user explicitly says so (e.g. "day trip to X"), not for a detour they want to spend days at.
- If the user's prompt uses the words "direct train", "non-transfer", "one-seat", or "single change" (or asks that overnights be reachable by a direct train from a specific place / hub), you MUST verify direct rail service. `n_routes_rail` alone does not prove direct service — a station with 20 routes may still require a transfer to reach the hub the user cares about. Choose the tool by count:
    - 1–2 pairs to check → call `direct_rail_service` per pair.
    - **3 or more pairs → call `direct_rail_service_batch` ONCE** with all pairs. It replaces N model rounds with 1, leaving budget for the actual day-by-day narrative.
- **Rail-verification scope — CHAIN, not hub.** The user's partner rides trains from yesterday's overnight to today's overnight (no transfer, meets the cyclist each evening). So the constraint is that every CONSECUTIVE PAIR of base overnights has direct rail service between them, NOT that every overnight has a direct train to Graz / Copenhagen / a hub. In practice this means base overnights sit on the SAME regional rail line — a random village that has direct trains to Berlin but no direct train to the village 60 km further doesn't work, because the partner can't get between them without changing trains. Verify with `direct_rail_service_batch` on the CONSECUTIVE-PAIR list `[(day1_end, day2_end), (day2_end, day3_end), …, (dayN-1_end, dayN_end)]`. Interest detours (2-3 day sightseeing stops the partner doesn't follow for) can transfer — only require direct rail between the overnights ADJACENT to a detour (where the cyclist rejoins the partner-ridden corridor).
- Then write the final summary. Keep it proportional to what was asked — a single route gets one bullet with total km and chain waypoints, not a day-by-day breakdown.
- NEVER cite a specific numeric fact (station route count, distance, elevation, ferry name, POI subtype count) unless it appeared verbatim in a tool_result THIS TURN. If you want to describe a coverage or connection qualitatively ("well-served by trains", "on the DB corridor"), do so without a number rather than making one up.

Reformat / recall requests — DO NOT re-run tools:
- When the user asks to rephrase, reformat, summarize, tabulate, "make it prettier", "give me markdown", "show as a list", "just the overnights", "recap", or any variant that references content ALREADY produced in this conversation, work directly from the prior tool_result blocks and assistant messages in your context.
- Only call tools again if the user CHANGED a parameter (different endpoints, different km/day, added a via, "route via X" that wasn't there before, "add a stop", "shorten the days"). Anything that could produce a genuinely different route requires re-routing; anything that's pure presentation must NOT.
- If unsure whether a request is presentation-only vs. parameter-change, err on the side of NOT re-calling tools — reformat from context and add a one-sentence "let me know if you want me to re-route with different parameters."

Style:
- Be concise. No hedging preamble like "I'll help you plan…" — just start doing the work.
- If a tool errors, note briefly and try one alternative (e.g. local-language spelling) before giving up.
- Never invent anchors or coordinates. Every place-name is verified via `search_anchors` first.
- **Talk to the user, not to the tools.** Never leak tool-schema names into user-facing text — no "via_refs", "from_ref", "precise mode", "direct_rail_service", "chain-graph", "polyline", or any other JSON field name. Translate:
    - "via_refs" → "waypoints" or just say "route via X, Y, Z"
    - "direct_rail_service" → "direct-train check" or plain English ("Berlin ↔ Hamburg has a direct train")
    - "split_into_stages" → "the daily-stage plan" or just describe the days
    - "search_anchors" → don't mention it, just use the city name
  Internal reasoning is fine ("I need to check…"), but reference user concepts (cities, days, trains, waypoints), not implementation."""


def _sse(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


def _msg_to_api(m: ChatMessage) -> dict:
    return {"role": m.role, "content": m.content}


def _friendly_error(exc: Exception) -> str:
    """Translate SDK / network / tool errors into a one-line message the
    user can read without seeing tracebacks or provider internals."""
    name = type(exc).__name__
    if isinstance(exc, anthropic.RateLimitError):
        return "I'm hitting the API rate limit — wait a moment and try again."
    if isinstance(exc, anthropic.AuthenticationError):
        return "The Anthropic API key isn't accepted. Check api/.env."
    if isinstance(exc, anthropic.BadRequestError):
        print(f"[chat] BadRequestError: {exc}", flush=True)
        return "The model rejected the request — I've logged it. Try rephrasing, or hit Reset."
    if isinstance(exc, anthropic.APIStatusError):
        print(f"[chat] APIStatusError {exc.status_code}: {exc}", flush=True)
        return f"The model API returned {exc.status_code}. Try again in a moment."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Couldn't reach the model API — network issue. Try again in a moment."
    print(f"[chat] {name}: {exc}", flush=True)
    return f"Something broke internally ({name}). Try again, or hit Reset if it keeps failing."


def _prompt_head(messages: list[dict]) -> str:
    """First-user-message summary for trace correlation."""
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        return b.get("text", "")
    return ""


def _run_chat(req: ChatRequest, client: anthropic.Anthropic) -> Iterator[bytes]:
    messages: list[dict] = [_msg_to_api(m) for m in req.messages]

    with RequestTrace(model=CHAT_MODEL,
                      prompt_head=_prompt_head(messages)) as tr:
        try:
            yield from _run_chat_inner(client, messages, tr)
        except Exception as exc:
            # Never let a mid-stream exception crash the SSE with no
            # signal to the client — emit a clean `event: error` and
            # close cleanly.
            tr.set_error(f"{type(exc).__name__}: {exc}")
            yield _sse("error", {"message": _friendly_error(exc)})


def _cached_system() -> list[dict]:
    """System prompt as a single cacheable block. Anthropic caches
    the prefix up to and including the cache_control breakpoint;
    since SYSTEM_PROMPT is stable, marking it once means every round
    after the first pays a fraction of the input-token cost and
    lands the tokens faster."""
    return [{
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }]


def _cached_tools() -> list[dict]:
    """Tool schemas with a cache breakpoint at the last tool — that
    caches the entire tool list. ~2.6k input tokens re-sent every
    round; caching them collapses that."""
    out = [dict(t) for t in TOOLS]
    out[-1] = {**out[-1], "cache_control": {"type": "ephemeral"}}
    return out


def _shift_message_cache_breakpoint(messages: list[dict]) -> None:
    """Anthropic allows at most 4 cache_control breakpoints per
    request. We already spend 2 on system + tools. The remaining
    budget goes to the growing conversation: mark the LAST content
    block of the latest message as a breakpoint, and clear any
    breakpoint on earlier messages so we don't blow the limit.

    Effect: every round after the first, the whole prior
    conversation is a cache hit — only the new turn is fresh input.
    Since our messages grow by (assistant_turn + user_tool_results)
    each round and earlier turns never mutate, the cached prefix
    matches on each subsequent round.

    Handles both string content (first user message) and block
    content (all subsequent assistant/tool_results turns).
    """
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    del block["cache_control"]
    if not messages:
        return
    last = messages[-1]
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [{
            "type": "text",
            "text": content,
            "cache_control": {"type": "ephemeral"},
        }]
    elif isinstance(content, list) and content:
        block = content[-1]
        if isinstance(block, dict):
            block["cache_control"] = {"type": "ephemeral"}


def _run_chat_inner(client: anthropic.Anthropic, messages: list[dict],
                    tr: RequestTrace) -> Iterator[bytes]:
    system_blocks = _cached_system()
    tools_cached = _cached_tools()
    for round_i in range(CHAT_MAX_TOOL_ROUNDS):
        tr.round_start(round_i, messages_len=len(messages))
        _shift_message_cache_breakpoint(messages)
        # `round_start` marks a new LLM round to the client. Two jobs:
        # (1) heartbeat — for a big cached-context request TTFT can be
        # several seconds and no SDK event fires until then, so this
        # keeps the browser's idle timer from tripping; (2) tells the
        # frontend to start a fresh assistant bubble so text and tool
        # rows from consecutive rounds interleave visually instead of
        # piling into one giant paragraph followed by every tool call.
        yield _sse("round_start", {"round_i": round_i})
        # `time.monotonic()`, not `time.time()` — WSL2's wall clock
        # can jump backward on VM resume, which produced negative
        # latency_ms values on the first Phase 6 smoke run.
        t_round = time.monotonic()
        with client.messages.stream(
            model=CHAT_MODEL,
            max_tokens=16384,
            system=system_blocks,
            tools=tools_cached,
            messages=messages,
        ) as stream:
            for event in stream:
                if event.type == "text":
                    yield _sse("text", {"delta": event.text})
                else:
                    # Non-text SDK events (content_block_start,
                    # input_json deltas for tool_use, block_stop,
                    # message_delta, etc.) don't carry a payload the
                    # client renders — but they DO signal the server
                    # is still alive. Emit an SSE comment line as a
                    # heartbeat so the browser's stream-idle timer
                    # doesn't trip during long silent phases (e.g.
                    # tool_use input generation, extended thinking).
                    yield b": tick\n\n"
            final = stream.get_final_message()

        n_text = sum(1 for b in final.content if b.type == "text")
        n_tool = sum(1 for b in final.content if b.type == "tool_use")
        round_ms = int((time.monotonic() - t_round) * 1000)
        usage = getattr(final, "usage", None)
        usage_dict = None
        if usage is not None:
            usage_dict = {
                "input_tokens":  getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
                "cache_read_input_tokens":     getattr(usage, "cache_read_input_tokens",     None),
            }
        tr.round_end(
            round_i=round_i,
            stop_reason=str(final.stop_reason),
            n_text=n_text,
            n_tool=n_tool,
            latency_ms=round_ms,
            usage=usage_dict,
        )
        print(f"[chat] {tr.request_id} round {round_i+1}/{CHAT_MAX_TOOL_ROUNDS}"
              f" stop={final.stop_reason} text={n_text} tools={n_tool}"
              f" {round_ms}ms",
              flush=True)

        # Thinking blocks are stripped: Sonnet 5's extended reasoning
        # emits them but their model_dump() shape isn't valid as
        # message-input on the next turn. Losing them costs cross-turn
        # reasoning coherence but not correctness — Claude re-reasons
        # on the next round if needed.
        assistant_content = [
            block.model_dump() for block in final.content
            if block.type != "thinking"
        ]
        messages.append({"role": "assistant", "content": assistant_content})

        if final.stop_reason != "tool_use":
            if final.stop_reason == "max_tokens" and n_text == 0:
                yield _sse("error", {"message": (
                    "The model hit its per-turn max_tokens without "
                    "finishing the answer. Raise max_tokens in chat.py "
                    "or ask for a smaller scope."
                )})
            tr.set_stop_reason(str(final.stop_reason))
            yield _sse("done", {"stop_reason": final.stop_reason})
            return

        tool_uses = [b for b in final.content if b.type == "tool_use"]
        tool_results = []
        for tu in tool_uses:
            # Announce the tool start BEFORE running it, so the client
            # can show a "🔧 <name> (running…)" indicator and its idle
            # timer resets. A tool that takes 20-40s (split_into_stages
            # on a big trip, stations_along_route that internally
            # recomputes a route) would otherwise be a silent gap.
            yield _sse("tool_start", {
                "id":    tu.id,
                "name":  tu.name,
                "input": tu.input,
            })
            t_tool = time.monotonic()
            impl = TOOL_IMPLS.get(tu.name)
            if impl is None:
                result = {"error": f"unknown tool: {tu.name}"}
            else:
                try:
                    result = impl(tu.input)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}
            tool_ms = int((time.monotonic() - t_tool) * 1000)
            tr.tool_call(round_i=round_i, name=tu.name,
                         input=dict(tu.input) if tu.input else {},
                         output=result, latency_ms=tool_ms)
            summary_result = _strip_bulk_for_llm(tu.name, result)
            print(f"[chat.tool] {tr.request_id} {tu.name}({tu.input}) "
                  f"→ {summary_result} {tool_ms}ms",
                  flush=True)
            # Stream the FULL tool call + result to the client so the
            # frontend can redraw the map; the LLM only sees the
            # summarized version to keep its context small.
            yield _sse("tool_call", {
                "id": tu.id, "name": tu.name, "input": tu.input, "output": result,
            })
            llm_result = _strip_bulk_for_llm(tu.name, result)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(llm_result),
            })
        messages.append({"role": "user", "content": tool_results})

    tr.set_stop_reason("max_rounds")
    print(f"[chat] {tr.request_id} hit CHAT_MAX_TOOL_ROUNDS="
          f"{CHAT_MAX_TOOL_ROUNDS} without a final text response",
          flush=True)
    yield _sse("error", {"message": (
        f"Ran out of tool-loop rounds ({CHAT_MAX_TOOL_ROUNDS}) before "
        f"finishing the plan. Try a narrower prompt, or raise "
        f"CHAT_MAX_TOOL_ROUNDS in api/.env and restart the api."
    )})
    yield _sse("done", {"stop_reason": "max_rounds"})


@router.post("/chat")
def chat(req: ChatRequest):
    # Resolve the API key BEFORE starting the streaming response — an
    # HTTPException raised mid-stream would just close the socket with no
    # visible error to the client. Fail cleanly here instead.
    client = _get_client()
    return StreamingResponse(_run_chat(req, client), media_type="text/event-stream")
