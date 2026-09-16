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
from typing import Any, Iterator

import anthropic
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from .tools import TOOLS, TOOL_IMPLS


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
        _client = anthropic.Anthropic(api_key=key, timeout=120.0)
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
        # Stages themselves are small (a dozen items × 6 fields) — keep.
        return result
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
- Call `split_into_stages` ONLY if the user asked for a multi-day plan, daily stages, km/day, or overnights — cues like "plan a X-day tour", "80 km/day", "break into stages". Do NOT split just because a route is long.
- Call `stations_near` ONLY if the user asked about rail, train, meeting the partner, or station-accessible overnights.
- When you need rail-accessible overnights across multiple candidate towns along a route, PREFER `stations_along_route` (one call, returns the ranked corridor) over N × `stations_near` calls.
- Then write the final summary. Keep it proportional to what was asked — a single route gets one bullet with total km and chain waypoints, not a day-by-day breakdown.

Reformat / recall requests — DO NOT re-run tools:
- When the user asks to rephrase, reformat, summarize, tabulate, "make it prettier", "give me markdown", "show as a list", "just the overnights", "recap", or any variant that references content ALREADY produced in this conversation, work directly from the prior tool_result blocks and assistant messages in your context.
- Only call tools again if the user CHANGED a parameter (different endpoints, different km/day, added a via, "route via X" that wasn't there before, "add a stop", "shorten the days"). Anything that could produce a genuinely different route requires re-routing; anything that's pure presentation must NOT.
- If unsure whether a request is presentation-only vs. parameter-change, err on the side of NOT re-calling tools — reformat from context and add a one-sentence "let me know if you want me to re-route with different parameters."

Style:
- Be concise. No hedging preamble like "I'll help you plan…" — just start doing the work.
- If a tool errors, note briefly and try one alternative (e.g. local-language spelling) before giving up.
- Never invent anchors or coordinates. Every place-name is verified via `search_anchors` first."""


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


def _run_chat(req: ChatRequest, client: anthropic.Anthropic) -> Iterator[bytes]:
    messages: list[dict] = [_msg_to_api(m) for m in req.messages]

    try:
        yield from _run_chat_inner(client, messages)
    except Exception as exc:
        # Never let a mid-stream exception crash the SSE with no signal
        # to the client — emit a clean `event: error` and close cleanly.
        yield _sse("error", {"message": _friendly_error(exc)})


def _run_chat_inner(client: anthropic.Anthropic, messages: list[dict]) -> Iterator[bytes]:
    for round_i in range(CHAT_MAX_TOOL_ROUNDS):
        with client.messages.stream(
            model=CHAT_MODEL,
            max_tokens=16384,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            for event in stream:
                if event.type == "text":
                    yield _sse("text", {"delta": event.text})
                elif event.type == "input_json":
                    pass
            final = stream.get_final_message()

        n_text = sum(1 for b in final.content if b.type == "text")
        n_tool = sum(1 for b in final.content if b.type == "tool_use")
        print(f"[chat] round {round_i+1}/{CHAT_MAX_TOOL_ROUNDS}  "
              f"stop_reason={final.stop_reason}  "
              f"text_blocks={n_text}  tool_uses={n_tool}",
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
            yield _sse("done", {"stop_reason": final.stop_reason})
            return

        tool_uses = [b for b in final.content if b.type == "tool_use"]
        tool_results = []
        for tu in tool_uses:
            impl = TOOL_IMPLS.get(tu.name)
            if impl is None:
                result = {"error": f"unknown tool: {tu.name}"}
            else:
                try:
                    result = impl(tu.input)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}
            summary_result = _strip_bulk_for_llm(tu.name, result)
            print(f"[chat.tool] {tu.name}({tu.input}) → {summary_result}",
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

    print(f"[chat] hit CHAT_MAX_TOOL_ROUNDS={CHAT_MAX_TOOL_ROUNDS} "
          f"without a final text response", flush=True)
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
