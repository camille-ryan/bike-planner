"""Structured trace logging for /chat.

Emits one JSON line per event to stdout under the `[chat-trace]`
prefix so it's:
  * easy for `scripts/trace_view.py` to filter by grep
  * captured by docker/k8s stdout log pipes without extra config
  * still parseable in-place when reading `docker logs bike-api`

Event types (`event` field):
  request_start   — one at the top of every /chat call
  round_start     — one per model round
  round_end       — one per model round (with latency + stop_reason)
  tool_call       — one per tool_use → tool_result cycle
  request_end     — one at the bottom of every /chat call

All events carry `ts` (iso), `request_id` (8-char UUID prefix), and
their event-specific fields. Field names are stable — trace_view.py
and any downstream consumer reads these by key.
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any


_PREFIX = "[chat-trace] "


def new_request_id() -> str:
    """Short-form request id — 8 hex chars is plenty for correlation
    inside one process's log stream."""
    return uuid.uuid4().hex[:8]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _emit(event: str, **fields: Any) -> None:
    """Write one JSON-line event to stdout. Never raises — a broken
    trace log must not take the /chat request down with it."""
    try:
        payload = {"ts": _now_iso(), "event": event, **fields}
        sys.stdout.write(_PREFIX + json.dumps(payload) + "\n")
        sys.stdout.flush()
    except Exception:
        # If serialization fails (an unserializable value slipped
        # into a tool input, say), fall back to a plain error line
        # rather than propagating.
        sys.stdout.write(f"{_PREFIX}{{\"event\": \"{event}\", "
                         f"\"error\": \"trace serialize failed\"}}\n")
        sys.stdout.flush()


def _redact(v: Any, max_str: int = 200) -> Any:
    """Trim long strings and drop obvious binary blobs before
    logging. Leaves numbers/bools/None alone; recurses dicts + lists."""
    if isinstance(v, str):
        return v if len(v) <= max_str else v[:max_str] + "…"
    if isinstance(v, dict):
        return {k: _redact(x, max_str) for k, x in v.items()}
    if isinstance(v, list):
        return [_redact(x, max_str) for x in v[:20]]
    return v


def _summarize_output(name: str, output: Any) -> str:
    """One-line output digest, mirrors the shape used by
    `eval/run_eval.py::_summarize_output` so eval traces and
    live-request traces read the same."""
    if isinstance(output, dict) and "error" in output:
        return f"error: {output['error']}"
    try:
        if name == "search_anchors":
            r = output.get("results") or []
            return f"{len(r)} hits"
        if name == "route":
            return (f"{output.get('total_km')} km, "
                    f"{len(output.get('chain_stops') or [])} stops")
        if name == "stations_near":
            s = output.get("stations") or []
            return f"{len(s)} stations"
        if name == "stations_along_route":
            r = output.get("anchors") or []
            return f"{len(r)} rail-served anchors"
        if name == "direct_rail_service":
            return (f"direct={output.get('direct_service')} "
                    f"shared={output.get('n_shared_routes')}")
        if name == "split_into_stages":
            s = output.get("stages") or []
            return f"{len(s)} stages"
        if name in ("pois_near_anchor", "pois_along_route"):
            p = output.get("pois") or []
            return f"{len(p)} POIs"
    except (AttributeError, IndexError, TypeError):
        pass
    return "ok"


class RequestTrace:
    """One /chat request's trace context. Tracks start time + round
    counter so events can be correlated. Use as a context manager:

        with RequestTrace(model=CHAT_MODEL, prompt_head=…) as tr:
            for i in range(rounds):
                tr.round_start(i)
                … stream …
                tr.round_end(i, stop_reason=…, n_text=…, n_tool=…,
                             usage=…)
                for tu in tool_uses:
                    tr.tool_call(round_i=i, name=tu.name,
                                 input=tu.input, output=result,
                                 latency_ms=…)
    """

    def __init__(self, model: str, prompt_head: str,
                 parent_request_id: str | None = None,
                 agent_role: str | None = None) -> None:
        self.request_id = new_request_id()
        self._model = model
        self._prompt_head = prompt_head
        # Parent + role fields for tree tracing across the multi-agent
        # planner: supervisor gets parent=None role="supervisor";
        # each segment sub-agent gets parent=<supervisor's id>
        # role=f"segment[{i}]". Downstream tooling filters by parent to
        # reconstruct the tree. None on both keeps single-agent output
        # byte-identical for backward compat.
        self.parent_request_id = parent_request_id
        self.agent_role = agent_role
        # monotonic, not wall clock — WSL2 wall time can jump on VM
        # resume, producing negative durations otherwise.
        self._t0 = time.monotonic()
        self._stop_reason: str | None = None
        self._error: str | None = None

    def _base_fields(self) -> dict[str, Any]:
        """Fields every event of this trace should carry."""
        out: dict[str, Any] = {"request_id": self.request_id}
        if self.parent_request_id is not None:
            out["parent_request_id"] = self.parent_request_id
        if self.agent_role is not None:
            out["agent_role"] = self.agent_role
        return out

    def __enter__(self) -> "RequestTrace":
        _emit(
            "request_start",
            **self._base_fields(),
            model=self._model,
            prompt_head=_redact(self._prompt_head, max_str=300),
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self._error = f"{type(exc).__name__}: {exc}"
        _emit(
            "request_end",
            **self._base_fields(),
            wall_ms=int((time.monotonic() - self._t0) * 1000),
            stop_reason=self._stop_reason,
            error=self._error,
        )

    def set_stop_reason(self, reason: str | None) -> None:
        self._stop_reason = reason

    def set_error(self, msg: str) -> None:
        self._error = msg

    def round_start(self, round_i: int, messages_len: int) -> None:
        _emit(
            "round_start",
            **self._base_fields(),
            round_i=round_i,
            messages_len=messages_len,
        )

    def round_end(self, round_i: int, stop_reason: str,
                  n_text: int, n_tool: int, latency_ms: int,
                  usage: dict | None) -> None:
        _emit(
            "round_end",
            **self._base_fields(),
            round_i=round_i,
            stop_reason=stop_reason,
            n_text=n_text,
            n_tool=n_tool,
            latency_ms=latency_ms,
            usage=usage,
        )

    def tool_call(self, round_i: int, name: str, input: dict,
                  output: Any, latency_ms: int) -> None:
        _emit(
            "tool_call",
            **self._base_fields(),
            round_i=round_i,
            name=name,
            input_keys=sorted((input or {}).keys()),
            input=_redact(input, max_str=120),
            output_summary=_summarize_output(name, output),
            latency_ms=latency_ms,
            is_error=isinstance(output, dict) and "error" in output,
        )
