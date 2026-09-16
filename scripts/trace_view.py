#!/usr/bin/env python3
"""Pretty-print one /chat request's structured trace timeline.

Reads `[chat-trace] {...}` JSON-line events from stdin (or a file)
and renders a human-readable timeline. Filters by request_id — pass
`--request-id <id>` to select one, or `--last` to auto-pick the most
recent request in the input.

Typical usage:

    docker logs bike-api | python3 scripts/trace_view.py --last
    docker logs bike-api 2>&1 | python3 scripts/trace_view.py \\
        --request-id a3f7b1c2

Output shape:

    [a3f7b1c2] START  model=claude-sonnet-5
        "Plan a bike tour from Graz to Copenhagen…"
      round 0  messages=1
      round 0  end  stop=tool_use  txt=0 tools=2  1420ms
        └─ search_anchors({query}) → 1 hits         45ms
        └─ search_anchors({query}) → 1 hits         38ms
      round 1  messages=3
      round 1  end  stop=tool_use  txt=0 tools=1  1120ms
        └─ route({from_ref,to_ref}) → 215.5 km, 10 stops  240ms
      …
    [a3f7b1c2] END  stop=end_turn  wall=12340ms  error=None
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


_PREFIX = "[chat-trace] "


def parse_events(fh) -> list[dict]:
    """Read all events from `fh`, ignoring non-trace lines."""
    out = []
    for line in fh:
        line = line.rstrip("\n")
        i = line.find(_PREFIX)
        if i < 0:
            continue
        try:
            out.append(json.loads(line[i + len(_PREFIX):]))
        except json.JSONDecodeError:
            continue
    return out


def format_input_shape(name: str, inp: dict | None) -> str:
    """Signature-style summary of a tool call's input."""
    if not inp:
        return "()"
    return "(" + ",".join(sorted(inp.keys())) + ")"


def render_request(events: list[dict], request_id: str) -> str:
    """Render one request's timeline as a multi-line string."""
    lines: list[str] = []
    by_round: dict[int, list[dict]] = defaultdict(list)
    start = None
    end = None
    round_events: dict[tuple[int, str], dict] = {}
    tools_by_round: dict[int, list[dict]] = defaultdict(list)

    for e in events:
        if e.get("request_id") != request_id:
            continue
        et = e.get("event")
        if et == "request_start":
            start = e
        elif et == "request_end":
            end = e
        elif et == "round_start":
            round_events[(e["round_i"], "start")] = e
        elif et == "round_end":
            round_events[(e["round_i"], "end")] = e
        elif et == "tool_call":
            tools_by_round[e["round_i"]].append(e)

    if start is None:
        return f"(no request_start seen for {request_id})"

    head = start.get("prompt_head") or ""
    lines.append(f"[{request_id}] START  model={start.get('model')}")
    if head:
        # Show prompt head indented, wrapped at 80 chars
        wrapped = head[:200]
        lines.append(f'    "{wrapped}"')

    rounds = sorted({k[0] for k in round_events})
    total_tool_ms = 0
    total_round_ms = 0
    tool_counts: dict[str, int] = defaultdict(int)
    for r in rounds:
        rs = round_events.get((r, "start"))
        re = round_events.get((r, "end"))
        if rs is not None:
            lines.append(f"  round {r}  messages={rs.get('messages_len')}")
        if re is not None:
            usage = re.get("usage") or {}
            usage_str = ""
            if usage:
                usage_str = (f"  tok(in={usage.get('input_tokens')},"
                             f"out={usage.get('output_tokens')})")
            lines.append(
                f"  round {r}  end  stop={re.get('stop_reason')}  "
                f"txt={re.get('n_text')} tools={re.get('n_tool')}  "
                f"{re.get('latency_ms')}ms{usage_str}"
            )
            total_round_ms += re.get("latency_ms") or 0
        for tc in tools_by_round.get(r, []):
            sig = format_input_shape(tc.get("name"), tc.get("input"))
            marker = "!!" if tc.get("is_error") else "└─"
            lines.append(
                f"    {marker} {tc.get('name')}{sig} → "
                f"{tc.get('output_summary')}  {tc.get('latency_ms')}ms"
            )
            total_tool_ms += tc.get("latency_ms") or 0
            tool_counts[tc.get("name")] += 1

    if end is not None:
        wall = end.get("wall_ms")
        lines.append(
            f"[{request_id}] END  stop={end.get('stop_reason')}  "
            f"wall={wall}ms  error={end.get('error')}"
        )
    else:
        lines.append(f"[{request_id}] END  (no request_end seen — request still in flight?)")

    if tool_counts:
        breakdown = ", ".join(f"{n}={c}" for n, c in
                              sorted(tool_counts.items(),
                                     key=lambda kv: -kv[1]))
        lines.append(
            f"    tool budget: {sum(tool_counts.values())} calls "
            f"({breakdown})  model={total_round_ms}ms  "
            f"tools={total_tool_ms}ms"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request-id",
                    help="Render just this request. Omit + use --last "
                         "to auto-pick the most recent one.")
    ap.add_argument("--last", action="store_true",
                    help="Render the most recent request in the input.")
    ap.add_argument("--list", action="store_true",
                    help="List all request_ids in the input and exit.")
    ap.add_argument("input", nargs="?",
                    help="File to read; defaults to stdin.")
    args = ap.parse_args()

    fh = open(args.input) if args.input else sys.stdin
    events = parse_events(fh)
    if not events:
        print("no trace events found (looked for '[chat-trace] '-prefixed "
              "JSON lines on stdin)", file=sys.stderr)
        return 2

    all_ids = []
    seen: set[str] = set()
    for e in events:
        rid = e.get("request_id")
        if rid and rid not in seen:
            seen.add(rid); all_ids.append(rid)

    if args.list:
        for rid in all_ids:
            starts = [e for e in events
                      if e.get("event") == "request_start"
                      and e.get("request_id") == rid]
            head = (starts[0].get("prompt_head") if starts else "")[:60]
            print(f"{rid}  {head}")
        return 0

    if args.request_id:
        target = args.request_id
    elif args.last:
        if not all_ids:
            print("no requests found", file=sys.stderr)
            return 2
        target = all_ids[-1]
    else:
        print("pass --request-id <id>, --last, or --list", file=sys.stderr)
        return 2

    print(render_request(events, target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
