"""Run the /chat endpoint against the gold-prompt suite and record
per-run traces.

Emits `analysis/data/traces.jsonl`, one line per (prompt_id × repeat):

    {
      "prompt_id": "graz_to_cph_50d",
      "run": 0,
      "prompt": "...",
      "response": "...",               # concatenated text deltas
      "tool_calls": [                   # in call order
        {"name": "search_anchors", "input": {...}, "output_summary": "..."},
        ...
      ],
      "n_tool_calls": 12,
      "stop_reason": "end_turn",
      "wall_ms": 92345,
      "error": null,
    }

Usage:

    python3 eval/run_eval.py                   # run entire suite
    python3 eval/run_eval.py --only fast       # only the fast tier
    python3 eval/run_eval.py --only flagship
    python3 eval/run_eval.py --prompt-id graz_wien_route

Requires the api container to be up at http://localhost:8001 (default
CHAT_URL). Reads no Anthropic key here — the api container holds it.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import httpx
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = _REPO_ROOT / "analysis" / "data" / "traces.jsonl"
_DEFAULT_PROMPTS = _REPO_ROOT / "eval" / "gold_prompts.yaml"
_DEFAULT_CHAT_URL = os.environ.get("EVAL_CHAT_URL", "http://localhost:8001/chat")


def _load_prompts(path: Path) -> list[dict]:
    """Return one flat list [{tier, id, repeats, prompt, judge_rubric}...]."""
    doc = yaml.safe_load(path.read_text())
    flat = []
    for tier in ("flagship", "fast"):
        for entry in doc.get(tier, []) or []:
            flat.append({
                "tier":         tier,
                "id":           entry["id"],
                "repeats":      int(entry.get("repeats", 1)),
                "prompt":       entry["prompt"].strip(),
                "judge_rubric": entry.get("judge_rubric", []),
            })
    return flat


def _summarize_output(name: str, output) -> str:
    """One-line output digest so traces.jsonl stays small. The judge and
    the dashboard both work off this — full tool payloads are kept only
    in the api's stdout log (not needed for scoring)."""
    if isinstance(output, dict) and "error" in output:
        return f"error: {output['error']}"
    try:
        if name == "search_anchors":
            r = output.get("results") or []
            names = [c.get("name") for c in r[:3]]
            return f"{len(r)} hits: {names}"
        if name == "route":
            return (f"{output.get('total_km')} km, "
                    f"{len(output.get('chain_stops') or [])} chain stops")
        if name == "stations_near":
            s = output.get("stations") or []
            return f"{len(s)} stations, top={s[0].get('name') if s else None}"
        if name == "stations_along_route":
            r = output.get("anchors") or []
            return f"{len(r)} rail-served anchors on route"
        if name == "direct_rail_service":
            return (f"direct={output.get('direct_service')} "
                    f"n_shared={output.get('n_shared_routes')}")
        if name == "split_into_stages":
            s = output.get("stages") or []
            return f"{len(s)} stages"
        if name in ("pois_near_anchor", "pois_along_route"):
            p = output.get("pois") or []
            return f"{len(p)} POIs"
    except (AttributeError, IndexError, TypeError):
        pass
    return "ok"


def _stream_chat(client: httpx.Client, url: str, prompt: str) -> dict:
    """POST prompt to /chat and collect the SSE stream. Returns the
    trace dict without prompt_id or run — the caller adds those."""
    messages = [{"role": "user", "content": prompt}]
    out = {
        "response":     "",
        "tool_calls":   [],
        "stop_reason":  None,
        "error":        None,
    }
    t0 = time.time()
    # The /chat endpoint now defaults to multi-agent for interactive
    # use; keep eval on the single-agent path for baseline
    # comparability. Issue #12 adds per-segment axes + sub-trace
    # packaging for a proper multi-agent eval.
    with client.stream("POST", url,
                       json={"messages": messages, "mode": "single"},
                       timeout=httpx.Timeout(1200.0, connect=30.0)) as r:
        r.raise_for_status()
        event_name: str | None = None
        for raw in r.iter_lines():
            if not raw:
                event_name = None
                continue
            if raw.startswith("event: "):
                event_name = raw[len("event: "):].strip()
                continue
            if raw.startswith("data: "):
                data = json.loads(raw[len("data: "):])
                if event_name == "text":
                    out["response"] += data.get("delta", "")
                elif event_name == "tool_call":
                    out["tool_calls"].append({
                        "name":            data.get("name"),
                        "input":           data.get("input"),
                        "output_summary":  _summarize_output(
                            data.get("name"), data.get("output")),
                    })
                elif event_name == "error":
                    out["error"] = data.get("message")
                elif event_name == "done":
                    out["stop_reason"] = data.get("stop_reason")
    out["wall_ms"] = int((time.time() - t0) * 1000)
    out["n_tool_calls"] = len(out["tool_calls"])
    return out


def _run_one(client: httpx.Client, url: str, prompt: str) -> dict:
    try:
        return _stream_chat(client, url, prompt)
    except (httpx.HTTPError, ValueError) as e:
        return {
            "response":     "",
            "tool_calls":   [],
            "stop_reason":  "transport_error",
            "error":        f"{type(e).__name__}: {e}",
            "wall_ms":      0,
            "n_tool_calls": 0,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, default=_DEFAULT_PROMPTS)
    ap.add_argument("--out",     type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--chat-url",           default=_DEFAULT_CHAT_URL)
    ap.add_argument("--only", choices=["flagship", "fast"],
                    help="Restrict to one tier.")
    ap.add_argument("--prompt-id", default=None,
                    help="Restrict to a single prompt id (across all tiers).")
    ap.add_argument("--append", action="store_true",
                    help="Append to the traces file instead of overwriting.")
    args = ap.parse_args()

    prompts = _load_prompts(args.prompts)
    if args.only:
        prompts = [p for p in prompts if p["tier"] == args.only]
    if args.prompt_id:
        prompts = [p for p in prompts if p["id"] == args.prompt_id]
    if not prompts:
        print("no prompts matched filters", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    n_runs = sum(p["repeats"] for p in prompts)
    print(f"[eval] {len(prompts)} prompt(s), "
          f"{n_runs} total runs → {args.out} "
          f"(chat={args.chat_url})")

    with contextlib.closing(httpx.Client()) as client, args.out.open(mode) as fout:
        run_i = 0
        for p in prompts:
            for rep in range(p["repeats"]):
                run_i += 1
                print(f"[eval] {run_i}/{n_runs}  "
                      f"{p['tier']}/{p['id']}  rep={rep+1}/{p['repeats']}")
                t0 = time.time()
                res = _run_one(client, args.chat_url, p["prompt"])
                trace = {
                    "prompt_id":   p["id"],
                    "tier":        p["tier"],
                    "run":         rep,
                    "prompt":      p["prompt"],
                    **res,
                }
                fout.write(json.dumps(trace) + "\n")
                fout.flush()
                dt = time.time() - t0
                mark = "OK" if not res["error"] else "ERR"
                print(f"[eval]   {mark}  {dt:.1f}s  "
                      f"tools={res['n_tool_calls']}  "
                      f"stop={res['stop_reason']}  "
                      f"error={res['error']}")
    print(f"[eval] done → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
