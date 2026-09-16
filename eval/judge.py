"""LLM-as-judge: score each trace against its prompt's rubric.

Reads `analysis/data/traces.jsonl`; writes `analysis/data/scores.jsonl`.

Uses Sonnet (via anthropic messages.stream with `tools=[]` and forced
structured output) — one call per (trace × rubric_axis). A score is an
integer 0–5 with a one-line rationale. The judge sees:
  * the user prompt
  * the assistant's final response text (concatenated deltas)
  * a compact tool-call summary (name + one-line output digest)
  * the single rubric axis being scored

The judge NEVER sees the whole tool payload or the system prompt — it
must reason about correctness from what the user would see plus what
the tools reported at the boundary.

Usage:

    python3 eval/judge.py                  # score every trace
    python3 eval/judge.py --model claude-sonnet-5

Reads ANTHROPIC_API_KEY from api/.env if present, else the environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TRACES = _REPO_ROOT / "analysis" / "data" / "traces.jsonl"
_DEFAULT_SCORES = _REPO_ROOT / "analysis" / "data" / "scores.jsonl"
_DEFAULT_PROMPTS = _REPO_ROOT / "eval" / "gold_prompts.yaml"
_DEFAULT_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "claude-sonnet-5")

_JUDGE_TOOL = {
    "name": "record_score",
    "description": (
        "Record the numeric score and a one-line rationale for the "
        "given rubric axis. Score is an integer 0–5 where 0=absent "
        "or wrong and 5=perfect."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 0, "maximum": 5},
            "rationale": {"type": "string",
                          "description": "One sentence justifying the score."},
        },
        "required": ["score", "rationale"],
    },
}


_JUDGE_SYSTEM = """You are grading a bike-tour planning assistant. You will
be given: a user prompt, the assistant's final text answer, a compact
tool-call summary, and ONE rubric axis to score.

Return exactly one `record_score` tool call. The score is an integer
0–5:
  0 = the axis is entirely absent, or the assistant's answer is wrong
      on this axis
  1 = minimal / obviously flawed
  2 = partial
  3 = adequate
  4 = strong, with a minor gap
  5 = complete and precise

Be strict but fair. If the tool_calls list is empty and the axis is
about grounding, that's usually a 0 or 1 (the answer must be
hallucinated). If the axis rewards NOT calling tools (e.g.
"no_tool_replay"), an empty tool list is the 5.

Never give a 5 for hedging. "The plan may need adjustment" is a 3, not
a 5.
"""


def _load_env_key() -> None:
    """Load ANTHROPIC_API_KEY from api/.env if it isn't in the
    environment already."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env_path = _REPO_ROOT / "api" / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip("\"'")
            return


def _prompt_index(prompts_path: Path) -> dict[str, list[dict]]:
    """{prompt_id: [rubric_axis, ...]}"""
    doc = yaml.safe_load(prompts_path.read_text())
    idx: dict[str, list[dict]] = {}
    for tier in ("flagship", "fast"):
        for entry in doc.get(tier, []) or []:
            idx[entry["id"]] = entry.get("judge_rubric", []) or []
    return idx


def _tool_summary(trace: dict) -> str:
    if not trace.get("tool_calls"):
        return "(no tool calls)"
    lines = []
    for tc in trace["tool_calls"]:
        args_shape = ", ".join(sorted((tc.get("input") or {}).keys()))
        lines.append(f"- {tc['name']}({args_shape}) → {tc['output_summary']}")
    return "\n".join(lines)


def _score_one(client: anthropic.Anthropic, model: str,
               trace: dict, axis: dict) -> dict:
    """One judge call. Returns {'axis': ..., 'score': int, 'rationale': str,
    'raw': str} — raw is the model's final message text (usually empty)."""
    user_msg = (
        f"USER PROMPT:\n{trace['prompt']}\n\n"
        f"ASSISTANT FINAL TEXT:\n{trace['response'] or '(empty)'}\n\n"
        f"TOOL CALLS:\n{_tool_summary(trace)}\n\n"
        f"RUBRIC AXIS: {axis['axis']}\n"
        f"WHAT TO CHECK: {axis['prompt']}\n"
    )
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_JUDGE_SYSTEM,
        tools=[_JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "record_score"},
        messages=[{"role": "user", "content": user_msg}],
    )
    score, rationale, raw_text = None, "", ""
    for block in resp.content:
        if block.type == "tool_use" and block.name == "record_score":
            score = block.input.get("score")
            rationale = block.input.get("rationale", "")
        elif block.type == "text":
            raw_text += block.text
    return {
        "axis":      axis["axis"],
        "score":     score,
        "rationale": rationale,
        "raw":       raw_text,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces",  type=Path, default=_DEFAULT_TRACES)
    ap.add_argument("--scores",  type=Path, default=_DEFAULT_SCORES)
    ap.add_argument("--prompts", type=Path, default=_DEFAULT_PROMPTS)
    ap.add_argument("--model",              default=_DEFAULT_MODEL)
    args = ap.parse_args()

    _load_env_key()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set (looked in env + api/.env)",
              file=sys.stderr)
        return 2
    if not args.traces.exists():
        print(f"no traces file at {args.traces}", file=sys.stderr)
        return 2
    rubric_by_id = _prompt_index(args.prompts)

    client = anthropic.Anthropic()
    args.scores.parent.mkdir(parents=True, exist_ok=True)

    n_traces = sum(1 for _ in args.traces.open())
    with args.traces.open() as fin, args.scores.open("w") as fout:
        for line_i, line in enumerate(fin, 1):
            trace = json.loads(line)
            rubric = rubric_by_id.get(trace["prompt_id"], [])
            if not rubric:
                print(f"[judge] {line_i}/{n_traces}  "
                      f"{trace['prompt_id']}  (no rubric — skipping)")
                continue
            print(f"[judge] {line_i}/{n_traces}  "
                  f"{trace['prompt_id']} run={trace['run']}  "
                  f"{len(rubric)} axes")
            scores = []
            for axis in rubric:
                t0 = time.time()
                try:
                    scored = _score_one(client, args.model, trace, axis)
                except anthropic.APIError as e:
                    scored = {"axis": axis["axis"], "score": None,
                              "rationale": f"api_error: {e}",
                              "raw": ""}
                dt = time.time() - t0
                print(f"[judge]   {axis['axis']}: "
                      f"{scored['score']}  ({dt:.1f}s)  "
                      f"{scored['rationale'][:80]}")
                scores.append(scored)
            fout.write(json.dumps({
                "prompt_id": trace["prompt_id"],
                "tier":      trace.get("tier"),
                "run":       trace["run"],
                "wall_ms":   trace.get("wall_ms"),
                "n_tool_calls": trace.get("n_tool_calls"),
                "error":     trace.get("error"),
                "scores":    scores,
            }) + "\n")
            fout.flush()
    print(f"[judge] done → {args.scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
