# Native SDK vs LangGraph — a benchmark on this project

**Question**: for a bike-tour planning agent (7 tools, streaming SSE
UI, single-user), does adopting LangGraph earn its complexity cost?

**Method**: same prompt, same tools, same model (`claude-sonnet-5`),
same SSE contract. Two backend implementations in the same repo,
switched via `CHAT_BACKEND=native|langgraph`:

- **`api/app/chat.py`** — 233 lines. Direct Anthropic SDK, one tool
  loop, model picks every next call.
- **`api/app/chat_langgraph.py`** — 570 lines. Explicit StateGraph
  with 5 nodes: `planner → router → enricher (fan-out per stage) →
  composer → critic (loop back on reject, cap 2 revisions)`.

Prompt:

> Plan a 5-day bike trip from Graz to Vienna. Include 1 rest day
> in a mid-size town along the way. Ride ~80 km/day. Enrich each
> riding day with 2-3 viewpoints within 2 km of the route.
> Rail-accessible overnights. Keep the summary concise — a table
> works.

## Numbers

| Metric | Native | LangGraph |
|---|---|---|
| Backend LOC (excl. shared `api.app.tools`) | **233** | 570 |
| Wall clock end-to-end | 138 s | **92 s** |
| Tool calls | 14 | 22 |
| Parallelism | serial | fan-out per stage (3 workers × 3 tools) |
| Revision loops | none | 2 (both critic rejections) |
| Final answer quality | **1-shot correct** | 2-shot; iter 2 produced empty table |

## The wall-clock win

LangGraph's critic loop makes it do **more work** (22 vs 14 tool
calls) yet completes **33 % faster** because the per-stage enricher
runs `stations_near` + `pois_along_route × 2` in parallel across
stages. Native calls them sequentially inside its single tool loop.

For any workload with N stage-scoped enrichments, the wall-clock
advantage scales roughly linearly with N. On a 30-day tour the gap
would widen from ~45 s to several minutes.

## The quality gap (surprising, but readable)

Both wall clocks are for the SAME prompt, but the ANSWERS are
different. Native came back with a cleaner one-shot itinerary
than LangGraph did after two critic-driven revisions.

**Why**: the user asked for `5 days × 80 km/day = 400 km`. The
actual Graz→Vienna corridor is **208 km**. The prompt is
physically over-constrained.

- **Native** absorbed the mismatch: "Total distance: 207.9 km (4
  riding days, avg ~52 km/day — the direct Graz–Wien corridor is
  shorter than 80 km/day×4, so daily distances below reflect the
  actual terrain/route rather than a forced target)." Then produced
  a clean 5-day table with a rest day at Wiener Neustadt, one-shot.
- **LangGraph** got closer on iteration 1 (rest day at Kapfenberg,
  mid-corridor), then the critic rejected because days 4 and 5 were
  35 – 45 km, ">25% deviation from target." The planner revised, the
  composer produced an EMPTY table (the feedback confused it), and
  iteration 2 aborted at `LG_MAX_REVISIONS`.

**Reading**: the critic is a **double-edged sword.**

- Real constraint violations that a single-shot model would miss?
  It catches them. If the user asks for rail-accessible overnights
  and a stage lands at a bus-only town, the critic can flag that.
- Constraints that are physically impossible? The critic keeps
  rejecting; the planner keeps trying; you burn tokens for no gain
  and hit the revision cap in a worse state than iteration 1. Native
  handles this by *reasoning about the trade-off in the answer*.

There's a fix — teach the critic to distinguish "planner error" from
"user error" (over-constrained prompt) — but that's more prompt
engineering **on top of** the framework, not something LangGraph gave
you for free.

## Where LangGraph is unambiguously the right call

- **Parallel workloads.** Any time you have N independent sub-tasks
  the model would otherwise serialise, `Send` fan-out earns its
  keep. The 33 % wall-clock win in this benchmark scales with N.
- **State machines that need to be readable code, not prompt.**
  The graph in `_build_graph()` fits on one screen:

  ```python
  g.add_edge(START, "planner")
  g.add_edge("planner", "router")
  g.add_conditional_edges("router", _route_after_router,
                          ["enrich_stage", "composer"])
  g.add_edge("enrich_stage", "composer")
  g.add_edge("composer", "critic")
  g.add_conditional_edges("critic", _after_critic,
                          {"planner": "planner", "end": END})
  ```

  An engineer walking in cold sees the whole control flow. The
  native tool loop's control flow lives in the model's head — you
  can *hope* it iterates the right way, or you can prompt-engineer
  until it does, but it's not in your code.
- **Explicit critic / self-refinement loops.** Native can't cleanly
  do "run the plan, evaluate it, revise if broken, cap iterations"
  without hand-rolling a Python state machine that IS-a bad LangGraph.
- **Structured output at graph boundaries.** `.with_structured_output`
  on the planner and critic makes those nodes return Pydantic models
  with schema validation. The native tool loop has no analog —
  everything is text or tool_use blocks.

## Where the native SDK stays the right call

- **Single-agent linear workflows.** If the whole workflow is "call
  a bunch of tools and write a summary," the native loop is 233 lines
  and reads top-to-bottom. LangGraph is 570 lines and requires
  understanding: reducers, `Annotated`, `Send`, `add_conditional_edges`
  returning list-of-Send-or-string, `stream_mode="updates"` deltas
  vs merged state. **The framework wins when you use its features.
  If your workflow is inherently linear, you're paying for weight
  you're not using.**
- **First-token latency.** Native starts streaming text within
  ~1 s (the model's first token). LangGraph doesn't yield anything
  visible until the planner node returns, ~5 s in. For interactive
  chat UX, that gap is felt.
- **Debugging path.** Native has one stack: model → tool loop →
  tool. LangGraph errors look like `InvalidUpdateError: At key
  'stream_events': Can receive only one value per step. Use an
  Annotated key to handle multiple values.` — a real error I hit
  and fixed during this benchmark, but not a friendly one for
  someone new to the framework.
- **Provider feature adoption speed.** Anthropic's extended-thinking
  blocks, prompt caching, streaming tool inputs, and `stop_reason`
  handling landed in `anthropic` before they landed in
  `langchain-anthropic`. If you need those, native gets them first.

## Verdict for this project

Ship the **native backend** as the default and keep the LangGraph
fork as an opt-in `CHAT_BACKEND=langgraph`. The default workload
(single route, chat context) doesn't benefit enough from LangGraph
to justify its cost. But the fork is a real portfolio artifact —
same tools, real parallel fan-out, real critic — and adopting the
framework would be a straight-line lift the day a use-case really
needs multi-agent orchestration.

If the workload evolved toward:
- Multi-region routing (dispatcher fan-out per region shard),
- Automated self-refinement (composer/critic loops that iterate on
  a real quality signal, not just constraint validation),
- Human-in-the-loop pauses (a checkpointer that lets a user approve
  each stage before continuing),

… I'd flip the default. Those are LangGraph's actual home ground.

## The interview-answer version

**"When would you pick LangGraph over the direct SDK?"**

> "When the workflow's control flow is worth writing as an
> explicit graph — parallel fan-out you want to see in the code,
> a critic loop, human-in-the-loop pauses, multi-agent handoffs.
> For a single agent calling a handful of tools in a linear
> fashion, the direct SDK is 2-3× less code and starts streaming
> to the user faster. I built both against the same tools in this
> project to prove out the trade-off; the writeup's in
> `NOTES/langchain-comparison.md`."

## Reproducing this benchmark

Both backends live in the `experiment/langchain-chat` branch. The
LangGraph fork won't run without adding its three dependencies to
`api/requirements.txt` (`langgraph`, `langchain-core`,
`langchain-anthropic`).

```bash
git checkout experiment/langchain-chat
docker compose build api                       # picks up new deps
docker compose up -d api
# Both backends now available:
curl -X POST http://localhost:8001/chat            -d '{...}'   # native
curl -X POST http://localhost:8001/chat_langgraph  -d '{...}'   # langgraph
```

Prompt used above, verbatim, is in the "Method" section.
