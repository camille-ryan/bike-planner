# Demo script

A short, curated sequence for showing this project to a reader (or an
interviewer) in ~5 minutes. Everything is chat-driven — the sidebar
is debug-only.

## Setup (once)

```
docker compose up -d postgres api web
open http://localhost:8080
```

Wait ~30 s on first boot for the api to preload its trunk-blob index
(much less now that trunks are lazy — see the boot log line
`indexed profile 'views': N trunk pairs (lazy; …)`).

The map opens on Central Europe. The left panel is **Overlays** (POI
toggles + paired-SPT debug); the right panel is **Chat**.

## Prompts, in order

Copy-paste each into the chat box. Give the map a beat between
prompts to finish painting.

### 1. Warm-up: single route

```
Route Graz to Wien, no daily stages.
```

**What to point at:**
- Chat panel: 3 tool_calls appear in order (`search_anchors` × 2,
  `route`).
- Map: the polyline paints in one sweep from Graz to Wien via the
  Mur/Mürz valley.
- Chat text: total km + chain waypoints, no unrequested stages.
- The trace: `docker logs bike-api | python3 scripts/trace_view.py --last`
  shows 3 rounds, wall < 10 s.

### 2. The one that shows the direct-rail tool earning its keep

```
I'm biking Praha → Wien over 5 days at ~80 km/day. My partner meets
me each evening by a direct train from Wien. Which overnights work?
```

**What to point at:**
- Chat panel: watch `direct_rail_service` fire once per candidate
  overnight (with `from_ref`, `to_ref` = the overnight, `Wien`).
- Chat text: each overnight is reported with either "direct service:
  yes, N shared routes" or a callout that a transfer is needed.
- Overlay panel (optional): toggle **Rail stations** — the picked
  overnights will each have a station within a few km.

### 3. The flagship: multi-day, multi-city, rail-verified

```
Plan a bike tour from Graz to Copenhagen over 50 days. Every
overnight must be reachable by non-transfer train (a direct train
from either Graz or Copenhagen, or a major hub on the corridor).
Ride 50 miles/day preferred, up to 80 miles if necessary. Detour to
nearby major cities and spend 2-3 days in each.
```

**What to point at:**
- This takes ~12–15 minutes end-to-end. Fine for a background
  demo; skip live if you're in an interview.
- The tool-call fanout: 100+ tool calls, dominated by
  `direct_rail_service` (57 of 108 in the last recorded run).
- The final plan: 50-day table with rest days per major city,
  every overnight tagged with rail-route count + direct-service
  verdict.

For the interviewer-live version, replace with the shorter #2 and
narrate that #3 is what the eval harness scores nightly.

### 4. The reformat trick — no wasted tools

Right after #1, type:

```
give that back as a markdown table with Day/From/To/Km columns
```

**What to point at:**
- Chat panel: zero new tool_calls. The LLM reformats from the
  prior turn's tool_results in context.
- This is `no_tool_replay` in the eval — scored 5/5 on every run.
  System-prompt rule doing its job.

### 5. Honest failure

```
Route Reykjavík to Tallinn by bike, 7 days.
```

**What to point at:**
- The LLM calls `search_anchors` twice, both come back with no hits
  inside our 4-country corridor.
- Chat text: honest "outside coverage" reply. No fabricated route.
- This is `honest_failure` in the eval, scored 5/5.

## Screenshot capture checklist

For the README's hero image and the `NOTES/demo.png` placeholder,
capture from a browser window sized ~1280×800:

- [ ] **hero.png** — the full app during or right after prompt #3,
  showing map painted with the corridor + chat panel with the
  50-day table visible. This is the "wow" shot.
- [ ] **tool-call-flow.png** — chat panel mid-stream on prompt #2,
  showing 3–4 `direct_rail_service` cards stacked (each with the
  station names + shared_routes count).
- [ ] **paired-spt-debug.png** — after any route, turn on the
  paired-SPT polygon overlay in the sidebar. Shows the polygonal
  corridors the router walked underneath the polyline.
- [ ] **trace-view.png** — a screenshot of a terminal running
  `docker logs bike-api | python3 scripts/trace_view.py --last`,
  showing the timeline output.

Save all under `NOTES/` (or `docs/img/` if we start a docs dir).
Update `README.md` to reference them.

## Optional: SSE-frame capture for a GIF

For a "watch the map paint itself" GIF, capture the browser at 10 fps
during prompt #2 (short enough to fit in a GitHub-preview GIF, long
enough to show the fanout):

```
# on macOS
brew install ffmpeg
ffmpeg -f avfoundation -framerate 10 -i "1" -t 30 -r 10 demo.gif

# on Linux
ffmpeg -f x11grab -framerate 10 -i :0.0 -t 30 -r 10 demo.gif
```

Trim in Kdenlive/Shotcut, target < 8 MB for a snappy README preview.
