# Future routing profiles — design notes

Captures architectural ideas for adding more routing profiles (scenic /
gravel / fast / etc.) without paying the full preprocess cost per
profile. None of this is implemented yet; this file is a reminder for
when the corridor lands and we want to expand beyond `lht`.

## The problem with the current architecture

`ways.cost` and `ways.reverse_cost` are computed at ingest time by
`pgrouting/cost.py:bike_edge_cost()` reading OSM tags. One value per
edge, one set of values per profile. Adding a second profile today
means re-running PBF ingest *and* the multi-source Bellman-Ford from
scratch — hours of work to redo something that's mostly profile-
invariant (the graph topology).

## Two refactors, in order

### 1. Separate cost from topology

Cheap, ~1 day, do this first whenever a second profile is wanted.

**Schema change:**
- `ways` keeps `gid, osm_way_id, source, target, length_m, is_oneway,
  is_ferry`, plus the relevant raw OSM tags (`highway`, `surface`,
  `tracktype`, `oneway`, `bicycle`, `cycleway`, `bicycle_road`,
  `access`). ~2 GB extra storage at corridor scale; computed once.
- Costs move out: `ways_cost_<profile> (gid, cost, reverse_cost)`.

**Per-profile pipeline becomes:**
```
docker compose --profile preprocess run --rm pgrouting compute-cost --profile gravel
docker compose --profile preprocess run --rm pgrouting spts         --profile gravel
```

The cost-recompute step is a single SQL UPDATE over `ways`, tens of
seconds. Then the wave loop runs against `ways JOIN ways_cost_gravel`.

PBF parsing, vertex dedup, anchor snap — all done once, shared.

### 2. Two-tier transit network

Bigger refactor, ~1 week, only worth it once profile #2 or #3 is
landing and the per-profile Bellman-Ford runtime is annoying.

**The insight:** for cycle touring, the segments where profile choice
*actually matters* are the curated cycle networks. OSM tags these
explicitly: `route=bicycle` relations with `network=icn|ncn|rcn|lcn`
(international / national / regional / local). EuroVelo 6/7/9 plus
national + regional networks cover most of what a tour rider wants.

**Architecture:**

| Layer | Edges | Cost | Profile-specific? | Used for |
|---|---|---|---|---|
| Base graph | Every cyclable way (~22M for AT) | "Minutes on bike" — distance × simple surface factor | No, shared | Approach routing: arbitrary point → nearest cycle network |
| Tour graph | Cycle-network member ways only (~1-2M for AT) | Profile-specific (`lht` / `gravel` / `scenic` / `fast`) | Yes | Tour routing: along the cycle network using the chosen profile |

**Per-query flow:**

1. Snap start + end to base graph.
2. **Approach phase** — bounded Dijkstra in base graph from start to
   nearest tour-network entry point. Live `pgr_dijkstra` query, no
   precompute. Bounded by ~30 km in practice. Same in reverse for end.
3. **Tour phase** — route within the tour graph using the chosen
   profile's SPT data. This is where the current Bellman-Ford pipeline
   would run, but on a 5-10% subset of edges.
4. Stitch the three segments.

**Sizing, ballpark for the corridor (AT+CZ+DE+DK):**

| Today (single profile) | Tier-2 architecture (3 profiles) |
|---|---|
| Postgres: ~25 GB at full AT, ~150 GB corridor | Base graph: ~5 GB shared. Tour graph + per-profile data: ~1 GB each. Three profiles ≈ 8 GB total. |
| Per-profile preprocess: hours | Per-profile preprocess: ~minutes |
| Wave loop runs over 22M edges | Wave loop runs over 1-2M edges |

**Schema additions:**
- `ways.cycle_network` (NULL | `icn` | `ncn` | `rcn` | `lcn`)
- `ways.is_tour_edge boolean` derived from `cycle_network IS NOT NULL`
- `tour_visited_<profile>` (mirrors current `visited`, restricted to
  tour edges)
- `base_cost` column on `ways` for the shared "minutes on bike" metric

**Ingest changes:** the existing `ingest/config.py:BIKE_ROUTE_FILTERS`
already pulls `r/route=bicycle` from PBFs. We'd need to walk those
relations and tag the member ways with `cycle_network`.

## The trade-off we're accepting

Beautiful roads that *aren't* part of any OSM cycle route relation
don't get profile-specific routing — they're routed via the
"approach" base graph, which doesn't know about scenicness or surface
preferences. For Austria/corridor this seems mostly fine; cycle
networks cover the main tour terrain. For wilderness gravel rides
along forest service roads, the gap is bigger and we'd need a
"supplementary edge set" per profile.

## Cost function as a SQL CASE

Even further out: writing the per-profile cost function as a SQL
CASE expression instead of Python lets profiles be entirely
declarative — no rebuilds, no Python container needed, just a single
DDL change. Trade-off: SQL is uglier than Python for branching logic
(highway type lookup, surface filter, ferry handling, etc.). Probably
keep `cost.py` and call it from a thin "recompute" script.

## Open questions

- **Approach radius**: how far is too far before we say "you're not
  near any cycle route"? Probably an upper bound around 50 km — beyond
  that it's not really a tour route anyway.
- **Profile per-segment**: can the user pick a different profile for
  each leg of a multi-day tour? (E.g., gravel day 1, paved day 2.)
  The tour-graph SPTs are per-profile, so yes — just route each leg
  with the appropriate profile's SPT data.
- **Profile blends**: can we query `0.7 × scenic_cost + 0.3 ×
  fast_cost` per edge? Mathematically yes if we keep cost columns
  per-profile and join. UI complexity is the question.
