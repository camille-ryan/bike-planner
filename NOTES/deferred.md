# Deferred work

Things we've intentionally shelved. Not-lost, just-not-now.

## Elevation profile

**Archived at commit** (search `git log --diff-filter=D --stat -- web/public/style.css` around the AI-native refactor)

**Why archived**: the old panel was a Strava-style "profile for one day of
riding". A tour spanning Graz → Copenhagen is thousands of km; showing
every meter of ascent as a squiggle at the bottom of the page is
information theater. Kept the DEM ingest + terrain tiles overlay
(hillshade toggle in Overlays panel), which are useful.

**When to revisit**: right before adding stage-level detail views to the
chat itinerary. "Day 3 profile: 780 m up / 620 m down; steepest 8%
between km 42-47." Per-stage, not per-tour.

**What we'd need to rebuild it**:

- A way to slice the polyline by day. `split_into_stages` already produces
  per-day from/to lonlat; we'd walk the polyline between those to isolate
  the day's coords.
- Elevation lookup along that slice. Options: (a) client-side raster tile
  lookup against the AWS terrarium DEM we already use for hillshade,
  (b) new API endpoint `/elevation?polyline=…` reading from the postgres
  DEM raster we ingested for scenicness.
- SVG or Canvas render — the old code used Canvas.

## Results panel (route cards)

**Archived at the same commit.**

**Why archived**: the panel rendered a card per profile after a sidebar-form
route completed (chain km, ferry legs, gap warnings). With the sidebar
form removed, no code path produces multiple parallel profiles to compare
in one card stack. The chat pane's "✓ Done" footer already conveys the
same info for a single route.

**When to revisit**: only if we bring back multi-profile comparison
(direct vs scenic vs fast). That's a possible portfolio feature — "here's
the same route re-priced through three cost functions" — but not
critical for the AI-native demo. If we do, the cards belong in the chat
pane's message flow, not a separate sidebar section.

**What we'd need to rebuild it**:

- Multi-profile support restored in `trunk_router.route()` (single-profile
  today per the deploy notes in `data/spt/views/`).
- A chat tool `compare_profiles(from_ref, to_ref, profiles=["direct","views"])`
  or extension to `route(profile=)`.
- Rendering in the chat log rather than a separate sidebar section — one
  card per profile inline with the assistant message that requested them.
