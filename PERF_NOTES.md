# Routing performance notes

Empirical timings driving the long-distance routing optimization.

## Graz → Copenhagen (1,624–1,800 km depending on path)

All measurements on the project Docker stack, `lht` profile, BRouter at
`-Xmx4G -DmaxRunningTime=1800`, 4 worker threads. BRouter's A* search
cost grows roughly exponentially in start–end distance, so per-leg
spacing is the primary lever.

| Date       | Approach                                     | Wall time | Output  | Notes |
|------------|----------------------------------------------|-----------|---------|-------|
| 2026-05-05 | A. Direct, 2 points (BRouter direct)         | 489 s     | 3.66 MB | Engine succeeds; previous 504s were proxy timeouts, not engine failures. |
| 2026-05-05 | B. Manual 7-pt (Maribor, Prague, Dresden, Berlin, Hamburg)  | 209 s | 4.75 MB | Curated through capitals; significant detour vs great-circle. |
| 2026-05-05 | C. Manual 10-pt (~120 km spacing)            | 140 s     | 4.45 MB | Hand-picked along corridor. |
| 2026-05-05 | D. **Auto-waypoint via API (7 inserted)**    | **119 s** | 2.95 MB / 1,625 km | End-to-end through `/route`; selector picks `place=city|town` anchors closest to great-circle line. |
| 2026-05-05 | E. Auto-waypoint, **cold per-leg cache**     | 108 s     | 2.97 MB | Phase 2: each leg fetched then cached to `/data/cache/legs.sqlite`. |
| 2026-05-05 | F. Auto-waypoint, **warm per-leg cache**     | **0.64 s** | 2.97 MB | All 8 legs served from disk; byte-identical output. **170× faster than cold.** |

### Takeaways

- **Speedup is sub-linear in waypoint count** — doubling waypoints
  doesn't halve time. Tracks per-leg search-space exponential.
- **Auto-selection produces shorter routes than manual curated.** Picking
  closest-to-line anchors avoids capital-city zigzag (1,625 km vs ~1,800
  km manual).
- **~120 km spacing is the empirical sweet spot.** Denser would shave more
  off but at diminishing returns and added detour cost.
- **"Target island detected"** errors come from a waypoint snapping to a
  small disconnected piece of road graph (Lübeck did this with one
  candidate coord). Not yet handled in auto-waypointer; will need a
  fallback to next-best anchor when BRouter returns this.

## Pre-optimization sizing

- Geofabrik PBFs: AT 750 MB, CZ 920 MB, DE 4.0 GB, DK 350 MB.
- BRouter segment tiles for the corridor: ~700 MB.
- POIs loaded: 348,643. Routing anchors loaded: 3,212 (113 city, 3,099 town).
