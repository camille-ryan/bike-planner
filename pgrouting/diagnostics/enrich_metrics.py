"""Per-variant metric expansion for graz_hainburg_compare.geojson.

Streams edges from postgres via a server-side cursor so memory stays
bounded at the size of the route-segment lookup table (~30 MB), not the
full 8.8M-edge corridor (which OOM-kills WSL if you fetchall it).

Reads the geojson, builds {(slon,slat,tlon,tlat) -> [variant_name, ...]}
for every route segment, then streams corridor edges and adds matching
edge stats into the per-variant accumulators.
"""
import json
import psycopg
from collections import defaultdict

GEOJSON_PATH = '/data/graz_hainburg_compare.geojson'
BBOX = (14.5, 46.8, 17.0, 48.5)


def _key(p1, p2):
    return (round(p1[0], 6), round(p1[1], 6), round(p2[0], 6), round(p2[1], 6))


def main():
    with open(GEOJSON_PATH) as h:
        fc = json.load(h)

    # Segment -> [variant names] map (both endpoint orderings so we can
    # match against the corridor stream without trying both ways per edge).
    seg_to_variants = defaultdict(list)
    for f in fc['features']:
        variant = f['properties']['variant']
        for seg in f['geometry']['coordinates']:
            seg_to_variants[_key(seg[0], seg[1])].append(variant)
            seg_to_variants[_key(seg[1], seg[0])].append(variant)
    print(f'segment lookup entries: {len(seg_to_variants):,}', flush=True)

    accum = {f['properties']['variant']: defaultdict(float)
             for f in fc['features']}

    pg = psycopg.connect('host=postgres user=bike password=bike dbname=bike_v2_test')
    with pg.cursor(name='enrich_scan') as cur:
        cur.itersize = 100_000
        cur.execute("""
          SELECT w.length_m, w.highway, w.canopy_frac, w.forest_local, w.vineyard_local,
                 w.water_local, w.waterway_along_edge, w.waterway_local, w.wetland_local,
                 w.local_relief, w.view_dominance,
                 w.viewpoint_local, w.viewpoint_regional,
                 ST_X(vs.the_geom), ST_Y(vs.the_geom),
                 ST_X(vt.the_geom), ST_Y(vt.the_geom)
          FROM ways w
          JOIN ways_vertices_pgr vs ON vs.id = w.source
          JOIN ways_vertices_pgr vt ON vt.id = w.target
          WHERE vs.lon BETWEEN %s AND %s AND vs.lat BETWEEN %s AND %s
            AND vt.lon BETWEEN %s AND %s AND vt.lat BETWEEN %s AND %s
        """, (BBOX[0], BBOX[2], BBOX[1], BBOX[3]) * 2)

        n_seen = 0; n_matched = 0
        for row in cur:
            n_seen += 1
            (length, highway, canopy, fl, vine, wl, wae, wlo, wet, relief, view,
             vp_l, vp_r, slon, slat, tlon, tlat) = row
            variants = seg_to_variants.get(_key([slon, slat], [tlon, tlat]))
            if not variants:
                continue
            n_matched += 1
            for v in variants:
                a = accum[v]
                a['length_m']         += length
                a['canopy_m']         += length * (canopy or 0)
                a['forest_local_m']   += length * (fl or 0)
                a['vineyard_m']       += length * (vine or 0)
                a['water_local_m']    += length * (wl or 0)
                a['waterway_along_m'] += length * (wae or 0)
                a['waterway_local_m'] += length * (wlo or 0)
                a['wetland_m']        += length * (wet or 0)
                a['relief_sum']       += length * (relief or 0)
                a['view_sum']         += length * max(view or 0, 0)
                a['vp_local_edges']   += 1 if (vp_l or 0) > 0 else 0
                a['vp_local_sum']     += length * (vp_l or 0)
                a['vp_regional_sum']  += length * (vp_r or 0)
                h = highway or ''
                if h == 'cycleway':                        a['cycleway_m']    += length
                elif h in ('track','path'):                a['track_path_m']  += length
                elif h in ('primary','trunk','secondary'): a['big_road_m']    += length
                elif h == 'residential':                   a['residential_m'] += length
                else:                                      a['other_m']       += length
            if n_seen % 1_000_000 == 0:
                print(f'  scanned {n_seen:,} edges, matched {n_matched:,}', flush=True)
    print(f'TOTAL scanned {n_seen:,}, matched {n_matched:,}', flush=True)

    for f in fc['features']:
        v = f['properties']['variant']
        a = accum[v]
        L = a['length_m'] or 1
        f['properties']['enriched'] = {
            'length_km':           round(a['length_m']/1000, 2),
            'canopy_km':           round(a['canopy_m']/1000, 2),
            'forest_local_km':     round(a['forest_local_m']/1000, 2),
            'vineyard_km':         round(a['vineyard_m']/1000, 3),
            'water_local_km':      round(a['water_local_m']/1000, 3),
            'waterway_along_km':   round(a['waterway_along_m']/1000, 2),
            'waterway_local_km':   round(a['waterway_local_m']/1000, 2),
            'wetland_km':          round(a['wetland_m']/1000, 4),
            'avg_relief_m':        round(a['relief_sum']/L, 1),
            'avg_view_dom_m':      round(a['view_sum']/L, 1),
            'vp_local_edges':      int(a['vp_local_edges']),
            'vp_regional_intensity': round(1000*a['vp_regional_sum']/L, 3),
            'pct_cycleway':        round(100*a['cycleway_m']/L, 1),
            'pct_track_path':      round(100*a['track_path_m']/L, 1),
            'pct_big_road':        round(100*a['big_road_m']/L, 1),
            'pct_residential':     round(100*a['residential_m']/L, 1),
            'pct_other':           round(100*a['other_m']/L, 1),
        }

    with open(GEOJSON_PATH, 'w') as h:
        json.dump(fc, h)

    print()
    order = ['direct','direct_minus3','direct_thresh5','direct_scenic_2x',
             'direct_scenic_5x','direct_scenic_10x','direct_interactive',
             'direct_interactive_vp','direct_multi','direct_minus3_multi',
             'vineyard_lover','forest_lover','views','water',
             'balanced','scenic']
    feats = {f['properties']['variant']: f for f in fc['features']}
    headers = ('variant','len','climb','canopy','forest','vineyard','wway_on',
               'avg_relf','avg_view','vp_edg','%cyc','%trk','%bigr')
    fmt = '{:>22} | ' + ' | '.join('{:>8}' for _ in headers[1:])
    print(fmt.format(*headers))
    for v in order:
        if v not in feats:
            continue
        p = feats[v]['properties']; e = p['enriched']
        print(fmt.format(v, p['length_km'], p.get('climb_m',0),
                         e['canopy_km'], e['forest_local_km'], e['vineyard_km'],
                         e['waterway_along_km'], e['avg_relief_m'], e['avg_view_dom_m'],
                         e['vp_local_edges'], e['pct_cycleway'], e['pct_track_path'],
                         e['pct_big_road']))


if __name__ == '__main__':
    main()
