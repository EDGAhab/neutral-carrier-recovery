"""Direction-stratified cargo analysis for the 1809-1811 boom (reviewer point 4).

Breaks Table 2 cargo mix by direction: US->Europe vs Europe->US,
for each period. Tests whether the "inversion" is a mechanical effect
of direction-structure change.
"""
import pandas as pd
import re
import sys
sys.path.insert(0, 'analysis')
from identify_us import us_voyages, is_certain_us_port
from boom_deepdive import cat_of, PERIODS, first_port, port_class

def main():
    us = us_voyages(tiers=('certain',))
    us['period'] = pd.cut(us['year'],
                          bins=[1782, 1800, 1808, 1811],
                          labels=['1783-1800', '1801-1808', '1809-1811'])
    ce = pd.read_csv('data/customs_entries.csv', sep=';',
                     usecols=['ce_id', 'year'])
    year_of = dict(zip(ce['ce_id'], ce['year']))

    # direction legs
    dep = first_port(pd.read_csv('data/departure.csv', sep=';', usecols=['cargo_id', 'ce_id', 'departure']), 'departure')
    dst = first_port(pd.read_csv('data/destination.csv', sep=';', usecols=['cargo_id', 'ce_id', 'destination']), 'destination')
    legs = dep.merge(dst, on='ce_id', suffixes=('_dep', '_dst'))
    legs['dep_c'] = legs['port_dep'].map(port_class)
    legs['dst_c'] = legs['port_dst'].map(port_class)

    def direction(r):
        d, s = r['dep_c'], r['dst_c']
        if d == 'us' and s == 'europe':
            return 'US_to_Europe'
        if d == 'europe' and s == 'us':
            return 'Europe_to_US'
        return 'other'

    legs['direction'] = legs.apply(direction, axis=1)
    dir_of = dict(zip(legs['ce_id'], legs['direction']))

    # cargo
    c = pd.read_csv('data/cargoes_regs.csv', sep=';', usecols=['ce_id', 'commodity'])
    c_us = c[c['ce_id'].isin(set(us['ce_id']))].copy()
    ce_period = dict(zip(us['ce_id'], us['period'].astype(str)))
    c_us['period'] = c_us['ce_id'].map(ce_period)
    c_us['direction'] = c_us['ce_id'].map(dir_of)
    vp = c_us.drop_duplicates(['ce_id', 'commodity']).copy()
    vp['cat'] = vp['commodity'].map(cat_of)

    rows = []
    for p, lo, hi in PERIODS:
        for d in ('US_to_Europe', 'Europe_to_US'):
            vps = vp[(vp['period'] == p) & (vp['direction'] == d)]
            n_cargo = vps['ce_id'].nunique()
            rec = {'period': p, 'direction': d, 'voyages': n_cargo}
            for cat in ('naval_stores', 'us_domestic', 'true_reexport',
                        'dyewood', 'ballast', 'grain'):
                n = vps[vps['cat'] == cat]['ce_id'].nunique()
                rec[f'{cat}_pct'] = round(n / n_cargo * 100, 1) if n_cargo else 0.0
            rows.append(rec)
            print(f"{p} {d} (n={n_cargo}): " +
                  ' '.join(f"{k}={rec[k+'_pct']}%" for k in
                            ('naval_stores', 'us_domestic', 'true_reexport',
                             'dyewood', 'ballast')))

    pd.DataFrame(rows).to_csv('notes/boom_cargo_by_direction.csv', index=False)
    print("\nSaved to notes/boom_cargo_by_direction.csv")

if __name__ == '__main__':
    main()
