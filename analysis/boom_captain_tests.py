"""Captain analysis improvements for Neutral-Carrier Boom (reviewer point 5).

1. Baseline: 1803-1805 captains vs pre-1800 (normal-period turnover rate)
2. All-tiers comparison pool (not just certain)
3. Round-trip matching rate within periods
4. 1812-1815 British domicile test (false neutrals?)
"""
import pandas as pd
import sys
sys.path.insert(0, 'analysis')
from identify_us import us_voyages
import re

def norm_name(x):
    return x.strip().lower() if isinstance(x, str) else ''

def load_captains(tiers):
    """Load captains with normalized keys for given tiers."""
    us = us_voyages(tiers=tiers)
    dmap = pd.read_csv('notes/us_domicile_dictionary.csv')
    dom_norm = {r['domicile_raw'].strip().lower(): r['normalized_port']
                for _, r in dmap.iterrows()}
    def norm_dom(d):
        if not isinstance(d, str):
            return ''
        return dom_norm.get(d.strip().lower(), d.strip().lower())
    cap = us[['ce_id', 'firstName', 'lastName', 'domicile', 'year']].copy()
    cap['key'] = (cap['firstName'].map(norm_name) + '|' +
                  cap['lastName'].map(norm_name) + '|' +
                  cap['domicile'].map(norm_dom))
    # filter empty names
    cap = cap[~cap['key'].str.startswith('|')]
    return cap

def turnover(new_period, old_period, label):
    """Calculate turnover: % of new_period captains not in old_period."""
    cap = load_captains(('certain', 'probable', 'ambiguous'))
    new_keys = set(cap[cap['year'].between(*new_period)]['key'])
    old_keys = set(cap[cap['year'].between(*old_period)]['key'])
    continuing = new_keys & old_keys
    new = new_keys - old_keys
    pct_new = len(new) / len(new_keys) * 100 if new_keys else 0
    print(f"{label}: {len(new_keys)} captains, {len(new)} new "
          f"({pct_new:.1f}%), {len(continuing)} continuing")
    return len(new_keys), len(new), len(continuing)

def main():
    print("=== 1. Baseline turnover (all tiers) ===")
    # Boom: 1809-1811 vs <=1806
    turnover((1809, 1811), (1783, 1806), "Boom (1809-1811 vs <=1806)")
    # Baseline: 1803-1805 vs <=1800 (normal period)
    turnover((1803, 1805), (1783, 1800), "Baseline (1803-1805 vs <=1800)")

    print("\n=== 2. Round-trip matching ===")
    # For each period, what % of captains appear exactly twice
    # (suggesting a recorded round trip) vs once?
    cap = load_captains(('certain',))
    for label, lo, hi in [('1801-1806', 1801, 1806), ('1809-1811', 1809, 1811)]:
        sub = cap[cap['year'].between(lo, hi)]
        counts = sub['key'].value_counts()
        n1 = (counts == 1).sum()
        n2 = (counts == 2).sum()
        n3p = (counts >= 3).sum()
        total = len(counts)
        print(f"{label}: {total} captains: 1x={n1} ({n1/total*100:.1f}%), "
              f"2x={n2} ({n2/total*100:.1f}%), 3+x={n3p} ({n3p/total*100:.1f}%)")

    print("\n=== 3. 1812-1815 British domicile test ===")
    # Do the 289 "new" boom captains reappear with British domiciles in 1812-1815?
    # This tests the "false neutral" hypothesis.
    cap_all = load_captains(('certain', 'probable', 'ambiguous'))
    boom = cap_all[cap_all['year'].between(1809, 1811)]
    boom_keys = set(boom['key'])
    old_keys = set(cap_all[cap_all['year'] <= 1806]['key'])
    new_boom_keys = boom_keys - old_keys
    print(f"New boom captains: {len(new_boom_keys)}")

    # Check 1812-1815: need British domicile data
    # For now, check if new boom captains appear at all in 1812-1815
    later = cap_all[cap_all['year'].between(1812, 1815)]
    later_keys = set(later['key'])
    reappear = new_boom_keys & later_keys
    print(f"  reappear 1812-1815 (any domicile): {len(reappear)} "
          f"({len(reappear)/len(new_boom_keys)*100:.1f}%)")
    # TODO: British domicile check needs domicile classification
    # Save list for manual inspection
    pd.DataFrame([{'captain_key': k} for k in sorted(reappear)]).to_csv(
        'notes/boom_new_captains_reappear_1812_1815.csv', index=False)
    print("  Saved to notes/boom_new_captains_reappear_1812_1815.csv")

if __name__ == '__main__':
    main()
