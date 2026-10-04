"""v10 verification: every number for manuscript v10, computed fresh from STRO 2.0.

Run: cd ~/workspace/yankees-in-the-sound && python3 analysis/v10_verification.py
"""
import pandas as pd, numpy as np, re, sys, difflib, math
sys.path.insert(0, 'analysis')
from identify_us import us_voyages, classify_domicile, is_certain_us_port, PROBABLE_US_PORTS, AMBIGUOUS_PORTS
from boom_deepdive import cat_of, port_class, first_port, is_toenningen, NAVAL_STORES, US_DOMESTIC, TRUE_REEXPORT

# ---------------- v10 category rule: indigo counts as re-export ----------------
# South Carolina indigo production had essentially ceased by 1800, so American
# vessels' indigo was re-exported (not domestic produce). Rum stays domestic.
def cat_of_v10(commodity):
    if commodity == 'Ballast':
        return 'ballast'
    if commodity in NAVAL_STORES:
        return 'naval_stores'
    if commodity == 'Indigo':
        return 'true_reexport'
    if commodity in (US_DOMESTIC - {'Indigo'}):
        return 'us_domestic'
    if commodity in TRUE_REEXPORT:
        return 'true_reexport'
    if re.search(r'camp|bras|blau|sandel|gelb|farve|stokf', commodity, re.I) and \
       re.search(r'holt|træ|trae', commodity, re.I):
        return 'dyewood'
    if commodity in {'Rug', 'Byg', 'Hvede', 'Havre', 'Ærter', 'Bønner', 'Malt'}:
        return 'grain'
    return 'other'

def is_stokf(commodity):
    return isinstance(commodity, str) and 'stokf' in commodity.lower()

def is_cotton(commodity):
    return commodity == 'Bomuld'

# ---------------- load ----------------
ce = pd.read_csv('data/customs_entries.csv', sep=';', usecols=['ce_id', 'year', 'month'])
year_of = dict(zip(ce['ce_id'], ce['year']))
month_of = dict(zip(ce['ce_id'], ce['month']))

us_cert = us_voyages(tiers=('certain',))
us_all3 = us_voyages(tiers=('certain', 'probable', 'ambiguous'))
CERT = set(us_cert['ce_id'])

def period_of(y):
    if y <= 1800: return '1783-1800'
    if y <= 1806: return '1801-1806'
    if y == 1807: return '1807'
    if y == 1808: return '1808'
    if y <= 1811: return '1809-1811'
    return 'other'

# legs (all flags, for traffic totals)
dep = first_port(pd.read_csv('data/departure.csv', sep=';', usecols=['cargo_id', 'ce_id', 'departure']), 'departure')
dst = first_port(pd.read_csv('data/destination.csv', sep=';', usecols=['cargo_id', 'ce_id', 'destination']), 'destination')
legs = dep.merge(dst, on='ce_id', suffixes=('_dep', '_dst'))
legs['year'] = legs['ce_id'].map(year_of)
legs['month'] = legs['ce_id'].map(month_of)
legs['dep_c'] = legs['port_dep'].map(port_class)
legs['dst_c'] = legs['port_dst'].map(port_class)

def direction(r):
    d, s = r['dep_c'], r['dst_c']
    if d == 'unknown' or s == 'unknown': return 'unknown'
    if d == 'region' or s == 'region': return 'involving_region'
    if d == 'us' and s == 'europe': return 'US_to_Europe'
    if d == 'europe' and s == 'us': return 'Europe_to_US'
    if d == 'europe' and s == 'europe': return 'intra_Europe'
    if d == 'us' and s == 'us': return 'US_to_US'
    return 'other'
legs['direction'] = legs.apply(direction, axis=1)
legs['period'] = legs['year'].apply(period_of)

def three_legs(df):
    """Merge American-origin region passages into outbound."""
    def leg(r):
        if r['direction'] == 'US_to_Europe': return 'A_outbound'
        if r['direction'] == 'involving_region':
            return 'A_outbound' if r['dep_c'] == 'us' else 'C_intra'
        if r['direction'] == 'Europe_to_US': return 'B_homeward'
        if r['direction'] == 'intra_Europe': return 'C_intra'
        return 'other'
    return df.apply(leg, axis=1)

legs_us = legs[legs['ce_id'].isin(CERT)].copy()
legs_us['leg3'] = three_legs(legs_us)

# cargo
c = pd.read_csv('data/cargoes_regs.csv', sep=';', usecols=['ce_id', 'commodity'])
vp = c.drop_duplicates(['ce_id', 'commodity']).copy()
vp['cat'] = vp['commodity'].map(cat_of_v10)
vp['is_dye'] = vp['cat'] == 'dyewood'
vp['is_stokf'] = vp['commodity'].map(is_stokf)
vp['is_cotton'] = vp['commodity'].map(is_cotton)
vp['is_indigo'] = vp['commodity'] == 'Indigo'
ce_leg = dict(zip(legs_us['ce_id'], legs_us['leg3']))
ce_per = dict(zip(legs_us['ce_id'], legs_us['period']))
ce_yr = dict(zip(legs_us['ce_id'], legs_us['year']))
vp_us = vp[vp['ce_id'].isin(CERT)].copy()
vp_us['period'] = vp_us['ce_id'].map(ce_per)
vp_us['leg3'] = vp_us['ce_id'].map(ce_leg)
vp_us['year'] = vp_us['ce_id'].map(ce_yr)

def P(sub, col):
    """Passage-level share: passages carrying col at least once / passages with any cargo row."""
    ids = set(sub['ce_id'].unique())
    n = len(ids)
    k = sub[sub[col]]['ce_id'].nunique() if col in sub else 0
    return n, k, (100.0 * k / n if n else 0.0)

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    den = 1 + z**2 / n
    ctr = (p + z**2 / (2*n)) / den
    hw = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / den
    return (100*(ctr-hw), 100*(ctr+hw))

print("=" * 70)
print("1. PERIOD COUNTS (certain tier)")
for p in ['1783-1800', '1801-1806', '1807', '1808', '1809-1811']:
    n = (legs_us['period'] == p).sum()
    print(f"  {p}: n={n}")
print("  yearly 1809-1811:", {y: int((legs_us['year'] == y).sum()) for y in (1809, 1810, 1811)})
print("  yearly 1801-1808:", {y: int((legs_us['year'] == y).sum()) for y in range(1801, 1809)})

print("=" * 70)
print("2. IDENTIFICATION TIERS by period (domicile-based)")
allw = us_all3.copy()
allw['period'] = allw['year'].apply(period_of)
for p in ['1783-1800', '1801-1806', '1807', '1808', '1809-1811']:
    s = allw[allw['period'] == p]
    vc = s['tier'].value_counts()
    c_, pr, am = int(vc.get('certain', 0)), int(vc.get('probable', 0)), int(vc.get('ambiguous', 0))
    tot = c_ + pr + am
    print(f"  {p}: certain={c_} probable={pr} ambiguous={am} | certain share of identified={100*c_/tot:.1f}% (n={tot})")

print("=" * 70)
print("3. DIRECTION x PERIOD (certain tier; 4-class then 3-leg)")
for p in ['1783-1800', '1801-1806', '1807', '1808', '1809-1811']:
    s = legs_us[legs_us['period'] == p]
    n = len(s)
    vc = s['direction'].value_counts()
    print(f"  {p} n={n}: " + " ".join(
        f"{d}={int(vc.get(d,0))}({100*vc.get(d,0)/n:.1f}%)" if n else f"{d}=0" for d in
        ('US_to_Europe', 'Europe_to_US', 'intra_Europe', 'involving_region', 'US_to_US', 'unknown', 'other')))
print("  three legs:")
for p in ['1783-1800', '1801-1806', '1809-1811']:
    s = legs_us[legs_us['period'] == p]
    n = len(s); vc = s['leg3'].value_counts()
    print(f"  {p} n={n}: " + " ".join(f"{d}={int(vc.get(d,0))}({100*vc.get(d,0)/n:.1f}%)" for d in ('A_outbound', 'B_homeward', 'C_intra', 'other')))
print("  region-leg origin split (dep_c):")
for p in ['1801-1806', '1809-1811']:
    s = legs_us[(legs_us['period'] == p) & (legs_us['direction'] == 'involving_region')]
    print(f"  {p}: " + str(s['dep_c'].value_counts().to_dict()))

print("=" * 70)
print("4. POOLED CARGO (v10: indigo = re-export)")
for p in ['1783-1800', '1801-1806', '1809-1811']:
    s = vp_us[vp_us['period'] == p]
    n = s['ce_id'].nunique()
    parts = []
    for cat, lab in [('naval_stores', 'naval'), ('us_domestic', 'dom'), ('true_reexport', 'reexp'),
                     ('dyewood', 'dye'), ('ballast', 'bal'), ('grain', 'grain')]:
        ss = s[s['cat'] == cat]
        k = ss['ce_id'].nunique()
        parts.append(f"{lab}={k}({100*k/n:.1f}%)")
    kcot = s[s['is_cotton']]['ce_id'].nunique()
    kdye = s[s['is_dye']]['ce_id'].nunique()
    kstok = s[s['is_stokf']]['ce_id'].nunique()
    kdx = s[s['is_dye'] & ~s['is_stokf']]['ce_id'].nunique()
    kunion = s[s['cat'].isin(('true_reexport', 'dyewood'))]['ce_id'].nunique()
    kind = s[s['is_indigo']]['ce_id'].nunique()
    print(f"  {p} n={n}: " + " ".join(parts) +
          f" | cotton={kcot}({100*kcot/n:.1f}%) dye_incl={kdye}({100*kdye/n:.1f}%) stokf={kstok} dye_excl={kdx}({100*kdx/n:.1f}%) union={kunion}({100*kunion/n:.1f}%) indigo={kind}")

print("=" * 70)
print("5. CARGO WITHIN 3 LEGS (v10 categories)")
for p in ['1801-1806', '1809-1811']:
    print(f"--- {p} ---")
    for leg in ('A_outbound', 'B_homeward', 'C_intra'):
        s = vp_us[(vp_us['period'] == p) & (vp_us['leg3'] == leg)]
        n = s['ce_id'].nunique()
        row = []
        for cat, lab in [('naval_stores', 'naval'), ('us_domestic', 'dom'), ('true_reexport', 'reexp'), ('dyewood', 'dye'), ('ballast', 'bal')]:
            k = s[s['cat'] == cat]['ce_id'].nunique()
            row.append(f"{lab}={100*k/n:.1f}%")
        kdx = s[s['is_dye'] & ~s['is_stokf']]['ce_id'].nunique()
        kcot = s[s['is_cotton']]['ce_id'].nunique()
        print(f"  {leg} n={n}: " + " ".join(row) + f" dye_excl={100*kdx/n:.1f}% cotton={100*kcot/n:.1f}%")

print("=" * 70)
print("6. KITAGAWA (three legs; dyewood incl / naval / domestic / reexport)")
def kitagawa(legs_list, s1, s2, r1, r2):
    comp = sum((b - a) * x for a, b, x in zip(s1, s2, r1))
    rate = sum(a * (y - x) for a, x, y in zip(s1, r1, r2))
    inter = sum((b - a) * (y - x) for a, b, x, y in zip(s1, s2, r1, r2))
    comp_h = sum((b - a) * (x + y) / 2 for a, b, x, y in zip(s1, s2, r1, r2))
    rate_h = sum((y - x) * (a + b) / 2 for a, b, x, y in zip(s1, s2, r1, r2))
    return comp, rate, inter, comp_h, rate_h
LEGS = ['A_outbound', 'B_homeward', 'C_intra']
for cat, lab in [('dyewood', 'DYEWOOD incl'), ('naval_stores', 'NAVAL'), ('us_domestic', 'DOMESTIC(v10)'), ('true_reexport', 'REEXPORT(v10)')]:
    s1, s2, r1, r2 = [], [], [], []
    for leg in LEGS:
        for p, ss, rr in (('1801-1806', s1, r1), ('1809-1811', s2, r2)):
            s = vp_us[(vp_us['period'] == p) & (vp_us['leg3'] == leg)]
            n = s['ce_id'].nunique()
            k = s[s['cat'] == cat]['ce_id'].nunique()
            tot = vp_us[vp_us['period'] == p]['ce_id'].nunique()
            ss.append(n / tot); rr.append(k / n if n else 0)
    comp, rate, inter, comp_h, rate_h = kitagawa(LEGS, s1, s2, r1, r2)
    d_tot = 100 * (sum(b*y for b, y in zip(s2, r2)) - sum(a*x for a, x in zip(s1, r1)))
    print(f"  {lab}: total change={d_tot:+.1f}pp | 3-term: comp={100*comp:+.1f} rate={100*rate:+.1f} inter={100*inter:+.1f} | half-split: comp={100*comp_h:+.1f} rate={100*rate_h:+.1f}")
    print(f"    shares 01-06: {[f'{x*100:.1f}%' for x in s1]} -> 09-11: {[f'{x*100:.1f}%' for x in s2]}")
    print(f"    rates  01-06: {[f'{x*100:.1f}%' for x in r1]} -> 09-11: {[f'{x*100:.1f}%' for x in r2]}")

print("=" * 70)
print("7. YEARLY Dyewood shares + Wilson 95% CI (inclusion)")
for y in (1809, 1810, 1811):
    s = vp_us[vp_us['year'] == y]
    n = s['ce_id'].nunique(); k = s[s['is_dye']]['ce_id'].nunique()
    lo, hi = wilson(k, n)
    print(f"  {y}: n={n} dye={k} ({100*k/n:.1f}%) Wilson95=[{lo:.1f},{hi:.1f}]")

print("=" * 70)
print("8. STOKF detail")
for p in ['1801-1806', '1809-1811']:
    s = vp_us[vp_us['period'] == p]
    k_stok = s[s['is_stokf']]['ce_id'].nunique()
    k_dye = s[s['is_dye']]['ce_id'].nunique()
    print(f"  {p}: passages w/ Stokf* commodity={k_stok}; dyewood passages={k_dye}; non-Stokf dyewood={k_dye-k_stok}")
print("  boom Stokf commodity spellings:")
print(vp_us[(vp_us['period'] == '1809-1811') & vp_us['is_stokf']]['commodity'].value_counts().to_string())
print("  boom Stokf x leg3:")
ss = vp_us[(vp_us['period'] == '1809-1811') & vp_us['is_stokf']].drop_duplicates('ce_id')
print(ss['leg3'].value_counts().to_string())
print("  baseline Stokf x leg3:")
ss0 = vp_us[(vp_us['period'] == '1801-1806') & vp_us['is_stokf']].drop_duplicates('ce_id')
print(ss0['leg3'].value_counts().to_string())

print("=" * 70)
print("9. INDIGO sensitivity (v10 main already excludes indigo from domestic)")
for p in ['1801-1806', '1809-1811']:
    s = vp_us[vp_us['period'] == p]
    n = s['ce_id'].nunique()
    ind = s[s['is_indigo']]['ce_id'].unique()
    dom_ex = s[(s['cat'] == 'us_domestic')]['ce_id'].nunique()
    reex = s[s['cat'] == 'true_reexport']['ce_id'].nunique()
    # naive (old) sensitivity: indigo back in domestic
    k_ind = len(ind)
    only_ind = s[s['is_indigo']].drop_duplicates('ce_id')
    print(f"  {p}: n={n} indigo passages={k_ind}; domestic(v10)={dom_ex}({100*dom_ex/n:.1f}%) reexport(v10)={reex}({100*reex/n:.1f}%)")

print("=" * 70)
print("10. TONNINGEN / KIEL")
tonn = legs_us[legs_us['port_dep'].map(is_toenningen)]
print("  Tonn dep by year:", tonn['year'].value_counts().sort_index().to_dict())
n1011 = int(((legs_us['year'] == 1810) | (legs_us['year'] == 1811)).sum())
nboom = int((legs_us['period'] == '1809-1811').sum())
k_tonn_boom = int(((tonn['year'] >= 1809)).sum())
print(f"  boom Tonn={k_tonn_boom}; expected in 1810-11 under uniform = {k_tonn_boom*n1011/nboom:.1f}; observed 1810+1811={int(((tonn['year']==1810)|(tonn['year']==1811)).sum())}")
kiel = legs_us[legs_us['port_dst'].str.lower() == 'kiel']
print("  Kiel dst by year:", kiel['year'].value_counts().sort_index().to_dict())
k_kiel = len(kiel)
print(f"  expected in 1810-11 under uniform = {k_kiel*n1011/nboom:.1f}; observed 1810+1811={int(((kiel['year']==1810)|(kiel['year']==1811)).sum())}")

print("=" * 70)
print("11. OSTER SOEN monthly 1810 (was the spike autumn?)")
ost = legs_us[(legs_us['direction'] == 'involving_region')]
print("  region passages by year:", ost['year'].value_counts().sort_index().to_dict())
o10 = ost[ost['year'] == 1810]
print("  1810 monthly:", o10['month'].value_counts().sort_index().to_dict())

print("=" * 70)
print("12. 1807: all-flag monthly; US tiers in 1807/1808; 1808 all-flag")
m07 = legs[legs['year'] == 1807]
print("  1807 all-flag monthly:", m07['month'].value_counts().sort_index().to_dict(), "total=", len(m07))
for p in ['1807', '1808']:
    s = allw[allw['period'] == p]
    print(f"  US tiers {p}:", s['tier'].value_counts().to_dict())
print("  1808 all-flag total:", int((legs['year'] == 1808).sum()))
print("  1806 all-flag total:", int((legs['year'] == 1806).sum()))

print("=" * 70)
print("13. ALL-FLAG totals 1809-1811 + US share of all traffic")
for y in (1809, 1810, 1811):
    a = int((legs['year'] == y).sum())
    uc = int(((legs['year'] == y) & (legs['ce_id'].isin(CERT))).sum())
    ua = int((allw['year'] == y).sum())
    print(f"  {y}: all-flag={a} US_certain={uc} ({100*uc/a:.1f}%) US_alltiers={ua} ({100*ua/a:.1f}%)")

print("=" * 70)
print("14. ST PETERSBURG concentration")
for y in (1809, 1810, 1811):
    s = legs_us[legs_us['year'] == y]
    spb = s[s['port_dst'].str.lower() == 'petersborg']
    spb_e = spb[spb['leg3'] == 'A_outbound']
    print(f"  {y}: n={len(s)} SPB-dst={len(spb)} ({100*len(spb)/len(s):.1f}%); SPB-dst outbound-leg={len(spb_e)}")
# SPB-bound with probable+ambiguous tiers, 1811 (coverage upper bound)
spb11_all = allw[(allw['year'] == 1811)]
print("  1811 SPB-dst all tiers (need legs): computed below")

print("=" * 70)
print("15. CAPTAINS (certain tier strict; baseline certain tier too)")
dmap = pd.read_csv('notes/us_domicile_dictionary.csv')
dom_norm = {r['domicile_raw'].strip().lower(): r['normalized_port'] for _, r in dmap.iterrows()}
def norm_dom(d):
    return dom_norm.get(d.strip().lower(), d.strip().lower()) if isinstance(d, str) else ''
def keyify(df):
    k = df.copy()
    k['fn'] = k['firstName'].map(lambda x: x.strip().lower() if isinstance(x, str) else '')
    k['ln'] = k['lastName'].map(lambda x: x.strip().lower() if isinstance(x, str) else '')
    k['dom'] = k['domicile'].map(norm_dom)
    k['key'] = k['fn'] + '|' + k['ln'] + '|' + k['dom']
    return k[~k['key'].str.startswith('|')]
cap_c = keyify(us_cert)
def turnover_strict(new_lo, new_hi, old_lo, old_hi, label):
    nk = set(cap_c[cap_c['year'].between(new_lo, new_hi)]['key'])
    ok = set(cap_c[cap_c['year'].between(old_lo, old_hi)]['key'])
    new = nk - ok
    print(f"  {label}: {len(nk)} captains, {len(new)} new ({100*len(new)/len(nk):.1f}%), {len(nk & ok)} continuing")
turnover_strict(1809, 1811, 1783, 1806, "certain: boom vs <=1806")
turnover_strict(1803, 1805, 1783, 1800, "certain: baseline 1803-05 vs <=1800")
# all tiers for comparison
cap_a = keyify(us_all3)
def turnover_all(new_lo, new_hi, old_lo, old_hi, label):
    nk = set(cap_a[cap_a['year'].between(new_lo, new_hi)]['key'])
    ok = set(cap_a[cap_a['year'].between(old_lo, old_hi)]['key'])
    new = nk - ok
    print(f"  {label}: {len(nk)} captains, {len(new)} new ({100*len(new)/len(nk):.1f}%), {len(nk & ok)} continuing")
turnover_all(1809, 1811, 1783, 1806, "all-tier: boom vs <=1806")
turnover_all(1803, 1805, 1783, 1800, "all-tier: baseline 1803-05 vs <=1800")

print("=" * 70)
print("16. FUZZY captain matching (difflib on name, domicile exact)")
boom = cap_c[cap_c['year'].between(1809, 1811)]
pre = cap_c[cap_c['year'] <= 1806]
pre_by_dom = {}
for _, r in pre.iterrows():
    pre_by_dom.setdefault(r['dom'], []).append(r['fn'] + ' ' + r['ln'])
strict_cont = set(boom['key']) & set(pre['key'])
fuzzy_extra = 0
for _, r in boom.iterrows():
    if r['key'] in strict_cont:
        continue
    nm = r['fn'] + ' ' + r['ln']
    cands = pre_by_dom.get(r['dom'], [])
    if cands and max((difflib.SequenceMatcher(None, nm, c).ratio() for c in cands), default=0) >= 0.88:
        fuzzy_extra += 1
n_boom = boom['key'].nunique()
print(f"  boom captain-keys={n_boom}; strict continuing={len(strict_cont)}; fuzzy-extra(name~0.88, same domicile)={fuzzy_extra}")
print(f"  fuzzy turnover = {100*(n_boom-len(strict_cont)-fuzzy_extra)/n_boom:.1f}%")
# baseline fuzzy
b05 = cap_c[cap_c['year'].between(1803, 1805)]
p00 = cap_c[cap_c['year'] <= 1800]
p00_by_dom = {}
for _, r in p00.iterrows():
    p00_by_dom.setdefault(r['dom'], []).append(r['fn'] + ' ' + r['ln'])
strict_b = set(b05['key']) & set(p00['key'])
fuzzy_b = 0
for _, r in b05.iterrows():
    if r['key'] in strict_b:
        continue
    nm = r['fn'] + ' ' + r['ln']
    cands = p00_by_dom.get(r['dom'], [])
    if cands and max((difflib.SequenceMatcher(None, nm, c).ratio() for c in cands), default=0) >= 0.88:
        fuzzy_b += 1
n_b = b05['key'].nunique()
print(f"  baseline captain-keys={n_b}; strict continuing={len(strict_b)}; fuzzy-extra={fuzzy_b}")
print(f"  baseline fuzzy turnover = {100*(n_b-len(strict_b)-fuzzy_b)/n_b:.1f}%")

print("=" * 70)
print("17. ROUND-TRIP test: same captain eastbound then westbound")
for lo, hi, lab in ((1801, 1806, '1801-1806'), (1809, 1811, '1809-1811')):
    sub = legs_us[legs_us['year'].between(lo, hi)].copy()
    sub = sub.merge(cap_c[['ce_id', 'key']].drop_duplicates('ce_id'), on='ce_id', how='left')
    east = set(sub[sub['leg3'] == 'A_outbound']['key'].dropna())
    west = set(sub[sub['leg3'] == 'B_homeward']['key'].dropna())
    both = east & west
    print(f"  {lab}: captains w/ eastbound={len(east)}, w/ westbound={len(west)}, both={len(both)} -> round-trip completion={100*len(both)/len(east):.1f}% of eastbound captains")
    # single-appearance distribution
    cnt = sub['key'].value_counts()
    n1 = (cnt == 1).sum(); tot = len(cnt)
    print(f"    appearance: 1x={n1} ({100*n1/tot:.1f}%) of {tot} captains")

print("=" * 70)
print("18. DIRECTION ROBUSTNESS: dictionary-based US departure ports")
USPORTS = set(PROBABLE_US_PORTS) | set(AMBIGUOUS_PORTS)
def dep_is_us_name(port):
    if not isinstance(port, str): return False
    pl = port.strip().lower()
    if is_certain_us_port(port): return True
    base = re.sub(r'\s+i\s+.*$', '', pl).strip()
    return base in USPORTS
for p in ['1801-1806', '1809-1811']:
    s = legs_us[legs_us['period'] == p]
    n = len(s)
    k = int(s['port_dep'].map(dep_is_us_name).sum())
    k_ann = int((s['dep_c'] == 'us').sum())
    print(f"  {p}: n={n} dep annotated-US={k_ann}({100*k_ann/n:.1f}%) dep US-by-name={k}({100*k/n:.1f}%)")
print("  (dictionary adds bare US port names without clerk annotation)")

print("=" * 70)
print("19. NAVAL STORES: observed commodities (boom, certain)")
ns_comms = vp_us[(vp_us['period'] == '1809-1811') & (vp_us['cat'] == 'naval_stores')]['commodity'].value_counts()
print(ns_comms.to_string())
print("  iron-family present:", any(x in ns_comms.index for x in ('Jern', 'Stang jern', 'Stangjern', 'St. jern', 'Staal')))

print("=" * 70)
print("20. 1811 SPB-dst coverage incl. probable+ambiguous (upper bound)")
legs_a = legs.copy()
legs_a['leg3'] = three_legs(legs_a)
sa = legs_a[(legs_a['year'] == 1811) & (legs_a['ce_id'].isin(set(us_all3['ce_id']))) & (legs_a['port_dst'].str.lower() == 'petersborg')]
print(f"  1811 all-tier US SPB-dst passages: {len(sa)}; eastbound-leg: {int((sa['leg3']=='A_outbound').sum())}")

print("=" * 70)
print("21. Single-appearance captains (certain, for text)")
for lo, hi, lab in ((1801, 1806, '1801-1806'), (1809, 1811, '1809-1811')):
    sub = cap_c[cap_c['year'].between(lo, hi)]
    cnt = sub['key'].value_counts()
    print(f"  {lab}: {(cnt==1).sum()}/ {len(cnt)} = {100*(cnt==1).sum()/len(cnt):.1f}% single-appearance")
