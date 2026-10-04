"""Identify US voyages in STRO 2.0 data (1783-1815 window), tiered by confidence.

STRO has no nationality/flag field. Identification uses masters.domicile
(captain's domicile) with hand-built disambiguation:

Tier 'certain'  : Danish clerk suffix "i America" / "i Amc." / "i Nord America"
                  / "i Amerika" etc. — the clerk's own disambiguation.
Tier 'probable' : bare US port name with no significant European homonym
                  (Philadelphia, Baltimore, Savannah, Nantucket, ...).
Tier 'ambiguous': bare name shared with a European port (Boston/Lincolnshire,
                  Portsmouth, Plymouth, Newport, Portland, Manchester, Norfolk,
                  Alexandria, Falmouth, ...). Reported separately, NOT in main count.
Tier 'non-us'   : everything else.

Excluded false friends (explicit non-US): Boston i Engeland/Engelland/England,
Charleston i Scotland, Carolina i Ostfrisland, etc.

Caveat (historiographical): domicile is the captain's residence, debated whether
it proxies ship registry. See soundtoll.nl/data/pdf/sc_source.pdf note 3.
"""
import re
import pandas as pd

FALSE_FRIENDS = re.compile(
    r'engeland|engelland|engl\.|england|scotland|ostfrisland|st\.\s*(thomas|croix|jan)', re.I)
# Danish clerk's "in America" qualifier, with all observed abbreviations:
# "i America", "i Amc.", "i Amr.", "i Am.", "i Amer.", "i Am.?)", "i Amerika",
# "i Ami.", "i Amca.", "i Nord America", "i Nord-America", ...
# Negative lookahead excludes "i Amsterdam" (port, not country).
AMERICA_I_FORM = re.compile(r'\bi\s+(nord[-\s]?)?am(?!sterdam)[a-z.]*', re.I)
# Trailing bare qualifier without "i": "Boston America", "Beverly Amc.",
# "Glochester in America", "Nord America", ...
AMERICA_TOKENS = {'america', 'amc', 'amer', 'am', 'amca', 'ami', 'amerika',
                  'ammerica', 'american', 'amerik', 'amr', 'ame', 'nordamercica'}
BARE_AMERICA = {'america', 'nord america', 'nord-america', 'nordamerica'}

# Bare US port names with no significant European homonym -> probable US
PROBABLE_US_PORTS = {
    'philadelphia', 'baltimore', 'savannah', 'newburyport', 'newberryport',
    'nantucket', 'nantuket', 'marblehead', 'salem', 'charleston', 'charlestown',
    'gloucester', 'glochester', 'glouchester', 'beverly', 'bewerley', 'beverley',
    'new bedford', 'newbedford', 'newbedfort', 'rappahannock', 'rappahanock',
    'rockshannock', 'kennebunk', 'kennybork', 'cohasset', 'cohasel',
    'fredericksburg', 'fridericksburg', 'friderichsburg', 'biddeford', 'biddeforth',
    'wilmington', 'welmington', 'cape ann', 'capeann', 'somersett', 'st. michaels',
    'new york', 'newyork', 'newyorck', 'new yorck', 'providence', 'rhode island',
}

# Bare names shared with European ports -> ambiguous, excluded from main count
AMBIGUOUS_PORTS = {
    'boston', 'portsmouth', 'plymouth', 'newport', 'newporth', 'neuport', 'neuwport',
    'portland', 'manchester', 'norfolk', 'alexandria', 'georgetown', 'bedford',
    'hudson', 'hampton', 'falmouth', 'newbury', 'somerset',
}


def classify_domicile(domicile) -> str:
    """Return 'certain' | 'probable' | 'ambiguous' | 'non-us'."""
    if not isinstance(domicile, str):
        return 'non-us'
    if FALSE_FRIENDS.search(domicile):
        return 'non-us'
    dl = domicile.lower().strip()
    if AMERICA_I_FORM.search(dl):
        return 'certain'
    # trailing bare qualifier, e.g. "Boston America", "Beverly Amc."
    toks = re.findall(r'[a-z.]+', dl)
    if toks and toks[-1].rstrip('.') in AMERICA_TOKENS:
        # make sure it is a qualifier, not a one-word place like "Edamer"
        if len(toks) > 1 or dl in BARE_AMERICA:
            return 'certain'
    if dl in BARE_AMERICA:
        return 'certain'
    if dl in PROBABLE_US_PORTS:
        return 'probable'
    if dl in AMBIGUOUS_PORTS:
        return 'ambiguous'
    return 'non-us'


def is_us_domicile(domicile) -> bool:
    """Legacy boolean: certain or probable."""
    return classify_domicile(domicile) in ('certain', 'probable')


def is_certain_us_port(port) -> bool:
    """Port-side US detection: only the clerk's explicit America qualifier.

    Bare English port names are NOT trusted here (Portsmouth/Plymouth/Newport/
    Boston all have European homonyms).
    """
    if not isinstance(port, str):
        return False
    if FALSE_FRIENDS.search(port):
        return False
    pl = port.lower().strip()
    if AMERICA_I_FORM.search(pl):
        return True
    toks = re.findall(r'[a-z.]+', pl)
    if toks and toks[-1].rstrip('.') in AMERICA_TOKENS:
        if len(toks) > 1 or pl in BARE_AMERICA:
            return True
    return pl in BARE_AMERICA


def load_window(data_dir='data', start=1783, end=1815):
    """Load customs entries + masters joined, filtered to year window."""
    ce = pd.read_csv(f'{data_dir}/customs_entries.csv', sep=';',
                     usecols=['ce_id', 'year', 'month', 'day'])
    m = pd.read_csv(f'{data_dir}/masters.csv', sep=';',
                    usecols=['ce_id', 'firstName', 'lastName', 'domicile'])
    df = m.merge(ce, on='ce_id')
    return df[(df['year'] >= start) & (df['year'] <= end)].copy()


def us_voyages(data_dir='data', start=1783, end=1815, tiers=('certain', 'probable')):
    """Return the US-identified voyages dataframe for the window."""
    df = load_window(data_dir, start, end)
    df['tier'] = df['domicile'].apply(classify_domicile)
    return df[df['tier'].isin(tiers)].copy()


if __name__ == '__main__':
    df = load_window()
    df['tier'] = df['domicile'].apply(classify_domicile)
    print(df['tier'].value_counts().to_string())
    for t in ('certain', 'probable'):
        sub = df[df['tier'] == t]
        print(f"\n{t}: {len(sub)} voyages")
        print(sub['year'].value_counts().sort_index().to_string())
