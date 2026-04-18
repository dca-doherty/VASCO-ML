#!/usr/bin/env python3
"""
nuclear_permutation_deciles.py
Decile-based nuclear-transient association with permutation tests.

Reproduces Figure 4 and nuclear-transient statistics:
  - Primary 10,000-permutation test on probability-weighted date-level counts
  - Mann-Whitney U confirmatory test
  - D9 vs pooled D0-D8 two-proportion z-test on percent in nuclear window
  - Lag sensitivity analysis, -3 to +3 days

Usage:
    python nuclear_permutation_deciles.py
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================================
# CONFIG -- edit paths for your environment
# ============================================================================
CATALOG_PATH    = "results/ml_pipeline_red_allcandidates/scored_catalog.csv"                    # .csv or .xlsb
OBS_DATES_PATH  = "data/poss_i_observing_dates_strict.csv"
NUCLEAR_CSV     = "data/nuclear_tests_1949_1957.csv"
OUTPUT_DIR      = "results/nuclear_permutation_deciles"

N_PERMUTATIONS  = 10000
RNG_SEED        = 42
COUNTRY_FILTER  = "USA"    # per paper
WINDOW_DAYS     = 1        # +/- 1 day
LAG_RANGE       = range(-3, 4)
# ============================================================================


def parse_obs_dates_to_str(df, col='obs_date'):
    sample = df[col].iloc[0]
    if isinstance(sample, (int, float, np.integer, np.floating)):
        base = datetime(1899, 12, 30)
        return df[col].apply(lambda x: (base + timedelta(days=int(x))).strftime('%Y-%m-%d'))
    return pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')


def find_date_col(df):
    for c in df.columns:
        lc = c.lower()
        if 'date' in lc and 'orig' not in lc and 'schedul' not in lc:
            return c
    raise ValueError("No date column found in nuclear test CSV.")


def load_nuclear_test_dates(path, country):
    nt = pd.read_csv(path)
    if 'Country' in nt.columns:
        nt = nt[nt['Country'].astype(str).str.upper() == country.upper()]
    date_col = find_date_col(nt)
    nt['_test_dt'] = pd.to_datetime(nt[date_col])
    return nt['_test_dt'].dropna()


def expand_window(test_dates, window_days):
    out = set()
    for td in test_dates:
        for lag in range(-window_days, window_days + 1):
            out.add((td + timedelta(days=lag)).strftime('%Y-%m-%d'))
    return out


def permutation_one_sided(values, labels, n_perm, seed):
    """One-sided permutation test: p = fraction of permutations with
       mean_lbl1 - mean_lbl0 >= observed."""
    rng = np.random.default_rng(seed)
    obs_diff = values[labels == 1].mean() - values[labels == 0].mean()
    n_pos = int(labels.sum())
    n = len(values)
    count_ge = 0
    idx = np.arange(n)
    for _ in range(n_perm):
        pos_idx = rng.choice(n, size=n_pos, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[pos_idx] = True
        perm_diff = values[mask].mean() - values[~mask].mean()
        if perm_diff >= obs_diff:
            count_ge += 1
    p = (count_ge + 1) / (n_perm + 1)   # add-one smoothing, standard
    return obs_diff, p


def two_prop_z(a_succ, a_n, b_succ, b_n):
    pa = a_succ / a_n
    pb = b_succ / b_n
    pp = (a_succ + b_succ) / (a_n + b_n)
    se = np.sqrt(pp * (1 - pp) * (1 / a_n + 1 / b_n)) if pp * (1 - pp) > 0 else 0
    z = (pa - pb) / se if se > 0 else 0
    p = 2 * stats.norm.sf(abs(z))
    return z, p, pa, pb


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("NUCLEAR-TRANSIENT -- DECILE PERMUTATION ANALYSIS")
    print("=" * 70)
    print(f"Catalog:   {CATALOG_PATH}")
    print(f"Nuclear:   {NUCLEAR_CSV}  (country={COUNTRY_FILTER}, +/-{WINDOW_DAYS}d)")
    print(f"Perms:     {N_PERMUTATIONS:,}, seed={RNG_SEED}")

    # Load catalog
    if CATALOG_PATH.lower().endswith('.xlsb'):
        df = pd.read_excel(CATALOG_PATH, engine='pyxlsb')
    else:
        df = pd.read_csv(CATALOG_PATH)
    if 'prob' not in df.columns:
        raise ValueError("Catalog must contain a 'prob' column.")
    df['obs_date_str'] = parse_obs_dates_to_str(df)
    print(f"Loaded:    {len(df):,} candidates on {df['obs_date_str'].nunique()} dates")

    # Filter to strict POSS-I dates (paper N=99,933)
    _poss = pd.read_csv(OBS_DATES_PATH)
    _poss_set = set(pd.to_datetime(_poss['obs_date']).dt.strftime('%Y-%m-%d'))
    df = df[df['obs_date_str'].isin(_poss_set)].reset_index(drop=True)
    print(f"After POSS-I filter: {len(df):,} candidates on {df['obs_date_str'].nunique()} dates")

    # Deciles
    df['decile'] = pd.qcut(df['prob'].rank(method='first'), 10, labels=False).astype(int)

    # Observation dates (the universe for permutation)
    obs = pd.read_csv(OBS_DATES_PATH)
    obs['obs_date'] = pd.to_datetime(obs['obs_date']).dt.strftime('%Y-%m-%d')
    all_dates = obs['obs_date'].values
    obs_set = set(all_dates)

    # Primary nuclear window (+/-1 day)
    test_dates = load_nuclear_test_dates(NUCLEAR_CSV, COUNTRY_FILTER)
    nuke_set_primary = expand_window(test_dates, WINDOW_DAYS) & obs_set
    print(f"Nuclear window (+/-{WINDOW_DAYS}d, {COUNTRY_FILTER}): "
          f"{len(nuke_set_primary)} of {len(all_dates)} dates")
    print(f"  Unique US tests: {len(test_dates)}")

    # Per-date probability-weighted transient counts
    per_date = df.groupby('obs_date_str')['prob'].sum().to_dict()
    values = np.array([per_date.get(d, 0.0) for d in all_dates])
    labels_primary = np.array([1 if d in nuke_set_primary else 0 for d in all_dates])

    if labels_primary.sum() == 0:
        print("ERROR: no nuclear-window dates intersect POSS-I observation dates.")
        sys.exit(1)

    # ---- Primary permutation test ----
    obs_diff, p_perm = permutation_one_sided(values, labels_primary, N_PERMUTATIONS, RNG_SEED)
    mean_nuke = values[labels_primary == 1].mean()
    mean_non  = values[labels_primary == 0].mean()
    ratio_primary = mean_nuke / mean_non if mean_non > 0 else float('nan')
    print()
    print("Primary permutation test (probability-weighted counts):")
    print(f"  Mean nuclear dates:     {mean_nuke:.3f}")
    print(f"  Mean non-nuclear dates: {mean_non:.3f}")
    print(f"  Ratio:                  {ratio_primary:.3f}")
    print(f"  Observed diff:          {obs_diff:.3f}")
    print(f"  p (permutation):        {p_perm:.4f}")

    # ---- Mann-Whitney U ----
    u_stat, p_mwu = stats.mannwhitneyu(
        values[labels_primary == 1], values[labels_primary == 0], alternative='greater')
    print()
    print("Mann-Whitney U (confirmatory):")
    print(f"  U = {u_stat:.1f}, p = {p_mwu:.4e}")

    # ---- D9 vs pooled D0-D8 z-test on percent in nuclear window ----
    def counts_in_window(subset, dates_set):
        arr = subset['obs_date_str'].values
        inw = np.array([d in dates_set for d in arr])
        return int(inw.sum()), len(arr)

    n_in_9, n_9 = counts_in_window(df[df['decile'] == 9], nuke_set_primary)
    n_in_o, n_o = counts_in_window(df[df['decile'] < 9],  nuke_set_primary)
    z, p_z, p9, po = two_prop_z(n_in_9, n_9, n_in_o, n_o)
    print()
    print("D9 vs pooled D0-D8 (% transients in nuclear window):")
    print(f"  D9 pct:        {100*p9:.2f}% ({n_in_9}/{n_9})")
    print(f"  D0-D8 pooled:  {100*po:.2f}% ({n_in_o}/{n_o})")
    print(f"  z = {z:+.3f}, p = {p_z:.4e}")

    # ---- Decile-wise percentages (for Figure 4) ----
    dec_rows = []
    for d in range(10):
        sub = df[df['decile'] == d]
        n_in, n_tot = counts_in_window(sub, nuke_set_primary)
        dec_rows.append({
            'decile': d,
            'n': n_tot,
            'n_in_nuclear_window': n_in,
            'pct_in_nuclear_window': 100 * n_in / n_tot if n_tot else 0.0,
            'prob_min': float(sub['prob'].min()),
            'prob_max': float(sub['prob'].max()),
        })
    dec_df = pd.DataFrame(dec_rows)

    # ---- Lag sensitivity -3..+3 ----
    print()
    print("Lag sensitivity (-3 to +3 days):")
    print(f"  {'Lag':>4} {'N dates':>8} {'Ratio':>8} {'p(perm)':>10} {'Sig':>5}")
    print("  " + "-" * 42)
    lag_rows = []
    for lag in LAG_RANGE:
        dates_lag = set((td + timedelta(days=lag)).strftime('%Y-%m-%d')
                        for td in test_dates) & obs_set
        labels_lag = np.array([1 if d in dates_lag else 0 for d in all_dates])
        n_lag = int(labels_lag.sum())
        if n_lag < 2:
            lag_rows.append({'lag': lag, 'n_dates': n_lag, 'mean_lag': np.nan,
                             'mean_non': np.nan, 'ratio': np.nan, 'p_perm': np.nan})
            print(f"  {lag:>+4d} {n_lag:>8} {'---':>8} {'---':>10} {'--':>5}")
            continue
        _, p_lag = permutation_one_sided(values, labels_lag, N_PERMUTATIONS, RNG_SEED)
        m1 = values[labels_lag == 1].mean()
        m0 = values[labels_lag == 0].mean()
        ratio = m1 / m0 if m0 > 0 else float('nan')
        sig = '***' if p_lag < 0.001 else '**' if p_lag < 0.01 else '*' if p_lag < 0.05 else 'ns'
        lag_rows.append({'lag': lag, 'n_dates': n_lag, 'mean_lag': m1,
                         'mean_non': m0, 'ratio': ratio, 'p_perm': p_lag})
        print(f"  {lag:>+4d} {n_lag:>8} {ratio:>8.3f} {p_lag:>10.4f} {sig:>5}")
    lag_df = pd.DataFrame(lag_rows)

    # ---- Save everything ----
    dec_df.to_csv(os.path.join(OUTPUT_DIR, 'decile_nuclear_pct.csv'), index=False)
    lag_df.to_csv(os.path.join(OUTPUT_DIR, 'lag_sensitivity.csv'), index=False)

    pd.DataFrame([
        {'analysis': 'primary_permutation',
         'p_value': p_perm, 'n_perm': N_PERMUTATIONS,
         'mean_nuke': mean_nuke, 'mean_non': mean_non,
         'ratio': ratio_primary, 'obs_diff': obs_diff,
         'n_nuke_dates': int(labels_primary.sum()),
         'n_non_dates': int((1 - labels_primary).sum())},
        {'analysis': 'mann_whitney_u',
         'p_value': p_mwu, 'u_stat': u_stat},
        {'analysis': 'd9_vs_d0-d8_zprop',
         'p_value': p_z, 'z_stat': z,
         'd9_pct_in_window': 100 * p9,
         'd0-d8_pooled_pct_in_window': 100 * po},
    ]).to_csv(os.path.join(OUTPUT_DIR, 'primary_tests_summary.csv'), index=False)

    # Figure 4
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(dec_df['decile'], dec_df['pct_in_nuclear_window'],
           color='#4c72b0', alpha=0.85, edgecolor='black')
    ax.set_xticks(range(10))
    ax.set_xlabel('ML-based Probability Decile')
    ax.set_ylabel('% Transients in Nuclear Window (+/- 1 day US test)')
    ax.set_title('Percentage of Transients in Nuclear Window by Decile')
    for _, r in dec_df.iterrows():
        ax.annotate(f"{r['pct_in_nuclear_window']:.1f}%",
                    (r['decile'], r['pct_in_nuclear_window']),
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'figure4_nuclear_window_pct.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

    print()
    print(f"Saved: {OUTPUT_DIR}/")
    print(f"  decile_nuclear_pct.csv")
    print(f"  lag_sensitivity.csv")
    print(f"  primary_tests_summary.csv")
    print(f"  figure4_nuclear_window_pct.png/.pdf")
    print("Done.")


if __name__ == '__main__':
    main()

