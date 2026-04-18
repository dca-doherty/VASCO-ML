#!/usr/bin/env python3
"""
shadow_deficit_deciles.py
Decile-based shadow deficit analysis for the VASCO ML paper.

Reproduces Figure 3 and shadow deficit statistics:
  - Per-decile one-sided binomial tests vs 0.692% expected (Bonferroni-corrected)
  - D9 vs pooled D0-D8 two-proportion z-test
  - Pairwise D9 vs D0 and D9 vs D1 z-tests (Bonferroni threshold 0.0056)

Usage:
    python shadow_deficit_deciles.py
"""

import os
import math
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
CATALOG_PATH = "results/ml_pipeline_red_allcandidates/scored_catalog.csv"              # .csv or .xlsb
OUTPUT_DIR   = "results/shadow_deficit_deciles"

EXPECTED_SHADOW_FRAC = 0.00692    # plate-aware Monte Carlo, 433/62,600
PENUMBRA_DEG         = 8.50       # geocentric anti-sun cone (matches deliverables_for_steve.py)
PALOMAR_LAT_DEG      = 33.3563
PALOMAR_LON_DEG      = -116.8650
GEO_ALT_KM           = 35786.0
# ============================================================================

R_EARTH_KM = 6371.0
R_SUN_KM   = 696000.0
AU_KM      = 149597870.7


# ------------------- Shadow geometry helpers -------------------

def datetime_to_jd(dt):
    year, month = dt.year, dt.month
    day = dt.day + dt.hour / 24 + dt.minute / 1440 + dt.second / 86400
    if month <= 2:
        year -= 1
        month += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    return int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5


def sun_position(jd):
    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360
    g = (357.528 + 0.9856003 * n) % 360
    g_rad = math.radians(g)
    lam = L + 1.915 * math.sin(g_rad) + 0.020 * math.sin(2 * g_rad)
    eps = math.radians(23.439 - 0.0000004 * n)
    lam_r = math.radians(lam)
    ra = math.degrees(math.atan2(math.cos(eps) * math.sin(lam_r), math.cos(lam_r))) % 360
    dec = math.degrees(math.asin(math.sin(eps) * math.sin(lam_r)))
    return ra, dec


def gmst_deg(jd):
    T = (jd - 2451545.0) / 36525.0
    theta = (280.46061837
             + 360.98564736629 * (jd - 2451545.0)
             + 0.000387933 * T**2
             - T**3 / 38710000.0)
    return theta % 360.0


def precess_j2000(ra_j2000, dec_j2000, dt):
    jd = datetime_to_jd(dt)
    T = (jd - 2451545.0) / 36525.0
    zeta = math.radians((2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3) / 3600.0)
    z = math.radians((2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3) / 3600.0)
    theta = math.radians((2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3) / 3600.0)
    ra0, dec0 = math.radians(ra_j2000), math.radians(dec_j2000)
    A = math.cos(dec0) * math.sin(ra0 + zeta)
    B = (math.cos(theta) * math.cos(dec0) * math.cos(ra0 + zeta)
         - math.sin(theta) * math.sin(dec0))
    C = (math.sin(theta) * math.cos(dec0) * math.cos(ra0 + zeta)
         + math.cos(theta) * math.sin(dec0))
    ra_ep = math.degrees(math.atan2(A, B) + z) % 360.0
    dec_ep = math.degrees(math.asin(min(1.0, max(-1.0, C))))
    return ra_ep, dec_ep


def topocentric_antisun(jd, lat=PALOMAR_LAT_DEG, lon=PALOMAR_LON_DEG,
                        dist=R_EARTH_KM + GEO_ALT_KM):
    sun_ra, sun_dec = sun_position(jd)
    geo_ra, geo_dec = (sun_ra + 180) % 360, -sun_dec
    lst = gmst_deg(jd) + lon
    lat_r, lst_r = math.radians(lat), math.radians(lst)
    ox = R_EARTH_KM * math.cos(lat_r) * math.cos(lst_r)
    oy = R_EARTH_KM * math.cos(lat_r) * math.sin(lst_r)
    oz = R_EARTH_KM * math.sin(lat_r)
    ra_r, dec_r = math.radians(geo_ra), math.radians(geo_dec)
    sx = dist * math.cos(dec_r) * math.cos(ra_r)
    sy = dist * math.cos(dec_r) * math.sin(ra_r)
    sz = dist * math.sin(dec_r)
    dx, dy, dz = sx - ox, sy - oy, sz - oz
    d = math.sqrt(dx**2 + dy**2 + dz**2)
    return (math.degrees(math.atan2(dy, dx)) % 360.0,
            math.degrees(math.asin(dz / d)))


def angular_sep(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(math.radians, [ra1, dec1, ra2, dec2])
    a = math.sin((d2 - d1) / 2)**2 + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2)**2
    return math.degrees(2 * math.asin(math.sqrt(min(1.0, a))))


def parse_obs_dates(df, col='obs_date'):
    sample = df[col].iloc[0]
    if isinstance(sample, (int, float, np.integer, np.floating)):
        base = datetime(1899, 12, 30)
        return df[col].apply(lambda x: base + timedelta(days=int(x)))
    return pd.to_datetime(df[col]).dt.to_pydatetime()


def compute_shadow_seps(ra_arr, dec_arr, dates):
    """Group by unique date for efficiency (once-per-date sun position)."""
    seps = np.zeros(len(ra_arr))
    date_idx = {}
    for i, d in enumerate(dates):
        key = d if not isinstance(d, np.datetime64) else pd.Timestamp(d).to_pydatetime()
        date_idx.setdefault(key, []).append(i)
    n_dates = len(date_idx)
    for di, (d, idxs) in enumerate(date_idx.items()):
        if (di + 1) % 50 == 0 or di == 0:
            print(f"    date {di+1}/{n_dates}...", flush=True)
        _d8 = datetime(d.year, d.month, d.day, 8, 0, 0)  # 08:00 UTC (matches paper)
        jd = datetime_to_jd(_d8)
        _sun_ra, _sun_dec = sun_position(jd)
        as_ra, as_dec = (_sun_ra + 180) % 360, -_sun_dec  # geocentric anti-sun (matches paper)
        for i in idxs:
            seps[i] = angular_sep(ra_arr[i], dec_arr[i], as_ra, as_dec)
    return seps


# ------------------- Stats helpers -------------------

def two_prop_z(a_success, a_n, b_success, b_n):
    pa = a_success / a_n
    pb = b_success / b_n
    pp = (a_success + b_success) / (a_n + b_n)
    se = math.sqrt(pp * (1 - pp) * (1 / a_n + 1 / b_n)) if pp * (1 - pp) > 0 else 0
    z = (pa - pb) / se if se > 0 else 0
    p = 2 * stats.norm.sf(abs(z))
    return z, p, pa, pb


# ------------------- Main -------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("SHADOW DEFICIT -- DECILE ANALYSIS")
    print("=" * 70)
    print(f"Catalog:      {CATALOG_PATH}")
    print(f"Expected:     {EXPECTED_SHADOW_FRAC*100:.3f}%")
    print(f"Penumbra:     {PENUMBRA_DEG} deg (3D topocentric, GEO)")

    # Load
    if CATALOG_PATH.lower().endswith('.xlsb'):
        df = pd.read_excel(CATALOG_PATH, engine='pyxlsb')
    else:
        df = pd.read_csv(CATALOG_PATH)
    print(f"Loaded:       {len(df):,} candidates")

    if 'prob' not in df.columns:
        raise ValueError("Catalog must contain a 'prob' column.")

    # Shadow separations
    df['obs_dt'] = parse_obs_dates(df)
    print("Computing shadow separations (topocentric 3D)...")
    ra = df['ra'].values.astype(float)
    dec = df['dec'].values.astype(float)
    df['shadow_sep'] = compute_shadow_seps(ra, dec, df['obs_dt'].values)
    df['in_shadow'] = (df['shadow_sep'] < PENUMBRA_DEG).astype(int)
    print(f"  In shadow:  {int(df['in_shadow'].sum()):,} / {len(df):,} ({100*df['in_shadow'].mean():.3f}%)")

    # Filter to strict POSS-I dates (paper N=99,933)
    poss = pd.read_csv('data/poss_i_observing_dates_strict.csv')
    poss_set = set(pd.to_datetime(poss['obs_date']).dt.strftime('%Y-%m-%d'))
    df = df[df['obs_dt'].apply(lambda d: pd.Timestamp(d).strftime('%Y-%m-%d')).isin(poss_set)].reset_index(drop=True)
    print(f"After POSS-I filter: {len(df):,} candidates")

    # Deciles from ML probability
    df['decile'] = pd.qcut(df['prob'].rank(method='first'), 10, labels=False).astype(int)

    # Per-decile binomial tests
    rows = []
    for d in range(10):
        sub = df[df['decile'] == d]
        n = len(sub)
        n_sh = int(sub['in_shadow'].sum())
        obs_frac = n_sh / n if n else 0
        deficit = 100 * (obs_frac - EXPECTED_SHADOW_FRAC) / EXPECTED_SHADOW_FRAC if EXPECTED_SHADOW_FRAC > 0 else 0
        p_binom = stats.binomtest(n_sh, n, EXPECTED_SHADOW_FRAC, alternative='less').pvalue
        rows.append({
            'decile': d,
            'n': n,
            'n_shadow': n_sh,
            'prob_min': float(sub['prob'].min()),
            'prob_max': float(sub['prob'].max()),
            'shadow_frac': obs_frac,
            'shadow_pct': 100 * obs_frac,
            'expected_pct': 100 * EXPECTED_SHADOW_FRAC,
            'deficit_pct': deficit,
            'p_binomial_one_sided': p_binom,
            'sig_bonferroni_005': bool(p_binom < 0.005),
        })
    res = pd.DataFrame(rows)

    # D9 vs pooled D0-D8
    d9 = df[df['decile'] == 9]
    pooled = df[df['decile'] < 9]
    z, p_z, p1, p2 = two_prop_z(
        int(d9['in_shadow'].sum()), len(d9),
        int(pooled['in_shadow'].sum()), len(pooled))

    # Pairwise D9 vs D0, D9 vs D1 (Bonferroni-adjusted threshold 0.0056)
    d0 = df[df['decile'] == 0]
    d1 = df[df['decile'] == 1]
    z90, p90, _, _ = two_prop_z(int(d9['in_shadow'].sum()), len(d9),
                                 int(d0['in_shadow'].sum()), len(d0))
    z91, p91, _, _ = two_prop_z(int(d9['in_shadow'].sum()), len(d9),
                                 int(d1['in_shadow'].sum()), len(d1))

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, 'shadow_deficit_deciles.csv')
    res.to_csv(csv_path, index=False)

    summary_rows = [
        {'test': 'D9_vs_D0-D8_pooled', 'z': z, 'p_value': p_z,
         'd9_pct': 100 * p1, 'pooled_pct': 100 * p2},
        {'test': 'D9_vs_D0', 'z': z90, 'p_value': p90},
        {'test': 'D9_vs_D1', 'z': z91, 'p_value': p91},
    ]
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'shadow_summary_tests.csv'), index=False)

    # Figure 3
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#d62728' if p < 0.005 else '#4c72b0' for p in res['p_binomial_one_sided']]
    ax.bar(res['decile'], -res['deficit_pct'], color=colors, alpha=0.85, edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(range(10))
    ax.set_xlabel('ML-based Probability Decile')
    ax.set_ylabel('Shadow Deficit (%)')
    ax.set_title(f'Shadow Deficit by Decile (expected = {EXPECTED_SHADOW_FRAC*100:.3f}%)')
    for _, r in res.iterrows():
        p = r['p_binomial_one_sided']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        if sig:
            ax.annotate(sig, (r['decile'], -r['deficit_pct']),
                        ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'figure3_shadow_deficit.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

    # Console report
    print()
    print("=" * 70)
    print("PER-DECILE RESULTS")
    print("=" * 70)
    print(f"{'Dec':>3} {'N':>8} {'Prob range':>22} {'Shadow%':>9} {'Deficit%':>10} {'p':>10} {'Sig':>5}")
    print("-" * 70)
    for _, r in res.iterrows():
        pr = f"[{r['prob_min']:.3f}, {r['prob_max']:.3f}]"
        p = r['p_binomial_one_sided']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"{int(r['decile']):>3} {r['n']:>8,} {pr:>22} "
              f"{r['shadow_pct']:>9.3f} {r['deficit_pct']:>+10.2f} "
              f"{p:>10.2e} {sig:>5}")

    print()
    print(f"D9 vs pooled D0-D8: z={z:+.3f}, p={p_z:.4f}")
    print(f"D9 vs D0:           z={z90:+.3f}, p={p90:.4f}  (Bonferroni thr 0.0056)")
    print(f"D9 vs D1:           z={z91:+.3f}, p={p91:.4f}  (Bonferroni thr 0.0056)")
    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")
    print("Done.")


if __name__ == '__main__':
    main()



