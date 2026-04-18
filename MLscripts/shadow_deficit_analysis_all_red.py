#!/usr/bin/env python3
"""
ML Threshold Shadow Deficit — 3D Conical Model
================================================

Computes the shadow deficit for VASCO transient candidates at increasing
ML probability thresholds using five shadow models (2D cylindrical,
3D umbra/penumbra, geocentric/topocentric).

Usage:
    1. Set PIPELINE_LABEL and paths in the configuration section below
    2. Run:  python shadow_deficit_analysis.py

Requires:
    - pandas, numpy, scipy, matplotlib, openpyxl (pip install if needed)
    - scored_catalog.csv (from the ML pipeline results folder)
    - brian_plate_aware_control_sample.csv (plate-aware random sky positions)

Author: Brian Doherty
Date: February 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


PIPELINE_LABEL = "Red All-Candidates"
PIPELINE_TAG = "red_allcandidates"
CATALOG_PATH = r"C:\Users\Brian Doherty\OneDrive\Documentos\python\patentInfo\results\ml_pipeline_red_allcandidates\scored_catalog.csv"
CONTROL_PATH = r"C:\Users\Brian Doherty\OneDrive\Documentos\python\patentInfo\data\brian_plate_aware_control_sample.csv"
OUTPUT_DIR = r"C:\Users\Brian Doherty\OneDrive\Documentos\python\patentInfo\results\shadow_deficit_red_allcandidates"

# ML probability thresholds to scan
ML_THRESHOLDS = [0.0, 0.25, 0.50, 0.65, 0.75, 0.85]

# Random seed for reproducibility
np.random.seed(42)

# ===========================================================================
# 3D CONICAL EARTH SHADOW MODEL
# ===========================================================================
# Physical constants
R_EARTH_KM = 6371.0
R_SUN_KM = 696000.0
AU_KM = 149597870.7
GEO_ALTITUDE_KM = 35786.0
L_UMBRA_KM = R_EARTH_KM * AU_KM / (R_SUN_KM - R_EARTH_KM)

# Palomar Observatory (POSS-I observation site)
PALOMAR_LAT_DEG = 33.3563
PALOMAR_LON_DEG = -116.8650


def shadow_cone_radii(altitude_km):
    """
    Angular radii of umbra, penumbra, and cylindrical shadow at a given
    altitude above Earth's surface.

    Returns (umbra_deg, penumbra_deg, cylindrical_deg).
    """
    d = R_EARTH_KM + altitude_km
    r_umbra_km = R_EARTH_KM * (1.0 - d / L_UMBRA_KM)
    umbra_deg = math.degrees(math.asin(r_umbra_km / d)) if r_umbra_km > 0 else 0.0
    r_penumbra_km = R_EARTH_KM + d * (R_SUN_KM + R_EARTH_KM) / AU_KM
    penumbra_deg = math.degrees(math.asin(min(1.0, r_penumbra_km / d)))
    cylindrical_deg = math.degrees(math.asin(R_EARTH_KM / d))
    return umbra_deg, penumbra_deg, cylindrical_deg


def sun_position(jd):
    """Sun's geocentric equatorial RA/Dec (degrees). Meeus algorithm."""
    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360
    g = (357.528 + 0.9856003 * n) % 360
    g_rad = math.radians(g)
    lam = L + 1.915 * math.sin(g_rad) + 0.020 * math.sin(2 * g_rad)
    lam_rad = math.radians(lam)
    eps = math.radians(23.439 - 0.0000004 * n)
    ra = math.degrees(math.atan2(
        math.cos(eps) * math.sin(lam_rad), math.cos(lam_rad))) % 360
    dec = math.degrees(math.asin(math.sin(eps) * math.sin(lam_rad)))
    return ra, dec


def datetime_to_jd(dt):
    """Python datetime -> Julian Date."""
    year, month = dt.year, dt.month
    day = dt.day + dt.hour / 24 + dt.minute / 1440 + dt.second / 86400
    if month <= 2:
        year -= 1
        month += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    return int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5


def gmst_deg(jd):
    """Greenwich Mean Sidereal Time in degrees."""
    T = (jd - 2451545.0) / 36525.0
    theta = (280.46061837
             + 360.98564736629 * (jd - 2451545.0)
             + 0.000387933 * T**2
             - T**3 / 38710000.0)
    return theta % 360.0


def precess_j2000_to_epoch(ra_j2000, dec_j2000, obs_datetime):
    """Precess J2000 coordinates to the observation epoch (IAU 1976)."""
    jd = datetime_to_jd(obs_datetime)
    T = (jd - 2451545.0) / 36525.0
    zeta_A = 2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3
    z_A = 2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3
    theta_A = 2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3
    zeta = math.radians(zeta_A / 3600.0)
    z = math.radians(z_A / 3600.0)
    theta = math.radians(theta_A / 3600.0)
    ra0, dec0 = math.radians(ra_j2000), math.radians(dec_j2000)
    A = math.cos(dec0) * math.sin(ra0 + zeta)
    B = (math.cos(theta) * math.cos(dec0) * math.cos(ra0 + zeta)
         - math.sin(theta) * math.sin(dec0))
    C = (math.sin(theta) * math.cos(dec0) * math.cos(ra0 + zeta)
         + math.cos(theta) * math.sin(dec0))
    ra_epoch = math.degrees(math.atan2(A, B) + z) % 360.0
    dec_epoch = math.degrees(math.asin(min(1.0, max(-1.0, C))))
    return ra_epoch, dec_epoch


def angular_separation(ra1, dec1, ra2, dec2):
    """Haversine angular separation (degrees, scalar)."""
    ra1r, dec1r = math.radians(ra1), math.radians(dec1)
    ra2r, dec2r = math.radians(ra2), math.radians(dec2)
    dra = ra2r - ra1r
    ddec = dec2r - dec1r
    a = (math.sin(ddec / 2)**2
         + math.cos(dec1r) * math.cos(dec2r) * math.sin(dra / 2)**2)
    return math.degrees(2 * math.asin(math.sqrt(min(1.0, a))))


def geocentric_antisun(jd):
    """Anti-sun point in geocentric equatorial coordinates."""
    sun_ra, sun_dec = sun_position(jd)
    return (sun_ra + 180.0) % 360.0, -sun_dec


def topocentric_antisun(jd, observer_lat=PALOMAR_LAT_DEG,
                        observer_lon=PALOMAR_LON_DEG,
                        shadow_distance_km=None):
    """
    Anti-sun direction corrected for observer parallax at Palomar.
    Shifts the apparent shadow center based on the observer's offset
    from Earth's center at the given shadow distance.
    """
    geo_ra, geo_dec = geocentric_antisun(jd)
    if shadow_distance_km is None:
        return geo_ra, geo_dec

    lst = gmst_deg(jd) + observer_lon
    lat_rad = math.radians(observer_lat)
    lst_rad = math.radians(lst)

    ox = R_EARTH_KM * math.cos(lat_rad) * math.cos(lst_rad)
    oy = R_EARTH_KM * math.cos(lat_rad) * math.sin(lst_rad)
    oz = R_EARTH_KM * math.sin(lat_rad)

    ra_rad = math.radians(geo_ra)
    dec_rad = math.radians(geo_dec)
    d = shadow_distance_km

    sx = d * math.cos(dec_rad) * math.cos(ra_rad)
    sy = d * math.cos(dec_rad) * math.sin(ra_rad)
    sz = d * math.sin(dec_rad)

    dx, dy, dz = sx - ox, sy - oy, sz - oz
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    topo_dec = math.degrees(math.asin(dz / dist))
    topo_ra = math.degrees(math.atan2(dy, dx)) % 360.0
    return topo_ra, topo_dec


# Shadow radii at GEO
GEO_UMBRA_DEG, GEO_PENUMBRA_DEG, GEO_CYLINDRICAL_DEG = shadow_cone_radii(GEO_ALTITUDE_KM)


# ===========================================================================
# SHADOW SEPARATION COMPUTATION
# ===========================================================================

def compute_shadow_seps(ra_arr, dec_arr, datetimes, use_topocentric=True):
    """
    Compute angular separation from the shadow center for each source,
    grouped by unique observation date for efficiency.
    """
    seps = np.full(len(ra_arr), np.nan)

    date_groups = {}
    for i, dt in enumerate(datetimes):
        key = dt
        if key not in date_groups:
            date_groups[key] = []
        date_groups[key].append(i)

    n_dates = len(date_groups)
    shadow_dist = R_EARTH_KM + GEO_ALTITUDE_KM

    for di, (dt, indices) in enumerate(date_groups.items()):
        if (di + 1) % 50 == 0 or di == 0:
            print(f"    Date {di+1}/{n_dates}...", flush=True)

        if isinstance(dt, np.datetime64):
            dt_py = pd.Timestamp(dt).to_pydatetime()
        elif isinstance(dt, pd.Timestamp):
            dt_py = dt.to_pydatetime()
        else:
            dt_py = dt

        jd = datetime_to_jd(dt_py)

        if use_topocentric:
            as_ra, as_dec = topocentric_antisun(jd, shadow_distance_km=shadow_dist)
        else:
            as_ra, as_dec = geocentric_antisun(jd)

        for idx in indices:
            ra_ep, dec_ep = precess_j2000_to_epoch(
                ra_arr[idx], dec_arr[idx], dt_py)
            seps[idx] = angular_separation(ra_ep, dec_ep, as_ra, as_dec)

    return seps


# ===========================================================================
# THRESHOLD SCAN
# ===========================================================================

def threshold_scan(df, ctrl_shadow_frac, ctrl_in, ctrl_out, shadow_col, radius, label):
    """Run the deficit calculation at each ML threshold."""
    rows = []
    for threshold in ML_THRESHOLDS:
        subset = df[df['prob'] >= threshold]
        n = len(subset)
        if n == 0:
            rows.append({
                'model': label, 'threshold': threshold,
                'n_retained': 0, 'n_shadow': 0, 'n_sunlit': 0,
                'shadow_pct': 0, 'control_pct': round(100 * ctrl_shadow_frac, 4),
                'deficit_pct': 0, 'chi2': 0, 'p_value': 1.0,
            })
            continue

        n_shadow = int((subset[shadow_col] <= radius).sum())
        n_sunlit = n - n_shadow
        shadow_frac = n_shadow / n

        deficit = ((shadow_frac - ctrl_shadow_frac) / ctrl_shadow_frac * 100
                   ) if ctrl_shadow_frac > 0 else 0

        contingency = np.array([
            [n_shadow, n_sunlit],
            [int(ctrl_in), int(ctrl_out)]
        ])
        if n_shadow > 0 or ctrl_in > 0:
            chi2, p_val, _, _ = stats.chi2_contingency(contingency)
        else:
            chi2, p_val = 0, 1.0

        rows.append({
            'model': label, 'threshold': threshold,
            'n_retained': n, 'n_shadow': n_shadow, 'n_sunlit': n_sunlit,
            'shadow_pct': round(100 * shadow_frac, 4),
            'control_pct': round(100 * ctrl_shadow_frac, 4),
            'deficit_pct': round(deficit, 2),
            'chi2': round(chi2, 4), 'p_value': p_val,
        })

    return rows


# ===========================================================================
# PROBABILITY-WEIGHTED ANALYSIS
# ===========================================================================

def weighted_analysis(df, ctrl_shadow_frac, shadow_col, radius, label):
    """Shadow analysis weighted by ML probability scores."""
    weights = df['prob'].values
    in_shadow = df[shadow_col].values <= radius

    w_shadow = np.sum(weights[in_shadow])
    w_sunlit = np.sum(weights[~in_shadow])
    w_total = w_shadow + w_sunlit
    w_frac = w_shadow / w_total if w_total > 0 else 0

    deficit = ((w_frac - ctrl_shadow_frac) / ctrl_shadow_frac * 100
               ) if ctrl_shadow_frac > 0 else 0

    expected_shadow = ctrl_shadow_frac * w_total
    expected_sunlit = (1 - ctrl_shadow_frac) * w_total
    observed = np.array([w_shadow, w_sunlit])
    expected = np.array([expected_shadow, expected_sunlit])
    chi2 = np.sum((observed - expected)**2 / expected)
    p_val = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        'model': label, 'threshold': 'weighted',
        'n_retained': round(w_total, 2),
        'n_shadow': round(w_shadow, 2), 'n_sunlit': round(w_sunlit, 2),
        'shadow_pct': round(100 * w_frac, 4),
        'control_pct': round(100 * ctrl_shadow_frac, 4),
        'deficit_pct': round(deficit, 2),
        'chi2': round(chi2, 4), 'p_value': p_val,
    }


# ===========================================================================
# DATE PARSING HELPERS
# ===========================================================================

def parse_obs_dates(df):
    """
    Convert the obs_date column to Python datetimes.
    Handles both Excel serial dates (integers like 19970) and
    ISO date strings (like '1954-09-03').
    """
    sample = df['obs_date'].iloc[0]

    if isinstance(sample, (int, float, np.integer, np.floating)):
        # Excel serial date — days since 1899-12-30
        excel_base = datetime(1899, 12, 30)
        df['obs_datetime'] = df['obs_date'].apply(
            lambda x: excel_base + timedelta(days=int(x)))
        print("  Date format: Excel serial (integer)")
    else:
        # Try parsing as date string
        df['obs_datetime'] = pd.to_datetime(df['obs_date'])
        df['obs_datetime'] = df['obs_datetime'].dt.to_pydatetime()
        print("  Date format: date string (parsed)")

    return df


# ===========================================================================
# MAIN ANALYSIS
# ===========================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 72)
    print("ML THRESHOLD SHADOW DEFICIT — 3D CONICAL MODEL")
    print(f"Pipeline: {PIPELINE_LABEL}")
    print("=" * 72)
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Observer: Palomar ({PALOMAR_LAT_DEG:.3f}N, {PALOMAR_LON_DEG:.3f}E)")
    print(f"\nGEO shadow radii:")
    print(f"  Umbra (3D):       {GEO_UMBRA_DEG:.3f} deg")
    print(f"  Cylindrical (2D): {GEO_CYLINDRICAL_DEG:.3f} deg")
    print(f"  Penumbra (3D):    {GEO_PENUMBRA_DEG:.3f} deg")
    print(f"  ML thresholds:    {ML_THRESHOLDS}")
    print()

    # ------------------------------------------------------------------
    # 1. Load scored catalog
    # ------------------------------------------------------------------
    print(f"Loading scored catalog: {os.path.basename(CATALOG_PATH)}")
    df = pd.read_csv(CATALOG_PATH)
    print(f"  Loaded {len(df):,} candidates")
    print(f"  prob range: {df['prob'].min():.4f} - {df['prob'].max():.4f}")
    print(f"  prob mean:  {df['prob'].mean():.4f}")

    if 'prob_raw' in df.columns:
        print(f"  prob == prob_raw: {(df['prob'] == df['prob_raw']).all()}")

    # Parse dates
    df = parse_obs_dates(df)
    print(f"  Date range: {df['obs_datetime'].min().strftime('%Y-%m-%d')} "
          f"to {df['obs_datetime'].max().strftime('%Y-%m-%d')}")
    print(f"  Unique dates: {df['obs_datetime'].nunique()}")

    # Threshold distribution
    print(f"\n  Candidates per threshold:")
    for t in ML_THRESHOLDS:
        n = (df['prob'] >= t).sum()
        print(f"    p >= {t:.2f}: {n:>8,}")
    print()

    # ------------------------------------------------------------------
    # 2. Load control sample
    # ------------------------------------------------------------------
    print(f"Loading control sample: {os.path.basename(CONTROL_PATH)}")
    ctrl = pd.read_csv(CONTROL_PATH)
    ctrl['datetime'] = pd.to_datetime(ctrl['UTobservation'])
    print(f"  Loaded {len(ctrl):,} control points")
    print()

    # ------------------------------------------------------------------
    # 3. Compute shadow separations for scored catalog
    # ------------------------------------------------------------------
    ra_arr = df['ra'].values.astype(float)
    dec_arr = df['dec'].values.astype(float)
    dates_arr = df['obs_datetime'].values

    print("Computing shadow separations for catalog (topocentric 3D)...")
    seps_topo = compute_shadow_seps(ra_arr, dec_arr, dates_arr, use_topocentric=True)
    df['shadow_sep_topo'] = seps_topo

    print("Computing shadow separations for catalog (geocentric 2D)...")
    seps_geo = compute_shadow_seps(ra_arr, dec_arr, dates_arr, use_topocentric=False)
    df['shadow_sep_geo'] = seps_geo

    print(f"  Topo sep range: {np.nanmin(seps_topo):.2f} - {np.nanmax(seps_topo):.2f} deg")
    print(f"  Geo  sep range: {np.nanmin(seps_geo):.2f} - {np.nanmax(seps_geo):.2f} deg")
    print()

    # ------------------------------------------------------------------
    # 4. Compute control shadow separations
    # ------------------------------------------------------------------
    print("Computing control shadow separations (topocentric 3D)...")
    ctrl_seps_topo = compute_shadow_seps(
        ctrl['RA'].values, ctrl['Dec'].values,
        ctrl['datetime'].values, use_topocentric=True)
    ctrl['shadow_sep_topo'] = ctrl_seps_topo

    print("Computing control shadow separations (geocentric 2D)...")
    ctrl_seps_geo = compute_shadow_seps(
        ctrl['RA'].values, ctrl['Dec'].values,
        ctrl['datetime'].values, use_topocentric=False)
    ctrl['shadow_sep_geo'] = ctrl_seps_geo
    print()

    # ------------------------------------------------------------------
    # 5. Control shadow fractions for each model
    # ------------------------------------------------------------------
    models = {
        '2D Cylindrical (geo)': ('shadow_sep_geo', GEO_CYLINDRICAL_DEG, ctrl_seps_geo),
        '3D Umbra (topo)': ('shadow_sep_topo', GEO_UMBRA_DEG, ctrl_seps_topo),
        '3D Penumbra (topo)': ('shadow_sep_topo', GEO_PENUMBRA_DEG, ctrl_seps_topo),
        '3D Umbra (geo)': ('shadow_sep_geo', GEO_UMBRA_DEG, ctrl_seps_geo),
        '3D Penumbra (geo)': ('shadow_sep_geo', GEO_PENUMBRA_DEG, ctrl_seps_geo),
    }

    ctrl_stats = {}
    print("Control shadow fractions:")
    for name, (col, radius, c_seps) in models.items():
        c_in = int(np.sum(c_seps <= radius))
        c_out = len(ctrl) - c_in
        c_frac = c_in / len(ctrl)
        ctrl_stats[name] = (c_frac, c_in, c_out)
        print(f"  {name:<25} {c_in:>6,} in shadow ({100*c_frac:.3f}%)")
    print()

    # ------------------------------------------------------------------
    # 6. Threshold scan (all models)
    # ------------------------------------------------------------------
    print("=" * 72)
    print(f"THRESHOLD SCAN: 2D vs 3D MODELS ({PIPELINE_LABEL.upper()})")
    print("=" * 72)
    print()

    all_rows = []

    for name, (col, radius, _) in models.items():
        c_frac, c_in, c_out = ctrl_stats[name]
        rows = threshold_scan(df, c_frac, c_in, c_out, col, radius, name)
        all_rows.extend(rows)

        w_row = weighted_analysis(df, c_frac, col, radius, name)
        all_rows.append(w_row)

    results_df = pd.DataFrame(all_rows)

    # Print formatted tables per model
    for name in models:
        model_df = results_df[(results_df['model'] == name) & (results_df['threshold'] != 'weighted')]
        w_df = results_df[(results_df['model'] == name) & (results_df['threshold'] == 'weighted')]

        print(f"--- {name} ---")
        print(f"  {'Thresh':>8} {'N':>10} {'Shadow':>8} {'Shad%':>9} {'Ctrl%':>9} "
              f"{'Deficit':>9} {'Chi2':>10} {'p-value':>12}")
        print(f"  {'':->8} {'':->10} {'':->8} {'':->9} {'':->9} "
              f"{'':->9} {'':->10} {'':->12}")

        for _, r in model_df.iterrows():
            sig = "***" if r['p_value'] < 0.001 else (
                "**" if r['p_value'] < 0.01 else (
                    "*" if r['p_value'] < 0.05 else ""))
            print(f"  {r['threshold']:>8.2f} {r['n_retained']:>10,} {r['n_shadow']:>8,} "
                  f"{r['shadow_pct']:>8.3f}% {r['control_pct']:>8.3f}% "
                  f"{r['deficit_pct']:>+8.1f}% {r['chi2']:>10.2f} {r['p_value']:>11.2e} {sig}")

        if len(w_df) > 0:
            wr = w_df.iloc[0]
            sig = "***" if wr['p_value'] < 0.001 else (
                "**" if wr['p_value'] < 0.01 else (
                    "*" if wr['p_value'] < 0.05 else ""))
            print(f"  {'wt':>8} {wr['n_retained']:>10.0f} {wr['n_shadow']:>8.1f} "
                  f"{wr['shadow_pct']:>8.3f}% {wr['control_pct']:>8.3f}% "
                  f"{wr['deficit_pct']:>+8.1f}% {wr['chi2']:>10.2f} {wr['p_value']:>11.2e} {sig}")
        print()

    # ------------------------------------------------------------------
    # 7. Monotonicity check
    # ------------------------------------------------------------------
    print("=" * 72)
    print(f"MONOTONICITY CHECK: DEFICIT vs ML THRESHOLD ({PIPELINE_LABEL.upper()})")
    print("=" * 72)
    print()

    for name in models:
        model_df = results_df[(results_df['model'] == name) & (results_df['threshold'] != 'weighted')]
        thresholds = model_df['threshold'].astype(float).values
        deficits = model_df['deficit_pct'].values

        monotonic = all(deficits[i] >= deficits[i+1] for i in range(len(deficits)-1))
        slope, intercept, r_value, p_slope, std_err = stats.linregress(thresholds, deficits)

        direction = "DEEPENS" if slope < 0 else "WEAKENS"
        mono_str = "STRICTLY MONOTONIC" if monotonic else "not strictly monotonic"

        print(f"  {name:<25} slope={slope:>+7.1f}%/unit  R²={r_value**2:.3f}  "
              f"p={p_slope:.4f}  {direction}  ({mono_str})")

    print()

    # ------------------------------------------------------------------
    # 8. Side-by-side comparison table
    # ------------------------------------------------------------------
    print("=" * 72)
    print(f"SIDE-BY-SIDE: DEFICIT AT EACH THRESHOLD ({PIPELINE_LABEL.upper()})")
    print("=" * 72)

    key_models = ['2D Cylindrical (geo)', '3D Umbra (topo)', '3D Penumbra (topo)']
    print(f"\n  {'Threshold':>10}", end="")
    for m in key_models:
        print(f"  {m:>25}", end="")
    print()
    print(f"  {'':->10}", end="")
    for _ in key_models:
        print(f"  {'':->25}", end="")
    print()

    for t in ML_THRESHOLDS:
        print(f"  {t:>10.2f}", end="")
        for m in key_models:
            row = results_df[(results_df['model'] == m) &
                             (results_df['threshold'] == t)]
            if len(row) > 0:
                d = row.iloc[0]['deficit_pct']
                p = row.iloc[0]['p_value']
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                print(f"  {d:>+8.1f}% (p={p:.1e}){sig:>4}", end="")
            else:
                print(f"  {'N/A':>25}", end="")
        print()

    # Weighted row
    print(f"  {'weighted':>10}", end="")
    for m in key_models:
        row = results_df[(results_df['model'] == m) &
                         (results_df['threshold'] == 'weighted')]
        if len(row) > 0:
            d = row.iloc[0]['deficit_pct']
            p = row.iloc[0]['p_value']
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"  {d:>+8.1f}% (p={p:.1e}){sig:>4}", end="")
        else:
            print(f"  {'N/A':>25}", end="")
    print()
    print()

    # ------------------------------------------------------------------
    # 9. Generate figure
    # ------------------------------------------------------------------
    print("Generating figure...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'ML Threshold Shadow Deficit — 2D vs 3D Shadow Models\n'
                 f'({PIPELINE_LABEL}, {len(df):,} candidates)',
                 fontsize=14, fontweight='bold')

    colors = {
        '2D Cylindrical (geo)': '#1f77b4',
        '3D Umbra (topo)': '#d62728',
        '3D Penumbra (topo)': '#ff7f0e',
    }
    markers = {
        '2D Cylindrical (geo)': 'o',
        '3D Umbra (topo)': 's',
        '3D Penumbra (topo)': '^',
    }

    # Panel A: Deficit vs threshold
    ax = axes[0, 0]
    for name in key_models:
        mdf = results_df[(results_df['model'] == name) &
                         (results_df['threshold'] != 'weighted')]
        ax.plot(mdf['threshold'].astype(float), mdf['deficit_pct'],
                f'{markers[name]}-', color=colors[name],
                linewidth=2, markersize=8, label=name)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('ML Probability Threshold (p_real)')
    ax.set_ylabel('Shadow Deficit (%)')
    ax.set_title(f'A) Shadow Deficit vs ML Threshold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Panel B: -log10(p-value) vs threshold
    ax = axes[0, 1]
    for name in key_models:
        mdf = results_df[(results_df['model'] == name) &
                         (results_df['threshold'] != 'weighted')]
        log_p = -np.log10(np.clip(mdf['p_value'].values.astype(float), 1e-300, 1))
        ax.plot(mdf['threshold'].astype(float), log_p,
                f'{markers[name]}-', color=colors[name],
                linewidth=2, markersize=8, label=name)
    ax.axhline(y=-np.log10(0.05), color='orange', linestyle='--',
               linewidth=1, label='p = 0.05')
    ax.axhline(y=-np.log10(0.001), color='red', linestyle='--',
               linewidth=1, label='p = 0.001')
    ax.set_xlabel('ML Probability Threshold (p_real)')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title(f'B) Statistical Significance vs Threshold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # Panel C: Shadow fraction (transients vs control)
    ax = axes[1, 0]
    for name in key_models:
        mdf = results_df[(results_df['model'] == name) &
                         (results_df['threshold'] != 'weighted')]
        ax.plot(mdf['threshold'].astype(float), mdf['shadow_pct'],
                f'{markers[name]}-', color=colors[name],
                linewidth=2, markersize=8, label=f'{name} (transients)')
    for name in key_models:
        c_frac = ctrl_stats[name][0]
        ax.axhline(y=100 * c_frac, color=colors[name],
                   linestyle=':', alpha=0.5)
    ax.set_xlabel('ML Probability Threshold (p_real)')
    ax.set_ylabel('Shadow Fraction (%)')
    ax.set_title(f'C) Shadow Fraction: Transients vs Control')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # Panel D: N candidates vs threshold
    ax = axes[1, 1]
    mdf = results_df[(results_df['model'] == '2D Cylindrical (geo)') &
                     (results_df['threshold'] != 'weighted')]
    ax.bar(mdf['threshold'].astype(float), mdf['n_retained'],
           width=0.08, color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('ML Probability Threshold (p_real)')
    ax.set_ylabel('N Candidates Retained')
    ax.set_title(f'D) Sample Size vs Threshold')
    ax.grid(True, alpha=0.3, axis='y')
    for _, row in mdf.iterrows():
        ax.annotate(f"{int(row['n_retained']):,}",
                    (row['threshold'], row['n_retained']),
                    textcoords='offset points', xytext=(0, 5),
                    ha='center', fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f'ml_threshold_shadow_3d_{PIPELINE_TAG}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    # ------------------------------------------------------------------
    # 10. Save results CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(OUTPUT_DIR, f'ml_threshold_shadow_3d_{PIPELINE_TAG}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ------------------------------------------------------------------
    # 11. Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print(f"SUMMARY ({PIPELINE_LABEL.upper()})")
    print("=" * 72)
    print(f"  Scored catalog:    {len(df):,} candidates")
    print(f"  Pipeline:          {PIPELINE_LABEL}")

    if 'prob_raw' in df.columns:
        if (df['prob'] == df['prob_raw']).all():
            print(f"  Calibration:       raw ensemble prob (calibration collapsed)")
        else:
            print(f"  Calibration:       calibrated (prob != prob_raw)")

    print(f"  Control sample:    {len(ctrl):,} points")
    print(f"  Unique obs dates:  {df['obs_datetime'].nunique()}")
    print()
    print(f"  Shadow radii at GEO:")
    print(f"    Umbra:       {GEO_UMBRA_DEG:.3f} deg")
    print(f"    Cylindrical: {GEO_CYLINDRICAL_DEG:.3f} deg")
    print(f"    Penumbra:    {GEO_PENUMBRA_DEG:.3f} deg")
    print()

    # Key findings at high thresholds
    print("  KEY FINDING — Deficit at highest thresholds:")
    for t in [0.50, 0.75, 0.85]:
        print(f"\n  p >= {t}:")
        for name in key_models:
            row = results_df[(results_df['model'] == name) &
                             (results_df['threshold'] == t)]
            if len(row) > 0:
                r = row.iloc[0]
                sig = "***" if r['p_value'] < 0.001 else (
                    "**" if r['p_value'] < 0.01 else (
                        "*" if r['p_value'] < 0.05 else "ns"))
                print(f"    {name:<25} deficit={r['deficit_pct']:>+7.1f}%  "
                      f"N={r['n_retained']:>6,}  shadow={r['n_shadow']:>4}  "
                      f"p={r['p_value']:.2e} {sig}")

    print()
    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == '__main__':
    main()
