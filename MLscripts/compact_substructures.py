#!/usr/bin/env python3
"""
compact_substructures.py
Identify compact groupings of high-probability transients on the same plate.

Reproduces the paper's finding for Decile 9: doublets (pairs), triplets, and
larger groupings within a small angular separation on the same observation date.

Usage:
    python compact_substructures.py
"""

import os
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ============================================================================
# CONFIG
# ============================================================================
CATALOG_PATH    = "results/ml_pipeline_red_allcandidates/scored_catalog.xlsb"
OUTPUT_CSV      = "results/compact_substructures.csv"
OUTPUT_SUMMARY  = "results/compact_substructures_summary.csv"

DECILE          = 9              # highest-probability decile
MAX_SEP_ARCMIN  = 10.0           # paper uses 10 arcmin same-night grouping
MIN_GROUP_SIZE  = 2
# ============================================================================


def haversine_deg(ra1, dec1, ra2, dec2):
    """Haversine angular separation in degrees (vectorized)."""
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = np.sin(ddec / 2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2)**2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def union_find_cluster(ra_arr, dec_arr, max_sep_deg):
    """Simple O(n^2) union-find clustering within max_sep_deg.
    For per-night groups (usually << 1000 points) this is adequate."""
    n = len(ra_arr)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_deg(ra_arr[i], dec_arr[i], ra_arr[j], dec_arr[j])
            if d <= max_sep_deg:
                union(i, j)
    return np.array([find(i) for i in range(n)])


def parse_obs_dates_to_str(df, col='obs_date'):
    s = df[col].iloc[0]
    if isinstance(s, (int, float, np.integer, np.floating)):
        base = datetime(1899, 12, 30)
        return df[col].apply(lambda x: (base + timedelta(days=int(x))).strftime('%Y-%m-%d'))
    return pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV) or '.', exist_ok=True)
    print("=" * 70)
    print(f"COMPACT SUBSTRUCTURES  (Decile {DECILE}, separation <= {MAX_SEP_ARCMIN} arcmin)")
    print("=" * 70)

    # Load catalog
    if CATALOG_PATH.lower().endswith('.xlsb'):
        df = pd.read_excel(CATALOG_PATH, engine='pyxlsb')
    else:
        df = pd.read_csv(CATALOG_PATH)
    df['obs_date_str'] = parse_obs_dates_to_str(df)
    print(f"Loaded {len(df):,} candidates on {df['obs_date_str'].nunique()} dates")

    # Filter to strict POSS-I dates (paper N=99,933)
    _poss = pd.read_csv('data/poss_i_observing_dates_strict.csv')
    _poss_set = set(pd.to_datetime(_poss['obs_date']).dt.strftime('%Y-%m-%d'))
    df = df[df['obs_date_str'].isin(_poss_set)].reset_index(drop=True)
    print(f"After POSS-I filter: {len(df):,} candidates")

    # Decile assignment (same rule as decile scripts)
    df['decile'] = pd.qcut(df['prob'].rank(method='first'), 10, labels=False).astype(int)
    sub = df[df['decile'] == DECILE].copy().reset_index(drop=True)
    print(f"Decile {DECILE}: {len(sub):,} transients")

    # Cluster per observation date
    max_sep_deg = MAX_SEP_ARCMIN / 60.0
    struct_records = []
    summary_counts = {'doublets': 0, 'triplets': 0, 'larger': 0}
    next_struct_id = 1

    for date, grp in sub.groupby('obs_date_str'):
        if len(grp) < MIN_GROUP_SIZE:
            continue
        ra = grp['ra'].values.astype(float)
        dec = grp['dec'].values.astype(float)
        labels = union_find_cluster(ra, dec, max_sep_deg)

        n_on_plate = len(grp)
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            size = int(mask.sum())
            if size < MIN_GROUP_SIZE:
                continue
            members = grp.iloc[np.where(mask)[0]].reset_index(drop=True)

            # Span (max pairwise separation, arcmin)
            span = 0.0
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    d = haversine_deg(members.loc[i, 'ra'], members.loc[i, 'dec'],
                                      members.loc[j, 'ra'], members.loc[j, 'dec'])
                    span = max(span, d * 60.0)

            # Count close pairs (within max_sep_deg)
            close_pairs = 0
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    d = haversine_deg(members.loc[i, 'ra'], members.loc[i, 'dec'],
                                      members.loc[j, 'ra'], members.loc[j, 'dec'])
                    if d <= max_sep_deg:
                        close_pairs += 1

            mean_prob = float(members['prob'].mean())

            if size == 2:
                summary_counts['doublets'] += 1
            elif size == 3:
                summary_counts['triplets'] += 1
            else:
                summary_counts['larger'] += 1

            for k, row in members.iterrows():
                struct_records.append({
                    'structure_id': next_struct_id,
                    'date': date,
                    'n_total': size,
                    'n_close_pairs': close_pairs,
                    'span_arcmin': round(span, 2),
                    'n_on_plate': n_on_plate,
                    'mean_prob': round(mean_prob, 3),
                    'source_num': k + 1,
                    'ra': float(row['ra']),
                    'dec': float(row['dec']),
                    'prob': round(float(row['prob']), 3),
                })
            next_struct_id += 1

    out = pd.DataFrame(struct_records)
    out.to_csv(OUTPUT_CSV, index=False)

    summary = pd.DataFrame([
        {'structure_type': 'doublets',        'count': summary_counts['doublets']},
        {'structure_type': 'triplets',        'count': summary_counts['triplets']},
        {'structure_type': 'larger_grouping', 'count': summary_counts['larger']},
    ])
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print()
    print("Summary:")
    print(f"  Doublets (pairs):       {summary_counts['doublets']}")
    print(f"  Triplets:               {summary_counts['triplets']}")
    print(f"  Larger groupings (>=4): {summary_counts['larger']}")
    print()
    print(f"Saved: {OUTPUT_CSV}  ({len(out)} member rows)")
    print(f"Saved: {OUTPUT_SUMMARY}")


if __name__ == '__main__':
    main()

