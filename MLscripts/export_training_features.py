#!/usr/bin/env python3
"""
export_training_features.py
Reconstructs and saves the training feature matrix using pipeline internals.
Run once before classifier_diagnostics.py.

Usage:
    python scripts/pipeline2/pipeline_red_allcandidates/export_training_features.py \
        --config scripts/pipeline2/pipeline_red_allcandidates/config_red_allcandidates.yaml
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_config, encode_model_b, prevalence_filter,
                    select_features, compute_plate_features, FEATURE_COLS)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
                   os.path.dirname(os.path.abspath(__file__)))))
    output_dir = os.path.join(base_dir, cfg['paths']['output'])
    cache_path = os.path.join(base_dir, cfg['paths']['feature_cache'])

    print(f"Loading feature cache: {cache_path}")
    vasco = pd.read_parquet(cache_path)

    # Inline import of load_labels from pipeline_train
    from pipeline_train import load_labels
    labels = load_labels(cfg, vasco)
    y = labels['label'].values.astype(int)

    # Merge red FITS features
    from scipy.spatial import cKDTree
    fits_cols = [c for c in vasco.columns if c.startswith('red_')]
    if fits_cols:
        cat_tree = cKDTree(vasco[['ra', 'dec']].values)
        dists, idxs = cat_tree.query(labels[['ra', 'dec']].values.astype(float))
        for col in fits_cols:
            labels[col] = np.nan
            mask = dists < 0.01
            labels.loc[labels.index[mask], col] = vasco.iloc[idxs[mask]][col].values

    # Plate features
    compute_plate_features(vasco, group_col='obs_date')
    for pcol in ['plate_low_snr_frac', 'plate_snr_std', 'plate_elongation_mean', 'plate_n_high_snr']:
        if pcol in vasco.columns and 'obs_date' in labels.columns:
            plate_map = vasco.groupby('obs_date')[pcol].first()
            labels[pcol] = labels['obs_date'].map(plate_map).fillna(0).values

    X_full, all_feature_names = encode_model_b(labels)
    X_cat_full, _ = encode_model_b(vasco)
    min_prev = cfg.get('model', {}).get('min_prevalence', 0.01)
    kept_idx, feature_names, dropped = prevalence_filter(
        X_full, X_cat_full, all_feature_names, min_prevalence=min_prev)
    X = X_full[:, kept_idx]

    fs_cfg = cfg.get('model', {}).get('feature_selection', {})
    sel_idx, feature_names, _ = select_features(
        X, y, feature_names,
        max_features=fs_cfg.get('max_features', 0),
        min_ratio=fs_cfg.get('min_ratio', 10.0),
        method=fs_cfg.get('method', 'mutual_info'))
    X = X[:, sel_idx]

    out_df = pd.DataFrame(X, columns=feature_names)
    out_df['label'] = y
    out_path = os.path.join(output_dir, 'training_features.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({out_df.shape[0]} rows x {out_df.shape[1]} cols)")

if __name__ == '__main__':
    main()