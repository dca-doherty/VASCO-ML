#!/usr/bin/env python3
"""
Active learning candidate selector for VASCO (red all-candidates variant).
Selects candidates for human labeling from three buckets:
  1. High-confidence real (highest prob, stratified by plate_quality for diversity)
  2. Decision boundary (most uncertain)
  3. Nuclear-date hard negatives (low-prob on test dates)

Usage:
    python pipeline_active_learning.py --config config_red_allcandidates.yaml [--n-total 150]
"""

import argparse
import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, setup_logging

logger = logging.getLogger("pipeline")


def select_candidates(vasco: pd.DataFrame, merged: pd.DataFrame,
                      n_total: int = 150, cfg: dict = None,
                      existing_labels: pd.DataFrame = None) -> pd.DataFrame:
    al_cfg = (cfg or {}).get('active_learning', {})
    frac_high = al_cfg.get('frac_high_confidence', 0.40)
    frac_boundary = al_cfg.get('frac_boundary', 0.35)
    frac_nuclear = al_cfg.get('frac_nuclear_negative', 0.25)

    n_high = int(n_total * frac_high)
    n_boundary = int(n_total * frac_boundary)
    n_nuclear = n_total - n_high - n_boundary

    exclude_ids = set()
    if existing_labels is not None and 'source_id' in existing_labels.columns:
        exclude_ids = set(existing_labels['source_id'].values)
    if existing_labels is not None and 'ra' in existing_labels.columns:
        from scipy.spatial import cKDTree
        if len(existing_labels) > 0:
            lab_tree = cKDTree(existing_labels[['ra', 'dec']].values)
            dists, _ = lab_tree.query(vasco[['ra', 'dec']].values)
            exclude_ids |= set(vasco.loc[dists < 0.01, 'source_id'].values)

    pool = vasco[~vasco['source_id'].isin(exclude_ids)].copy()
    logger.info(f"Candidate pool: {len(pool):,d} (excluded {len(exclude_ids)} already labeled)")

    # Filter to actual measurements only (no median-imputed)
    if 'imputation_source' in pool.columns:
        actual_pool = pool[pool['imputation_source'] == 'actual']
        logger.info(f"  Actual-measurement pool: {len(actual_pool):,d}")
    else:
        actual_pool = pool

    nuc_dates = set()
    if merged is not None:
        nuc_col = 'Nuclear_Testing_YN_Window_Plus_Minus_1_Day'
        if nuc_col in merged.columns:
            nuc_dates = set(merged.loc[merged[nuc_col] == 1, 'date_str'].values)

    selected = []

    # Bucket 1: High-confidence real, stratified by plate_quality for diversity
    high_pool = actual_pool.sort_values('prob', ascending=False)
    strat_col = 'plate_quality' if 'plate_quality' in high_pool.columns else None

    if strat_col and high_pool[strat_col].nunique() > 1:
        classes = high_pool[strat_col].value_counts()
        per_class = max(1, n_high // len(classes))
        bucket1 = []
        for cls_name in classes.index:
            cls_pool = high_pool[high_pool[strat_col] == cls_name].head(per_class)
            bucket1.append(cls_pool)
        bucket1 = pd.concat(bucket1).sort_values('prob', ascending=False).head(n_high)
    else:
        bucket1 = high_pool.head(n_high)

    bucket1 = bucket1.copy()
    bucket1['al_bucket'] = 'high_confidence'
    bucket1['al_priority'] = range(1, len(bucket1) + 1)
    selected.append(bucket1)
    logger.info(f"  Bucket 1 (high confidence): {len(bucket1)} candidates, "
                f"prob range [{bucket1['prob'].min():.3f}, {bucket1['prob'].max():.3f}]")

    # Bucket 2: Decision boundary
    report_path = None
    if cfg:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        report_path = os.path.join(base_dir, cfg['paths']['output'], 'training_report.json')

    opt_t = 0.5
    if report_path and os.path.exists(report_path):
        with open(report_path) as f:
            opt_t = json.load(f).get('optimal_threshold', 0.5)

    already_selected = set(bucket1['source_id'].values)
    boundary_pool = actual_pool[~actual_pool['source_id'].isin(already_selected)].copy()
    boundary_pool['dist_to_threshold'] = (boundary_pool['prob'] - opt_t).abs()
    boundary_pool = boundary_pool.sort_values('dist_to_threshold')
    bucket2 = boundary_pool.head(n_boundary).copy()
    bucket2['al_bucket'] = 'boundary'
    bucket2['al_priority'] = range(1, len(bucket2) + 1)
    selected.append(bucket2)
    logger.info(f"  Bucket 2 (boundary, t={opt_t:.3f}): {len(bucket2)} candidates, "
                f"prob range [{bucket2['prob'].min():.3f}, {bucket2['prob'].max():.3f}]")

    # Bucket 3: Nuclear-date hard negatives
    already_selected |= set(bucket2['source_id'].values)
    nuc_pool = actual_pool[~actual_pool['source_id'].isin(already_selected)].copy()

    if 'obs_date' in nuc_pool.columns and nuc_dates:
        nuc_pool = nuc_pool[nuc_pool['obs_date'].astype(str).isin(nuc_dates)]
        logger.info(f"  Nuclear-date pool: {len(nuc_pool):,d} candidates")

    nuc_pool = nuc_pool.sort_values('prob', ascending=True)
    bucket3 = nuc_pool.head(n_nuclear).copy()
    bucket3['al_bucket'] = 'nuclear_hard_negative'
    bucket3['al_priority'] = range(1, len(bucket3) + 1)
    selected.append(bucket3)
    logger.info(f"  Bucket 3 (nuclear hard neg): {len(bucket3)} candidates, "
                f"prob range [{bucket3['prob'].min():.3f}, {bucket3['prob'].max():.3f}]")

    result = pd.concat(selected, ignore_index=True)

    result['label'] = ''
    result['notes'] = ''
    result['reviewed_by'] = ''
    result['review_date'] = ''

    return result


def run_active_learning(cfg: dict, n_total: int = 150) -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    output_dir = os.path.join(base_dir, cfg['paths']['output'])
    os.makedirs(output_dir, exist_ok=True)

    log = setup_logging(output_dir, "active_learning")
    log.info("=" * 60)
    log.info("ACTIVE LEARNING CANDIDATE SELECTION (RED ALL-CANDIDATES)")
    log.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Target: {n_total} candidates")
    log.info("=" * 60)

    scored_path = os.path.join(output_dir, "scored_catalog.parquet")
    if not os.path.exists(scored_path):
        scored_path = os.path.join(output_dir, "scored_catalog.csv")
    if not os.path.exists(scored_path):
        raise FileNotFoundError("Scored catalog not found. Run scoring first.")

    vasco = pd.read_parquet(scored_path) if scored_path.endswith('.parquet') else pd.read_csv(scored_path)
    log.info(f"Scored catalog: {len(vasco):,d} candidates")

    nuc_path = os.path.join(base_dir, cfg['paths']['nuclear_timeline'])
    merged = pd.read_csv(nuc_path)
    merged['date_str'] = merged['Date'].astype(str).str.strip()
    log.info(f"Nuclear timeline: {len(merged)} dates")

    existing = []
    for label_path in cfg['paths']['labels']:
        full_path = os.path.join(base_dir, label_path)
        if os.path.exists(full_path):
            if full_path.endswith('.csv'):
                df = pd.read_csv(full_path)
                if 'label' in df.columns or 'is_real_transient' in df.columns:
                    existing.append(df)
    existing_labels = pd.concat(existing, ignore_index=True) if existing else None
    if existing_labels is not None:
        log.info(f"Existing labels: {len(existing_labels)} candidates")

    result = select_candidates(vasco, merged, n_total=n_total, cfg=cfg,
                               existing_labels=existing_labels)

    # Review columns (red-only: no spectral features)
    review_cols = ['source_id', 'ra', 'dec', 'obs_date',
                   'prob', 'prob_raw', 'classification', 'plate_quality',
                   'psf_verdict', 'detection_class', 'imputation_source',
                   'snr', 'red_fits_snr', 'red_fits_fwhm',
                   'shap_top1_name', 'shap_top1_val',
                   'shap_top2_name', 'shap_top2_val',
                   'al_bucket', 'al_priority',
                   'label', 'notes', 'reviewed_by', 'review_date']
    review_cols = [c for c in review_cols if c in result.columns]

    out_path = os.path.join(output_dir, "active_learning_candidates.csv")
    result[review_cols].to_csv(out_path, index=False)
    log.info(f"\nSaved: {out_path}")

    log.info(f"\n  Selection summary:")
    for bucket in ['high_confidence', 'boundary', 'nuclear_hard_negative']:
        mask = result['al_bucket'] == bucket
        n = mask.sum()
        if n > 0:
            log.info(f"    {bucket:25s}: n={n:>4d}  "
                    f"prob=[{result.loc[mask, 'prob'].min():.3f}, "
                    f"{result.loc[mask, 'prob'].max():.3f}]")

    log.info("\nActive learning selection complete")
    return result


def main():
    parser = argparse.ArgumentParser(description="VASCO Active Learning (Red All-Candidates)")
    parser.add_argument('--config', required=True, help='Path to config_red_allcandidates.yaml')
    parser.add_argument('--n-total', type=int, default=150,
                        help='Total candidates to select (default: 150)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_active_learning(cfg, n_total=args.n_total)


if __name__ == "__main__":
    main()
