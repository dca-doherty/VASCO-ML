#!/usr/bin/env python3
"""
Scoring pipeline for VASCO transient classification (red all-candidates variant).
Scores the filtered catalog with the trained ensemble, applies calibration,
and computes per-candidate SHAP top-3 features.

Usage:
    python pipeline_score.py --config config_red_allcandidates.yaml
"""

import argparse
import os
import sys
import logging
import warnings
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_config, setup_logging, encode_model_b, score_with_ensemble,
                    bimodality_coefficient, compute_plate_features)

logger = logging.getLogger("pipeline")


def compute_per_candidate_shap(models: dict, X: np.ndarray,
                                feature_names: List[str],
                                top_k: int = 3) -> pd.DataFrame:
    """Compute per-candidate SHAP top-k features using RF TreeExplainer."""
    try:
        import shap
        Xs = models['scaler'].transform(X)
        explainer = shap.TreeExplainer(models['rf'])
        shap_out = explainer(Xs)
        sv = shap_out.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        rows = []
        for i in range(sv.shape[0]):
            row_sv = sv[i]
            abs_sv = np.abs(row_sv)
            top_idx = np.argsort(-abs_sv)[:top_k]
            entry = {}
            for k, idx in enumerate(top_idx):
                entry[f'shap_top{k+1}_name'] = feature_names[idx]
                entry[f'shap_top{k+1}_val'] = float(row_sv[idx])
            rows.append(entry)
        return pd.DataFrame(rows)
    except Exception as e:
        logger.warning(f"Per-candidate SHAP failed: {e}")
        cols = {}
        for k in range(top_k):
            cols[f'shap_top{k+1}_name'] = [''] * X.shape[0]
            cols[f'shap_top{k+1}_val'] = [np.nan] * X.shape[0]
        return pd.DataFrame(cols)


def run_scoring(cfg: dict) -> pd.DataFrame:
    """Run full scoring pipeline on the filtered catalog."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    output_dir = os.path.join(base_dir, cfg['paths']['output'])
    cache_path = os.path.join(base_dir, cfg['paths']['feature_cache'])
    os.makedirs(output_dir, exist_ok=True)

    log = setup_logging(output_dir, "scoring")
    log.info("=" * 60)
    log.info("SCORING PIPELINE (RED ALL-CANDIDATES)")
    log.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # Load feature cache
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Feature cache not found: {cache_path}. Run extraction first.")
    vasco = pd.read_parquet(cache_path)
    log.info(f"Feature cache: {len(vasco):,d} rows, {len(vasco.columns)} columns")

    # Load trained model
    model_path = os.path.join(output_dir, "model_final.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")
    model_bundle = joblib.load(model_path)
    models = model_bundle['models']
    feature_names = model_bundle['feature_names']
    kept_idx = model_bundle.get('kept_feature_idx', None)
    dropped = model_bundle.get('dropped_features', [])
    log.info(f"Model loaded: {len(feature_names)} features")
    if dropped:
        log.info(f"  Dropped by prevalence/selection filter: {dropped}")

    # Load calibrator
    cal_path = os.path.join(output_dir, "calibrator.joblib")
    calibrator = None
    if os.path.exists(cal_path):
        calibrator = joblib.load(cal_path)
        log.info("Calibrator loaded")

    # Encode features
    log.info("Computing plate-level features...")
    compute_plate_features(vasco, group_col='obs_date')

    log.info("Encoding features...")
    X_full, all_names = encode_model_b(vasco)
    if kept_idx is not None:
        X = X_full[:, kept_idx]
        names = [all_names[i] for i in kept_idx]
    else:
        X = X_full
        names = all_names
    assert names == feature_names, f"Feature mismatch: {names} vs {feature_names}"
    log.info(f"Feature matrix: {X.shape[0]:,d} x {X.shape[1]}")

    # Score
    log.info("Scoring all candidates...")
    raw_probs = score_with_ensemble(models, X)

    # Proxy shrinkage: median-imputed candidates (no FITS data) get shrunk
    if 'imputation_source' in vasco.columns:
        imputed_mask = (vasco['imputation_source'] == 'median_imputed').values
        n_imputed = imputed_mask.sum()
        if n_imputed > 0:
            shrinkage = cfg.get('model', {}).get('proxy_shrinkage', 0.7)
            actual_median = float(np.median(raw_probs[~imputed_mask]))
            imputed_before = raw_probs[imputed_mask].mean()
            raw_probs[imputed_mask] = shrinkage * actual_median + (1 - shrinkage) * raw_probs[imputed_mask]
            imputed_after = raw_probs[imputed_mask].mean()
            log.info(f"  Imputation shrinkage (alpha={shrinkage:.2f}): {n_imputed:,d} candidates, "
                    f"mean {imputed_before:.4f} -> {imputed_after:.4f} "
                    f"(actual median: {actual_median:.4f})")

    vasco['prob_raw'] = raw_probs

    # Calibrate
    use_calibrated = False
    if calibrator is not None:
        log.info("Applying calibration...")
        from sklearn.isotonic import IsotonicRegression
        if isinstance(calibrator, IsotonicRegression):
            cal_probs = calibrator.predict(raw_probs)
        else:
            cal_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

        raw_unique = len(np.unique(np.round(raw_probs, 4)))
        cal_unique = len(np.unique(np.round(cal_probs, 4)))
        resolution_ratio = cal_unique / max(raw_unique, 1)
        top3_frac = sum(sorted(pd.Series(np.round(cal_probs, 6)).value_counts().values,
                               reverse=True)[:3]) / len(cal_probs)
        log.info(f"  Raw unique values: {raw_unique:,d}, Calibrated: {cal_unique:,d} "
                f"(resolution ratio: {resolution_ratio:.2%}, top-3 coverage: {top3_frac:.1%})")

        if np.std(cal_probs) < 1e-6:
            log.warning("Calibration is degenerate (constant output). Using raw probabilities.")
        elif top3_frac > 0.50:
            log.warning(f"Calibration collapsed distribution: top 3 values cover "
                       f"{top3_frac:.1%}. Using raw probabilities to preserve discrimination.")
        else:
            use_calibrated = True
            log.info(f"Calibrated prob range: [{cal_probs.min():.4f}, {cal_probs.max():.4f}]")

    if use_calibrated:
        vasco['prob'] = cal_probs
        vasco['prob_calibrated'] = cal_probs
    else:
        vasco['prob'] = raw_probs
        if calibrator is not None:
            vasco['prob_calibrated'] = cal_probs
            log.info("Using raw ensemble probabilities as primary scores.")
        else:
            log.info("No calibrator -- using raw probabilities")

    # Optimal threshold
    report_path = os.path.join(output_dir, "training_report.json")
    opt_t = 0.5
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        opt_t = report.get('optimal_threshold', 0.5)
    vasco['prediction'] = (vasco['prob'] >= opt_t).astype(int)
    log.info(f"Threshold: {opt_t:.3f}")

    # Summary by classification
    log.info("\n  Probability by classification:")
    for cls in ['EXCELLENT', 'GOOD', 'MARGINAL', 'POOR', 'UNCERTAIN']:
        mask = vasco['classification'] == cls
        if mask.sum() > 0:
            log.info(f"    {cls:15s}: n={mask.sum():>7,d}  "
                    f"mean={vasco.loc[mask, 'prob'].mean():.4f}  "
                    f"median={vasco.loc[mask, 'prob'].median():.4f}")

    # Summary by imputation source
    if 'imputation_source' in vasco.columns:
        log.info("\n  Probability by imputation source:")
        for src in ['actual', 'median_imputed']:
            mask = vasco['imputation_source'] == src
            if mask.sum() > 0:
                log.info(f"    {src:15s}: n={mask.sum():>7,d}  "
                        f"mean={vasco.loc[mask, 'prob'].mean():.4f}  "
                        f"median={vasco.loc[mask, 'prob'].median():.4f}")

    # Bimodality
    bc = bimodality_coefficient(vasco['prob'].values)
    log.info(f"\n  Bimodality coefficient: {bc:.4f} {'(bimodal)' if bc > 0.555 else '(unimodal)'}")

    # Per-candidate SHAP top-3
    log.info("\nComputing per-candidate SHAP top-3 features...")
    shap_df = compute_per_candidate_shap(models, X, feature_names, top_k=3)
    for col in shap_df.columns:
        vasco[col] = shap_df[col].values

    # Charts
    log.info("Generating scoring charts...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Scoring Results (Red All-Candidates) -- {len(vasco):,d} candidates, threshold={opt_t:.2f}",
                 fontsize=13, fontweight='bold')

    ax1 = axes[0]
    colors = {'EXCELLENT': 'green', 'GOOD': 'blue', 'MARGINAL': 'orange',
              'POOR': 'red', 'UNCERTAIN': 'gray'}
    for cls in ['EXCELLENT', 'GOOD', 'MARGINAL', 'POOR', 'UNCERTAIN']:
        mask = vasco['classification'] == cls
        if mask.sum() > 0:
            ax1.hist(vasco.loc[mask, 'prob'], bins=50, alpha=0.5,
                    label=f"{cls} (n={mask.sum():,d})", color=colors.get(cls, 'gray'))
    ax1.axvline(x=opt_t, color='black', linestyle='--', linewidth=1.5, label=f't={opt_t:.2f}')
    ax1.set_xlabel('Probability')
    ax1.set_ylabel('Count')
    ax1.set_title('Probability by Classification')
    ax1.legend(fontsize=7)

    ax2 = axes[1]
    imp_colors = {'actual': 'blue', 'median_imputed': 'orange'}
    if 'imputation_source' in vasco.columns:
        for src in ['actual', 'median_imputed']:
            mask = vasco['imputation_source'] == src
            if mask.sum() > 0:
                ax2.hist(vasco.loc[mask, 'prob'], bins=50, alpha=0.5,
                        label=f"{src} (n={mask.sum():,d})", color=imp_colors.get(src, 'gray'))
    ax2.axvline(x=opt_t, color='black', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Probability by Imputation Source')
    ax2.legend(fontsize=7)

    ax3 = axes[2]
    if calibrator is not None:
        ax3.scatter(vasco['prob_raw'], vasco['prob'], s=0.1, alpha=0.1, color='blue')
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax3.set_xlabel('Raw Probability')
        ax3.set_ylabel('Calibrated Probability')
        ax3.set_title('Calibration Effect')
    else:
        ax3.text(0.5, 0.5, 'No calibrator', ha='center', va='center', fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    chart_path = os.path.join(output_dir, "chart_scoring.png")
    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Chart saved: {chart_path}")

    # Output columns (red-only: no p2_*, no spectral, no blue, no Bruehl)
    out_cols = ['source_id', 'ra', 'dec', 'obs_date', 'classification', 'classification_orig',
                'psf_verdict', 'plate_quality', 'diff_verdict',
                'snr', 'psf_ratio', 'elongation', 'compactness', 'sharpness',
                'candidate_score', 'in_red_only', 'in_blue_only',
                'n_comparison_stars',
                'gal_lat', 'gal_lon', 'dist_from_center_deg',
                'plate_n_candidates', 'plate_median_snr',
                'imputation_source', 'detection_class',
                # Red FITS morphometry
                'red_fits_snr', 'red_fits_fwhm', 'red_ellipticity', 'red_sharpness_2nd',
                'red_n_connected_px', 'red_aperture_flux', 'red_dist_to_edge_px',
                'red_symmetry_score', 'red_gradient_magnitude', 'red_near_bright_star',
                'prob_raw', 'prob', 'prediction',
                'shap_top1_name', 'shap_top1_val',
                'shap_top2_name', 'shap_top2_val',
                'shap_top3_name', 'shap_top3_val']
    out_cols = [c for c in out_cols if c in vasco.columns]

    catalog_path = os.path.join(output_dir, "scored_catalog.csv")
    vasco[out_cols].to_csv(catalog_path, index=False)
    log.info(f"Scored catalog saved: {catalog_path} ({len(out_cols)} columns)")

    vasco[out_cols].to_parquet(os.path.join(output_dir, "scored_catalog.parquet"), index=False)

    n_above = (vasco['prob'] >= opt_t).sum()
    n_below = (vasco['prob'] < opt_t).sum()
    log.info(f"\n  Summary:")
    log.info(f"    Above threshold (predicted real): {n_above:,d} ({100*n_above/len(vasco):.1f}%)")
    log.info(f"    Below threshold (predicted artifact): {n_below:,d} ({100*n_below/len(vasco):.1f}%)")
    log.info(f"    Mean probability: {vasco['prob'].mean():.4f}")
    log.info(f"    Median probability: {vasco['prob'].median():.4f}")

    log.info("\nScoring complete")
    return vasco


def main():
    parser = argparse.ArgumentParser(description="VASCO Scoring Pipeline (Red All-Candidates)")
    parser.add_argument('--config', required=True, help='Path to config_red_allcandidates.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_scoring(cfg)


if __name__ == "__main__":
    main()
