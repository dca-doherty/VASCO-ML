#!/usr/bin/env python3
"""
Model training pipeline for VASCO transient classification (red all-candidates variant).
Trains RF + GBM + XGB ensemble with CV, calibration, and SHAP analysis.
No spectral features, no blue-band features, no color features.

Usage:
    python pipeline_train.py --config config_red_allcandidates.yaml
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
from typing import Dict, List, Tuple, Any

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_config, setup_logging, encode_model_b, encode_pipeline1,
                    train_ensemble_cv, train_final_ensemble, score_with_ensemble,
                    match_batch_to_catalog, prevalence_filter,
                    select_features, compute_plate_features, tune_ensemble_hyperparams,
                    FEATURE_COLS)

logger = logging.getLogger("pipeline")


def load_labels(cfg: dict, vasco: pd.DataFrame) -> pd.DataFrame:
    """Load and merge all training label files.

    Training labels are kept unchanged (including any BOTH_BANDS candidates).
    Re-matched to catalog by RA/Dec for current feature values.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    tol_deg = cfg.get('crossmatch', {}).get('catalog_match_deg', 0.01)
    all_labeled = []

    for label_path in cfg['paths']['labels']:
        full_path = os.path.join(base_dir, label_path)
        if not os.path.exists(full_path):
            logger.warning(f"Label file not found: {full_path}")
            continue

        if full_path.endswith('.csv'):
            df = pd.read_csv(full_path)
            if 'label' in df.columns:
                logger.info(f"Loaded {full_path}: {len(df)} rows "
                           f"({(df['label']==1).sum()} pos / {(df['label']==0).sum()} neg)")
                feat_df, n_matched = match_batch_to_catalog(df, vasco, tol_deg)
                n_unmatched = len(df) - n_matched
                for col in FEATURE_COLS:
                    if col in df.columns:
                        unmatched = feat_df[col].isna() & df[col].notna()
                        if unmatched.any():
                            feat_df.loc[unmatched, col] = df.loc[unmatched, col].values
                feat_df['label'] = df['label'].values
                if 'source' in df.columns:
                    feat_df['source'] = df['source'].values
                if 'candidate_id' in df.columns:
                    feat_df['candidate_id'] = df['candidate_id'].values
                logger.info(f"  Re-matched to catalog: {n_matched}/{len(df)} "
                           f"({n_unmatched} use original CSV features)")
                matched_mask = feat_df['matched'] == True
                if n_unmatched > 0:
                    m_pos = int(df.loc[matched_mask, 'label'].sum())
                    m_neg = int((~df.loc[matched_mask, 'label'].astype(bool)).sum())
                    u_pos = int(df.loc[~matched_mask, 'label'].sum())
                    u_neg = int((~df.loc[~matched_mask, 'label'].astype(bool)).sum())
                    logger.info(f"    Matched:   {m_pos} pos / {m_neg} neg")
                    logger.info(f"    Unmatched: {u_pos} pos / {u_neg} neg (original CSV features)")
                all_labeled.append(feat_df)
            elif 'is_real_transient' in df.columns:
                df['label'] = df['is_real_transient'].astype(int)
                feat_df, n_matched = match_batch_to_catalog(df, vasco, tol_deg)
                for col in FEATURE_COLS:
                    if col in df.columns:
                        unmatched = feat_df[col].isna() & df[col].notna()
                        if unmatched.any():
                            feat_df.loc[unmatched, col] = df.loc[unmatched, col].values
                feat_df['label'] = df['label'].values
                logger.info(f"  Re-matched to catalog: {n_matched}/{len(df)}")
                all_labeled.append(feat_df)

        elif full_path.endswith('.xlsx'):
            df = pd.read_excel(full_path, sheet_name='Candidates')
            df = df.dropna(subset=['ra', 'dec']).reset_index(drop=True)
            logger.info(f"Loaded {full_path}: {len(df)} rows (Steve's batch)")

            steve_feat, n_matched = match_batch_to_catalog(df, vasco, tol_deg)
            logger.info(f"  Steve matched: {n_matched}/{len(df)}")

            steve_seed = cfg.get('steve_labels', {}).get('seed', 50)
            steve_n_pos = cfg.get('steve_labels', {}).get('n_positive', 7)
            rng = np.random.RandomState(steve_seed)
            perm = rng.permutation(len(steve_feat))
            steve_feat['label'] = 0
            steve_feat.iloc[perm[:steve_n_pos], steve_feat.columns.get_loc('label')] = 1
            steve_feat['source'] = 'steve_batch'
            all_labeled.append(steve_feat)
            logger.info(f"  Steve labels: {steve_n_pos} pos / {len(steve_feat)-steve_n_pos} neg")

    if not all_labeled:
        raise ValueError("No label files loaded. Check config.yaml paths.")

    combined = pd.concat(all_labeled, ignore_index=True)
    logger.info(f"Combined training set: {len(combined)} "
               f"({(combined['label']==1).sum()} pos / {(combined['label']==0).sum()} neg)")
    return combined


def compute_shap_values(models: dict, X: np.ndarray, feature_names: List[str],
                        output_dir: str) -> np.ndarray:
    """Compute SHAP values for the RF model and save summary plot."""
    try:
        import shap
        Xs = models['scaler'].transform(X)
        explainer = shap.TreeExplainer(models['rf'])
        shap_out = explainer(Xs)

        sv = shap_out.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        np.save(os.path.join(output_dir, "shap_values.npy"), sv)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, Xs, feature_names=feature_names, show=False, max_display=23)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "chart_shap_summary.png"), dpi=150, bbox_inches='tight')
        plt.close()

        mean_abs_shap = np.abs(sv).mean(axis=0)
        shap_imp = sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1])
        shap_df = pd.DataFrame(shap_imp, columns=['feature', 'mean_abs_shap'])
        shap_df.to_csv(os.path.join(output_dir, "feature_importance_shap.csv"), index=False)

        logger.info(f"SHAP values computed. Top 5: {[f'{n}: {v:.4f}' for n, v in shap_imp[:5]]}")
        return sv
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None


def run_training(cfg: dict, tune: bool = False, tune_iter: int = 30) -> dict:
    """Run full training pipeline."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    output_dir = os.path.join(base_dir, cfg['paths']['output'])
    cache_path = os.path.join(base_dir, cfg['paths']['feature_cache'])
    os.makedirs(output_dir, exist_ok=True)

    log = setup_logging(output_dir, "training")
    log.info("=" * 60)
    log.info("MODEL TRAINING PIPELINE (RED ALL-CANDIDATES)")
    log.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # Load feature cache
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Feature cache not found: {cache_path}. Run extraction first.")
    vasco = pd.read_parquet(cache_path)
    log.info(f"Feature cache: {len(vasco):,d} rows, {len(vasco.columns)} columns")

    # Load and merge labels (training set kept unchanged, including BOTH_BANDS)
    labels = load_labels(cfg, vasco)
    n_pos = (labels['label'] == 1).sum()
    n_neg = (labels['label'] == 0).sum()

    # Merge red FITS features from cache into training labels (no blue, no color)
    fits_cols = [c for c in vasco.columns if c.startswith('red_')]
    if fits_cols:
        from scipy.spatial import cKDTree
        cat_tree = cKDTree(vasco[['ra', 'dec']].values)
        label_coords = labels[['ra', 'dec']].values.astype(float)
        finite_mask = np.isfinite(label_coords).all(axis=1)
        if not finite_mask.all():
            log.warning(f"Dropping {(~finite_mask).sum()} labels with NaN/inf coordinates")
            labels = labels[finite_mask].reset_index(drop=True)
            label_coords = labels[['ra', 'dec']].values.astype(float)
        dists, idxs = cat_tree.query(label_coords)
        for col in fits_cols:
            labels[col] = np.nan
            mask = dists < 0.01
            labels.loc[labels.index[mask], col] = vasco.iloc[idxs[mask]][col].values
        n_fits_matched = (dists < 0.01).sum()
        log.info(f"Red FITS features merged into training labels: {n_fits_matched}/{len(labels)} "
                 f"({len(fits_cols)} columns)")

    # Compute plate-level quality features from full catalog, map to training labels
    log.info("Computing plate-level features...")
    compute_plate_features(vasco, group_col='obs_date')
    for pcol in ['plate_low_snr_frac', 'plate_snr_std', 'plate_elongation_mean', 'plate_n_high_snr']:
        if pcol in vasco.columns and 'obs_date' in labels.columns:
            plate_map = vasco.groupby('obs_date')[pcol].first()
            labels[pcol] = labels['obs_date'].map(plate_map).fillna(0).values

    # Encode features
    X_full, all_feature_names = encode_model_b(labels)
    y = labels['label'].values.astype(int)
    obs_dates = labels['obs_date'].values if 'obs_date' in labels.columns else None
    log.info(f"Encoded feature matrix: {X_full.shape[0]} samples x {X_full.shape[1]} features")

    # Encode catalog for prevalence comparison
    X_cat_full, _ = encode_model_b(vasco)

    # Apply minimum-prevalence filter
    min_prev = cfg.get('model', {}).get('min_prevalence', 0.01)
    kept_idx, feature_names, dropped = prevalence_filter(
        X_full, X_cat_full, all_feature_names, min_prevalence=min_prev)
    X = X_full[:, kept_idx]
    if dropped:
        log.info(f"\n  Prevalence filter (min={min_prev:.0%}): dropped {len(dropped)} features:")
        for d in dropped:
            log.info(f"    - {d}")
    log.info(f"Final feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    log.info(f"Features: {feature_names}")

    # Feature selection
    fs_cfg = cfg.get('model', {}).get('feature_selection', {})
    fs_max = fs_cfg.get('max_features', 0)
    fs_ratio = fs_cfg.get('min_ratio', 10.0)
    fs_method = fs_cfg.get('method', 'mutual_info')
    sel_idx, feature_names, fs_dropped = select_features(
        X, y, feature_names, max_features=fs_max, min_ratio=fs_ratio, method=fs_method)
    if fs_dropped:
        log.info(f"\n  Feature selection dropped {len(fs_dropped)} features:")
        for d in fs_dropped:
            log.info(f"    - {d}")
        kept_idx = [kept_idx[i] for i in sel_idx]
        X = X[:, sel_idx]
    log.info(f"Training matrix: {X.shape[0]} samples x {X.shape[1]} features")

    # Hyperparameter tuning (optional)
    if tune:
        log.info(f"\nHyperparameter tuning ({tune_iter} iters per model)...")
        cfg = tune_ensemble_hyperparams(X, y, cfg, n_iter=tune_iter)
        log.info(f"  Tuned RF:  {cfg['model']['rf']}")
        log.info(f"  Tuned GBM: {cfg['model']['gbm']}")
        log.info(f"  Tuned XGB: {cfg['model']['xgb']}")

    # Cross-validation
    log.info(f"\nCross-validation ({cfg['cv']['split_method']}, {cfg['cv']['n_folds']} folds)...")
    cv_result = train_ensemble_cv(X, y, cfg, obs_dates)

    fold_df = pd.DataFrame(cv_result['fold_metrics'])
    log.info(f"\n  Per-fold metrics:")
    for _, row in fold_df.iterrows():
        log.info(f"    Fold {int(row['fold'])}: AUC={row['auc']:.3f} Sens={row['sens']:.3f} "
                f"Spec={row['spec']:.3f} F1={row['f1']:.3f}")

    mean_auc = fold_df['auc'].mean()
    mean_sens = fold_df['sens'].mean()
    mean_spec = fold_df['spec'].mean()
    mean_f1 = fold_df['f1'].mean()
    log.info(f"\n  Mean: AUC={mean_auc:.3f} +/- {fold_df['auc'].std():.3f}")
    log.info(f"        Sens={mean_sens:.3f} Spec={mean_spec:.3f} F1={mean_f1:.3f}")

    brier = brier_score_loss(y, cv_result['probs'])
    log.info(f"  Brier score: {brier:.4f}")

    # Feature importance
    imp = cv_result['importances'].mean(axis=0)
    imp_sorted = sorted(zip(feature_names, imp), key=lambda x: -x[1])
    log.info(f"\n  Feature importance (top 10):")
    for name, val in imp_sorted[:10]:
        log.info(f"    {name:30s}: {val:.4f}")

    imp_df = pd.DataFrame(imp_sorted, columns=['feature', 'rf_importance'])
    imp_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    # Train final model
    log.info(f"\nTraining final model on all {len(y)} samples...")
    final_models = train_final_ensemble(X, y, cfg)

    model_path = os.path.join(output_dir, "model_final.joblib")
    joblib.dump({
        'models': final_models,
        'feature_names': feature_names,
        'kept_feature_idx': kept_idx,
        'dropped_features': dropped + fs_dropped,
    }, model_path)
    log.info(f"Model saved: {model_path}")

    # Calibration
    log.info("\nCalibrating probabilities...")
    oof_probs = cv_result['probs']
    cal_method = cfg['model'].get('calibration', 'sigmoid')

    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    if cal_method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(oof_probs, y)
    else:
        calibrator = LogisticRegression()
        calibrator.fit(oof_probs.reshape(-1, 1), y)

    cal_path = os.path.join(output_dir, "calibrator.joblib")
    joblib.dump(calibrator, cal_path)
    log.info(f"Calibrator saved: {cal_path}")

    # SHAP
    log.info("\nComputing SHAP values...")
    shap_vals = compute_shap_values(final_models, X, feature_names, output_dir)

    # Charts
    log.info("\nGenerating diagnostic charts...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Training Diagnostics (Red All-Candidates) -- AUC={mean_auc:.3f}, {len(y)} samples",
                 fontsize=14, fontweight='bold')

    ax1 = axes[0, 0]
    fpr, tpr, _ = roc_curve(y, oof_probs)
    ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC={mean_auc:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('FPR')
    ax1.set_ylabel('TPR')
    ax1.set_title('ROC Curve (OOF)')
    ax1.legend()

    ax2 = axes[0, 1]
    try:
        prob_true, prob_pred = calibration_curve(y, oof_probs, n_bins=8)
        ax2.plot(prob_pred, prob_true, 'bo-', label='Uncalibrated')
        if cal_method == 'isotonic':
            cal_probs = calibrator.predict(oof_probs)
        else:
            cal_probs = calibrator.predict_proba(oof_probs.reshape(-1, 1))[:, 1]
        prob_true_c, prob_pred_c = calibration_curve(y, cal_probs, n_bins=8)
        ax2.plot(prob_pred_c, prob_true_c, 'rs-', label='Calibrated')
    except Exception:
        pass
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('Mean predicted')
    ax2.set_ylabel('Fraction positive')
    ax2.set_title('Calibration Curve')
    ax2.legend()

    ax3 = axes[0, 2]
    j_scores = tpr - fpr
    opt_idx = np.argmax(j_scores)
    thresholds = roc_curve(y, oof_probs)[2]
    opt_t = thresholds[min(opt_idx, len(thresholds) - 1)] if len(thresholds) > 0 else 0.5
    if opt_t > 1.0 or opt_t < 0.0:
        opt_t = 0.5
    preds_opt = (oof_probs >= opt_t).astype(int)
    cm = confusion_matrix(y, preds_opt, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Artifact', 'Real'], yticklabels=['Artifact', 'Real'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title(f'Confusion Matrix (t={opt_t:.2f})')

    ax4 = axes[1, 0]
    top15 = imp_sorted[:15]
    ax4.barh(range(len(top15)), [v for _, v in top15][::-1], color='steelblue')
    ax4.set_yticks(range(len(top15)))
    ax4.set_yticklabels([n for n, _ in top15][::-1], fontsize=8)
    ax4.set_xlabel('Importance')
    ax4.set_title('Top 15 Features (RF)')

    ax5 = axes[1, 1]
    ax5.hist(oof_probs[y == 1], bins=30, alpha=0.6, label=f'Real (n={n_pos})', color='green', density=True)
    ax5.hist(oof_probs[y == 0], bins=30, alpha=0.6, label=f'Artifact (n={n_neg})', color='red', density=True)
    ax5.axvline(x=opt_t, color='black', linestyle='--', label=f'Optimal t={opt_t:.2f}')
    ax5.set_xlabel('Predicted Probability')
    ax5.set_ylabel('Density')
    ax5.set_title('OOF Probability Distribution')
    ax5.legend()

    ax6 = axes[1, 2]
    top10_idx = [feature_names.index(n) for n, _ in imp_sorted[:10] if n in feature_names]
    if len(top10_idx) >= 2:
        corr_data = pd.DataFrame(X[:, top10_idx], columns=[feature_names[i] for i in top10_idx])
        corr_mat = corr_data.corr()
        sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    ax=ax6, xticklabels=True, yticklabels=True)
        ax6.tick_params(labelsize=7)
    ax6.set_title('Feature Correlation (top 10)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart_path = os.path.join(output_dir, "chart_training_diagnostics.png")
    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Charts saved: {chart_path}")

    # Save training report
    report = {
        'date': datetime.now().isoformat(),
        'pipeline_variant': 'red_allcandidates',
        'n_samples': len(y), 'n_positive': int(n_pos), 'n_negative': int(n_neg),
        'n_features': len(feature_names), 'feature_names': feature_names,
        'dropped_features': dropped + fs_dropped,
        'n_dropped_prevalence': len(dropped), 'n_dropped_selection': len(fs_dropped),
        'cv_method': cfg['cv']['split_method'], 'n_folds': cfg['cv']['n_folds'],
        'mean_auc': float(mean_auc), 'std_auc': float(fold_df['auc'].std()),
        'mean_sensitivity': float(mean_sens), 'mean_specificity': float(mean_spec),
        'mean_f1': float(mean_f1), 'brier_score': float(brier),
        'optimal_threshold': float(opt_t),
        'calibration_method': cal_method,
        'tuned': tune,
        'confusion_matrix': cm.tolist(),
    }
    if tune:
        report['tuned_params'] = {k: cfg['model'][k] for k in ('rf', 'gbm', 'xgb')}
    with open(os.path.join(output_dir, "training_report.json"), 'w') as f:
        json.dump(report, f, indent=2)

    fold_df.to_csv(os.path.join(output_dir, "cv_fold_results.csv"), index=False)

    log.info("\nTraining complete")
    return report


def main():
    parser = argparse.ArgumentParser(description="VASCO Model Training (Red All-Candidates)")
    parser.add_argument('--config', required=True, help='Path to config_red_allcandidates.yaml')
    parser.add_argument('--tune', action='store_true',
                        help='Run RandomizedSearchCV on each ensemble model before training')
    parser.add_argument('--tune-iter', type=int, default=30,
                        help='Number of RandomizedSearchCV iterations per model (default: 30)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_training(cfg, tune=args.tune, tune_iter=args.tune_iter)


if __name__ == "__main__":
    main()
