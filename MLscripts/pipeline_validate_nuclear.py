#!/usr/bin/env python3
"""
Nuclear association validation pipeline for VASCO (red all-candidates variant).
Runs threshold sweep, quintile analysis, D-1 anticipatory signal,
probability-weighted regression, and three-way comparison.

Spectral class breakdown is removed (no spectral features in red-only pipeline).

Usage:
    python pipeline_validate_nuclear.py --config config_red_allcandidates.yaml
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
import seaborn as sns

from scipy import stats as scipy_stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_config, setup_logging, run_nb_glm, run_nuclear_threshold,
                    bimodality_coefficient)

logger = logging.getLogger("pipeline")


def load_nuclear_timeline(cfg: dict, base_dir: str) -> pd.DataFrame:
    nuc_path = os.path.join(base_dir, cfg['paths']['nuclear_timeline'])
    merged = pd.read_csv(nuc_path)
    merged['date_str'] = merged['Date'].astype(str).str.strip()
    obs_dates_path = os.path.join(base_dir, cfg['paths'].get('observation_dates', ''))
    if obs_dates_path and os.path.exists(obs_dates_path):
        obs = pd.read_csv(obs_dates_path)
        obs_set = set(obs['obs_date'].astype(str).str.strip())
        merged = merged[merged['date_str'].isin(obs_set)]
    return merged


def threshold_sweep(vasco: pd.DataFrame, merged: pd.DataFrame,
                    cfg: dict, log: logging.Logger) -> pd.DataFrame:
    n_thresh = cfg['validation']['n_thresholds']
    t_min = cfg['validation']['threshold_min']
    t_max = cfg['validation']['threshold_max']
    thresholds = np.linspace(t_min, t_max, n_thresh)

    log.info(f"\nThreshold sweep: {n_thresh} thresholds from {t_min} to {t_max}")
    results = []

    for t in thresholds:
        result = run_nuclear_threshold(vasco, merged, threshold=t)
        irr = result.get('irr', float('nan'))
        irr_lo = result.get('irr_lo', float('nan'))
        irr_hi = result.get('irr_hi', float('nan'))
        fisher = result.get('fisher_or', float('nan'))
        pval = result.get('pval', float('nan'))
        sig = '*' if not np.isnan(pval) and pval < 0.05 else ' '
        n = result['n_retained']
        log.info(f"  t={t:.3f}: N={n:>7,d}  IRR={irr:>6.3f}{sig} "
                f"({irr_lo:.2f}-{irr_hi:.2f})  Fisher={fisher:.3f}")
        results.append({
            'threshold': t, 'n_retained': n,
            'pct_rejected': result.get('pct_rejected', 0),
            'irr': irr, 'irr_lo': irr_lo, 'irr_hi': irr_hi,
            'fisher_or': fisher, 'fisher_pval': result.get('fisher_pval', float('nan')),
            'nb_pval': pval,
            'n_dates': result.get('n_dates', 0),
        })

    sweep_df = pd.DataFrame(results)

    valid_irr = sweep_df.dropna(subset=['irr'])
    valid_irr = valid_irr[valid_irr['irr'] > 0]
    if len(valid_irr) > 1:
        irr_vals = valid_irr['irr'].values
        n_inc = sum(1 for i in range(1, len(irr_vals)) if irr_vals[i] > irr_vals[i-1])
        n_steps = len(irr_vals) - 1
        pct_mono = 100 * n_inc / n_steps if n_steps > 0 else 0
        log.info(f"\n  Monotonicity: {n_inc}/{n_steps} steps increasing ({pct_mono:.0f}%)")
    else:
        pct_mono = 0

    sweep_df.attrs['monotonicity_pct'] = pct_mono
    return sweep_df


def quintile_analysis(vasco: pd.DataFrame, merged: pd.DataFrame,
                      log: logging.Logger) -> pd.DataFrame:
    log.info("\nQuintile analysis:")

    vasco_copy = vasco.copy()
    vasco_copy['prob_rank'] = vasco_copy['prob'].rank(method='first')
    vasco_copy['quintile'] = pd.qcut(vasco_copy['prob_rank'], q=5,
                                      labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)'])

    results = []
    for q_label in ['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)']:
        q_data = vasco_copy[vasco_copy['quintile'] == q_label]
        prob_lo = q_data['prob'].min()
        prob_hi = q_data['prob'].max()

        fc = q_data.groupby('obs_date').size().to_dict()
        m = merged.copy()
        m['count'] = m['date_str'].map(fc).fillna(0).astype(int)
        mdf = m[['count', 'Nuclear_Testing_YN_Window_Plus_Minus_1_Day',
                  'moon_illumination', 'cloud_cover_estimate', 'PRCP']].dropna()

        nb = run_nb_glm(mdf)
        irr = nb['irr']
        irr_lo = nb['irr_lo']
        irr_hi = nb['irr_hi']
        pval = nb['pval']
        sig = '*' if not np.isnan(pval) and pval < 0.05 else ' '

        log.info(f"  {q_label}: n={len(q_data):>7,d}  "
                f"prob=[{prob_lo:.3f},{prob_hi:.3f}]  "
                f"IRR={irr:.3f}{sig} ({irr_lo:.2f}-{irr_hi:.2f})")

        results.append({
            'quintile': q_label, 'n': len(q_data),
            'prob_min': prob_lo, 'prob_max': prob_hi,
            'prob_mean': q_data['prob'].mean(),
            'irr': irr, 'irr_lo': irr_lo, 'irr_hi': irr_hi,
            'pval': pval,
        })

    return pd.DataFrame(results)


def d_minus_1_analysis(vasco: pd.DataFrame, merged: pd.DataFrame,
                       log: logging.Logger) -> dict:
    log.info("\nD-1 anticipatory signal analysis:")

    if 'One_Day_BEFORE_Nuclear_Testing_YN' not in merged.columns:
        log.warning("  No D-1 column in nuclear timeline. Skipping.")
        return {}

    fc = vasco.groupby('obs_date').size().to_dict()
    m = merged.copy()
    m['count'] = m['date_str'].map(fc).fillna(0).astype(int)

    has_t = (m['count'] > 0).astype(int)
    d1 = m['One_Day_BEFORE_Nuclear_Testing_YN'].fillna(0).astype(int)
    ct = pd.crosstab(d1, has_t)
    for idx_val in [0, 1]:
        for col_val in [0, 1]:
            if idx_val not in ct.index: ct.loc[idx_val, :] = 0
            if col_val not in ct.columns: ct[col_val] = 0
    ct = ct.sort_index(axis=0).sort_index(axis=1)
    table = np.array([[ct.loc[1, 1], ct.loc[1, 0]], [ct.loc[0, 1], ct.loc[0, 0]]])
    fisher_or, fisher_p = scipy_stats.fisher_exact(table)
    log.info(f"  Full catalog: D-1 Fisher OR={fisher_or:.3f} (p={fisher_p:.4f})")

    results = {'d1_full_or': fisher_or, 'd1_full_pval': fisher_p}

    prob_fc = vasco.groupby('obs_date')['prob'].sum().to_dict()
    m['wt_count'] = m['date_str'].map(prob_fc).fillna(0)
    try:
        model = smf.glm(
            'wt_count ~ One_Day_BEFORE_Nuclear_Testing_YN + '
            'moon_illumination + cloud_cover_estimate + PRCP',
            data=m[['wt_count', 'One_Day_BEFORE_Nuclear_Testing_YN',
                     'moon_illumination', 'cloud_cover_estimate', 'PRCP']].dropna(),
            family=sm.families.Gaussian()).fit()
        d1_coef = model.params['One_Day_BEFORE_Nuclear_Testing_YN']
        d1_pval = model.pvalues['One_Day_BEFORE_Nuclear_Testing_YN']
        log.info(f"  Weighted D-1: coef={d1_coef:.4f} (p={d1_pval:.4f})")
        results['d1_weighted_coef'] = float(d1_coef)
        results['d1_weighted_pval'] = float(d1_pval)
    except Exception as e:
        log.warning(f"  Weighted D-1 failed: {e}")

    high_prob = vasco[vasco['prob'] >= 0.5]
    if len(high_prob) > 0:
        hfc = high_prob.groupby('obs_date').size().to_dict()
        mh = merged.copy()
        mh['count'] = mh['date_str'].map(hfc).fillna(0).astype(int)
        has_th = (mh['count'] > 0).astype(int)
        d1h = mh['One_Day_BEFORE_Nuclear_Testing_YN'].fillna(0).astype(int)
        cth = pd.crosstab(d1h, has_th)
        for idx_val in [0, 1]:
            for col_val in [0, 1]:
                if idx_val not in cth.index: cth.loc[idx_val, :] = 0
                if col_val not in cth.columns: cth[col_val] = 0
        cth = cth.sort_index(axis=0).sort_index(axis=1)
        table_h = np.array([[cth.loc[1, 1], cth.loc[1, 0]], [cth.loc[0, 1], cth.loc[0, 0]]])
        fisher_h, fisher_hp = scipy_stats.fisher_exact(table_h)
        log.info(f"  High-prob (>=0.5) D-1: Fisher OR={fisher_h:.3f} (p={fisher_hp:.4f})")
        results['d1_highprob_or'] = float(fisher_h)
        results['d1_highprob_pval'] = float(fisher_hp)

    return results


def probability_weighted_regression(vasco: pd.DataFrame, merged: pd.DataFrame,
                                     log: logging.Logger) -> dict:
    log.info("\nProbability-weighted regression:")

    wt = vasco.groupby('obs_date')['prob'].sum().to_dict()
    m = merged.copy()
    m['wt_count'] = m['date_str'].map(wt).fillna(0)

    uw = vasco.groupby('obs_date').size().to_dict()
    m['raw_count'] = m['date_str'].map(uw).fillna(0).astype(int)

    results = {}

    mdf_raw = m[['raw_count', 'Nuclear_Testing_YN_Window_Plus_Minus_1_Day',
                  'moon_illumination', 'cloud_cover_estimate', 'PRCP']].dropna().copy()
    mdf_raw.columns = ['count', 'Nuclear_Testing_YN_Window_Plus_Minus_1_Day',
                        'moon_illumination', 'cloud_cover_estimate', 'PRCP']
    nb_raw = run_nb_glm(mdf_raw)
    log.info(f"  Unweighted: IRR={nb_raw['irr']:.3f} (p={nb_raw['pval']:.4f})")
    results['raw_irr'] = nb_raw['irr']
    results['raw_pval'] = nb_raw['pval']

    try:
        model = smf.glm(
            'wt_count ~ Nuclear_Testing_YN_Window_Plus_Minus_1_Day + '
            'moon_illumination + cloud_cover_estimate + PRCP',
            data=m[['wt_count', 'Nuclear_Testing_YN_Window_Plus_Minus_1_Day',
                     'moon_illumination', 'cloud_cover_estimate', 'PRCP']].dropna(),
            family=sm.families.Gaussian()).fit()

        nuc_coef = model.params['Nuclear_Testing_YN_Window_Plus_Minus_1_Day']
        nuc_pval = model.pvalues['Nuclear_Testing_YN_Window_Plus_Minus_1_Day']
        log.info(f"  Weighted (Gaussian): coef={nuc_coef:.4f} (p={nuc_pval:.4f})")
        results['weighted_coef'] = float(nuc_coef)
        results['weighted_pval'] = float(nuc_pval)
    except Exception as e:
        log.warning(f"  Weighted regression failed: {e}")

    return results


def load_baselines(cfg: dict, base_dir: str, log: logging.Logger) -> dict:
    baselines = {}
    baseline_cfg = cfg['paths'].get('baselines', {})

    for name, path in baseline_cfg.items():
        full_path = os.path.join(base_dir, path)
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            baselines[name] = df
            log.info(f"  Loaded baseline: {name} ({len(df)} rows)")
        else:
            log.warning(f"  Baseline not found: {name} -> {full_path}")

    return baselines


def three_way_comparison(sweep_df: pd.DataFrame, baselines: dict,
                          vasco: pd.DataFrame, training_report: dict,
                          log: logging.Logger) -> pd.DataFrame:
    log.info("\nThree-way comparison table:")

    rows = []

    irr_at_t = {}
    for t_target in [0.50, 0.75]:
        closest = sweep_df.iloc[(sweep_df['threshold'] - t_target).abs().argsort()[:1]]
        if len(closest) > 0:
            irr_at_t[t_target] = closest.iloc[0]['irr']
        else:
            irr_at_t[t_target] = float('nan')

    s5 = {'auc': 0.874, 'irr_050': 1.947, 'irr_075': 7.672,
          'mono': 65, 'n_uncertain': 107662}
    v1 = {'auc': 0.889, 'irr_050': 2.706, 'irr_075': 2.539,
          'mono': 28, 'n_uncertain': 40365}

    cur_auc = training_report.get('mean_auc', float('nan'))
    cur_mono = sweep_df.attrs.get('monotonicity_pct', 0)
    n_unc = int((vasco['classification'] == 'UNCERTAIN').sum())

    metrics = [
        ('AUC', s5['auc'], v1['auc'], cur_auc),
        ('IRR at t=0.50', s5['irr_050'], v1['irr_050'], irr_at_t.get(0.50, float('nan'))),
        ('IRR at t=0.75', s5['irr_075'], v1['irr_075'], irr_at_t.get(0.75, float('nan'))),
        ('Monotonicity %', s5['mono'], v1['mono'], cur_mono),
        ('N UNCERTAIN', s5['n_uncertain'], v1['n_uncertain'], n_unc),
        ('N scored (catalog size)', 107875, 107875, len(vasco)),
        ('Mean prob', 0.194, 0.178, float(vasco['prob'].mean())),
        ('Prob std', 0.104, 0.178, float(vasco['prob'].std())),
    ]

    log.info(f"\n  {'Metric':30s} {'S5 Baseline':>14s} {'Reclass v1':>14s} {'Red-AllCand':>14s}")
    log.info(f"  {'-' * 74}")
    for name, s5_val, v1_val, cur_val in metrics:
        def fmt(v):
            if isinstance(v, (int, np.integer)):
                return f"{v:>14,d}"
            elif np.isnan(v):
                return f"{'nan':>14s}"
            elif abs(v) >= 100:
                return f"{int(v):>14,d}"
            else:
                return f"{v:>14.3f}"
        log.info(f"  {name:30s} {fmt(s5_val)} {fmt(v1_val)} {fmt(cur_val)}")
        rows.append({'metric': name, 's5_baseline': s5_val, 'reclass_v1': v1_val, 'red_allcandidates': cur_val})

    return pd.DataFrame(rows)


def generate_charts(sweep_df: pd.DataFrame, quintile_df: pd.DataFrame,
                    baselines: dict, vasco: pd.DataFrame,
                    output_dir: str, log: logging.Logger) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Nuclear Validation Results (Red All-Candidates)", fontsize=14, fontweight='bold')

    # Panel 1: IRR vs Threshold
    ax1 = axes[0, 0]
    valid = sweep_df.dropna(subset=['irr'])
    valid = valid[valid['irr'] > 0]
    ax1.plot(valid['threshold'], valid['irr'], 'bo-', label='Red-Only', markersize=4)
    ax1.fill_between(valid['threshold'], valid['irr_lo'], valid['irr_hi'], alpha=0.2)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    for name, color in [('s5_sweep', 'green'), ('reclass_v1_sweep', 'red'), ('proxy_sweep', 'purple')]:
        if name in baselines:
            bl = baselines[name]
            bl_valid = bl.dropna(subset=['irr'])
            bl_valid = bl_valid[bl_valid['irr'] > 0]
            if len(bl_valid) > 0:
                ax1.plot(bl_valid['threshold'], bl_valid['irr'], '--',
                        color=color, label=name.replace('_sweep', ''), alpha=0.7)
    ax1.set_xlabel('Probability Threshold')
    ax1.set_ylabel('IRR')
    ax1.set_title('IRR vs Threshold')
    ax1.legend(fontsize=7)
    ax1.set_ylim(bottom=0, top=12)

    # Panel 2: N retained vs Threshold
    ax2 = axes[0, 1]
    ax2.plot(sweep_df['threshold'], sweep_df['n_retained'], 'b-', linewidth=2)
    ax2.set_xlabel('Probability Threshold')
    ax2.set_ylabel('N Retained')
    ax2.set_title('Catalog Size vs Threshold')
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    # Panel 3: Quintile IRR
    ax3 = axes[0, 2]
    if len(quintile_df) > 0:
        x = range(len(quintile_df))
        bars = ax3.bar(x, quintile_df['irr'], color='steelblue', alpha=0.8)
        if 'irr_lo' in quintile_df.columns and 'irr_hi' in quintile_df.columns:
            yerr_lo = quintile_df['irr'] - quintile_df['irr_lo']
            yerr_hi = quintile_df['irr_hi'] - quintile_df['irr']
            ax3.errorbar(x, quintile_df['irr'], yerr=[yerr_lo, yerr_hi],
                        fmt='none', color='black', capsize=3)
        ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(quintile_df['quintile'].values, fontsize=8, rotation=15)
        ax3.set_ylabel('IRR')
        ax3.set_title('IRR by Probability Quintile')

    # Panel 4: Probability distribution
    ax4 = axes[1, 0]
    ax4.hist(vasco['prob'], bins=100, color='steelblue', alpha=0.7, edgecolor='none')
    bc = bimodality_coefficient(vasco['prob'].values)
    ax4.set_xlabel('Probability')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Probability Distribution (BC={bc:.3f})')

    # Panel 5: Fisher OR significance
    ax5 = axes[1, 1]
    valid_fisher = sweep_df.dropna(subset=['fisher_or'])
    if len(valid_fisher) > 0:
        colors = ['red' if p < 0.05 else 'gray'
                  for p in valid_fisher['fisher_pval'].fillna(1)]
        ax5.bar(range(len(valid_fisher)), valid_fisher['fisher_or'], color=colors, alpha=0.7)
        ax5.axhline(y=1.0, color='gray', linestyle='--')
        ax5.set_xlabel('Threshold Index')
        ax5.set_ylabel('Fisher OR')
        ax5.set_title('Fisher OR by Threshold (red = p<0.05)')

    # Panel 6: Probability by classification
    ax6 = axes[1, 2]
    class_order = ['EXCELLENT', 'GOOD', 'MARGINAL', 'UNCERTAIN', 'POOR']
    data_for_box = []
    labels_for_box = []
    for cls in class_order:
        mask = vasco['classification'] == cls
        if mask.sum() > 0:
            data_for_box.append(vasco.loc[mask, 'prob'].values)
            labels_for_box.append(f"{cls}\n(n={mask.sum():,d})")
    if data_for_box:
        bp = ax6.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        box_colors = ['green', 'blue', 'orange', 'gray', 'red']
        for patch, color in zip(bp['boxes'], box_colors[:len(data_for_box)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        ax6.tick_params(labelsize=7)
    ax6.set_ylabel('Probability')
    ax6.set_title('Probability by Classification')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart_path = os.path.join(output_dir, "chart_nuclear_validation.png")
    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Chart saved: {chart_path}")


def run_validation(cfg: dict) -> dict:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    output_dir = os.path.join(base_dir, cfg['paths']['output'])
    os.makedirs(output_dir, exist_ok=True)

    log = setup_logging(output_dir, "validation")
    log.info("=" * 60)
    log.info("NUCLEAR VALIDATION PIPELINE (RED ALL-CANDIDATES)")
    log.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    scored_path = os.path.join(output_dir, "scored_catalog.parquet")
    if not os.path.exists(scored_path):
        scored_path = os.path.join(output_dir, "scored_catalog.csv")
    if not os.path.exists(scored_path):
        raise FileNotFoundError("Scored catalog not found. Run scoring first.")

    if scored_path.endswith('.parquet'):
        vasco = pd.read_parquet(scored_path)
    else:
        vasco = pd.read_csv(scored_path)
    log.info(f"Scored catalog: {len(vasco):,d} candidates")
    log.info(f"Probability: mean={vasco['prob'].mean():.4f}, "
            f"median={vasco['prob'].median():.4f}, "
            f"std={vasco['prob'].std():.4f}")

    merged = load_nuclear_timeline(cfg, base_dir)
    log.info(f"Nuclear timeline: {len(merged)} dates")

    report_path = os.path.join(output_dir, "training_report.json")
    training_report = {}
    if os.path.exists(report_path):
        with open(report_path) as f:
            training_report = json.load(f)
        log.info(f"Training report: AUC={training_report.get('mean_auc', '?')}")

    log.info("\nLoading baselines:")
    baselines = load_baselines(cfg, base_dir, log)

    # 1. Threshold sweep
    log.info("\n" + "=" * 60)
    log.info("1. THRESHOLD SWEEP")
    log.info("=" * 60)
    sweep_df = threshold_sweep(vasco, merged, cfg, log)
    sweep_df.to_csv(os.path.join(output_dir, "threshold_sweep.csv"), index=False)

    # 2. Quintile analysis
    log.info("\n" + "=" * 60)
    log.info("2. QUINTILE ANALYSIS")
    log.info("=" * 60)
    quintile_df = quintile_analysis(vasco, merged, log)
    quintile_df.to_csv(os.path.join(output_dir, "quintile_analysis.csv"), index=False)

    # 3. D-1 anticipatory signal
    log.info("\n" + "=" * 60)
    log.info("3. D-1 ANTICIPATORY SIGNAL")
    log.info("=" * 60)
    d1_results = d_minus_1_analysis(vasco, merged, log)

    # 4. Spectral class breakdown -- SKIPPED in red-only pipeline
    log.info("\n" + "=" * 60)
    log.info("4. SPECTRAL CLASS BREAKDOWN -- SKIPPED (red all-candidates pipeline)")
    log.info("=" * 60)

    # 5. Probability-weighted regression
    log.info("\n" + "=" * 60)
    log.info("5. PROBABILITY-WEIGHTED REGRESSION")
    log.info("=" * 60)
    pw_results = probability_weighted_regression(vasco, merged, log)

    # 6. Three-way comparison
    log.info("\n" + "=" * 60)
    log.info("6. THREE-WAY COMPARISON")
    log.info("=" * 60)
    comparison_df = three_way_comparison(sweep_df, baselines, vasco, training_report, log)
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    # 7. Charts
    log.info("\n" + "=" * 60)
    log.info("7. GENERATING CHARTS")
    log.info("=" * 60)
    generate_charts(sweep_df, quintile_df, baselines, vasco, output_dir, log)

    # Save validation report
    val_report = {
        'date': datetime.now().isoformat(),
        'pipeline_variant': 'red_allcandidates',
        'n_candidates': len(vasco),
        'mean_prob': float(vasco['prob'].mean()),
        'bimodality_coefficient': float(bimodality_coefficient(vasco['prob'].values)),
        'monotonicity_pct': float(sweep_df.attrs.get('monotonicity_pct', 0)),
        'd1_results': d1_results,
        'prob_weighted': pw_results,
    }

    for t_target in [0.50, 0.75]:
        closest = sweep_df.iloc[(sweep_df['threshold'] - t_target).abs().argsort()[:1]]
        if len(closest) > 0:
            row = closest.iloc[0]
            val_report[f'irr_at_{t_target}'] = float(row['irr'])
            val_report[f'n_retained_at_{t_target}'] = int(row['n_retained'])

    with open(os.path.join(output_dir, "validation_report.json"), 'w') as f:
        json.dump(val_report, f, indent=2, default=str)

    summary_path = os.path.join(output_dir, "validation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("VASCO ML Pipeline (Red All-Candidates) -- Nuclear Validation Summary\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Candidates: {len(vasco):,d}\n")
        f.write(f"AUC: {training_report.get('mean_auc', '?')}\n")
        f.write(f"Monotonicity: {sweep_df.attrs.get('monotonicity_pct', 0):.0f}%\n")
        f.write(f"Bimodality: {bimodality_coefficient(vasco['prob'].values):.4f}\n\n")

        f.write("Classification distribution:\n")
        for cls in ['EXCELLENT', 'GOOD', 'MARGINAL', 'UNCERTAIN', 'POOR']:
            cnt = (vasco['classification'] == cls).sum()
            f.write(f"  {cls}: {cnt:,d}\n")

        f.write(f"\nThreshold sweep saved to: threshold_sweep.csv\n")
        f.write(f"Quintile analysis saved to: quintile_analysis.csv\n")
        f.write(f"Comparison saved to: model_comparison.csv\n")

    log.info(f"\nValidation summary saved: {summary_path}")
    log.info("\nValidation complete")
    return val_report


def main():
    parser = argparse.ArgumentParser(description="VASCO Nuclear Validation (Red All-Candidates)")
    parser.add_argument('--config', required=True, help='Path to config_red_allcandidates.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_validation(cfg)


if __name__ == "__main__":
    main()
