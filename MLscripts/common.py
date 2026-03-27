"""
Shared utilities for the VASCO ML pipeline (red-only variant).
Removes all spectral cross-matching, blue-band features, color features,
and external catalog dependencies. Only red-plate FITS morphometry,
catalog-level features, and plate-level aggregates are retained.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from scipy import stats as scipy_stats

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import statsmodels.api as sm
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)

# ---- Pipeline 1 feature definitions ----
NUMERIC_FEATURES = ['snr', 'psf_ratio', 'elongation', 'compactness', 'sharpness',
                    'n_comparison_stars', 'candidate_score']
BINARY_FEATURES = ['in_red_only', 'in_blue_only']
PLATE_QUALITY_CATS = ['GOOD', 'MODERATE', 'POOR', 'NO_PLATE']
DIFF_VERDICT_CATS = ['CONFIRMED_TRANSIENT', 'SIGNIFICANT_DIFF', 'NO_DIFF', 'OUT_OF_BOUNDS', 'NOT_CHECKED']
PSF_VERDICT_CATS = ['EXCELLENT', 'GOOD', 'NORMAL', 'UNKNOWN']

FEATURE_COLS = ['ra', 'dec', 'plate_quality', 'psf_ratio', 'psf_verdict',
                'n_comparison_stars', 'elongation', 'compactness', 'sharpness', 'snr',
                'candidate_score', 'classification', 'diff_verdict', 'in_red_only',
                'in_blue_only', 'obs_date']


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: str, name: str = "pipeline") -> logging.Logger:
    import os
    os.makedirs(output_dir, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, f"{name}.log"), mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    log.addHandler(ch)
    return log


# =====================================================================
# FEATURE ENCODERS
# =====================================================================

def encode_pipeline1(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Encode Pipeline 1 features (catalog-level)."""
    X_parts, names = [], []
    for c in NUMERIC_FEATURES:
        X_parts.append(pd.to_numeric(df[c], errors='coerce').fillna(0).values.reshape(-1, 1))
        names.append(c)
    for c in BINARY_FEATURES:
        X_parts.append(df[c].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).values.reshape(-1, 1))
        names.append(c)
    for cat in PLATE_QUALITY_CATS:
        X_parts.append((df.get('plate_quality', pd.Series([''] * len(df))) == cat).astype(int).values.reshape(-1, 1))
        names.append(f'plate_quality_{cat}')
    for cat in DIFF_VERDICT_CATS:
        X_parts.append((df.get('diff_verdict', pd.Series([''] * len(df))) == cat).astype(int).values.reshape(-1, 1))
        names.append(f'diff_{cat}')
    for cat in PSF_VERDICT_CATS:
        X_parts.append((df.get('psf_verdict', pd.Series([''] * len(df))) == cat).astype(int).values.reshape(-1, 1))
        names.append(f'psf_{cat}')
    return np.hstack(X_parts), names


def compute_plate_features(df: pd.DataFrame, group_col: str = 'obs_date') -> pd.DataFrame:
    """Compute plate-level quality features grouped by observation date."""
    snr_col = 'snr'
    elong_col = 'elongation'
    if group_col not in df.columns or snr_col not in df.columns:
        logger.warning("Cannot compute plate features: missing columns")
        for col in ['plate_low_snr_frac', 'plate_snr_std', 'plate_elongation_mean', 'plate_n_high_snr']:
            df[col] = 0.0
        return df

    snr_vals = pd.to_numeric(df[snr_col], errors='coerce').fillna(0)
    plate_stats = pd.DataFrame({
        'plate_low_snr_frac': (snr_vals < 5).groupby(df[group_col]).mean(),
        'plate_snr_std': snr_vals.groupby(df[group_col]).std().fillna(0),
        'plate_n_high_snr': (snr_vals > 15).groupby(df[group_col]).sum(),
    })

    if elong_col in df.columns:
        elong_vals = pd.to_numeric(df[elong_col], errors='coerce').fillna(0)
        plate_stats['plate_elongation_mean'] = elong_vals.groupby(df[group_col]).mean()
    else:
        plate_stats['plate_elongation_mean'] = 0.0

    for col in plate_stats.columns:
        df[col] = df[group_col].map(plate_stats[col]).fillna(0).values

    return df


def encode_model_b(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Encode red-only features: Pipeline 1 + plate aggregates + red FITS morphometry.

    Dropped vs original:
      - p2_fwhm, p2_red_snr, p2_br_color, p2_point_source (spectral-derived)
      - p2_spectral_class one-hots (spectral-derived)
      - All blue_* FITS features
      - br_color_fits, flux_ratio_red, flux_ratio_blue, detected_both (color/dual-band)
      - All Bruehl experimental features (spectral-derived)
    """
    X_p1, names = encode_pipeline1(df)
    X_parts = [X_p1]

    # Plate-level quality features
    PLATE_FEATURES = ['plate_low_snr_frac', 'plate_snr_std', 'plate_elongation_mean', 'plate_n_high_snr']
    for col in PLATE_FEATURES:
        if col in df.columns:
            raw = pd.to_numeric(df[col], errors='coerce').fillna(0)
            X_parts.append(raw.values.reshape(-1, 1))
            names.append(col)

    # Red-band FITS morphometric features only
    RED_FITS_NUMERIC = [
        'red_fits_snr', 'red_fits_fwhm', 'red_ellipticity', 'red_sharpness_2nd',
        'red_n_connected_px', 'red_aperture_flux', 'red_dist_to_edge_px',
        'red_symmetry_score', 'red_gradient_magnitude', 'red_near_bright_star',
    ]
    for col in RED_FITS_NUMERIC:
        if col in df.columns:
            raw = pd.to_numeric(df[col], errors='coerce')
            med = raw.median()
            if np.isnan(med): med = 0
            X_parts.append(raw.fillna(med).values.reshape(-1, 1))
            names.append(col)

    return np.hstack(X_parts), names


def prevalence_filter(X_train: np.ndarray, X_catalog: np.ndarray,
                      feature_names: List[str],
                      min_prevalence: float = 0.01) -> Tuple[List[int], List[str], List[str]]:
    """Filter out binary/one-hot features where minority class is below min_prevalence."""
    kept_idx, kept_names, dropped = [], [], []
    for i, name in enumerate(feature_names):
        col_cat = X_catalog[:, i]
        n_unique = len(np.unique(col_cat[~np.isnan(col_cat)]))
        if n_unique > 2:
            kept_idx.append(i)
            kept_names.append(name)
            continue
        cat_frac = np.nanmean(col_cat != 0)
        cat_min_prev = min(cat_frac, 1 - cat_frac)
        col_tr = X_train[:, i]
        tr_frac = np.nanmean(col_tr != 0)
        tr_min_prev = min(tr_frac, 1 - tr_frac)
        if cat_min_prev < min_prevalence or tr_min_prev < min_prevalence:
            dropped.append(name)
        else:
            kept_idx.append(i)
            kept_names.append(name)
    return kept_idx, kept_names, dropped


def select_features(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                    max_features: int = 0, min_ratio: float = 10.0,
                    method: str = 'mutual_info',
                    protected_prefixes: Optional[List[str]] = None
                    ) -> Tuple[List[int], List[str], List[str]]:
    """Select top features to maintain a healthy sample-to-feature ratio."""
    n_samples, n_features = X.shape
    if max_features <= 0:
        max_features = max(int(n_samples / min_ratio), 10)

    if n_features <= max_features:
        logger.info(f"  Feature selection: {n_features} features <= cap {max_features}, keeping all")
        return list(range(n_features)), list(feature_names), []

    # Red-only: only red FITS and plate-aggregate features are selectable
    SELECTABLE_PREFIXES = ('red_',)
    SELECTABLE_NAMES = {'plate_low_snr_frac', 'plate_snr_std', 'plate_elongation_mean', 'plate_n_high_snr'}
    protected_idx, selectable_idx = [], []
    for i, name in enumerate(feature_names):
        is_selectable = any(name.startswith(p) for p in SELECTABLE_PREFIXES) or name in SELECTABLE_NAMES
        if not is_selectable:
            protected_idx.append(i)
        else:
            selectable_idx.append(i)

    n_protected = len(protected_idx)
    budget = max_features - n_protected
    logger.info(f"  Feature selection: {n_protected} protected + {len(selectable_idx)} selectable, "
                f"budget for extras: {budget}")

    if budget <= 0 or not selectable_idx:
        kept_idx = sorted(protected_idx)
        kept_names = [feature_names[i] for i in kept_idx]
        dropped_names = [feature_names[i] for i in selectable_idx]
        logger.info(f"  No budget for extra features, keeping {n_protected} protected only")
        return kept_idx, kept_names, dropped_names

    if budget >= len(selectable_idx):
        logger.info(f"  All {len(selectable_idx)} red FITS features fit within budget, keeping all")
        return list(range(n_features)), list(feature_names), []

    from sklearn.feature_selection import mutual_info_classif, f_classif
    X_sel = X[:, selectable_idx]
    if method == 'f_classif':
        scores, _ = f_classif(np.nan_to_num(X_sel), y)
    else:
        scores = mutual_info_classif(np.nan_to_num(X_sel), y, random_state=42, n_neighbors=5)

    ranked = np.argsort(scores)[::-1]
    top_sel = [selectable_idx[r] for r in ranked[:budget]]

    kept_idx = sorted(protected_idx + top_sel)
    kept_names = [feature_names[i] for i in kept_idx]
    dropped_names = [feature_names[i] for i in range(n_features) if i not in kept_idx]

    logger.info(f"  Feature selection: {n_features} -> {len(kept_idx)} features "
                f"({n_protected} protected + {budget} red FITS, "
                f"ratio: {n_samples}/{len(kept_idx)} = {n_samples/len(kept_idx):.1f}:1)")
    return kept_idx, kept_names, dropped_names


# =====================================================================
# CATALOG MATCHING (for training labels)
# =====================================================================

def match_batch_to_catalog(batch_df: pd.DataFrame, vasco: pd.DataFrame,
                           tol_deg: float = 0.01) -> Tuple[pd.DataFrame, int]:
    """Match a batch of candidates to the VASCO catalog by RA/Dec."""
    matched = []
    n_matched = 0
    for _, row in batch_df.iterrows():
        ra, dec = row['ra'], row['dec']
        dist = np.sqrt((vasco['ra'] - ra)**2 + (vasco['dec'] - dec)**2)
        min_idx = dist.idxmin()
        if dist[min_idx] < tol_deg:
            m = vasco.loc[min_idx, FEATURE_COLS].to_dict()
            m['matched'] = True
            n_matched += 1
        else:
            m = {col: np.nan for col in FEATURE_COLS}
            m['matched'] = False
        matched.append(m)
    return pd.DataFrame(matched), n_matched


# =====================================================================
# MODEL TRAINING
# =====================================================================

def build_ensemble(cfg: dict, weight_ratio: float, seed: int = 42,
                   fold_offset: int = 0) -> dict:
    """Build the three-model ensemble from config hyperparameters."""
    rf_cfg = cfg['model']['rf']
    gbm_cfg = cfg['model']['gbm']
    xgb_cfg = cfg['model']['xgb']

    rf = RandomForestClassifier(
        n_estimators=rf_cfg['n_estimators'], max_depth=rf_cfg['max_depth'],
        min_samples_leaf=rf_cfg['min_samples_leaf'],
        class_weight='balanced', random_state=seed + fold_offset, n_jobs=-1)

    gb = GradientBoostingClassifier(
        n_estimators=gbm_cfg['n_estimators'], max_depth=gbm_cfg['max_depth'],
        learning_rate=gbm_cfg['learning_rate'],
        min_samples_leaf=gbm_cfg['min_samples_leaf'], random_state=seed + fold_offset)

    xg = XGBClassifier(
        n_estimators=xgb_cfg['n_estimators'], max_depth=xgb_cfg['max_depth'],
        learning_rate=xgb_cfg['learning_rate'],
        min_child_weight=xgb_cfg.get('min_child_weight', 3),
        scale_pos_weight=weight_ratio, random_state=seed + fold_offset,
        use_label_encoder=False, eval_metric='logloss', verbosity=0)

    return {'rf': rf, 'gbm': gb, 'xgb': xg}


def tune_ensemble_hyperparams(X: np.ndarray, y: np.ndarray, cfg: dict,
                              n_iter: int = 30, cv_folds: int = 5) -> dict:
    """Run RandomizedSearchCV on each ensemble model, update cfg with best params."""
    seed = cfg['seed']
    weight_ratio = (y == 0).sum() / max((y == 1).sum(), 1)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    results = {}

    logger.info(f"  Tuning RandomForest ({n_iter} iters, {cv_folds}-fold)...")
    rf_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2', None],
    }
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=seed, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        rf_base, rf_grid, n_iter=n_iter, cv=cv, scoring='roc_auc',
        n_jobs=-1, random_state=seed, error_score='raise')
    rf_search.fit(Xs, y)
    results['rf'] = rf_search.best_params_
    logger.info(f"    RF best AUC={rf_search.best_score_:.4f}  params={rf_search.best_params_}")

    logger.info(f"  Tuning GradientBoosting ({n_iter} iters)...")
    gbm_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_samples_leaf': [1, 2, 4, 6],
        'subsample': [0.7, 0.8, 0.9, 1.0],
    }
    sw = np.where(y == 1, weight_ratio, 1.0)
    gbm_base = GradientBoostingClassifier(random_state=seed)
    gbm_search = RandomizedSearchCV(
        gbm_base, gbm_grid, n_iter=n_iter, cv=cv, scoring='roc_auc',
        n_jobs=-1, random_state=seed, error_score='raise')
    gbm_search.fit(Xs, y, sample_weight=sw)
    results['gbm'] = gbm_search.best_params_
    logger.info(f"    GBM best AUC={gbm_search.best_score_:.4f}  params={gbm_search.best_params_}")

    logger.info(f"  Tuning XGBoost ({n_iter} iters)...")
    xgb_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 2.0],
    }
    xgb_base = XGBClassifier(
        scale_pos_weight=weight_ratio, random_state=seed,
        use_label_encoder=False, eval_metric='logloss', verbosity=0)
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_grid, n_iter=n_iter, cv=cv, scoring='roc_auc',
        n_jobs=-1, random_state=seed, error_score='raise')
    xgb_search.fit(Xs, y)
    results['xgb'] = xgb_search.best_params_
    logger.info(f"    XGB best AUC={xgb_search.best_score_:.4f}  params={xgb_search.best_params_}")

    for model_key in ('rf', 'gbm', 'xgb'):
        for param, val in results[model_key].items():
            cfg['model'][model_key][param] = val

    logger.info("  Hyperparameter tuning complete -- cfg updated.")
    return cfg


def train_ensemble_cv(X: np.ndarray, y: np.ndarray, cfg: dict,
                      obs_dates: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Train ensemble with cross-validation."""
    n_folds = cfg['cv']['n_folds']
    seed = cfg['seed']
    weight_ratio = (y == 0).sum() / max((y == 1).sum(), 1)

    split_method = cfg['cv'].get('split_method', 'stratified')

    if split_method == 'time' and obs_dates is not None:
        obs_dates_str = np.array([str(d) if pd.notna(d) else '' for d in obs_dates])
        valid_dates = obs_dates_str[obs_dates_str != '']
        unique_dates = np.sort(np.unique(valid_dates))
        date_folds = np.array_split(unique_dates, n_folds)
        folds = []
        for fold_i in range(n_folds):
            test_dates = set(date_folds[fold_i])
            test_idx = np.array([i for i in range(len(y)) if obs_dates_str[i] in test_dates])
            train_idx = np.array([i for i in range(len(y)) if obs_dates_str[i] not in test_dates])
            if len(test_idx) > 0 and len(train_idx) > 0:
                folds.append((train_idx, test_idx))
        logger.info(f"Time-based CV: {len(folds)} folds from {len(unique_dates)} unique dates")
        if len(folds) < n_folds:
            logger.warning(f"Time-based CV produced only {len(folds)} folds. "
                          "Falling back to stratified CV.")
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            folds = list(skf.split(X, y))
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = list(skf.split(X, y))

    all_probs = np.zeros(len(y))
    fold_metrics = []
    fold_importances = []

    for fi, (tri, tei) in enumerate(folds):
        sc = StandardScaler()
        Xtr, Xte = sc.fit_transform(X[tri]), sc.transform(X[tei])
        yt_train, yt_test = y[tri], y[tei]

        models = build_ensemble(cfg, weight_ratio, seed, fi)
        models['rf'].fit(Xtr, yt_train)
        sw = np.where(yt_train == 1, weight_ratio, 1.0)
        models['gbm'].fit(Xtr, yt_train, sample_weight=sw)
        models['xgb'].fit(Xtr, yt_train)

        p = (models['rf'].predict_proba(Xte)[:, 1] +
             models['gbm'].predict_proba(Xte)[:, 1] +
             models['xgb'].predict_proba(Xte)[:, 1]) / 3
        all_probs[tei] = p
        preds = (p >= 0.5).astype(int)

        if len(np.unique(yt_test)) > 1:
            tn, fp, fn, tp = confusion_matrix(yt_test, preds, labels=[0, 1]).ravel()
            fold_metrics.append({
                'fold': fi, 'auc': roc_auc_score(yt_test, p),
                'sens': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'spec': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'f1': f1_score(yt_test, preds, zero_division=0),
                'n_train': len(tri), 'n_test': len(tei),
            })
        fold_importances.append(models['rf'].feature_importances_)

    return {
        'probs': all_probs,
        'fold_metrics': fold_metrics,
        'importances': np.array(fold_importances),
    }


def train_final_ensemble(X: np.ndarray, y: np.ndarray, cfg: dict) -> dict:
    """Train final ensemble on all data."""
    seed = cfg['seed']
    weight_ratio = (y == 0).sum() / max((y == 1).sum(), 1)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    models = build_ensemble(cfg, weight_ratio, seed)
    models['rf'].fit(Xs, y)
    sw = np.where(y == 1, weight_ratio, 1.0)
    models['gbm'].fit(Xs, y, sample_weight=sw)
    models['xgb'].fit(Xs, y)
    models['scaler'] = sc
    return models


def score_with_ensemble(models: dict, X: np.ndarray) -> np.ndarray:
    """Score candidates using trained ensemble."""
    Xs = models['scaler'].transform(X)
    return (models['rf'].predict_proba(Xs)[:, 1] +
            models['gbm'].predict_proba(Xs)[:, 1] +
            models['xgb'].predict_proba(Xs)[:, 1]) / 3


# =====================================================================
# NUCLEAR VALIDATION
# =====================================================================

def run_nb_glm(mdf: pd.DataFrame) -> dict:
    """Run negative binomial GLM for nuclear association."""
    try:
        nb = smf.glm('count ~ Nuclear_Testing_YN_Window_Plus_Minus_1_Day + '
                     'moon_illumination + cloud_cover_estimate + PRCP',
                     data=mdf, family=sm.families.NegativeBinomial()).fit(maxiter=300)
        coef = nb.params['Nuclear_Testing_YN_Window_Plus_Minus_1_Day']
        se = nb.bse['Nuclear_Testing_YN_Window_Plus_Minus_1_Day']
        return {'irr': float(np.exp(coef)), 'irr_lo': float(np.exp(coef - 1.96 * se)),
                'irr_hi': float(np.exp(coef + 1.96 * se)),
                'pval': float(nb.pvalues['Nuclear_Testing_YN_Window_Plus_Minus_1_Day'])}
    except Exception:
        return {'irr': float('nan'), 'irr_lo': float('nan'),
                'irr_hi': float('nan'), 'pval': float('nan')}


def run_nuclear_threshold(vasco: pd.DataFrame, merged: pd.DataFrame,
                          threshold: float = 0.5) -> dict:
    """Run nuclear association at a single probability threshold."""
    retained = vasco[vasco['prob'] >= threshold]
    n_ret = len(retained)
    pct_bad = 100 * (1 - n_ret / len(vasco))
    result = {'threshold': threshold, 'n_retained': n_ret, 'pct_rejected': pct_bad}

    if pct_bad <= 5:
        result['note'] = 'Rejection < 5%'
        return result

    fc = retained.groupby('obs_date').size().to_dict()
    m = merged.copy()
    m['count'] = m['date_str'].map(fc).fillna(0).astype(int)
    result['n_dates'] = int((m['count'] > 0).sum())

    mdf = m[['count', 'Nuclear_Testing_YN_Window_Plus_Minus_1_Day',
             'moon_illumination', 'cloud_cover_estimate', 'PRCP']].dropna().copy()
    nb = run_nb_glm(mdf)
    result.update(nb)

    has_t = (m['count'] > 0).astype(int)
    ct = pd.crosstab(m['Nuclear_Testing_YN_Window_Plus_Minus_1_Day'], has_t)
    for idx_val in [0, 1]:
        for col_val in [0, 1]:
            if idx_val not in ct.index: ct.loc[idx_val, :] = 0
            if col_val not in ct.columns: ct[col_val] = 0
    ct = ct.sort_index(axis=0).sort_index(axis=1)
    table = np.array([[ct.loc[1, 1], ct.loc[1, 0]], [ct.loc[0, 1], ct.loc[0, 0]]])
    fisher_or, fisher_p = scipy_stats.fisher_exact(table)
    result['fisher_or'] = float(fisher_or)
    result['fisher_pval'] = float(fisher_p)
    return result


def bimodality_coefficient(data: np.ndarray) -> float:
    """Compute bimodality coefficient. Values > 0.555 suggest bimodality."""
    n = len(data)
    skew = scipy_stats.skew(data)
    kurt = scipy_stats.kurtosis(data, fisher=True)
    return (skew**2 + 1) / (kurt + 3 * (n - 1)**2 / ((n - 2) * (n - 3)))
