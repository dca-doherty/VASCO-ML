#!/usr/bin/env python3
"""
Feature extraction pipeline for VASCO transient candidates (red all-candidates variant).
Extracts red-plate FITS morphometry for ALL 107,875 candidates -- no BOTH_BANDS
cross-match filter, no spectral filtering, no blue plate data.

For candidates without a red FITS file (~6k), median imputation is applied.

Usage:
    python pipeline_extract_features.py --config config_red_allcandidates.yaml
"""

import argparse
import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, setup_logging, compute_plate_features

logger = logging.getLogger("pipeline")


# =====================================================================
# FITS FEATURE EXTRACTION (red plate only)
# =====================================================================

def extract_fits_features(fits_path: str, cfg: dict) -> Dict[str, float]:
    """Extract morphometric features from a single FITS cutout."""
    features = {}
    nan_keys = ['peak_px', 'local_bg_mean', 'local_bg_std',
                'fits_snr', 'fits_fwhm', 'ellipticity', 'sharpness_2nd',
                'n_connected_px', 'aperture_flux', 'dist_to_edge_px',
                'symmetry_score', 'gradient_magnitude', 'near_bright_star']
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            if data is None:
                return {k: np.nan for k in nan_keys}
            data = data.astype(float)
            ny, nx = data.shape

            border = max(int(min(ny, nx) * 0.2), 5)
            bg_mask = np.zeros_like(data, dtype=bool)
            bg_mask[:border, :] = True
            bg_mask[-border:, :] = True
            bg_mask[:, :border] = True
            bg_mask[:, -border:] = True
            bg_vals = data[bg_mask]
            bg_mean = np.nanmedian(bg_vals)
            bg_std = np.nanstd(bg_vals)
            features['local_bg_mean'] = float(bg_mean)
            features['local_bg_std'] = float(bg_std)

            cy, cx = ny // 2, nx // 2
            r = min(20, ny // 4, nx // 4)
            cutout = data[max(0, cy-r):cy+r, max(0, cx-r):cx+r]
            features['peak_px'] = float(np.nanmax(cutout))

            source_flux = features['peak_px'] - bg_mean
            features['fits_snr'] = float(source_flux / bg_std) if bg_std > 0 else 0.0

            yy, xx = np.ogrid[0:cutout.shape[0], 0:cutout.shape[1]]
            cy_c, cx_c = cutout.shape[0] // 2, cutout.shape[1] // 2
            half_max = (features['peak_px'] + bg_mean) / 2
            above_half = cutout > half_max
            if above_half.any():
                features['fits_fwhm'] = float(2 * np.sqrt(above_half.sum() / np.pi))
            else:
                features['fits_fwhm'] = np.nan

            sub = cutout - bg_mean
            sub[sub < 0] = 0
            total = sub.sum()
            if total > 0:
                yg, xg = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
                xc = (xg * sub).sum() / total
                yc = (yg * sub).sum() / total
                Mxx = ((xg - xc)**2 * sub).sum() / total
                Myy = ((yg - yc)**2 * sub).sum() / total
                Mxy = ((xg - xc) * (yg - yc) * sub).sum() / total
                a2 = (Mxx + Myy) / 2 + np.sqrt(((Mxx - Myy) / 2)**2 + Mxy**2)
                b2 = (Mxx + Myy) / 2 - np.sqrt(((Mxx - Myy) / 2)**2 + Mxy**2)
                a2 = max(a2, 1e-10)
                features['ellipticity'] = float(1 - np.sqrt(max(b2, 0) / a2))
                features['sharpness_2nd'] = float(total / (Mxx + Myy)) if (Mxx + Myy) > 0 else 0
            else:
                features['ellipticity'] = np.nan
                features['sharpness_2nd'] = np.nan

            sigma_thresh = cfg['extraction']['sigma_threshold']
            above_thresh = cutout > (bg_mean + sigma_thresh * bg_std)
            features['n_connected_px'] = int(above_thresh.sum())

            ap_r = cfg['extraction']['aperture_radius_px']
            ann_in = cfg['extraction']['annulus_inner_px']
            ann_out = cfg['extraction']['annulus_outer_px']
            yf, xf = np.ogrid[0:ny, 0:nx]
            r_full = np.sqrt((yf - cy)**2 + (xf - cx)**2)
            ap_mask = r_full <= ap_r
            ann_mask = (r_full >= ann_in) & (r_full <= ann_out)
            ann_bg = np.nanmedian(data[ann_mask]) if ann_mask.any() else bg_mean
            ap_flux = np.nansum(data[ap_mask] - ann_bg)
            features['aperture_flux'] = float(ap_flux)

            peak_y, peak_x = np.unravel_index(np.nanargmax(cutout), cutout.shape)
            peak_y += max(0, cy - r)
            peak_x += max(0, cx - r)
            features['dist_to_edge_px'] = float(min(peak_x, nx - peak_x, peak_y, ny - peak_y))

            size_s = min(10, cutout.shape[0] // 2, cutout.shape[1] // 2)
            if size_s >= 3:
                sc = cutout - bg_mean
                flip_x = np.fliplr(sc)
                flip_y = np.flipud(sc)
                denom = np.sum(np.abs(sc)) + 1e-10
                sym = 1.0 - (np.sum(np.abs(sc - flip_x)) + np.sum(np.abs(sc - flip_y))) / (4 * denom)
                features['symmetry_score'] = float(np.clip(sym, 0, 1))
            else:
                features['symmetry_score'] = np.nan

            from scipy.ndimage import sobel as _sobel
            gx = _sobel(cutout, axis=1)
            gy = _sobel(cutout, axis=0)
            grad = np.sqrt(gx**2 + gy**2)
            source_mask = cutout > (bg_mean + 3 * bg_std)
            if source_mask.any():
                features['gradient_magnitude'] = float(np.mean(grad[source_mask]))
            else:
                features['gradient_magnitude'] = float(np.mean(grad))

            ch, cw = cutout.shape
            bright_thresh = bg_mean + 15 * bg_std
            yb, xb = np.where(cutout > bright_thresh)
            if len(xb) > 0:
                dist_from_cen = np.sqrt((xb - cw / 2)**2 + (yb - ch / 2)**2)
                far_bright = dist_from_cen > 0.6 * min(ch, cw) / 2
                features['near_bright_star'] = 1.0 if far_bright.any() else 0.0
            else:
                features['near_bright_star'] = 0.0

    except Exception as e:
        return {k: np.nan for k in nan_keys}

    return features


# =====================================================================
# MAIN EXTRACTION
# =====================================================================

def run_extraction(cfg: dict) -> pd.DataFrame:
    """Run red all-candidates feature extraction pipeline.

    Unlike the red-only variant, this does NOT filter by detection_class.
    All 107,875 candidates from the VASCO catalog are retained and scored.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    output_dir = os.path.join(base_dir, cfg['paths']['output'])
    cache_path = os.path.join(base_dir, cfg['paths']['feature_cache'])
    os.makedirs(output_dir, exist_ok=True)

    log = setup_logging(output_dir, "extraction")
    log.info("=" * 60)
    log.info("FEATURE EXTRACTION PIPELINE (RED ALL-CANDIDATES)")
    log.info("All 107,875 candidates -- no BOTH_BANDS filter")
    log.info("=" * 60)

    # Load catalog
    catalog_path = os.path.join(base_dir, cfg['paths']['catalog'])
    vasco = pd.read_csv(catalog_path)
    log.info(f"Loaded VASCO v4: {len(vasco):,d} candidates")

    # NO BOTH_BANDS FILTER -- keep every candidate in the catalog.
    # The red-only pipeline filtered via spectral detection_class here;
    # this variant intentionally skips that step to score the full catalog.
    vasco['detection_class'] = 'ALL'
    log.info(f"BOTH_BANDS filter: DISABLED -- retaining all {len(vasco):,d} candidates")

    # Generate composite source_id
    vasco['source_id'] = vasco.apply(
        lambda r: f"{r['ra']:.6f}_{r['dec']:.6f}_{r['obs_date']}", axis=1)

    # Check for existing cache (resume capability)
    already_done = set()
    if os.path.exists(cache_path):
        existing = pd.read_parquet(cache_path)
        already_done = set(existing['source_id'].values)
        log.info(f"Found existing cache with {len(already_done):,d} entries")
        has_fits_cols = any(c.startswith('red_') for c in existing.columns)
        if len(already_done) == len(vasco) and has_fits_cols:
            log.info("Cache is complete with FITS features -- skipping extraction")
            log.info(f"Feature cache: {len(existing):,d} rows, {len(existing.columns)} columns")
            log.info("Feature extraction complete")
            return existing

    # ---- Positional features ----
    log.info("Computing positional features...")
    coords = SkyCoord(ra=vasco['ra'].values * u.degree, dec=vasco['dec'].values * u.degree, frame='icrs')
    galactic = coords.galactic
    vasco['gal_lat'] = galactic.b.degree
    vasco['gal_lon'] = galactic.l.degree

    plate_centers = vasco.groupby('obs_date').agg(
        center_ra=('ra', 'median'), center_dec=('dec', 'median')).reset_index()
    vasco = vasco.merge(plate_centers, on='obs_date', how='left')
    vasco['dist_from_center_deg'] = np.sqrt(
        (vasco['ra'] - vasco['center_ra'])**2 +
        (vasco['dec'] - vasco['center_dec'])**2 * np.cos(np.radians(vasco['dec']))**2)
    vasco = vasco.drop(columns=['center_ra', 'center_dec'])

    # ---- Plate-level features ----
    log.info("Computing plate-level features...")
    plate_stats = vasco.groupby('obs_date').agg(
        plate_n_candidates=('ra', 'count'),
        plate_median_snr=('snr', 'median'),
    ).reset_index()
    vasco = vasco.merge(plate_stats, on='obs_date', how='left')

    # Keep original classification for reference
    vasco['classification_orig'] = vasco['classification'].copy()

    log.info(f"Classification distribution (unchanged from catalog):")
    for val in ['EXCELLENT', 'GOOD', 'MARGINAL', 'UNCERTAIN', 'POOR']:
        cnt = (vasco['classification'] == val).sum()
        log.info(f"  {val}: {cnt:,d}")

    # ---- Imputation source tracking ----
    vasco['imputation_source'] = 'median_imputed'

    # ---- FITS features (red only) ----
    fits_dir = cfg['paths']['fits_cutouts']
    if not os.path.isabs(fits_dir):
        fits_dir = os.path.join(base_dir, fits_dir)
    if os.path.isdir(fits_dir):
        n_fits = len([f for f in os.listdir(fits_dir) if f.endswith('.fits')])
        log.info(f"FITS cutout directory found: {n_fits} files")

        if n_fits > 0:
            import re as _re
            from scipy.spatial import cKDTree
            _pat = _re.compile(r'plate_ra([\d.]+)_dec([\d.-]+)_(RED|BLUE)\.fits')
            _file_coords = {}
            for fn in os.listdir(fits_dir):
                m = _pat.match(fn)
                if m:
                    fra, fdec, band = float(m.group(1)), float(m.group(2)), m.group(3)
                    if band == 'RED':
                        key = (round(fra, 4), round(fdec, 4))
                        _file_coords[key] = os.path.join(fits_dir, fn)
            log.info(f"Red FITS file index: {len(_file_coords)} unique positions")
            _keys = list(_file_coords.keys())
            _tree = cKDTree(np.array(_keys)) if _keys else None
            _max_dist = 0.01

            fits_features_all = []
            candidates = vasco[['source_id', 'ra', 'dec']].copy()
            n_matched = 0
            batch_size = cfg['extraction']['batch_size']
            n_batches = (len(candidates) + batch_size - 1) // batch_size

            for batch_i in tqdm(range(n_batches), desc="Red FITS extraction"):
                start = batch_i * batch_size
                end = min(start + batch_size, len(candidates))
                batch = candidates.iloc[start:end]

                for _, row in batch.iterrows():
                    sid = row['source_id']
                    if sid in already_done:
                        continue
                    red_path = None
                    _try_red = os.path.join(fits_dir, f"{sid}_red.fits")
                    if os.path.exists(_try_red):
                        red_path = _try_red
                    elif _tree is not None:
                        dist, idx = _tree.query([row['ra'], row['dec']])
                        if dist <= _max_dist:
                            red_path = _file_coords[_keys[idx]]
                            n_matched += 1

                    feats = {'source_id': sid}
                    if red_path and os.path.exists(red_path):
                        red_f = extract_fits_features(red_path, cfg)
                        for k, v in red_f.items():
                            feats[f'red_{k}'] = v
                        feats['has_red_fits'] = 1
                    else:
                        feats['has_red_fits'] = 0
                    fits_features_all.append(feats)

            log.info(f"Red FITS KD-tree matched: {n_matched} candidates")
            if fits_features_all:
                fits_df = pd.DataFrame(fits_features_all)
                vasco = vasco.merge(fits_df, on='source_id', how='left')
                n_with_fits = (vasco.get('has_red_fits', 0) == 1).sum()
                vasco.loc[vasco.get('has_red_fits', 0) == 1, 'imputation_source'] = 'actual'
                log.info(f"Red FITS features extracted for {n_with_fits:,d} candidates")
                log.info(f"Median-imputed (no FITS): {len(vasco) - n_with_fits:,d} candidates")
    else:
        log.info(f"No FITS directory at {fits_dir} -- skipping FITS extraction")
        log.info("Pipeline uses catalog-level + plate-aggregate features only")

    # ---- Save feature cache ----
    log.info(f"Saving feature cache to {cache_path}")
    vasco.to_parquet(cache_path, index=False)
    log.info(f"Feature cache: {len(vasco):,d} rows, {len(vasco.columns)} columns")

    log.info("Feature extraction complete")
    return vasco


def main():
    parser = argparse.ArgumentParser(description="VASCO Feature Extraction (Red All-Candidates)")
    parser.add_argument('--config', required=True, help='Path to config_red_allcandidates.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_extraction(cfg)


if __name__ == "__main__":
    main()
