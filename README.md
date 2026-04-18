# VASCO-ML

Machine learning classification pipeline for anomalous transient sources detected on historical Palomar Observatory Sky Survey (POSS-I) photographic plates (1949-1957), developed as part of the VASCO (Vanishing and Appearing Sources during a Century of Observations) project.

This repository contains the code and data used in:

> Bruehl, S., Doherty, B., Streblyanska, A., & Villarroel, B. (2026). *Machine Learning Supports Existence of Previously Unrecognized Transient Astronomical Phenomena in Historical Observatory Images.* Scientific Reports, in press.

## What this does

The pipeline takes a catalog of 107,875 transient candidates detected on 1950s red-sensitive photographic plates and classifies each one as a likely real transient or a likely artifact (emulsion defect, cosmic ray hit, plate flaw). It does this using 23 morphometric features extracted from catalog metadata and directly from digitized FITS plate scans.

The classifier is a four-model ensemble (Random Forest, Gradient Boosting, XGBoost, LightGBM) trained on 250 human-labeled examples (134 real, 116 defects) and validated against two physical signals that were never used during training:

1. **Earth's shadow deficit.** Real transients in geostationary orbit should disappear when Earth blocks the Sun. The highest-probability decile (D9) shows the largest shadow deficit: 0.31% in shadow versus 0.692% geometric expectation (55.2% deficit, p = 2.0e-07 by binomial test). D9 shadow fraction is significantly lower than the pooled D0-D8 average (two-proportion z = 2.78, p = 0.003).
2. **Nuclear testing association.** Transient counts are elevated on nights within +/- 1 day of US atmospheric nuclear tests. Date-level permutation (10,000 iterations, probability-weighted counts): ratio = 1.84, p = 0.024. Mann-Whitney U: p = 0.002. The percentage of transients in D9 falling within a nuclear window (13.5%) exceeds D0 (8.3%).

Neither signal was used during training. A morphology-only classifier recovers both independently, which is the core finding of the paper.

## Reproduce in one command

Every paper number can be regenerated from the raw inputs with a single command. `reproduce.py` orchestrates the full pipeline, then verifies 22 key numeric outputs against `expected_outputs.json`.

```bash
git clone https://github.com/dca-doherty/VASCO-ML.git
cd VASCO-ML
pip install -r requirements.txt
python reproduce.py --verify-only
```

Expected output on a fresh clone:

```
[ PASS ] N candidates in scored catalog: actual=107875 expected=107875
[ PASS ] Cross-validation mean AUC: actual=0.8085 expected=0.809
[ PASS ] Cross-validation mean sensitivity: actual=0.7088 expected=0.709
[ PASS ] D9 shadow fraction: actual=0.3102 expected=0.310
[ PASS ] D9 shadow deficit percent: actual=-55.18 expected=-55.2
[ PASS ] D9 shadow binomial p-value: actual=2.0e-07 expected=2.0e-07
[ PASS ] D9 vs pooled D0-D8 shadow z-stat: actual=-2.779 expected=-2.78
[ PASS ] Nuclear permutation test p-value: actual=0.0249 expected=0.024
[ PASS ] Nuclear permutation ratio: actual=1.84 expected=1.84
[ PASS ] Nuclear Mann-Whitney U p-value: actual=0.0023 expected=0.002
[ PASS ] D9 nuclear percent: actual=13.55 expected=13.55
[ PASS ] D0 nuclear percent: actual=8.31 expected=8.30
[ PASS ] Compact substructures doublets in D9: actual=549 expected=549
[ PASS ] Compact substructures triplets in D9: actual=143 expected=143
... (22 checks total)
------------------------------------------------------------------------
  22 passed, 0 failed, 0 skipped
------------------------------------------------------------------------
[PASS] ALL CHECKS PASSED -- paper numbers reproduced
```

Flags:

- `--force` re-run every stage from scratch (ignore cached outputs)
- `--skip-pipeline` assume `scored_catalog.xlsb` already exists, only run analyses
- `--verify-only` check existing outputs without re-running anything

Total runtime on a modern workstation: approximately 100 minutes for the full pipeline, 5-10 minutes with `--skip-pipeline`, under 30 seconds for `--verify-only`.

## Pipeline stages

1. **`pipeline_extract_features.py`** - Extracts 23 morphometric features from FITS plate scans
2. **`pipeline_train.py`** - Trains the four-model ensemble with 5-fold cross-validation, computes SHAP values
3. **`pipeline_score.py`** - Assigns ML probability to each of the 107,875 candidates
4. **`shadow_deficit_deciles.py`** - Per-decile binomial tests, D9 vs pooled D0-D8 z-test
5. **`nuclear_permutation_deciles.py`** - 10,000-iteration permutation test on date-level probability-weighted counts, Mann-Whitney U, D9 vs pooled z-test, lag sensitivity from -3 to +3 days
6. **`compact_substructures.py`** - Doublets, triplets, and larger groupings within D9 (within 10 arcmin, same night)

Supporting modules:

- `common.py` - Shared utilities, four-model ensemble definition, feature encoding
- `pipeline_active_learning.py` - Active learning strategy for label acquisition
- `pipeline_validate_nuclear.py` - Secondary nuclear validation (exploratory)
- `shadow_deficit_analysis_all_red.py` - Exploratory shadow analysis across 2D/3D models
- `config_red_allcandidates.yaml` - All hyperparameters and file paths

## Features

The classifier uses 23 features:

- **Catalog-level** (from Solano et al. detection pipeline): SNR, sharpness, elongation, compactness, PSF ratio, candidate score, number of comparison stars
- **Plate-level aggregates**: low-SNR fraction, SNR standard deviation, mean elongation, high-SNR count, plate quality indicators
- **Red FITS morphometry**: FWHM, aperture flux, ellipticity, second-order sharpness, connected pixel count, gradient magnitude, radial symmetry, distance to plate edge, bright-star proximity, pixel-level SNR

All features are red-band only. Blue-band and spectral features were excluded to avoid color-dependent biases on photographic emulsions.

## Data

Included in `data/`:

- `SUPERVIKTIG_HELAVASCO_validated_v4.csv` - 107,875 transient candidates from the Solano et al. detection pipeline
- `poss_i_observing_dates_strict.csv` - 370 POSS-I observation dates
- `nuclear_tests_1949_1957.csv` - US, USSR, UK atmospheric test catalog
- `san_diego_daily_cloud_cover_1949_1957.csv` - NOAA ISD WBAN 23188 Lindbergh Field daily records
- `gfz_solar_1949_1957_filled.csv` - GFZ Potsdam solar data (Matzka et al. 2021)
- `merged_validation_data_2.csv` - per-date covariate data

Not included (regenerated by pipeline or external):

- FITS plate scans (available from IRSA/STScI)
- Trained model artifacts (regenerated by pipeline)
- Validated label files containing expert annotations

## Committed results

Included in `results/`:

- `shadow_deficit_deciles/shadow_deficit_deciles.csv` - per-decile binomial tests
- `shadow_deficit_deciles/shadow_summary_tests.csv` - D9 z-tests
- `shadow_deficit_deciles/figure3_shadow_deficit.png/.pdf` - publication Figure 3
- `nuclear_permutation_deciles/decile_nuclear_pct.csv` - per-decile nuclear fractions
- `nuclear_permutation_deciles/primary_tests_summary.csv` - permutation, MWU, D9 z-test
- `nuclear_permutation_deciles/lag_sensitivity.csv` - -3 to +3 day lag results
- `nuclear_permutation_deciles/figure4_nuclear_window_pct.png/.pdf` - publication Figure 4
- `compact_substructures.csv`, `compact_substructures_summary.csv` - D9 doublets, triplets, larger groupings
- `ml_pipeline_red_allcandidates/training_report.json` - 5-fold CV metrics
- `ml_pipeline_red_allcandidates/scored_catalog_summary.json` - catalog size summary

## Configuration

ML parameters in `config_red_allcandidates.yaml`:

- **Ensemble:** RF + GBM + XGBoost + LightGBM, mean probability aggregation
- **Calibration:** Isotonic attempted but collapsed probability distribution; raw mean ensemble probabilities used
- **RF:** max_depth=8, min_samples_leaf=3, n_estimators=300
- **GBM:** max_depth=4, min_samples_leaf=3, n_estimators=300
- **XGB:** max_depth=4, learning_rate=0.05, min_child_weight=3, n_estimators=300
- **LightGBM:** max_depth=4, learning_rate=0.05, min_child_samples=3, n_estimators=300
- **Cross-validation:** 5-fold stratified, seed=42
- **Decile assignment:** 10 equal-sized bins of the 99,933 transients on strict POSS-I dates (107,875 catalog minus 7,942 on 34 dates outside the strict list)
- **Shadow model:** Geocentric anti-sun direction at 35,786 km GEO, 8.50 degree cone
- **Shadow expected rate:** 0.692% (plate-aware Monte Carlo, 626 plates x 100 positions = 62,600 control positions)
- **Nuclear window:** +/- 1 day around US atmospheric tests, 26 of 370 POSS-I dates, 95 unique test dates
- **Permutation:** 10,000 iterations, seed=42, probability-weighted date-level counts

## Requirements

Python 3.11 or newer. Install with:

```bash
pip install -r requirements.txt
```

Key dependencies: scikit-learn, xgboost, lightgbm, shap, numpy, pandas, scipy, statsmodels, matplotlib, astropy, photutils, joblib, pyxlsb.

## Reproducibility

All random seeds fixed at 42. YAML config records hyperparameters and file paths. Feature cache (Parquet) preserves the exact feature matrix. CV fold assignments are deterministic. Approximate runtimes: ~100 minutes for full pipeline, ~5 minutes shadow + nuclear + compact, under 30 seconds for `--verify-only`.

## Methods documentation

`STATISTICAL_METHODS.md` contains the full statistical methods description matching the submitted paper.

## Authors

- Stephen Bruehl, Ph.D. - Vanderbilt University Medical Center (corresponding author)
- Brian Doherty - Independent Researcher, Dallas, TX (ML pipeline, statistical validation)
- Alina Streblyanska - Instituto de Astrofisica de Canarias (independent verification)
- Beatriz Villarroel - Stockholm University / NORDITA (VASCO project PI)

## License

MIT License for code. Research datasets in this repository are made available for reproducibility of the associated publication. Please cite the paper if you use the code or data.
