# VASCO-ML

Machine learning classification pipeline for anomalous transient sources detected on historical Palomar Observatory Sky Survey (POSS-I) photographic plates (1949-1958), developed as part of the VASCO (Vanishing and Appearing Sources during a Century of Observations) project.

This repository contains the code used in:

> Bruehl, S., Doherty, B., Villarroel, B., et al. (2026). *Machine Learning Validation of Anomalous Photographic Plate Transients and Their Association with Nuclear Testing.* [In preparation for Nature.]

## What This Does

The pipeline takes a catalog of ~108,000 transient candidates detected on 1950s red-sensitive photographic plates and classifies each one as a likely real transient or a likely artifact (emulsion defect, cosmic ray hit, plate flaw, etc.). It does this using 23 morphometric features extracted from the original catalog metadata and directly from digitized FITS plate scans.

The classifier is a calibrated ensemble of four models (Random Forest, Gradient Boosting, XGBoost, LightGBM) trained on a small set of human-labeled examples and validated against two independent physical signals:

1. **Earth's shadow deficit** -- real transients in geostationary orbit should avoid Earth's shadow zone. The classifier-selected candidates show a monotonically deepening shadow deficit with ML confidence, reaching 100% absence at high probability thresholds.

2. **Nuclear testing association** -- transient counts are elevated on nights within one day of US atmospheric nuclear tests. This association strengthens monotonically with ML confidence (IRR from 1.76 at the full catalog to 15.70 at the highest probability threshold).

Neither of these physical signals was used during training. The fact that a morphology-only classifier recovers them independently is the core finding of the paper.

## Pipeline Structure

The pipeline runs in four stages, orchestrated by `pipeline_run_all.py`:

```
pipeline_extract_features.py   Extract morphometric features from FITS plate scans
        |
pipeline_train.py              Train ensemble classifier, cross-validate, compute SHAP values
        |
pipeline_score.py              Score all ~108k candidates, apply calibration
        |
pipeline_validate_nuclear.py   Validate against nuclear testing timeline and Earth's shadow
```

Supporting modules:

- `common.py` -- shared utilities, feature encoding, plate-level aggregation
- `shadow_deficit_analysis_all_red.py` -- standalone shadow deficit analysis with Monte Carlo control
- `pipeline_active_learning.py` -- active learning sampling strategy for label acquisition
- `config_red_allcandidates.yaml` -- all hyperparameters, file paths, and pipeline settings

## Requirements

Python 3.11+

Key dependencies:

```
scikit-learn>=1.7
xgboost>=3.1
shap>=0.50
numpy>=2.3
pandas>=2.3
scipy>=1.16
statsmodels>=0.14
matplotlib>=3.10
astropy>=7.2
astroquery>=0.4
photutils>=2.3
sep>=1.4
joblib>=1.5
openpyxl>=3.1
PyYAML>=6.0
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

All commands assume you are in the pipeline directory and have the config file and data in place.

**Full pipeline** (extraction through validation):

```bash
python pipeline_run_all.py --config config_red_allcandidates.yaml
```

**Skip FITS extraction** (reuse cached features):

```bash
python pipeline_run_all.py --config config_red_allcandidates.yaml --no-extract
```

**Retrain + score + validate** (most common during development):

```bash
python pipeline_run_all.py --config config_red_allcandidates.yaml --retrain-only
```

**Individual stages:**

```bash
python pipeline_train.py --config config_red_allcandidates.yaml
python pipeline_score.py --config config_red_allcandidates.yaml
python pipeline_validate_nuclear.py --config config_red_allcandidates.yaml
```

## Features

The classifier uses 23 features organized into three groups:

**Catalog-level** (from the Solano et al. detection pipeline): signal-to-noise ratio, sharpness, elongation, compactness, PSF ratio, candidate score, number of comparison stars.

**Plate-level aggregates** (computed per observation date): fraction of low-SNR detections, SNR standard deviation, mean elongation, count of high-SNR detections, plate quality indicators.

**Red FITS morphometry** (extracted directly from digitized plate scans): FWHM, aperture flux, ellipticity, second-order sharpness, connected pixel count, gradient magnitude, radial symmetry, distance to plate edge, proximity to bright stars, SNR from pixel data.

All features are red-band only. Blue-band and spectral features were deliberately excluded to avoid color-dependent biases on photographic emulsions.

## Data

The input catalog (`SUPERVIKTIG_HELAVASCO_validated_v4.csv`) is derived from the original VASCO detection pipeline (Solano et al.) and contains 107,875 transient candidates with coordinates, observation dates, and catalog-level morphometric measurements.

The scored output (`scored_catalog.xlsb`) adds ML probability, prediction, and per-candidate SHAP feature attributions for every candidate.

FITS plate scans are accessed from IRSA (STScI Digitized Sky Survey) and are not included in this repository due to size.

## Output

The pipeline produces:

- `scored_catalog.xlsb` -- full catalog with ML probabilities
- `model_final.joblib` -- trained ensemble model
- `calibrator.joblib` -- probability calibration model
- `threshold_sweep.csv` -- nuclear association results at 19+ thresholds
- `feature_importance.csv` / `feature_importance_shap.csv` -- feature rankings
- `chart_shap_summary.png` -- SHAP beeswarm plot (23 features)
- `chart_training_diagnostics.png` -- CV and calibration diagnostics
- `chart_nuclear_validation.png` -- IRR vs threshold with confidence intervals
- Training, scoring, and validation logs

## Configuration

All parameters are in `config_red_allcandidates.yaml`. Key settings:

- **Ensemble**: RF + GBM + XGBoost + LightGBM, mean probability aggregation
- **Calibration**: Isotonic
- **RF**: max_depth=8, min_samples_leaf=3, n_estimators=300
- **GBM**: max_depth=4, min_samples_leaf=3, n_estimators=300
- **XGB**: max_depth=4, learning_rate=0.05, min_child_weight=3, n_estimators=300
- **LightGBM**: max_depth=4, learning_rate=0.05, min_child_samples=3, n_estimators=300
- **Cross-validation**: 5-fold stratified
- **Nuclear window**: +/-1 day around US atmospheric tests (1949-1957)
- **Shadow model**: 3D topocentric penumbra at GEO altitude (8.96 deg half-angle)

## Authors

- Stephen Bruehl, Ph.D. -- Vanderbilt University Medical Center (lead author)
- Brian Doherty -- Independent Researcher, Dallas, TX (ML pipeline)
- Beatriz Villarroel -- Stockholm University / NORDITA (PI, VASCO project)

## License

This code is provided for research reproducibility. Please cite the associated publication if you use it in your work.
