#!/usr/bin/env python3
"""
Master runner for the VASCO ML pipeline (red all-candidates variant).
Orchestrates extraction, training, scoring, and nuclear validation.

Usage:
    python pipeline_run_all.py --config config_red_allcandidates.yaml
    python pipeline_run_all.py --config config_red_allcandidates.yaml --no-extract
    python pipeline_run_all.py --config config_red_allcandidates.yaml --validate-only
    python pipeline_run_all.py --config config_red_allcandidates.yaml --score-only
"""

import argparse
import os
import sys
import time
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="VASCO ML Pipeline (Red All-Candidates) -- Master Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full pipeline:        python pipeline_run_all.py --config config_red_allcandidates.yaml
  Skip FITS extraction: python pipeline_run_all.py --config config_red_allcandidates.yaml --no-extract
  Retrain only:         python pipeline_run_all.py --config config_red_allcandidates.yaml --retrain-only
  Validate only:        python pipeline_run_all.py --config config_red_allcandidates.yaml --validate-only
  Score only:           python pipeline_run_all.py --config config_red_allcandidates.yaml --score-only
""")
    parser.add_argument('--config', required=True, help='Path to config_red_allcandidates.yaml')
    parser.add_argument('--no-extract', action='store_true',
                       help='Skip feature extraction (reuse existing cache)')
    parser.add_argument('--retrain-only', action='store_true',
                       help='Retrain model + score + validate (skip extraction)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run nuclear validation (scored catalog must exist)')
    parser.add_argument('--score-only', action='store_true',
                       help='Only run scoring (trained model must exist)')
    parser.add_argument('--n-workers', type=int, default=None,
                       help='Override n_workers for FITS extraction')
    parser.add_argument('--verbose', action='store_true',
                       help='Increase logging verbosity')
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    output_dir = os.path.join(base_dir, cfg['paths']['output'])
    os.makedirs(output_dir, exist_ok=True)

    if args.n_workers is not None:
        cfg['extraction']['n_workers'] = args.n_workers

    log = setup_logging(output_dir, "pipeline_run")
    if args.verbose:
        for handler in log.handlers:
            handler.setLevel(logging.DEBUG)

    log.info("=" * 70)
    log.info("VASCO ML PIPELINE (RED ALL-CANDIDATES) -- MASTER RUNNER")
    log.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Config: {args.config}")
    log.info(f"Output: {output_dir}")
    log.info("=" * 70)

    run_extract = True
    run_train = True
    run_score = True
    run_validate = True

    if args.validate_only:
        run_extract = run_train = run_score = False
    elif args.score_only:
        run_extract = run_train = False
    elif args.retrain_only or args.no_extract:
        run_extract = False

    stages = []
    if run_extract: stages.append('extract')
    if run_train: stages.append('train')
    if run_score: stages.append('score')
    if run_validate: stages.append('validate')
    log.info(f"Stages: {' -> '.join(stages)}")

    total_start = time.time()
    results = {}

    if run_extract:
        log.info(f"\n{'='*70}")
        log.info("STAGE 1: FEATURE EXTRACTION (RED ALL-CANDIDATES)")
        log.info("=" * 70)
        t0 = time.time()
        try:
            from pipeline_extract_features import run_extraction
            run_extraction(cfg)
            elapsed = time.time() - t0
            log.info(f"Extraction completed in {elapsed:.1f}s")
            results['extract'] = 'OK'
        except Exception as e:
            log.error(f"Extraction failed: {e}")
            results['extract'] = f'FAILED: {e}'
            if not args.no_extract:
                log.error("Cannot continue without feature cache.")
                _print_summary(log, results, total_start)
                sys.exit(1)
    else:
        cache_path = os.path.join(base_dir, cfg['paths']['feature_cache'])
        if not os.path.exists(cache_path) and run_train:
            log.error(f"Feature cache not found: {cache_path}")
            log.error("Run with extraction first, or provide an existing cache.")
            sys.exit(1)
        log.info("Skipping extraction (using existing cache)")
        results['extract'] = 'SKIPPED'

    if run_train:
        log.info(f"\n{'='*70}")
        log.info("STAGE 2: MODEL TRAINING (RED ALL-CANDIDATES)")
        log.info("=" * 70)
        t0 = time.time()
        try:
            from pipeline_train import run_training
            report = run_training(cfg)
            elapsed = time.time() - t0
            log.info(f"Training completed in {elapsed:.1f}s")
            log.info(f"  AUC: {report.get('mean_auc', '?'):.3f}")
            results['train'] = f"OK (AUC={report.get('mean_auc', 0):.3f})"
        except Exception as e:
            log.error(f"Training failed: {e}")
            import traceback
            log.error(traceback.format_exc())
            results['train'] = f'FAILED: {e}'
            if run_score:
                log.error("Cannot score without trained model.")
                _print_summary(log, results, total_start)
                sys.exit(1)
    else:
        results['train'] = 'SKIPPED'

    if run_score:
        log.info(f"\n{'='*70}")
        log.info("STAGE 3: SCORING (RED ALL-CANDIDATES)")
        log.info("=" * 70)
        t0 = time.time()
        try:
            from pipeline_score import run_scoring
            run_scoring(cfg)
            elapsed = time.time() - t0
            log.info(f"Scoring completed in {elapsed:.1f}s")
            results['score'] = 'OK'
        except Exception as e:
            log.error(f"Scoring failed: {e}")
            import traceback
            log.error(traceback.format_exc())
            results['score'] = f'FAILED: {e}'
            if run_validate:
                log.error("Cannot validate without scored catalog.")
                _print_summary(log, results, total_start)
                sys.exit(1)
    else:
        results['score'] = 'SKIPPED'

    if run_validate:
        log.info(f"\n{'='*70}")
        log.info("STAGE 4: NUCLEAR VALIDATION (RED ALL-CANDIDATES)")
        log.info("=" * 70)
        t0 = time.time()
        try:
            from pipeline_validate_nuclear import run_validation
            val_report = run_validation(cfg)
            elapsed = time.time() - t0
            log.info(f"Validation completed in {elapsed:.1f}s")
            mono = val_report.get('monotonicity_pct', '?')
            results['validate'] = f"OK (mono={mono}%)"
        except Exception as e:
            log.error(f"Validation failed: {e}")
            import traceback
            log.error(traceback.format_exc())
            results['validate'] = f'FAILED: {e}'
    else:
        results['validate'] = 'SKIPPED'

    _print_summary(log, results, total_start)


def _print_summary(log: logging.Logger, results: dict, start_time: float):
    total_elapsed = time.time() - start_time
    log.info(f"\n{'='*70}")
    log.info("PIPELINE SUMMARY (RED ALL-CANDIDATES)")
    log.info("=" * 70)
    for stage, status in results.items():
        icon = '[OK]' if status.startswith('OK') else '[SKIP]' if status == 'SKIPPED' else '[FAIL]'
        log.info(f"  {icon} {stage:15s}: {status}")
    log.info(f"\n  Total time: {total_elapsed:.1f}s")

    n_failed = sum(1 for v in results.values() if 'FAILED' in str(v))
    if n_failed > 0:
        log.error(f"\n  {n_failed} stage(s) FAILED")
    else:
        log.info("\n  All stages completed successfully")


if __name__ == "__main__":
    main()
