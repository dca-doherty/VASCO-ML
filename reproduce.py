#!/usr/bin/env python3
"""
reproduce.py
One-command reproduction of all VASCO-ML paper results.

Workflow:
  1. Check Python version and required packages
  2. Check data files exist in data/
  3. Run pipeline (skip if scored_catalog exists unless --force)
  4. Run Framework A scripts (shadow + nuclear + combine WLS)
  5. Run Framework B scripts (deciles + permutation + compact substructures)
  6. Verify every output against expected_outputs.json tolerances
  7. Print PASS / FAIL summary with deltas

Usage:
    python reproduce.py                       # full run, skip completed stages
    python reproduce.py --force               # re-run everything from scratch
    python reproduce.py --verify-only         # skip running, just check outputs
    python reproduce.py --skip-pipeline       # assume scored_catalog.xlsb exists
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MLSCRIPTS = REPO_ROOT / "MLscripts"
DATA_DIR  = REPO_ROOT / "data"
RESULTS   = REPO_ROOT / "results"
CONFIG    = MLSCRIPTS / "config_red_allcandidates.yaml"
EXPECTED  = REPO_ROOT / "expected_outputs.json"

# ----- environment checks -----

MIN_PYTHON = (3, 11)
REQUIRED_PKGS = [
    "numpy", "pandas", "scipy", "sklearn", "xgboost",
    "lightgbm", "statsmodels", "matplotlib", "astropy", "pyxlsb",
]

# ----- required data files (small, should live in data/) -----

REQUIRED_DATA = [
    "SUPERVIKTIG_HELAVASCO_validated_v4.csv",
    "poss_i_observing_dates_strict.csv",
    "nuclear_tests_1949_1957.csv",
    "san_diego_daily_cloud_cover_1949_1957.csv",
    "gfz_solar_1949_1957_filled.csv",
    "merged_validation_data_2.csv",
]

# ----- pipeline stages (label, script_path_relative_to_mlscripts, key_output_relative_to_repo) -----

STAGES = [
    ("Core pipeline (train + score)",
        ["pipeline_run_all.py", "--config", str(CONFIG), "--no-extract"],
        "results/ml_pipeline_red_allcandidates/scored_catalog.csv"),
    ("Shadow deficit by decile",
        ["shadow_deficit_deciles.py"],
        "results/shadow_deficit_deciles/shadow_deficit_deciles.csv"),
    ("Nuclear permutation by decile",
        ["nuclear_permutation_deciles.py"],
        "results/nuclear_permutation_deciles/primary_tests_summary.csv"),
    ("Compact substructures",
        ["compact_substructures.py"],
        "results/compact_substructures_summary.csv"),
]


# =====================================================================

def run_command(cmd, cwd=None, label=""):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    start = time.time()
    r = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - start
    if r.returncode != 0:
        print(f"[FAIL] {label} failed (exit {r.returncode}, {elapsed:.1f}s)")
        sys.exit(1)
    print(f"[OK  ] {label} ({elapsed:.1f}s)")


def check_environment():
    print("Environment check:")
    if sys.version_info < MIN_PYTHON:
        print(f"[FAIL] Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, "
              f"got {sys.version_info.major}.{sys.version_info.minor}")
        sys.exit(1)
    print(f"  python: {sys.version.split()[0]}")
    print(f"  platform: {platform.platform()}")

    missing = []
    for pkg in REQUIRED_PKGS:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[FAIL] Missing packages: {', '.join(missing)}")
        print("       Run: pip install -r requirements.txt")
        sys.exit(1)
    print(f"  packages: all {len(REQUIRED_PKGS)} present")


def check_data():
    print("Data check:")
    missing = [f for f in REQUIRED_DATA if not (DATA_DIR / f).exists()]
    if missing:
        print(f"[FAIL] Missing data files in {DATA_DIR}:")
        for f in missing:
            print(f"    data/{f}")
        print("       See README 'Data' section for sources.")
        sys.exit(1)
    print(f"  all {len(REQUIRED_DATA)} data files present in data/")


def run_stages(force=False, skip_pipeline=False):
    print("Running pipeline stages:")
    for i, (label, args, output_rel) in enumerate(STAGES, 1):
        if skip_pipeline and i == 1:
            print(f"[SKIP] {label} (--skip-pipeline)")
            continue
        output = REPO_ROOT / output_rel
        if output.exists() and not force:
            print(f"[SKIP] {label} (output exists: {output_rel})")
            continue
        cmd = [sys.executable, str(MLSCRIPTS / args[0])] + list(args[1:])
        print(f"\nStage {i}/{len(STAGES)}: {label}")
        run_command(cmd, cwd=REPO_ROOT, label=label)


# ----- output verification -----

def get_value(spec):
    """Extract a single numeric value from a JSON or CSV per the spec.

    Spec shape:
      {"file": "path/to/file.json", "key": "mean_auc"}
      {"file": "path/to/file.csv",  "row": {"threshold": 0.0}, "col": "irr"}
      {"file": "path/to/file.csv",  "col": "count", "agg": "sum"}
    """
    path = REPO_ROOT / spec["file"]
    if not path.exists():
        return None, f"file not found: {spec['file']}"

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        key = spec["key"]
        for part in key.split("."):
            if isinstance(data, dict) and part in data:
                data = data[part]
            else:
                return None, f"key '{key}' not found in {spec['file']}"
        return float(data), None

    if path.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        if "row" in spec:
            filt = df.copy()
            for k, v in spec["row"].items():
                if isinstance(v, float):
                    filt = filt[(filt[k] - v).abs() < 1e-6]
                else:
                    filt = filt[filt[k] == v]
            if len(filt) == 0:
                return None, f"no row matching {spec['row']} in {spec['file']}"
            return float(filt[spec["col"]].iloc[0]), None
        if "agg" in spec:
            if spec["agg"] == "sum":
                return float(df[spec["col"]].sum()), None
            if spec["agg"] == "count":
                return float(len(df)), None
        return float(df[spec["col"]].iloc[0]), None

    return None, f"unsupported file type: {path.suffix}"


def verify_outputs():
    print("\n" + "=" * 72)
    print("VERIFICATION (against expected_outputs.json)")
    print("=" * 72)

    if not EXPECTED.exists():
        print(f"[FAIL] {EXPECTED.name} not found")
        sys.exit(1)

    with open(EXPECTED) as f:
        spec = json.load(f)

    passed = []
    failed = []
    skipped = []

    for check in spec["checks"]:
        name = check["name"]
        actual, err = get_value(check)
        expected = float(check["value"])
        tol = float(check["tolerance"])

        if err:
            skipped.append((name, err))
            print(f"[SKIP] {name}: {err}")
            continue

        delta = abs(actual - expected)
        status = "PASS" if delta <= tol else "FAIL"
        msg = (f"{name}: actual={actual:.4f}  expected={expected:.4f} "
               f"+/-{tol}  delta={delta:.4f}")
        if status == "PASS":
            passed.append(name)
            print(f"[ PASS ] {msg}")
        else:
            failed.append((name, msg))
            print(f"[ FAIL ] {msg}")

    print()
    print("-" * 72)
    print(f"  {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    print("-" * 72)
    return failed, skipped


def main():
    ap = argparse.ArgumentParser(
        description="One-command reproduction of VASCO-ML paper results")
    ap.add_argument("--force", action="store_true",
                    help="Re-run all stages even if outputs exist")
    ap.add_argument("--verify-only", action="store_true",
                    help="Skip running stages, only verify existing outputs")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="Skip core ML pipeline stage (assume scored_catalog.xlsb exists)")
    args = ap.parse_args()

    print("=" * 72)
    print("VASCO-ML ONE-COMMAND REPRODUCTION")
    print("=" * 72)

    check_environment()
    check_data()

    if not args.verify_only:
        run_stages(force=args.force, skip_pipeline=args.skip_pipeline)

    failed, skipped = verify_outputs()

    if failed:
        print(f"\n[FAIL] Reproduction incomplete: {len(failed)} numeric checks failed")
        print("       Likely causes: stale outputs, dependency version drift, "
              "or pipeline not re-run after LightGBM patch")
        sys.exit(1)
    if skipped:
        print(f"\n[WARN] {len(skipped)} checks skipped (missing outputs). "
              "Run without --verify-only to generate them.")
        sys.exit(2)
    print("\n[PASS] ALL CHECKS PASSED -- paper numbers reproduced")


if __name__ == "__main__":
    main()
