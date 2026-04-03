#!/usr/bin/env python
"""Run temporal drift experiment on the Lending Club dataset (FLAGSHIP).

This is the primary temporal experiment from the paper:
    - Train a LightGBM model on 2012 reference data
    - Monitor quarterly cohorts from 2013-Q1 through 2018-Q4
    - Also run retrospective analysis on 2008-2011 (crisis period)
    - Compute per-window: SWIFT scores, PSI, SSI, KS, Raw W₁, MMD, BBSD
    - Measure actual model AUC per window
    - Correlate drift scores with AUC degradation (Spearman ρ)

Key research questions:
    RQ1: Does SWIFT correlate better with model degradation than PSI/KS?
    RQ2: Does SWIFT have fewer false alarms during stable periods?

Usage:
    uv run python experiments/run_lending_club.py
    uv run python experiments/run_lending_club.py --fast   # Quick run

Results are saved to results/lending_club_temporal.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lending_club_experiment")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lending Club temporal experiment")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: fewer permutations (100 instead of 500), smaller sample",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/lending_club_temporal.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Max pooled samples for permutation test (0 = no limit)",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Fraction of data to sample (for development; e.g., 0.1)",
    )
    parser.add_argument(
        "--ref-year",
        type=int,
        default=2012,
        help="Year to use as reference period (default: 2012)",
    )
    args = parser.parse_args()

    n_permutations = 100 if args.fast else 500
    max_samples = None if args.max_samples == 0 else args.max_samples

    logger.info("=" * 60)
    logger.info("SWIFT Temporal Experiment: Lending Club Dataset")
    logger.info("=" * 60)
    logger.info(
        "Mode: %s (n_permutations=%d, max_samples=%s)",
        "FAST" if args.fast else "FULL",
        n_permutations,
        max_samples or "unlimited",
    )

    # --- Load dataset ---
    logger.info("Loading Lending Club dataset...")
    t0 = time.time()
    from experiments.data_loader import load_lending_club, create_temporal_splits

    bundle = load_lending_club(sample_frac=args.sample_frac)
    logger.info(
        "Dataset loaded in %.1fs: %s",
        time.time() - t0,
        bundle,
    )

    feature_names = bundle.feature_names
    numeric_features = bundle.numeric_features
    categorical_features = bundle.categorical_features

    # --- Create temporal splits ---
    logger.info("Creating quarterly temporal splits...")
    splits = create_temporal_splits(bundle, period="Q", min_window_size=500)
    logger.info("Created %d quarterly windows", len(splits))
    for label, X_w, y_w in splits:
        logger.info(
            "  %s: n=%d, default_rate=%.3f",
            label, len(X_w), y_w.mean(),
        )

    # --- Identify reference period ---
    ref_year = args.ref_year
    ref_windows = [
        (label, X_w, y_w) for label, X_w, y_w in splits
        if label.startswith(str(ref_year))
    ]
    if not ref_windows:
        logger.error("No windows found for reference year %d!", ref_year)
        raise ValueError(f"No data found for reference year {ref_year}")

    # Combine reference windows
    X_ref = pd.concat([xw for _, xw, _ in ref_windows], ignore_index=True)
    y_ref = pd.concat([yw for _, _, yw in ref_windows], ignore_index=True)
    ref_labels = [label for label, _, _ in ref_windows]
    logger.info(
        "Reference period: %s (n=%d, default_rate=%.3f)",
        "+".join(ref_labels), len(X_ref), y_ref.mean(),
    )

    # Monitoring windows = everything except reference
    ref_label_set = set(ref_labels)
    mon_windows = [
        (label, X_w, y_w) for label, X_w, y_w in splits
        if label not in ref_label_set
    ]
    logger.info("Monitoring windows: %d", len(mon_windows))

    # --- Train model on reference data ---
    logger.info("Training LightGBM model on reference data...")
    from experiments.runner_base import train_model

    model, ref_auc = train_model(
        X_ref, y_ref, feature_names,
        categorical_features=categorical_features,
        random_state=42,
        num_boost_round=200,
    )
    logger.info("Reference model AUC: %.4f", ref_auc)

    # --- Fit SWIFT on reference data ---
    logger.info("Fitting SWIFTMonitor on reference data...")
    from swift.pipeline import SWIFTMonitor

    t_fit = time.time()
    monitor = SWIFTMonitor(
        model=model,
        n_permutations=n_permutations,
        alpha=0.05,
        max_samples=max_samples,
        random_state=42,
    )
    monitor.fit(X_ref[feature_names])
    fit_time = time.time() - t_fit
    logger.info("SWIFT fit complete in %.1fs", fit_time)

    # Get SHAP values for baselines
    shap_values_ref = monitor.shap_values_

    # --- Run per-window analysis ---
    logger.info("Running per-window drift analysis...")
    from experiments.baselines import run_all_baselines
    from experiments.evaluation import (
        compute_model_performance,
        compute_temporal_drift_analysis,
    )

    period_labels: list[str] = []
    swift_max_scores: list[float] = []
    swift_mean_scores: list[float] = []
    model_aucs: list[float] = []
    all_baseline_max_scores: dict[str, list[float]] = {
        "PSI": [], "SSI": [], "KS": [], "RawW1": [], "MMD": [],
        "BBSD_KS": [], "BBSD_PSI": [],
    }

    # Per-window detailed results
    window_results: list[dict] = []

    test_rng = np.random.default_rng(42 + 2000)

    for label, X_w, y_w in mon_windows:
        logger.info("Processing window %s (n=%d)...", label, len(X_w))

        t_win = time.time()

        # SWIFT scoring
        scenario_seed = test_rng.integers(0, 2**31)
        monitor.set_params(random_state=int(scenario_seed))
        swift_result = monitor.test(X_w[feature_names])

        swift_scores = {
            r.feature_name: r.swift_score
            for r in swift_result.feature_results
        }
        swift_pvalues = {
            r.feature_name: r.p_value
            for r in swift_result.feature_results
        }

        # Baselines
        baseline_scores = run_all_baselines(
            X_ref=X_ref[feature_names],
            X_mon=X_w[feature_names],
            feature_names=feature_names,
            shap_values=shap_values_ref,
            bucket_sets=monitor.bucket_sets_,
            model=model,
        )

        # Model performance on this window
        y_pred_w = model.predict(X_w[feature_names])
        win_auc = compute_model_performance(y_w.values, y_pred_w)

        # Record
        period_labels.append(label)
        swift_max_scores.append(swift_result.swift_max)
        swift_mean_scores.append(swift_result.swift_mean)
        model_aucs.append(win_auc)

        # Max baseline scores per method
        for method in all_baseline_max_scores:
            if method in baseline_scores and baseline_scores[method]:
                max_score = max(baseline_scores[method].values())
            else:
                max_score = 0.0
            all_baseline_max_scores[method].append(max_score)

        win_time = time.time() - t_win

        logger.info(
            "  %s: AUC=%.4f (ΔAUC=%.4f), SWIFT_max=%.4f, "
            "PSI_max=%.4f, KS_max=%.4f, BBSD_KS=%.4f [%.1fs]",
            label, win_auc, ref_auc - win_auc,
            swift_result.swift_max,
            all_baseline_max_scores["PSI"][-1],
            all_baseline_max_scores["KS"][-1],
            all_baseline_max_scores["BBSD_KS"][-1],
            win_time,
        )

        # Store detailed per-window result
        window_results.append({
            "period": label,
            "n_observations": len(X_w),
            "default_rate": float(y_w.mean()),
            "model_auc": win_auc,
            "auc_degradation": ref_auc - win_auc,
            "swift_max": swift_result.swift_max,
            "swift_mean": swift_result.swift_mean,
            "swift_n_drifted": len(swift_result.drifted_features),
            "swift_scores": swift_scores,
            "swift_pvalues": swift_pvalues,
            "baseline_max_scores": {
                method: all_baseline_max_scores[method][-1]
                for method in all_baseline_max_scores
            },
        })

    # --- Compute Spearman correlations ---
    logger.info("Computing Spearman correlations...")
    drift_scores_by_method: dict[str, np.ndarray] = {
        "SWIFT_max": np.array(swift_max_scores),
        "SWIFT_mean": np.array(swift_mean_scores),
    }
    for method, scores in all_baseline_max_scores.items():
        drift_scores_by_method[f"{method}_max"] = np.array(scores)

    temporal_result = compute_temporal_drift_analysis(
        period_labels=period_labels,
        drift_scores_by_method=drift_scores_by_method,
        model_aucs=np.array(model_aucs),
        ref_auc=ref_auc,
    )

    # --- Save results ---
    total_time = time.time() - t0

    output_data = {
        "dataset_name": "lending_club",
        "experiment_type": "temporal",
        "ref_year": ref_year,
        "ref_labels": ref_labels,
        "ref_n": len(X_ref),
        "ref_auc": ref_auc,
        "fit_time_seconds": fit_time,
        "total_time_seconds": total_time,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_monitoring_windows": len(mon_windows),
        "window_results": window_results,
        "spearman_correlations": {
            method: {
                "rho": temporal_result.spearman_rho[method],
                "pvalue": temporal_result.spearman_pvalue[method],
            }
            for method in temporal_result.spearman_rho
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def json_default(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    output_path.write_text(json.dumps(output_data, indent=2, default=json_default))
    logger.info("Results saved to %s", args.output)

    # --- Print summary ---
    print()
    print("=" * 70)
    print("RESULTS SUMMARY: Lending Club Temporal Drift Experiment")
    print("=" * 70)
    print(f"  Reference period: {'+'.join(ref_labels)} (n={len(X_ref)}, AUC={ref_auc:.4f})")
    print(f"  Monitoring windows: {len(mon_windows)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Total time: {total_time:.1f}s (fit: {fit_time:.1f}s)")
    print()

    # --- Per-window results table ---
    print("-" * 100)
    print(
        f"{'Period':<10} {'n':>7} {'AUC':>7} {'ΔAUC':>7} "
        f"{'SWIFT_max':>10} {'PSI_max':>9} {'KS_max':>8} "
        f"{'BBSD_KS':>9} {'Drifted':>8}"
    )
    print("-" * 100)
    for wr in window_results:
        print(
            f"{wr['period']:<10} {wr['n_observations']:>7} "
            f"{wr['model_auc']:>7.4f} {wr['auc_degradation']:>+7.4f} "
            f"{wr['swift_max']:>10.4f} "
            f"{wr['baseline_max_scores']['PSI']:>9.4f} "
            f"{wr['baseline_max_scores']['KS']:>8.4f} "
            f"{wr['baseline_max_scores'].get('BBSD_KS', 0.0):>9.4f} "
            f"{wr['swift_n_drifted']:>4}/{len(feature_names)}"
        )

    # --- Spearman correlations ---
    print()
    print("-" * 50)
    print("SPEARMAN ρ (drift score vs AUC degradation)")
    print("-" * 50)
    for method, rho_info in sorted(
        output_data["spearman_correlations"].items(),
        key=lambda x: abs(x[1]["rho"]),
        reverse=True,
    ):
        rho = rho_info["rho"]
        pval = rho_info["pvalue"]
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {method:<15} ρ={rho:+.4f}  (p={pval:.4f}) {sig}")

    print()
    logger.info("Done!")


if __name__ == "__main__":
    main()
