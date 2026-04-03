#!/usr/bin/env python
"""Run controlled drift experiment on the Bank Marketing dataset.

This script runs all drift scenarios (S1-S9) at multiple magnitudes,
comparing SWIFT against PSI, SSI, KS, Raw W₁, and MMD baselines.

Bank Marketing (UCI id=222) has 45K rows and 20 features (after removing
'duration' for data-leakage reasons).  It includes a mix of numeric and
categorical features, making it a good complement to the Taiwan Credit
dataset which is almost entirely numeric.

Usage:
    uv run python experiments/run_bank_marketing.py
    uv run python experiments/run_bank_marketing.py --fast   # Quick run (fewer permutations)

Results are saved to results/bank_marketing_controlled.json
"""

from __future__ import annotations

import argparse
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bank_marketing_experiment")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bank Marketing controlled experiment")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: fewer permutations (200 instead of 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/bank_marketing_controlled.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Max pooled samples for permutation test (0 = no limit)",
    )
    args = parser.parse_args()

    n_permutations = 200 if args.fast else 1000
    max_samples = None if args.max_samples == 0 else args.max_samples

    logger.info("=" * 60)
    logger.info("SWIFT Controlled Experiment: Bank Marketing Dataset")
    logger.info("=" * 60)
    logger.info(
        "Mode: %s (n_permutations=%d, max_samples=%s)",
        "FAST" if args.fast else "FULL",
        n_permutations,
        max_samples or "unlimited",
    )

    # --- Load dataset ---
    logger.info("Loading Bank Marketing dataset...")
    t0 = time.time()
    from experiments.data_loader import load_bank_marketing

    bundle = load_bank_marketing()
    logger.info(
        "Dataset loaded in %.1fs: %s",
        time.time() - t0,
        bundle,
    )

    # --- Configure experiment ---
    from experiments.runner_base import ExperimentConfig
    from experiments.runner_controlled import run_controlled_experiment

    # Bank Marketing has both numeric and categorical features, so include S6
    config = ExperimentConfig(
        dataset_name="bank_marketing",
        scenarios=["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"],
        magnitudes=[0.5, 1.0, 2.0, 3.0],
        # Per-scenario magnitude overrides are in ExperimentConfig defaults:
        # S5: [0.05, 0.10, 0.20, 0.50]
        # S6: [0.10, 0.25, 0.50, 1.00]
        # S7: [0.10, 0.20, 0.50, 1.00]
        # S8: [0.25, 0.50, 0.75, 1.00]
        n_permutations=n_permutations,
        alpha=0.05,
        ref_fraction=0.6,
        n_features_to_drift=3,
        max_samples=max_samples,
        random_state=42,
    )

    logger.info("Experiment config:")
    logger.info("  Scenarios: %s", config.scenarios)
    for sc in config.scenarios:
        logger.info("    %s magnitudes: %s", sc, config.magnitudes_for(sc))
    logger.info("  n_permutations: %d", config.n_permutations)
    logger.info("  ref_fraction: %.2f", config.ref_fraction)

    # --- Run experiment ---
    logger.info("Starting experiment...")
    result = run_controlled_experiment(
        X=bundle.X,
        y=bundle.y,
        feature_names=bundle.feature_names,
        numeric_features=bundle.numeric_features,
        categorical_features=bundle.categorical_features,
        config=config,
    )

    # --- Print summary ---
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(result.summary())
    print()

    # --- Print detection highlights ---
    print("-" * 70)
    print("DETECTION HIGHLIGHTS")
    print("-" * 70)

    for sr in result.scenario_results:
        n_drifted = len(sr.swift_drifted)
        n_features = len(sr.swift_scores)
        gt_drifted = len(sr.drifted_features)

        # Get max PSI and max KS for comparison
        max_psi = (
            max(sr.baseline_scores.get("PSI", {}).values())
            if sr.baseline_scores.get("PSI")
            else 0
        )
        max_ks = (
            max(sr.baseline_scores.get("KS", {}).values())
            if sr.baseline_scores.get("KS")
            else 0
        )

        print(
            f"  {sr.scenario} (mag={sr.magnitude:.2f}): "
            f"SWIFT_max={sr.swift_max:.4f}, "
            f"PSI_max={max_psi:.4f}, "
            f"KS_max={max_ks:.4f}, "
            f"BBSD_KS={sr.baseline_scores.get('BBSD_KS', {}).get('_model_output', 0):.4f} | "
            f"SWIFT detected {n_drifted}/{n_features}, "
            f"ground truth: {gt_drifted} features drifted"
        )

    # --- Save results ---
    result.save(args.output)
    logger.info("Results saved to %s", args.output)

    # --- Summary table (for paper Table 2) ---
    print()
    print("-" * 70)
    print("TABLE: Average SWIFT score by scenario and magnitude")
    print("-" * 70)
    print(
        f"{'Scenario':<6} {'Mag':>5} {'SWIFT_max':>11} {'SWIFT_mean':>12} "
        f"{'PSI_max':>9} {'BBSD_KS':>9} {'Detected':>10}"
    )
    print("-" * 65)
    for sr in result.scenario_results:
        max_psi = (
            max(sr.baseline_scores.get("PSI", {}).values())
            if sr.baseline_scores.get("PSI")
            else 0
        )
        bbsd_ks = sr.baseline_scores.get("BBSD_KS", {}).get("_model_output", 0)
        print(
            f"{sr.scenario:<6} {sr.magnitude:>5.2f} {sr.swift_max:>11.6f} "
            f"{sr.swift_mean:>12.6f} {max_psi:>9.4f} {bbsd_ks:>9.4f} "
            f"{len(sr.swift_drifted):>4}/{len(sr.swift_scores)}"
        )

    print()
    logger.info("Done! Total time: %.1fs", result.total_time_seconds)


if __name__ == "__main__":
    main()
