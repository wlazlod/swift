#!/usr/bin/env python
"""Run S10 gradual drift experiment on all controlled-experiment datasets.

S10 injects a linearly increasing mean shift over 12 monitoring periods on
the top-3 SHAP-important features.  The key metric is **detection delay**:
how many periods until each method first flags drift.

Usage:
    uv run python experiments/run_gradual_drift.py
    uv run python experiments/run_gradual_drift.py --fast
    uv run python experiments/run_gradual_drift.py --dataset taiwan_credit

Results are saved to results/<dataset>_gradual_drift.json
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
logger = logging.getLogger("gradual_drift_experiment")


_DATASETS = ["taiwan_credit", "bank_marketing", "home_credit"]


def run_dataset(
    dataset_name: str,
    n_permutations: int,
    n_null_runs: int,
    max_samples: int | None,
    output_dir: str,
) -> None:
    """Run S10 on a single dataset."""
    from experiments.data_loader import (
        load_bank_marketing,
        load_home_credit,
        load_taiwan_credit,
    )
    from experiments.runner_gradual import (
        GradualDriftConfig2,
        run_gradual_drift_experiment,
    )

    # Load dataset
    logger.info("Loading %s dataset...", dataset_name)
    t0 = time.time()

    if dataset_name == "taiwan_credit":
        bundle = load_taiwan_credit()
    elif dataset_name == "bank_marketing":
        bundle = load_bank_marketing()
    elif dataset_name == "home_credit":
        bundle = load_home_credit()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info("Dataset loaded in %.1fs: %s", time.time() - t0, bundle)

    # Configure S10 experiment
    config = GradualDriftConfig2(
        dataset_name=dataset_name,
        n_steps=12,
        max_magnitude=3.0,
        n_features_to_drift=3,
        n_permutations=n_permutations,
        n_null_runs=n_null_runs,
        alpha=0.05,
        ref_fraction=0.6,
        max_samples=max_samples,
        random_state=42,
    )

    logger.info("S10 config: %d steps, max_mag=%.1fσ, n_perm=%d, n_null=%d",
                config.n_steps, config.max_magnitude,
                config.n_permutations, config.n_null_runs)

    # Run experiment
    result = run_gradual_drift_experiment(
        X=bundle.X,
        y=bundle.y,
        feature_names=bundle.feature_names,
        numeric_features=bundle.numeric_features,
        categorical_features=bundle.categorical_features,
        config=config,
    )

    # Print summary
    print()
    print("=" * 70)
    print(f"S10 GRADUAL DRIFT RESULTS: {dataset_name.upper()}")
    print("=" * 70)
    print(result.summary())
    print()

    # Print detection delay comparison table
    print("-" * 70)
    print("DETECTION DELAY COMPARISON")
    print("-" * 70)
    print(f"{'Method':<20} {'Delay':>8} {'Threshold':>12} {'Status':<12}")
    print("-" * 54)
    for method, delay in sorted(result.detection_delay.items()):
        threshold = result.null_threshold.get(method, float("nan"))
        if delay is not None:
            status = f"step {delay}/{result.n_steps}"
        else:
            status = "NOT DETECTED"
        print(f"{method:<20} {str(delay or '-'):>8} {threshold:>12.6f} {status:<12}")

    # Print step-by-step scores
    print()
    print("-" * 70)
    print("STEP-BY-STEP SCORES (SWIFT_max)")
    print("-" * 70)
    for sr in result.step_results:
        marker = " *" if sr.swift_max > result.null_threshold.get("SWIFT_max", float("inf")) else ""
        psi_max = max(sr.baseline_scores.get("PSI", {}).values()) if sr.baseline_scores.get("PSI") else 0
        bbsd_ks = sr.baseline_scores.get("BBSD_KS", {}).get("_model_output", 0)
        print(
            f"  Step {sr.step:>2}/{result.n_steps}: "
            f"mag={sr.magnitude:.3f}σ, "
            f"SWIFT_max={sr.swift_max:.6f}, "
            f"PSI_max={psi_max:.4f}, "
            f"BBSD_KS={bbsd_ks:.4f}"
            f"{marker}"
        )

    # Save
    out_path = f"{output_dir}/{dataset_name}_gradual_drift.json"
    result.save(out_path)
    logger.info("Results saved to %s", out_path)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="S10 Gradual Drift experiment")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: fewer permutations and null runs",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=_DATASETS,
        default=None,
        help="Run on a single dataset (default: all 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Max pooled samples for permutation test (0 = no limit)",
    )
    args = parser.parse_args()

    n_permutations = 100 if args.fast else 500
    n_null_runs = 10 if args.fast else 30
    max_samples = None if args.max_samples == 0 else args.max_samples

    datasets = [args.dataset] if args.dataset else _DATASETS

    logger.info("=" * 60)
    logger.info("SWIFT S10 Gradual Drift Experiment")
    logger.info("=" * 60)
    logger.info("Mode: %s", "FAST" if args.fast else "FULL")
    logger.info("Datasets: %s", datasets)
    logger.info("n_permutations=%d, n_null_runs=%d, max_samples=%s",
                n_permutations, n_null_runs, max_samples or "unlimited")

    t_total = time.time()

    for dataset_name in datasets:
        logger.info("-" * 60)
        logger.info("Running S10 on %s...", dataset_name)
        logger.info("-" * 60)
        try:
            run_dataset(
                dataset_name=dataset_name,
                n_permutations=n_permutations,
                n_null_runs=n_null_runs,
                max_samples=max_samples,
                output_dir=args.output_dir,
            )
        except Exception:
            logger.exception("Failed on %s", dataset_name)

    logger.info("=" * 60)
    logger.info("All S10 experiments complete. Total time: %.1fs",
                time.time() - t_total)


if __name__ == "__main__":
    main()
