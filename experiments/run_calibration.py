#!/usr/bin/env python
"""Type I error calibration study for SWIFT.

Validates that the permutation-test p-values are well-calibrated under
the null hypothesis (S9: no drift).  For each (dataset, α) combination,
runs R independent repetitions of:

    new random split → train LightGBM → fit SWIFTMonitor → S9 (no drift)
    → permutation test (B permutations) → record if H₀ rejected

The expected empirical rejection rate should be close to α.

Usage:
    uv run python experiments/run_calibration.py
    uv run python experiments/run_calibration.py --fast          # 20 reps, B=200
    uv run python experiments/run_calibration.py --reps 200      # Full run (overnight)
    uv run python experiments/run_calibration.py --dataset taiwan_credit

Results are saved to results/v2/calibration_{dataset}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("calibration")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for a calibration experiment.

    Attributes:
        dataset_name: Dataset identifier.
        alphas: Significance levels to calibrate.
        n_reps: Number of repetitions per (dataset, α).
        n_permutations: Number of permutations per test (B).
        ref_fraction: Fraction of data used as reference.
        max_samples: Max pooled samples for permutation test.
        base_seed: Base random seed (each rep gets base_seed + rep_idx).
    """

    dataset_name: str
    alphas: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.10])
    n_reps: int = 200
    n_permutations: int = 1000
    ref_fraction: float = 0.6
    max_samples: int | None = 5000
    base_seed: int = 42


# ---------------------------------------------------------------------------
# Core calibration logic
# ---------------------------------------------------------------------------


@dataclass
class RepResult:
    """Result from a single calibration repetition.

    Attributes:
        rep_idx: Repetition index (0-based).
        seed: Random seed used for this rep.
        model_auc: Model AUC on validation data.
        swift_max: Model-level SWIFT_max score.
        swift_mean: Model-level SWIFT_mean score.
        pvalues: Per-feature p-values.
        rejected_at: Dict of α → bool (whether H₀ was rejected).
        n_drifted_at: Dict of α → int (number of features flagged).
    """

    rep_idx: int
    seed: int
    model_auc: float
    swift_max: float
    swift_mean: float
    pvalues: dict[str, float]
    rejected_at: dict[float, bool]
    n_drifted_at: dict[float, int]


def run_single_rep(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    categorical_features: list[str],
    config: CalibrationConfig,
    rep_idx: int,
) -> RepResult:
    """Run a single calibration repetition.

    1. Random split with unique seed
    2. Train LightGBM
    3. Fit SWIFTMonitor
    4. Run permutation test on clean monitoring data (no drift)
    5. Record rejection decisions at each α

    Args:
        X: Full feature DataFrame.
        y: Full target Series.
        feature_names: Feature names.
        categorical_features: Categorical feature names.
        config: Calibration configuration.
        rep_idx: Repetition index (0-based).

    Returns:
        RepResult with rejection decisions.
    """
    from experiments.runner_base import train_model
    from swift.pipeline import SWIFTMonitor
    from swift.types import CorrectionMethod

    seed = config.base_seed + rep_idx
    rng = np.random.default_rng(seed)

    # Split into reference and monitoring
    n = len(X)
    n_ref = int(n * config.ref_fraction)
    indices = rng.permutation(n)
    ref_idx = indices[:n_ref]
    mon_idx = indices[n_ref:]

    X_ref = X.iloc[ref_idx].reset_index(drop=True)
    y_ref = y.iloc[ref_idx].reset_index(drop=True)
    X_mon = X.iloc[mon_idx].reset_index(drop=True)

    # Train model
    model, model_auc = train_model(
        X_ref, y_ref, feature_names,
        categorical_features=categorical_features,
        random_state=seed,
    )

    # Fit SWIFT
    monitor = SWIFTMonitor(
        model=model,
        n_permutations=config.n_permutations,
        max_samples=config.max_samples,
        random_state=seed,
    )
    monitor.fit(X_ref[feature_names])

    # Test on clean monitoring data (no drift = S9)
    # Use the max alpha to get the most generous p-value set
    # (p-values don't change with alpha; only decisions do)
    max_alpha = max(config.alphas)
    monitor.set_params(alpha=max_alpha, random_state=seed + 10000)
    swift_result = monitor.test(X_mon[feature_names])

    # Extract p-values
    pvalues = {
        r.feature_name: r.p_value
        for r in swift_result.feature_results
    }

    # Determine rejection at each α (using BH correction)
    from swift.threshold import correct_pvalues

    rejected_at: dict[float, bool] = {}
    n_drifted_at: dict[float, int] = {}

    for alpha in config.alphas:
        decisions = correct_pvalues(pvalues, CorrectionMethod.BH, alpha)
        n_drifted = sum(1 for v in decisions.values() if v)
        rejected_at[alpha] = n_drifted > 0
        n_drifted_at[alpha] = n_drifted

    return RepResult(
        rep_idx=rep_idx,
        seed=seed,
        model_auc=model_auc,
        swift_max=swift_result.swift_max,
        swift_mean=swift_result.swift_mean,
        pvalues=pvalues,
        rejected_at=rejected_at,
        n_drifted_at=n_drifted_at,
    )


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------


def compute_calibration_summary(
    rep_results: list[RepResult],
    config: CalibrationConfig,
) -> dict[str, Any]:
    """Aggregate repetition results into calibration summary.

    Computes empirical FPR ± 95% CI for each α level.

    Args:
        rep_results: List of per-repetition results.
        config: Calibration configuration.

    Returns:
        Summary dict with calibration metrics.
    """
    n_reps = len(rep_results)

    summary: dict[str, Any] = {
        "dataset": config.dataset_name,
        "n_reps": n_reps,
        "n_permutations": config.n_permutations,
        "calibration": {},
    }

    for alpha in config.alphas:
        n_rejected = sum(1 for r in rep_results if r.rejected_at[alpha])
        empirical_fpr = n_rejected / n_reps

        # Wilson score interval for binomial proportion (95% CI)
        z = 1.96
        denominator = 1 + z**2 / n_reps
        centre = (empirical_fpr + z**2 / (2 * n_reps)) / denominator
        margin = z * np.sqrt(
            (empirical_fpr * (1 - empirical_fpr) + z**2 / (4 * n_reps)) / n_reps
        ) / denominator
        ci_lower = max(0.0, centre - margin)
        ci_upper = min(1.0, centre + margin)

        summary["calibration"][str(alpha)] = {
            "nominal_alpha": alpha,
            "empirical_fpr": float(empirical_fpr),
            "n_rejected": n_rejected,
            "n_reps": n_reps,
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper),
            "well_calibrated": ci_lower <= alpha <= ci_upper,
        }

    # Collect all p-values for QQ-plot
    all_pvalues: list[float] = []
    for r in rep_results:
        all_pvalues.extend(r.pvalues.values())
    summary["all_pvalues"] = sorted(all_pvalues)

    # Model AUC summary
    aucs = [r.model_auc for r in rep_results]
    summary["model_auc_mean"] = float(np.mean(aucs))
    summary["model_auc_std"] = float(np.std(aucs))

    # SWIFT score summary under null
    swift_maxs = [r.swift_max for r in rep_results]
    swift_means = [r.swift_mean for r in rep_results]
    summary["swift_max_under_null"] = {
        "mean": float(np.mean(swift_maxs)),
        "std": float(np.std(swift_maxs)),
        "median": float(np.median(swift_maxs)),
        "q95": float(np.quantile(swift_maxs, 0.95)),
    }
    summary["swift_mean_under_null"] = {
        "mean": float(np.mean(swift_means)),
        "std": float(np.std(swift_means)),
        "median": float(np.median(swift_means)),
        "q95": float(np.quantile(swift_means, 0.95)),
    }

    # Per-rep details
    summary["rep_details"] = [
        {
            "rep_idx": r.rep_idx,
            "seed": r.seed,
            "model_auc": r.model_auc,
            "swift_max": r.swift_max,
            "swift_mean": r.swift_mean,
            "rejected_at": {str(k): v for k, v in r.rejected_at.items()},
            "n_drifted_at": {str(k): v for k, v in r.n_drifted_at.items()},
        }
        for r in rep_results
    ]

    return summary


def save_results(summary: dict[str, Any], path: Path) -> None:
    """Save calibration results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def json_default(obj: Any) -> Any:
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    path.write_text(json.dumps(summary, indent=2, default=json_default))
    logger.info("Results saved to %s", path)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


DATASET_LOADERS = {
    "taiwan_credit": "experiments.data_loader:load_taiwan_credit",
    "bank_marketing": "experiments.data_loader:load_bank_marketing",
    "home_credit": "experiments.data_loader:load_home_credit",
}


def load_dataset(name: str) -> Any:
    """Load a dataset by name.

    Args:
        name: Dataset identifier (taiwan_credit, bank_marketing, home_credit).

    Returns:
        DatasetBundle with X, y, feature_names, etc.
    """
    if name not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {list(DATASET_LOADERS.keys())}"
        )

    module_path, func_name = DATASET_LOADERS[name].split(":")
    import importlib
    module = importlib.import_module(module_path)
    loader = getattr(module, func_name)
    return loader()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Type I error calibration study for SWIFT"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "taiwan_credit", "bank_marketing", "home_credit"],
        help="Dataset to run (default: all)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=200,
        help="Number of repetitions per (dataset, α) (default: 200)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: 20 reps, B=200",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/v2",
        help="Output directory for results (default: results/v2)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutations per test (default: 1000)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Max pooled samples for permutation test (0 = no limit)",
    )
    args = parser.parse_args()

    # Determine datasets to run
    if args.dataset == "all":
        datasets = ["taiwan_credit", "bank_marketing", "home_credit"]
    else:
        datasets = [args.dataset]

    # Apply fast mode overrides
    n_reps = 20 if args.fast else args.reps
    n_permutations = 200 if args.fast else args.n_permutations
    max_samples = None if args.max_samples == 0 else args.max_samples

    output_dir = Path(args.output_dir)

    logger.info("=" * 70)
    logger.info("SWIFT Type I Error Calibration Study")
    logger.info("=" * 70)
    logger.info("Datasets: %s", datasets)
    logger.info("Repetitions: %d per dataset", n_reps)
    logger.info("Permutations: %d per test", n_permutations)
    logger.info("Alpha levels: [0.01, 0.05, 0.10]")
    logger.info("Mode: %s", "FAST" if args.fast else "FULL")

    t_total_start = time.time()

    for dataset_name in datasets:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Dataset: %s", dataset_name)
        logger.info("=" * 60)

        # Load dataset
        logger.info("Loading %s...", dataset_name)
        t0 = time.time()
        bundle = load_dataset(dataset_name)
        logger.info("Loaded in %.1fs: %s", time.time() - t0, bundle)

        # Configure
        config = CalibrationConfig(
            dataset_name=dataset_name,
            n_reps=n_reps,
            n_permutations=n_permutations,
            max_samples=max_samples,
        )

        # Run repetitions
        rep_results: list[RepResult] = []
        for rep_idx in range(n_reps):
            t_rep = time.time()
            logger.info(
                "  Rep %d/%d (seed=%d)...",
                rep_idx + 1, n_reps, config.base_seed + rep_idx,
            )

            result = run_single_rep(
                X=bundle.X,
                y=bundle.y,
                feature_names=bundle.feature_names,
                categorical_features=bundle.categorical_features,
                config=config,
                rep_idx=rep_idx,
            )
            rep_results.append(result)

            # Log intermediate results
            rejected_str = ", ".join(
                f"α={a}: {'REJ' if result.rejected_at[a] else 'ok'}"
                for a in config.alphas
            )
            logger.info(
                "    AUC=%.4f, SWIFT_max=%.6f | %s (%.1fs)",
                result.model_auc, result.swift_max, rejected_str,
                time.time() - t_rep,
            )

            # Periodic summary every 20 reps
            if (rep_idx + 1) % 20 == 0:
                for alpha in config.alphas:
                    n_rej = sum(1 for r in rep_results if r.rejected_at[alpha])
                    logger.info(
                        "    Running FPR at α=%.2f: %d/%d = %.3f",
                        alpha, n_rej, len(rep_results), n_rej / len(rep_results),
                    )

        # Compute summary
        summary = compute_calibration_summary(rep_results, config)

        # Print results
        print()
        print("=" * 60)
        print(f"CALIBRATION RESULTS: {dataset_name}")
        print("=" * 60)
        print(f"{'α (nominal)':<15} {'FPR (empirical)':<18} {'95% CI':<20} {'Calibrated?'}")
        print("-" * 65)

        for alpha_key, cal in summary["calibration"].items():
            alpha = cal["nominal_alpha"]
            fpr = cal["empirical_fpr"]
            ci_lo = cal["ci_95_lower"]
            ci_hi = cal["ci_95_upper"]
            ok = cal["well_calibrated"]
            print(
                f"{alpha:<15.2f} {fpr:<18.4f} "
                f"[{ci_lo:.4f}, {ci_hi:.4f}]     "
                f"{'YES' if ok else 'NO'}"
            )

        print()
        print(f"Model AUC: {summary['model_auc_mean']:.4f} ± {summary['model_auc_std']:.4f}")
        print(f"SWIFT_max under null: {summary['swift_max_under_null']['mean']:.6f} "
              f"± {summary['swift_max_under_null']['std']:.6f}")
        print(f"SWIFT_mean under null: {summary['swift_mean_under_null']['mean']:.6f} "
              f"± {summary['swift_mean_under_null']['std']:.6f}")

        # Save
        output_path = output_dir / f"calibration_{dataset_name}.json"
        save_results(summary, output_path)

    total_time = time.time() - t_total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("All calibration experiments complete in %.1fs (%.1f min)",
                total_time, total_time / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
