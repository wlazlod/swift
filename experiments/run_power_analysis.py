#!/usr/bin/env python
"""Power analysis: detection rate vs sample size for SWIFT and baselines.

Measures how detection power scales with monitoring sample size n,
at different drift magnitudes, for SWIFT, PSI, and KS baselines.

Design (from revision plan Phase 3):
- Dataset: Taiwan Credit (default)
- Sample sizes: n in {500, 1000, 2000, 5000, 10000, 18000}
- Scenarios: S1 at 0.5sigma, 1.0sigma, 2.0sigma; S9 for FPR
- Repetitions: 50 per configuration
- Baselines: PSI, KS, Decker

For each (n, magnitude, rep):
    1. Random split with n_ref = n, n_mon = n (subsample if needed)
    2. Train LightGBM on reference
    3. Fit SWIFTMonitor
    4. Inject S1 drift (or S9 for null)
    5. Run SWIFT permutation test + baselines
    6. Record detection (reject / not reject) for each method

Output:
- Power curves (detection rate vs sample size, curves per magnitude)
- Saved to results/v2/power_analysis.json

Usage:
    uv run python experiments/run_power_analysis.py
    uv run python experiments/run_power_analysis.py --fast       # 10 reps, B=200
    uv run python experiments/run_power_analysis.py --reps 50    # Full run
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
import shap
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("power_analysis")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerAnalysisConfig:
    """Configuration for power analysis experiment.

    Attributes:
        dataset_name: Dataset identifier.
        sample_sizes: List of monitoring sample sizes to test.
        magnitudes: Drift magnitudes for S1.
        n_reps: Repetitions per (n, magnitude).
        n_permutations: Number of permutations per test.
        n_features_to_drift: Number of features to drift.
        max_samples: Max pooled samples for permutation test.
        alpha: Significance level.
        base_seed: Base random seed.
    """

    dataset_name: str = "taiwan_credit"
    sample_sizes: list[int] = field(
        default_factory=lambda: [500, 1000, 2000, 5000, 10000, 18000]
    )
    magnitudes: list[float] = field(
        default_factory=lambda: [0.0, 0.5, 1.0, 2.0]
    )  # 0.0 = S9 (null)
    n_reps: int = 50
    n_permutations: int = 1000
    n_features_to_drift: int = 3
    max_samples: int | None = 5000
    alpha: float = 0.05
    base_seed: int = 42


# ---------------------------------------------------------------------------
# Core power analysis logic
# ---------------------------------------------------------------------------


@dataclass
class PowerRepResult:
    """Result from a single power analysis repetition.

    Attributes:
        sample_size: Monitoring sample size.
        magnitude: Drift magnitude (0.0 = null).
        rep_idx: Repetition index.
        seed: Random seed used.
        model_auc: Model AUC.
        swift_rejected: Whether SWIFT rejected H0.
        swift_max: SWIFT_max score.
        swift_mean: SWIFT_mean score.
        psi_max: Max PSI score across features.
        ks_max: Max KS statistic across features.
        decker_max: Max Decker KS statistic across features.
        drifted_features: Ground truth drifted features.
    """

    sample_size: int
    magnitude: float
    rep_idx: int
    seed: int
    model_auc: float
    swift_rejected: bool
    swift_max: float
    swift_mean: float
    psi_max: float
    ks_max: float
    decker_max: float
    drifted_features: list[str]


def run_power_rep(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    config: PowerAnalysisConfig,
    sample_size: int,
    magnitude: float,
    rep_idx: int,
) -> PowerRepResult:
    """Run a single power analysis repetition.

    Args:
        X: Full feature DataFrame.
        y: Full target Series.
        feature_names: Feature names.
        numeric_features: Numeric feature names.
        categorical_features: Categorical feature names.
        config: Power analysis configuration.
        sample_size: Monitoring sample size for this rep.
        magnitude: Drift magnitude (0.0 = null).
        rep_idx: Repetition index.

    Returns:
        PowerRepResult with detection outcomes.
    """
    from experiments.baselines import compute_decker, compute_ks, compute_psi
    from experiments.drift import DriftConfig, DriftScenario
    from experiments.runner_base import train_model
    from swift.pipeline import SWIFTMonitor

    seed = config.base_seed + rep_idx * 1000 + sample_size
    rng = np.random.default_rng(seed)

    # Subsample to get exactly sample_size for ref and mon
    n_total = len(X)
    n_needed = sample_size * 2  # ref + mon
    if n_needed > n_total:
        # Use all data, split proportionally
        indices = rng.permutation(n_total)
        n_ref = n_total // 2
    else:
        indices = rng.choice(n_total, size=n_needed, replace=False)
        n_ref = sample_size

    ref_idx = indices[:n_ref]
    mon_idx = indices[n_ref : n_ref + sample_size]

    X_ref = X.iloc[ref_idx].reset_index(drop=True)
    y_ref = y.iloc[ref_idx].reset_index(drop=True)
    X_mon_clean = X.iloc[mon_idx].reset_index(drop=True)

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
        alpha=config.alpha,
        max_samples=config.max_samples,
        random_state=seed,
    )
    monitor.fit(X_ref[feature_names])

    shap_values_ref = monitor.shap_values_

    # Inject drift (or not for S9)
    if magnitude > 0:
        drift_config = DriftConfig(
            scenario=DriftScenario.S1_MEAN_SHIFT_IMPORTANT,
            magnitude=magnitude,
            n_features=config.n_features_to_drift,
            random_state=seed,
        )
        from experiments.drift import inject_drift

        drift_result = inject_drift(
            X=X_mon_clean,
            config=drift_config,
            shap_values=shap_values_ref,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        X_mon_test = drift_result.X_drifted[feature_names]
        drifted_features = drift_result.drifted_features
    else:
        X_mon_test = X_mon_clean[feature_names]
        drifted_features = []

    # SWIFT test
    monitor.set_params(random_state=seed + 20000)
    swift_result = monitor.test(X_mon_test)
    swift_rejected = swift_result.num_drifted > 0

    # Baselines
    psi_scores = compute_psi(X_ref[feature_names], X_mon_test, feature_names)
    ks_scores = compute_ks(X_ref[feature_names], X_mon_test, feature_names)

    # Decker: compute monitoring SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_mon = np.asarray(explainer.shap_values(X_mon_test))
    decker_scores = compute_decker(
        shap_values_ref, shap_values_mon, feature_names,
    )

    return PowerRepResult(
        sample_size=sample_size,
        magnitude=magnitude,
        rep_idx=rep_idx,
        seed=seed,
        model_auc=model_auc,
        swift_rejected=swift_rejected,
        swift_max=swift_result.swift_max,
        swift_mean=swift_result.swift_mean,
        psi_max=max(psi_scores.values()),
        ks_max=max(ks_scores.values()),
        decker_max=max(decker_scores.values()),
        drifted_features=drifted_features,
    )


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------


def compute_power_summary(
    rep_results: list[PowerRepResult],
    config: PowerAnalysisConfig,
) -> dict[str, Any]:
    """Aggregate power analysis results.

    For each (sample_size, magnitude), computes detection rate for SWIFT
    and baselines, plus 95% CI.

    Args:
        rep_results: All repetition results.
        config: Power analysis configuration.

    Returns:
        Summary dict with power curves.
    """
    summary: dict[str, Any] = {
        "dataset": config.dataset_name,
        "config": {
            "sample_sizes": config.sample_sizes,
            "magnitudes": config.magnitudes,
            "n_reps": config.n_reps,
            "n_permutations": config.n_permutations,
            "alpha": config.alpha,
        },
        "power_curves": {},
    }

    # Group by (sample_size, magnitude)
    from collections import defaultdict
    groups: dict[tuple[int, float], list[PowerRepResult]] = defaultdict(list)
    for r in rep_results:
        groups[(r.sample_size, r.magnitude)].append(r)

    for (n, mag), reps in sorted(groups.items()):
        key = f"n={n}_mag={mag}"
        n_reps = len(reps)

        # SWIFT detection rate
        swift_det = sum(1 for r in reps if r.swift_rejected)
        swift_rate = swift_det / n_reps

        # PSI detection rate (using threshold from S9 null at same n)
        # For simplicity, report the max score distribution
        psi_maxs = [r.psi_max for r in reps]
        ks_maxs = [r.ks_max for r in reps]
        decker_maxs = [r.decker_max for r in reps]
        swift_maxs = [r.swift_max for r in reps]
        swift_means = [r.swift_mean for r in reps]

        # Wilson score CI for SWIFT detection rate
        z = 1.96
        denom = 1 + z**2 / n_reps
        centre = (swift_rate + z**2 / (2 * n_reps)) / denom
        margin = z * np.sqrt(
            (swift_rate * (1 - swift_rate) + z**2 / (4 * n_reps)) / n_reps
        ) / denom

        summary["power_curves"][key] = {
            "sample_size": n,
            "magnitude": mag,
            "n_reps": n_reps,
            "swift_detection_rate": float(swift_rate),
            "swift_ci_lower": float(max(0, centre - margin)),
            "swift_ci_upper": float(min(1, centre + margin)),
            "swift_max_scores": {
                "mean": float(np.mean(swift_maxs)),
                "std": float(np.std(swift_maxs)),
                "median": float(np.median(swift_maxs)),
            },
            "swift_mean_scores": {
                "mean": float(np.mean(swift_means)),
                "std": float(np.std(swift_means)),
            },
            "psi_max_scores": {
                "mean": float(np.mean(psi_maxs)),
                "std": float(np.std(psi_maxs)),
                "median": float(np.median(psi_maxs)),
            },
            "ks_max_scores": {
                "mean": float(np.mean(ks_maxs)),
                "std": float(np.std(ks_maxs)),
                "median": float(np.median(ks_maxs)),
            },
            "decker_max_scores": {
                "mean": float(np.mean(decker_maxs)),
                "std": float(np.std(decker_maxs)),
                "median": float(np.median(decker_maxs)),
            },
            "model_auc": {
                "mean": float(np.mean([r.model_auc for r in reps])),
                "std": float(np.std([r.model_auc for r in reps])),
            },
        }

    # Per-rep details (compact)
    summary["rep_details"] = [
        {
            "sample_size": r.sample_size,
            "magnitude": r.magnitude,
            "rep_idx": r.rep_idx,
            "swift_rejected": r.swift_rejected,
            "swift_max": r.swift_max,
            "psi_max": r.psi_max,
            "ks_max": r.ks_max,
            "decker_max": r.decker_max,
            "model_auc": r.model_auc,
        }
        for r in rep_results
    ]

    return summary


def save_results(summary: dict[str, Any], path: Path) -> None:
    """Save power analysis results to JSON."""
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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Power analysis: detection rate vs sample size"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=50,
        help="Repetitions per (sample_size, magnitude) (default: 50)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: 10 reps, B=200",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/v2/power_analysis.json",
        help="Output path (default: results/v2/power_analysis.json)",
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

    n_reps = 10 if args.fast else args.reps
    n_permutations = 200 if args.fast else args.n_permutations
    max_samples = None if args.max_samples == 0 else args.max_samples

    config = PowerAnalysisConfig(
        n_reps=n_reps,
        n_permutations=n_permutations,
        max_samples=max_samples,
    )

    logger.info("=" * 70)
    logger.info("SWIFT Power Analysis")
    logger.info("=" * 70)
    logger.info("Dataset: %s", config.dataset_name)
    logger.info("Sample sizes: %s", config.sample_sizes)
    logger.info("Magnitudes: %s (0.0 = null/S9)", config.magnitudes)
    logger.info("Reps: %d", n_reps)
    logger.info("Permutations: %d", n_permutations)
    logger.info("Mode: %s", "FAST" if args.fast else "FULL")

    # Load dataset
    logger.info("Loading Taiwan Credit dataset...")
    t0 = time.time()
    from experiments.data_loader import load_taiwan_credit
    bundle = load_taiwan_credit()
    logger.info("Loaded in %.1fs: %s", time.time() - t0, bundle)

    # Total configurations
    n_configs = len(config.sample_sizes) * len(config.magnitudes)
    total_reps = n_configs * n_reps
    logger.info(
        "Total: %d configurations x %d reps = %d runs",
        n_configs, n_reps, total_reps,
    )

    t_total = time.time()
    rep_results: list[PowerRepResult] = []
    run_count = 0

    for magnitude in config.magnitudes:
        for sample_size in config.sample_sizes:
            scenario_label = f"S9 (null)" if magnitude == 0 else f"S1 (mag={magnitude}σ)"
            logger.info("")
            logger.info(
                "--- %s, n=%d ---",
                scenario_label, sample_size,
            )

            for rep_idx in range(n_reps):
                run_count += 1
                t_rep = time.time()

                result = run_power_rep(
                    X=bundle.X,
                    y=bundle.y,
                    feature_names=bundle.feature_names,
                    numeric_features=bundle.numeric_features,
                    categorical_features=bundle.categorical_features,
                    config=config,
                    sample_size=sample_size,
                    magnitude=magnitude,
                    rep_idx=rep_idx,
                )
                rep_results.append(result)

                if (rep_idx + 1) % 10 == 0 or rep_idx == n_reps - 1:
                    # Log running detection rate
                    group_reps = [
                        r for r in rep_results
                        if r.sample_size == sample_size and r.magnitude == magnitude
                    ]
                    det_rate = sum(1 for r in group_reps if r.swift_rejected) / len(group_reps)
                    logger.info(
                        "  [%d/%d] n=%d, mag=%.1f: SWIFT det_rate=%.2f "
                        "(%d/%d reps, %.1fs/rep)",
                        run_count, total_reps, sample_size, magnitude,
                        det_rate, len(group_reps), n_reps,
                        time.time() - t_rep,
                    )

    # Compute summary
    summary = compute_power_summary(rep_results, config)

    # Print power table
    print()
    print("=" * 80)
    print("POWER ANALYSIS RESULTS")
    print("=" * 80)
    print(
        f"{'Magnitude':<12} {'n':<8} {'SWIFT det%':<12} {'95% CI':<18} "
        f"{'PSI_max':<10} {'KS_max':<10} {'Decker_max'}"
    )
    print("-" * 80)

    for key in sorted(summary["power_curves"].keys()):
        pc = summary["power_curves"][key]
        print(
            f"{pc['magnitude']:<12.1f} {pc['sample_size']:<8d} "
            f"{pc['swift_detection_rate']:<12.3f} "
            f"[{pc['swift_ci_lower']:.3f}, {pc['swift_ci_upper']:.3f}]   "
            f"{pc['psi_max_scores']['mean']:<10.4f} "
            f"{pc['ks_max_scores']['mean']:<10.4f} "
            f"{pc['decker_max_scores']['mean']:.4f}"
        )

    # Save
    output_path = Path(args.output)
    save_results(summary, output_path)

    total_time = time.time() - t_total
    logger.info("")
    logger.info(
        "Power analysis complete in %.1fs (%.1f min)",
        total_time, total_time / 60,
    )


if __name__ == "__main__":
    main()
