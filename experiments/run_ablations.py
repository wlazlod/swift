#!/usr/bin/env python
"""Run ablation experiments A1-A5 on controlled drift scenarios.

For each dataset, trains a LightGBM model, fits SWIFT, then for each
(scenario × magnitude) pair runs:
  - Full SWIFT (reference)
  - A1: No SHAP normalization (W₁ on raw values in model-aware buckets)
  - A2: No model-aware buckets (SHAP + W₁ on equal-frequency bins)
  - A3: PSI on model-aware buckets (KL divergence instead of Wasserstein)
  - A4: W₂ instead of W₁ (sensitivity to Wasserstein order)
  - A5: Importance-weighted aggregation

Results are saved to <output-dir>/<dataset>_ablations.json.

Usage:
    uv run python experiments/run_ablations.py                     # All 3 datasets
    uv run python experiments/run_ablations.py --dataset taiwan    # Single dataset
    uv run python experiments/run_ablations.py --fast              # Fewer permutations
    uv run python experiments/run_ablations.py --output-dir results/v2
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation_experiment")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Scenarios to test ablations on (subset of controlled scenarios).
# We focus on S1 (important drift), S2 (unimportant drift), S3 (variance),
# S7 (null injection), and S9 (no drift) to cover the key ablation axes.
ABLATION_SCENARIOS = ["S1", "S2", "S3", "S7", "S9"]

# Default magnitudes per scenario — same as controlled experiments
DEFAULT_MAGNITUDES: dict[str, list[float]] = {
    "S1": [0.5, 1.0, 2.0, 3.0],
    "S2": [0.5, 1.0, 2.0, 3.0],
    "S3": [0.5, 1.0, 2.0, 3.0],
    "S7": [0.10, 0.20, 0.50, 1.00],
    "S9": [0.0],
}

# Number of equal-frequency bins for A2
A2_N_BINS = 10


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for ablation experiments."""

    dataset_name: str
    scenarios: list[str]
    magnitudes: dict[str, list[float]]
    n_permutations: int
    alpha: float
    ref_fraction: float
    n_features_to_drift: int
    max_samples: int | None
    random_state: int
    a2_n_bins: int

    def magnitudes_for(self, scenario: str) -> list[float]:
        if scenario == "S9":
            return [0.0]
        return self.magnitudes.get(scenario, [0.5, 1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Single-scenario ablation runner
# ---------------------------------------------------------------------------


def _run_single_ablation_scenario(
    X_ref: pd.DataFrame,
    X_mon_drifted: pd.DataFrame,
    monitor: Any,
    feature_names: list[str],
    config: AblationConfig,
    scenario_seed: int,
) -> dict[str, Any]:
    """Run full SWIFT + all ablation variants on one drifted monitoring set.

    Returns a dict with keys: swift, A1, A2, A3, A4, A5, each containing
    max and mean scores (and per_feature for detailed analysis).
    """
    from swift.aggregation import aggregate_scores
    from swift.distance import compute_swift_scores
    from experiments.ablations import (
        compute_a1_no_shap_normalization,
        compute_a2_no_model_buckets,
        compute_a3_psi_on_model_buckets,
        compute_a4_w2_instead_of_w1,
        compute_a5_importance_weighted,
    )

    bucket_sets = monitor.bucket_sets_
    shap_values = monitor.shap_values_

    results: dict[str, Any] = {}

    # Full SWIFT (reference)
    monitor.set_params(random_state=scenario_seed)
    swift_result = monitor.test(X_mon_drifted)
    results["SWIFT"] = {
        "max": swift_result.swift_max,
        "mean": swift_result.swift_mean,
        "per_feature": {
            r.feature_name: r.swift_score
            for r in swift_result.feature_results
        },
        "n_detected": len(swift_result.drifted_features),
    }

    # A1: No SHAP normalization
    a1_scores = compute_a1_no_shap_normalization(
        X_ref[feature_names], X_mon_drifted, bucket_sets,
    )
    a1_agg = aggregate_scores(a1_scores)
    results["A1"] = {
        "max": a1_agg.swift_max,
        "mean": a1_agg.swift_mean,
        "per_feature": a1_scores,
    }

    # A2: No model-aware buckets
    a2_scores = compute_a2_no_model_buckets(
        X_ref[feature_names], X_mon_drifted, shap_values,
        feature_names, n_bins=config.a2_n_bins,
    )
    a2_agg = aggregate_scores(a2_scores)
    results["A2"] = {
        "max": a2_agg.swift_max,
        "mean": a2_agg.swift_mean,
        "per_feature": a2_scores,
    }

    # A3: PSI on model-aware buckets
    a3_scores = compute_a3_psi_on_model_buckets(
        X_ref[feature_names], X_mon_drifted, bucket_sets,
    )
    a3_agg = aggregate_scores(a3_scores)
    results["A3"] = {
        "max": a3_agg.swift_max,
        "mean": a3_agg.swift_mean,
        "per_feature": a3_scores,
    }

    # A4: W₂ instead of W₁
    a4_scores = compute_a4_w2_instead_of_w1(
        X_ref[feature_names], X_mon_drifted, bucket_sets,
    )
    a4_agg = aggregate_scores(a4_scores)
    results["A4"] = {
        "max": a4_agg.swift_max,
        "mean": a4_agg.swift_mean,
        "per_feature": a4_scores,
    }

    # A5: Importance-weighted aggregation
    a5_result = compute_a5_importance_weighted(
        X_ref[feature_names], X_mon_drifted, bucket_sets,
        shap_values, feature_names,
    )
    results["A5"] = {
        "max": a5_result["max"],
        "mean": a5_result["mean"],
        "weighted": a5_result["swift_weighted"],
        "per_feature": a5_result["per_feature"],
    }

    return results


# ---------------------------------------------------------------------------
# Dataset-level ablation runner
# ---------------------------------------------------------------------------


def run_dataset_ablations(
    dataset_name: str,
    config: AblationConfig,
) -> dict[str, Any]:
    """Run ablation experiments on a single dataset.

    Returns a JSON-serializable result dict.
    """
    from experiments.data_loader import (
        load_bank_marketing,
        load_home_credit,
        load_taiwan_credit,
    )
    from experiments.drift import DriftConfig, DriftScenario, inject_drift
    from experiments.runner_base import train_model
    from swift.pipeline import SWIFTMonitor

    _SCENARIO_MAP: dict[str, DriftScenario] = {
        "S1": DriftScenario.S1_MEAN_SHIFT_IMPORTANT,
        "S2": DriftScenario.S2_MEAN_SHIFT_UNIMPORTANT,
        "S3": DriftScenario.S3_VARIANCE_CHANGE,
        "S7": DriftScenario.S7_NULL_RATE_INCREASE,
        "S9": DriftScenario.S9_NO_DRIFT,
    }

    t_start = time.time()

    # --- Load dataset ---
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

    X = bundle.X
    y = bundle.y
    feature_names = bundle.feature_names
    numeric_features = bundle.numeric_features
    categorical_features = bundle.categorical_features

    # --- Split ref / mon ---
    rng = np.random.default_rng(config.random_state)
    n = len(X)
    n_ref = int(n * config.ref_fraction)
    indices = rng.permutation(n)
    ref_idx = indices[:n_ref]
    mon_idx = indices[n_ref:]

    X_ref = X.iloc[ref_idx].reset_index(drop=True)
    y_ref = y.iloc[ref_idx].reset_index(drop=True)
    X_mon_clean = X.iloc[mon_idx].reset_index(drop=True)
    y_mon = y.iloc[mon_idx].reset_index(drop=True)

    logger.info("Data split: n_ref=%d, n_mon=%d", len(X_ref), len(X_mon_clean))

    # --- Train model ---
    logger.info("Training LightGBM model...")
    model, model_auc = train_model(
        X_ref, y_ref, feature_names,
        categorical_features=categorical_features,
        random_state=config.random_state,
    )

    # --- Fit SWIFT ---
    logger.info("Fitting SWIFTMonitor...")
    t_fit = time.time()
    monitor = SWIFTMonitor(
        model=model,
        n_permutations=config.n_permutations,
        alpha=config.alpha,
        max_samples=config.max_samples,
        random_state=config.random_state,
    )
    monitor.fit(X_ref[feature_names])
    fit_time = time.time() - t_fit
    logger.info("SWIFT fit complete in %.1fs", fit_time)

    shap_values_ref = monitor.shap_values_

    # --- Run ablations for each scenario × magnitude ---
    test_rng = np.random.default_rng(config.random_state + 2000)
    scenario_results: list[dict[str, Any]] = []

    for scenario_code in config.scenarios:
        drift_scenario = _SCENARIO_MAP[scenario_code]
        magnitudes = config.magnitudes_for(scenario_code)

        for magnitude in magnitudes:
            logger.info(
                "Running ablations for %s (mag=%.2f)...",
                scenario_code, magnitude,
            )

            # Inject drift
            drift_config = DriftConfig(
                scenario=drift_scenario,
                magnitude=magnitude,
                n_features=config.n_features_to_drift,
                random_state=config.random_state,
            )

            drift_result = inject_drift(
                X=X_mon_clean,
                config=drift_config,
                shap_values=shap_values_ref,
                feature_names=feature_names,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                y=y_mon,
            )

            X_mon_drifted = drift_result.X_drifted[feature_names]

            # Run all ablation variants
            scenario_seed = int(test_rng.integers(0, 2**31))
            ablation_scores = _run_single_ablation_scenario(
                X_ref=X_ref,
                X_mon_drifted=X_mon_drifted,
                monitor=monitor,
                feature_names=feature_names,
                config=config,
                scenario_seed=scenario_seed,
            )

            scenario_results.append({
                "scenario": scenario_code,
                "magnitude": magnitude,
                "drifted_features": drift_result.drifted_features,
                "description": drift_result.description,
                "ablation_scores": ablation_scores,
            })

            # Log summary
            swift_max = ablation_scores["SWIFT"]["max"]
            a1_max = ablation_scores["A1"]["max"]
            a2_max = ablation_scores["A2"]["max"]
            a3_max = ablation_scores["A3"]["max"]
            a4_max = ablation_scores["A4"]["max"]
            a5_max = ablation_scores["A5"]["max"]

            logger.info(
                "  %s (mag=%.2f): SWIFT=%.4f, A1=%.4f, A2=%.4f, "
                "A3=%.4f, A4=%.4f, A5=%.4f",
                scenario_code, magnitude,
                swift_max, a1_max, a2_max, a3_max, a4_max, a5_max,
            )

    total_time = time.time() - t_start

    result = {
        "dataset_name": dataset_name,
        "model_auc": model_auc,
        "n_ref": len(X_ref),
        "n_mon": len(X_mon_clean),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "fit_time_seconds": fit_time,
        "total_time_seconds": total_time,
        "a2_n_bins": config.a2_n_bins,
        "scenario_results": scenario_results,
    }

    return result


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON serialization helper for numpy types."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


DATASETS = ["taiwan_credit", "bank_marketing", "home_credit"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ablation experiments A1-A5 on controlled drift scenarios",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS + ["all"],
        default="all",
        help="Dataset(s) to run ablations on (default: all)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: fewer permutations (50 instead of 200)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Max pooled samples for permutation test (0 = no limit)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    n_permutations = 50 if args.fast else 200
    max_samples = None if args.max_samples == 0 else args.max_samples
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("SWIFT Ablation Experiments (A1-A5)")
    logger.info("=" * 60)
    logger.info("Datasets: %s", datasets)
    logger.info("Mode: %s (n_permutations=%d)", "FAST" if args.fast else "FULL", n_permutations)
    logger.info("Scenarios: %s", ABLATION_SCENARIOS)

    for dataset_name in datasets:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Dataset: %s", dataset_name)
        logger.info("=" * 60)

        # Match n_features_to_drift to the controlled experiment configs:
        # home_credit uses 5 (120+ features), others use 3.
        n_features_to_drift = 5 if dataset_name == "home_credit" else 3

        config = AblationConfig(
            dataset_name=dataset_name,
            scenarios=list(ABLATION_SCENARIOS),
            magnitudes=dict(DEFAULT_MAGNITUDES),
            n_permutations=n_permutations,
            alpha=0.05,
            ref_fraction=0.6,
            n_features_to_drift=n_features_to_drift,
            max_samples=max_samples,
            random_state=42,
            a2_n_bins=A2_N_BINS,
        )

        result = run_dataset_ablations(dataset_name, config)

        # Save results
        output_path = output_dir / f"{dataset_name}_ablations.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, default=_json_default),
        )
        logger.info("Results saved to %s", output_path)

        # Print summary table
        print()
        print(f"{'='*70}")
        print(f"ABLATION RESULTS: {dataset_name}")
        print(f"{'='*70}")
        print(
            f"{'Scenario':<6} {'Mag':>5} {'SWIFT':>8} {'A1':>8} "
            f"{'A2':>8} {'A3':>8} {'A4':>8} {'A5':>8}"
        )
        print("-" * 62)

        for sr in result["scenario_results"]:
            s = sr["ablation_scores"]
            print(
                f"{sr['scenario']:<6} {sr['magnitude']:>5.2f} "
                f"{s['SWIFT']['max']:>8.4f} {s['A1']['max']:>8.4f} "
                f"{s['A2']['max']:>8.4f} {s['A3']['max']:>8.4f} "
                f"{s['A4']['max']:>8.4f} {s['A5']['max']:>8.4f}"
            )

        print()
        logger.info(
            "Done with %s! Total time: %.1fs",
            dataset_name, result["total_time_seconds"],
        )

    logger.info("")
    logger.info("All ablation experiments complete!")


if __name__ == "__main__":
    main()
