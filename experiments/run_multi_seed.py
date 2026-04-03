#!/usr/bin/env python
"""Multi-seed stability study for SWIFT.

Runs the full controlled experiment across multiple random seeds to compute
confidence intervals (mean +/- std) for all metrics. Each seed triggers a
new random split -> retrain -> recompute SHAP -> full experiment cycle.

Design (from revision plan Phase 4):
- Datasets: Taiwan Credit, Bank Marketing, Home Credit
- Seeds: 10 (configurable)
- Scenarios: S1, S2, S4, S7, S8, S9 (representative subset)
- Magnitudes: per-scenario defaults from ExperimentConfig
- Per seed: full ExperimentConfig with random_state = base_seed + seed_idx

Output:
- Per (dataset, scenario, magnitude): mean +/- std for SWIFT_max, SWIFT_mean,
  PSI_max, KS_max, SSI_max, Decker_max, BBSD_KS, BBSD_PSI, n_drifted
- Saved to results/v2/multi_seed_{dataset}.json
- Aggregated summary to results/v2/multi_seed_summary.json

Usage:
    uv run python experiments/run_multi_seed.py
    uv run python experiments/run_multi_seed.py --fast          # 3 seeds, B=200
    uv run python experiments/run_multi_seed.py --seeds 10      # Full run
    uv run python experiments/run_multi_seed.py --dataset taiwan_credit
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("multi_seed")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiSeedConfig:
    """Configuration for multi-seed experiment.

    Attributes:
        dataset_name: Dataset identifier.
        n_seeds: Number of random seeds to run.
        base_seed: Starting seed (each run uses base_seed + seed_idx).
        scenarios: Scenario codes to include.
        n_permutations: Number of permutations per SWIFT test.
        alpha: Significance level.
        ref_fraction: Fraction of data for reference set.
        n_features_to_drift: Features to drift per scenario.
        max_samples: Max pooled samples for permutation test.
    """

    dataset_name: str = "taiwan_credit"
    n_seeds: int = 10
    base_seed: int = 42
    scenarios: list[str] = field(
        default_factory=lambda: ["S1", "S2", "S4", "S7", "S8", "S9"]
    )
    n_permutations: int = 1000
    alpha: float = 0.05
    ref_fraction: float = 0.6
    n_features_to_drift: int = 3
    max_samples: int | None = 5000


# ---------------------------------------------------------------------------
# Core multi-seed logic
# ---------------------------------------------------------------------------


@dataclass
class SeedResult:
    """Result from a single seed run.

    Attributes:
        seed_idx: Seed index (0-based).
        random_state: Actual random_state used.
        model_auc: Model AUC for this seed.
        scenario_metrics: Dict of (scenario, magnitude) -> metric dict.
        total_time: Wall-clock time for this seed run.
    """

    seed_idx: int
    random_state: int
    model_auc: float
    scenario_metrics: dict[tuple[str, float], dict[str, float]]
    total_time: float


def extract_scenario_metrics(
    scenario_result: Any,
) -> dict[str, float]:
    """Extract key metrics from a ScenarioResult.

    Args:
        scenario_result: A ScenarioResult from runner_controlled.

    Returns:
        Dict of metric_name -> value.
    """
    baselines = scenario_result.baseline_scores

    # Safely extract max values from baseline dicts
    def safe_max(scores: dict[str, float] | None) -> float:
        if not scores:
            return 0.0
        vals = [v for v in scores.values() if not np.isnan(v)]
        return max(vals) if vals else 0.0

    psi_max = safe_max(baselines.get("PSI"))
    ks_max = safe_max(baselines.get("KS"))
    ssi_max = safe_max(baselines.get("SSI"))
    decker_max = safe_max(baselines.get("Decker_KS"))
    bbsd_ks = baselines.get("BBSD_KS", {}).get("_model_output", 0.0)
    bbsd_psi = baselines.get("BBSD_PSI", {}).get("_model_output", 0.0)

    return {
        "swift_max": scenario_result.swift_max,
        "swift_mean": scenario_result.swift_mean,
        "n_drifted": len(scenario_result.swift_drifted),
        "n_features": len(scenario_result.swift_scores),
        "psi_max": psi_max,
        "ks_max": ks_max,
        "ssi_max": ssi_max,
        "decker_max": decker_max,
        "bbsd_ks": bbsd_ks,
        "bbsd_psi": bbsd_psi,
    }


def run_single_seed(
    bundle: Any,
    config: MultiSeedConfig,
    seed_idx: int,
) -> SeedResult:
    """Run a full controlled experiment with one seed.

    Args:
        bundle: DatasetBundle from data_loader.
        config: Multi-seed configuration.
        seed_idx: Seed index (0-based).

    Returns:
        SeedResult with per-scenario metrics.
    """
    from experiments.runner_base import ExperimentConfig
    from experiments.runner_controlled import run_controlled_experiment

    random_state = config.base_seed + seed_idx

    exp_config = ExperimentConfig(
        dataset_name=config.dataset_name,
        scenarios=list(config.scenarios),
        n_permutations=config.n_permutations,
        alpha=config.alpha,
        ref_fraction=config.ref_fraction,
        n_features_to_drift=config.n_features_to_drift,
        max_samples=config.max_samples,
        random_state=random_state,
    )

    logger.info(
        "  Seed %d/%d (random_state=%d)...",
        seed_idx + 1, config.n_seeds, random_state,
    )

    t0 = time.time()
    result = run_controlled_experiment(
        X=bundle.X,
        y=bundle.y,
        feature_names=bundle.feature_names,
        numeric_features=bundle.numeric_features,
        categorical_features=bundle.categorical_features,
        config=exp_config,
    )
    elapsed = time.time() - t0

    # Extract metrics per (scenario, magnitude)
    scenario_metrics: dict[tuple[str, float], dict[str, float]] = {}
    for sr in result.scenario_results:
        key = (sr.scenario, sr.magnitude)
        scenario_metrics[key] = extract_scenario_metrics(sr)

    return SeedResult(
        seed_idx=seed_idx,
        random_state=random_state,
        model_auc=result.model_auc,
        scenario_metrics=scenario_metrics,
        total_time=elapsed,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_seed_results(
    seed_results: list[SeedResult],
    config: MultiSeedConfig,
) -> dict[str, Any]:
    """Aggregate results across seeds into summary statistics.

    Args:
        seed_results: List of per-seed results.
        config: Multi-seed configuration.

    Returns:
        Summary dict with mean/std/CI for all metrics.
    """
    n_seeds = len(seed_results)

    # Group metrics by (scenario, magnitude)
    grouped: dict[
        tuple[str, float], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))

    for sr in seed_results:
        for (scenario, magnitude), metrics in sr.scenario_metrics.items():
            for metric_name, value in metrics.items():
                grouped[(scenario, magnitude)][metric_name].append(value)

    # Compute summary stats
    scenario_summaries: dict[str, Any] = {}

    for (scenario, magnitude), metrics_dict in sorted(grouped.items()):
        key = f"{scenario}_mag={magnitude:.2f}"
        summary: dict[str, Any] = {
            "scenario": scenario,
            "magnitude": magnitude,
            "n_seeds": n_seeds,
        }

        for metric_name, values in metrics_dict.items():
            arr = np.array(values)
            summary[metric_name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "values": [float(v) for v in arr],
            }

            # 95% CI (t-distribution)
            if len(arr) > 1:
                sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
                t_crit = sp_stats.t.ppf(0.975, df=len(arr) - 1)
                ci_lower = float(np.mean(arr) - t_crit * sem)
                ci_upper = float(np.mean(arr) + t_crit * sem)
                summary[metric_name]["ci_95_lower"] = ci_lower
                summary[metric_name]["ci_95_upper"] = ci_upper

        scenario_summaries[key] = summary

    # Model AUC across seeds
    aucs = [sr.model_auc for sr in seed_results]
    auc_arr = np.array(aucs)

    result: dict[str, Any] = {
        "dataset": config.dataset_name,
        "n_seeds": n_seeds,
        "base_seed": config.base_seed,
        "scenarios": list(config.scenarios),
        "config": {
            "n_permutations": config.n_permutations,
            "alpha": config.alpha,
            "ref_fraction": config.ref_fraction,
            "n_features_to_drift": config.n_features_to_drift,
            "max_samples": config.max_samples,
        },
        "model_auc": {
            "mean": float(np.mean(auc_arr)),
            "std": float(np.std(auc_arr, ddof=1)) if n_seeds > 1 else 0.0,
            "values": [float(v) for v in auc_arr],
        },
        "scenario_summaries": scenario_summaries,
        "seed_details": [
            {
                "seed_idx": sr.seed_idx,
                "random_state": sr.random_state,
                "model_auc": sr.model_auc,
                "total_time": sr.total_time,
            }
            for sr in seed_results
        ],
    }

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_results(summary: dict[str, Any], path: Path) -> None:
    """Save multi-seed results to JSON."""
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


def print_summary_table(summary: dict[str, Any]) -> None:
    """Print a formatted summary table to stdout."""
    print()
    print("=" * 100)
    print(f"MULTI-SEED RESULTS: {summary['dataset']} ({summary['n_seeds']} seeds)")
    print("=" * 100)
    print(
        f"Model AUC: {summary['model_auc']['mean']:.4f} "
        f"+/- {summary['model_auc']['std']:.4f}"
    )
    print()
    print(
        f"{'Scenario':<6} {'Mag':>5} "
        f"{'SWIFT_max':>14} {'SWIFT_mean':>14} "
        f"{'PSI_max':>12} {'KS_max':>12} {'SSI_max':>12} "
        f"{'Decker_max':>12} {'BBSD_KS':>12} {'BBSD_PSI':>12} {'n_drifted':>12}"
    )
    print("-" * 138)

    for key in sorted(summary["scenario_summaries"].keys()):
        ss = summary["scenario_summaries"][key]
        scenario = ss["scenario"]
        magnitude = ss["magnitude"]

        def fmt(metric: str) -> str:
            m = ss.get(metric, {})
            if not m:
                return "N/A"
            return f"{m['mean']:.4f}+/-{m['std']:.4f}"

        def fmt_int(metric: str) -> str:
            m = ss.get(metric, {})
            if not m:
                return "N/A"
            return f"{m['mean']:.1f}+/-{m['std']:.1f}"

        print(
            f"{scenario:<6} {magnitude:>5.2f} "
            f"{fmt('swift_max'):>14} {fmt('swift_mean'):>14} "
            f"{fmt('psi_max'):>12} {fmt('ks_max'):>12} {fmt('ssi_max'):>12} "
            f"{fmt('decker_max'):>12} {fmt('bbsd_ks'):>12} {fmt('bbsd_psi'):>12} "
            f"{fmt_int('n_drifted'):>12}"
        )


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
        name: Dataset identifier.

    Returns:
        DatasetBundle.
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
        description="Multi-seed stability study for SWIFT"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "taiwan_credit", "bank_marketing", "home_credit"],
        help="Dataset to run (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of random seeds (default: 10)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: 3 seeds, B=200",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/v2",
        help="Output directory (default: results/v2)",
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
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    args = parser.parse_args()

    # Determine datasets
    if args.dataset == "all":
        datasets = ["taiwan_credit", "bank_marketing", "home_credit"]
    else:
        datasets = [args.dataset]

    # Apply fast mode
    n_seeds = 3 if args.fast else args.seeds
    n_permutations = 200 if args.fast else args.n_permutations
    max_samples = None if args.max_samples == 0 else args.max_samples

    output_dir = Path(args.output_dir)

    logger.info("=" * 70)
    logger.info("SWIFT Multi-Seed Stability Study")
    logger.info("=" * 70)
    logger.info("Datasets: %s", datasets)
    logger.info("Seeds: %d (base=%d)", n_seeds, args.base_seed)
    logger.info("Scenarios: S1, S2, S7, S8, S9")
    logger.info("Permutations: %d per test", n_permutations)
    logger.info("Mode: %s", "FAST" if args.fast else "FULL")

    t_total_start = time.time()
    all_summaries: dict[str, Any] = {}

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
        config = MultiSeedConfig(
            dataset_name=dataset_name,
            n_seeds=n_seeds,
            base_seed=args.base_seed,
            n_permutations=n_permutations,
            max_samples=max_samples,
        )

        # Run all seeds
        seed_results: list[SeedResult] = []
        for seed_idx in range(n_seeds):
            result = run_single_seed(bundle, config, seed_idx)
            seed_results.append(result)

            logger.info(
                "    Seed %d complete: AUC=%.4f, time=%.1fs",
                seed_idx + 1, result.model_auc, result.total_time,
            )

        # Aggregate
        summary = aggregate_seed_results(seed_results, config)
        all_summaries[dataset_name] = summary

        # Print
        print_summary_table(summary)

        # Save per-dataset
        output_path = output_dir / f"multi_seed_{dataset_name}.json"
        save_results(summary, output_path)

    # Save combined summary
    combined_path = output_dir / "multi_seed_summary.json"
    save_results(all_summaries, combined_path)

    total_time = time.time() - t_total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info(
        "All multi-seed experiments complete in %.1fs (%.1f min)",
        total_time, total_time / 60,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
