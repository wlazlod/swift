"""S10 gradual drift experiment runner.

Runs gradually increasing drift over N monitoring periods and measures
detection delay for SWIFT and all baselines.

Typical usage:
    from experiments.runner_gradual import (
        GradualDriftConfig2,
        run_gradual_drift_experiment,
    )

    cfg = GradualDriftConfig2(dataset_name="taiwan_credit")
    result = run_gradual_drift_experiment(X, y, feature_names,
                                          numeric_features,
                                          categorical_features, cfg)
    result.save("results/taiwan_credit_gradual.json")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from experiments.baselines import run_all_baselines
from experiments.drift import GradualDriftConfig, inject_gradual_drift
from experiments.runner_base import json_default, train_model
from swift.pipeline import SWIFTMonitor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GradualDriftStepResult:
    """Result from one step of the gradual drift experiment.

    Attributes:
        step: Step number (1-indexed).
        magnitude: Drift magnitude at this step (in sigma).
        swift_scores: Per-feature SWIFT W1 scores.
        swift_pvalues: Per-feature p-values.
        swift_drifted: Features flagged as drifted.
        swift_max: Model-level max SWIFT score.
        swift_mean: Model-level mean SWIFT score.
        baseline_scores: Baseline scores (method -> feature -> score).
    """

    step: int
    magnitude: float
    swift_scores: dict[str, float]
    swift_pvalues: dict[str, float]
    swift_drifted: list[str]
    swift_max: float
    swift_mean: float
    baseline_scores: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "step": self.step,
            "magnitude": self.magnitude,
            "swift_scores": self.swift_scores,
            "swift_pvalues": self.swift_pvalues,
            "swift_drifted": self.swift_drifted,
            "swift_max": self.swift_max,
            "swift_mean": self.swift_mean,
            "baseline_scores": self.baseline_scores,
        }


def compute_detection_delay(
    step_scores: list[float],
    threshold: float,
) -> int | None:
    """Find the first step where the score exceeds the threshold.

    Args:
        step_scores: Score at each step (1-indexed conceptually, 0-indexed in list).
        threshold: Detection threshold.

    Returns:
        1-indexed step number of first detection, or None if never detected.
    """
    for i, score in enumerate(step_scores):
        if score > threshold:
            return i + 1  # 1-indexed
    return None


@dataclass
class GradualDriftExperimentResult:
    """Result of an S10 gradual drift experiment.

    Attributes:
        dataset_name: Dataset identifier.
        step_results: List of per-step results.
        n_steps: Total number of monitoring steps.
        max_magnitude: Maximum drift magnitude at the final step.
        drifted_features: Feature names that were drifted.
        model_auc: Model AUC on validation data.
        n_ref: Number of reference observations.
        n_mon: Number of monitoring observations.
        n_features: Number of features.
        feature_names: List of feature names.
        detection_delay: Method name -> step of first detection (None = never).
        null_threshold: Threshold from S9 null baseline used for detection.
        total_time_seconds: Total wall-clock time.
        fit_time_seconds: Time spent fitting SWIFT.
    """

    dataset_name: str
    step_results: list[GradualDriftStepResult]
    n_steps: int
    max_magnitude: float
    drifted_features: list[str]
    model_auc: float
    n_ref: int
    n_mon: int
    n_features: int
    feature_names: list[str]
    detection_delay: dict[str, int | None]
    null_threshold: dict[str, float]
    total_time_seconds: float = 0.0
    fit_time_seconds: float = 0.0

    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "dataset_name": self.dataset_name,
            "n_steps": self.n_steps,
            "max_magnitude": self.max_magnitude,
            "drifted_features": self.drifted_features,
            "model_auc": self.model_auc,
            "n_ref": self.n_ref,
            "n_mon": self.n_mon,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "detection_delay": {
                k: v for k, v in self.detection_delay.items()
            },
            "null_threshold": self.null_threshold,
            "total_time_seconds": self.total_time_seconds,
            "fit_time_seconds": self.fit_time_seconds,
            "step_results": [sr.to_dict() for sr in self.step_results],
        }

        path.write_text(json.dumps(data, indent=2, default=json_default))
        logger.info("Gradual drift results saved to %s", path)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"S10 Gradual Drift: {self.dataset_name}",
            f"  Model AUC: {self.model_auc:.4f}",
            f"  n_ref={self.n_ref}, n_mon={self.n_mon}, p={self.n_features}",
            f"  Steps: {self.n_steps}, max_magnitude: {self.max_magnitude}sigma",
            f"  Drifted features: {self.drifted_features}",
            f"  Total time: {self.total_time_seconds:.1f}s",
            "",
            "  Detection delays (step of first detection):",
        ]
        for method, delay in sorted(self.detection_delay.items()):
            threshold = self.null_threshold.get(method, float("nan"))
            if delay is not None:
                step_r = self.step_results[delay - 1]
                lines.append(
                    f"    {method}: step {delay}/{self.n_steps} "
                    f"(mag={step_r.magnitude:.3f}sigma, threshold={threshold:.6f})"
                )
            else:
                lines.append(
                    f"    {method}: NOT DETECTED (threshold={threshold:.6f})"
                )

        lines.append("")
        lines.append("  Per-step SWIFT_max scores:")
        for sr in self.step_results:
            lines.append(
                f"    Step {sr.step}: mag={sr.magnitude:.3f}sigma, "
                f"SWIFT_max={sr.swift_max:.6f}"
            )

        return "\n".join(lines)


@dataclass(frozen=True)
class GradualDriftConfig2:
    """Configuration for the S10 gradual drift experiment runner.

    Attributes:
        dataset_name: Identifier for the dataset.
        n_steps: Number of monitoring periods.
        max_magnitude: Maximum drift magnitude (in sigma) at the final step.
        n_features_to_drift: Number of features to drift.
        n_permutations: Permutations for SWIFT p-values.
        n_null_runs: Number of S9 (no-drift) runs for null threshold calibration.
        alpha: Significance level.
        ref_fraction: Fraction of data for reference set.
        max_samples: Max pool size for permutation test.
        random_state: Global random seed.
    """

    dataset_name: str
    n_steps: int = 12
    max_magnitude: float = 3.0
    n_features_to_drift: int = 3
    n_permutations: int = 200
    n_null_runs: int = 20
    alpha: float = 0.05
    ref_fraction: float = 0.6
    max_samples: int | None = 5000
    random_state: int = 42


# ---------------------------------------------------------------------------
# Gradual drift experiment
# ---------------------------------------------------------------------------


def run_gradual_drift_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    config: GradualDriftConfig2,
) -> GradualDriftExperimentResult:
    """Run an S10 gradual drift experiment.

    Steps:
        1. Split data into reference and monitoring sets
        2. Train LightGBM on reference data
        3. Fit SWIFTMonitor on reference data
        4. Calibrate null thresholds by running N null (no-drift) tests
        5. Inject gradual drift (N steps of increasing magnitude)
        6. For each step, score with SWIFT + baselines
        7. Compute detection delay per method

    Args:
        X: Full feature DataFrame.
        y: Full target Series.
        feature_names: Feature names.
        numeric_features: Numeric feature names.
        categorical_features: Categorical feature names.
        config: Gradual drift experiment configuration.

    Returns:
        GradualDriftExperimentResult with per-step scores and detection delays.
    """
    t_start = time.time()
    rng = np.random.default_rng(config.random_state)

    # --- Step 1: Split into reference and monitoring ---
    n = len(X)
    n_ref = int(n * config.ref_fraction)
    indices = rng.permutation(n)
    ref_idx = indices[:n_ref]
    mon_idx = indices[n_ref:]

    X_ref = X.iloc[ref_idx].reset_index(drop=True)
    y_ref = y.iloc[ref_idx].reset_index(drop=True)
    X_mon_clean = X.iloc[mon_idx].reset_index(drop=True)

    logger.info(
        "S10: Data split: n_ref=%d, n_mon=%d",
        len(X_ref), len(X_mon_clean),
    )

    # --- Step 2: Train model ---
    logger.info("S10: Training LightGBM model...")
    model, model_auc = train_model(
        X_ref, y_ref, feature_names,
        categorical_features=categorical_features,
        random_state=config.random_state,
    )

    # --- Step 3: Fit SWIFT ---
    logger.info("S10: Fitting SWIFTMonitor...")
    t_fit_start = time.time()
    monitor = SWIFTMonitor(
        model=model,
        n_permutations=config.n_permutations,
        alpha=config.alpha,
        max_samples=config.max_samples,
        random_state=config.random_state,
    )
    monitor.fit(X_ref[feature_names])
    fit_time = time.time() - t_fit_start

    shap_values_ref = monitor.shap_values_

    # --- Step 4: Null threshold calibration ---
    logger.info("S10: Calibrating null thresholds (%d runs)...", config.n_null_runs)
    null_rng = np.random.default_rng(config.random_state + 2000)

    null_swift_max_scores: list[float] = []
    null_baseline_max_scores: dict[str, list[float]] = {}

    for null_run in range(config.n_null_runs):
        null_seed = null_rng.integers(0, 2**31)

        # Resample monitoring data (bootstrap of clean monitoring set)
        resample_rng = np.random.default_rng(null_seed)
        resample_idx = resample_rng.choice(
            len(X_mon_clean), size=len(X_mon_clean), replace=True,
        )
        X_null = X_mon_clean.iloc[resample_idx].reset_index(drop=True)

        # SWIFT score on null data
        null_seed_int = int(null_seed)
        monitor.set_params(random_state=null_seed_int + 1)
        null_result = monitor.test(X_null[feature_names])
        null_swift_max_scores.append(null_result.swift_max)

        # Compute SHAP on null monitoring data (for Decker baseline)
        null_explainer = shap.TreeExplainer(model)
        null_shap_mon = np.asarray(
            null_explainer.shap_values(X_null[feature_names])
        )

        # Baseline scores on null data
        null_baselines = run_all_baselines(
            X_ref=X_ref[feature_names],
            X_mon=X_null[feature_names],
            feature_names=feature_names,
            shap_values=shap_values_ref,
            shap_values_mon=null_shap_mon,
            bucket_sets=monitor.bucket_sets_,
            model=model,
        )

        for method, feat_scores in null_baselines.items():
            if method not in null_baseline_max_scores:
                null_baseline_max_scores[method] = []
            max_score = max(feat_scores.values()) if feat_scores else 0.0
            null_baseline_max_scores[method].append(max_score)

    # Threshold = (1 - alpha) quantile of null distribution
    null_thresholds: dict[str, float] = {
        "SWIFT_max": float(
            np.quantile(null_swift_max_scores, 1 - config.alpha)
        ),
    }
    for method, null_scores in null_baseline_max_scores.items():
        null_thresholds[f"{method}_max"] = float(
            np.quantile(null_scores, 1 - config.alpha)
        )

    logger.info("S10: Null thresholds: %s", null_thresholds)

    # --- Step 5: Inject gradual drift ---
    logger.info(
        "S10: Injecting gradual drift (%d steps, max_mag=%.1fsigma)...",
        config.n_steps, config.max_magnitude,
    )
    gradual_config = GradualDriftConfig(
        n_steps=config.n_steps,
        max_magnitude=config.max_magnitude,
        n_features=config.n_features_to_drift,
        random_state=config.random_state,
    )
    gradual_result = inject_gradual_drift(
        X=X_mon_clean,
        config=gradual_config,
        shap_values=shap_values_ref,
        feature_names=feature_names,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    # --- Step 6: Score each step ---
    test_rng = np.random.default_rng(config.random_state + 3000)
    step_results: list[GradualDriftStepResult] = []

    for step_idx, drift_step in enumerate(gradual_result.steps):
        step_num = step_idx + 1
        logger.info(
            "S10: Scoring step %d/%d (mag=%.3fsigma)...",
            step_num, config.n_steps, drift_step.magnitude,
        )

        X_drifted = drift_step.X_drifted[feature_names]
        step_seed = test_rng.integers(0, 2**31)

        # SWIFT
        monitor.set_params(random_state=int(step_seed))
        swift_result = monitor.test(X_drifted)

        swift_scores = {
            r.feature_name: r.swift_score
            for r in swift_result.feature_results
        }
        swift_pvalues = {
            r.feature_name: r.p_value
            for r in swift_result.feature_results
        }

        # Compute SHAP on drifted monitoring data (for Decker baseline)
        step_explainer = shap.TreeExplainer(model)
        step_shap_mon = np.asarray(
            step_explainer.shap_values(X_drifted)
        )

        # Baselines
        baseline_scores = run_all_baselines(
            X_ref=X_ref[feature_names],
            X_mon=X_drifted,
            feature_names=feature_names,
            shap_values=shap_values_ref,
            shap_values_mon=step_shap_mon,
            bucket_sets=monitor.bucket_sets_,
            model=model,
        )

        sr = GradualDriftStepResult(
            step=step_num,
            magnitude=drift_step.magnitude,
            swift_scores=swift_scores,
            swift_pvalues=swift_pvalues,
            swift_drifted=list(swift_result.drifted_features),
            swift_max=swift_result.swift_max,
            swift_mean=swift_result.swift_mean,
            baseline_scores=baseline_scores,
        )
        step_results.append(sr)

    # --- Step 7: Compute detection delays ---
    swift_max_series = [sr.swift_max for sr in step_results]
    detection_delays: dict[str, int | None] = {
        "SWIFT_max": compute_detection_delay(
            swift_max_series, null_thresholds["SWIFT_max"],
        ),
    }

    # Baselines
    baseline_methods = list(step_results[0].baseline_scores.keys())
    for method in baseline_methods:
        method_max_series = []
        for sr in step_results:
            feat_scores = sr.baseline_scores.get(method, {})
            method_max_series.append(
                max(feat_scores.values()) if feat_scores else 0.0,
            )
        threshold_key = f"{method}_max"
        threshold = null_thresholds.get(threshold_key, float("inf"))
        detection_delays[f"{method}_max"] = compute_detection_delay(
            method_max_series, threshold,
        )

    total_time = time.time() - t_start

    result = GradualDriftExperimentResult(
        dataset_name=config.dataset_name,
        step_results=step_results,
        n_steps=config.n_steps,
        max_magnitude=config.max_magnitude,
        drifted_features=gradual_result.drifted_features,
        model_auc=model_auc,
        n_ref=len(X_ref),
        n_mon=len(X_mon_clean),
        n_features=len(feature_names),
        feature_names=feature_names,
        detection_delay=detection_delays,
        null_threshold=null_thresholds,
        total_time_seconds=total_time,
        fit_time_seconds=fit_time,
    )

    logger.info("S10 experiment complete.\n%s", result.summary())
    return result
