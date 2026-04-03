"""Shared experiment infrastructure — dataclasses, model training, utilities.

Contains the core building blocks used by both controlled and gradual drift runners:
- ExperimentConfig / ScenarioResult / ExperimentResult dataclasses
- LightGBM model training
- JSON serialization helpers
- Scenario mapping constants
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from swift.pipeline import SWIFTMonitor

logger = logging.getLogger(__name__)

# Map scenario short codes to DriftScenario enum
# Imported lazily to keep this module lightweight; re-exported for convenience.
_SCENARIO_MAP_CACHE: dict[str, Any] | None = None


def _get_scenario_map() -> dict[str, Any]:
    """Lazily build and cache the scenario short-code → enum mapping."""
    global _SCENARIO_MAP_CACHE
    if _SCENARIO_MAP_CACHE is None:
        from experiments.drift import DriftScenario

        _SCENARIO_MAP_CACHE = {
            "S1": DriftScenario.S1_MEAN_SHIFT_IMPORTANT,
            "S2": DriftScenario.S2_MEAN_SHIFT_UNIMPORTANT,
            "S3": DriftScenario.S3_VARIANCE_CHANGE,
            "S4": DriftScenario.S4_COVARIATE_ROTATION,
            "S5": DriftScenario.S5_SUBPOPULATION_SHIFT,
            "S6": DriftScenario.S6_CATEGORY_FREQ_SHIFT,
            "S7": DriftScenario.S7_NULL_RATE_INCREASE,
            "S8": DriftScenario.S8_BENIGN_DRIFT,
            "S9": DriftScenario.S9_NO_DRIFT,
            "S10": DriftScenario.S10_GRADUAL_DRIFT,
        }
    return _SCENARIO_MAP_CACHE


# Scenarios where magnitude is irrelevant (run once with magnitude=0)
NO_MAGNITUDE_SCENARIOS = {"S9"}

# Default drift scenarios from paper design
DEFAULT_SCENARIOS = ["S1", "S2", "S3", "S4", "S5", "S7", "S8", "S9"]
DEFAULT_MAGNITUDES = [0.5, 1.0, 2.0]

# Per-scenario magnitude overrides for scenarios where the default range
# is inappropriate (e.g., S4 rotation > π/4 wraps, S5 frac > 1.0 is
# meaningless, S7 null rate > 1.0 is impossible, S8 jitter fraction is 0-1).
DEFAULT_SCENARIO_MAGNITUDES: dict[str, list[float]] = {
    "S4": [0.25, 0.50, 0.75, 1.00],
    "S5": [0.05, 0.10, 0.20, 0.50],
    "S6": [0.10, 0.25, 0.50, 1.00],
    "S7": [0.10, 0.20, 0.50, 1.00],
    "S8": [0.25, 0.50, 0.75, 1.00],
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a controlled experiment.

    Attributes:
        dataset_name: Identifier for the dataset.
        scenarios: List of scenario codes (e.g., ["S1", "S2", "S9"]).
        magnitudes: Default list of drift magnitudes to test.
        scenario_magnitudes: Per-scenario magnitude overrides.
            Keys are scenario codes; values are lists of magnitudes.
            If a scenario is not listed, ``magnitudes`` is used.
        n_permutations: Number of permutations for SWIFT p-values.
        alpha: Significance level for SWIFT testing.
        ref_fraction: Fraction of data used as reference (rest = monitoring).
        n_features_to_drift: Number of features to drift in each scenario.
        max_samples: Maximum pool size for the permutation test (for perf).
            None = no limit.
        random_state: Global random seed.
    """

    dataset_name: str
    scenarios: list[str] = field(default_factory=lambda: list(DEFAULT_SCENARIOS))
    magnitudes: list[float] = field(default_factory=lambda: list(DEFAULT_MAGNITUDES))
    scenario_magnitudes: dict[str, list[float]] = field(
        default_factory=lambda: dict(DEFAULT_SCENARIO_MAGNITUDES),
    )
    n_permutations: int = 200
    alpha: float = 0.05
    ref_fraction: float = 0.6
    n_features_to_drift: int = 3
    max_samples: int | None = 5000
    random_state: int = 42

    def magnitudes_for(self, scenario_code: str) -> list[float]:
        """Return the magnitudes list for a given scenario code."""
        if scenario_code in NO_MAGNITUDE_SCENARIOS:
            return [0.0]
        return self.scenario_magnitudes.get(scenario_code, self.magnitudes)


@dataclass
class ScenarioResult:
    """Result from running a single (scenario x magnitude) combination.

    Attributes:
        scenario: Scenario code (e.g., "S1").
        magnitude: Drift magnitude applied.
        swift_scores: Per-feature SWIFT scores.
        swift_pvalues: Per-feature p-values from permutation test.
        swift_drifted: Features flagged as drifted by SWIFT.
        swift_max: Model-level max SWIFT score.
        swift_mean: Model-level mean SWIFT score.
        baseline_scores: Baseline scores (method -> feature -> score).
        drifted_features: Features that had drift injected (ground truth).
        description: Human-readable description of the drift.
    """

    scenario: str
    magnitude: float
    swift_scores: dict[str, float]
    swift_pvalues: dict[str, float]
    swift_drifted: list[str]
    swift_max: float
    swift_mean: float
    baseline_scores: dict[str, dict[str, float]]
    drifted_features: list[str]
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "scenario": self.scenario,
            "magnitude": self.magnitude,
            "swift_scores": self.swift_scores,
            "swift_pvalues": self.swift_pvalues,
            "swift_drifted": self.swift_drifted,
            "swift_max": self.swift_max,
            "swift_mean": self.swift_mean,
            "baseline_scores": self.baseline_scores,
            "drifted_features": self.drifted_features,
            "description": self.description,
        }


@dataclass
class ExperimentResult:
    """Result of a full controlled experiment.

    Attributes:
        dataset_name: Dataset identifier.
        scenario_results: List of per-scenario results.
        model_auc: Model AUC on validation data.
        n_ref: Number of reference observations.
        n_mon: Number of monitoring observations (before drift).
        n_features: Number of features.
        feature_names: List of feature names.
        total_time_seconds: Total experiment wall-clock time.
        fit_time_seconds: Time spent fitting SWIFT.
        config: The experiment configuration used.
    """

    dataset_name: str
    scenario_results: list[ScenarioResult]
    model_auc: float
    n_ref: int
    n_mon: int
    n_features: int
    feature_names: list[str]
    total_time_seconds: float = 0.0
    fit_time_seconds: float = 0.0
    config: Optional[ExperimentConfig] = None

    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "dataset_name": self.dataset_name,
            "model_auc": self.model_auc,
            "n_ref": self.n_ref,
            "n_mon": self.n_mon,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "total_time_seconds": self.total_time_seconds,
            "fit_time_seconds": self.fit_time_seconds,
            "scenario_results": [sr.to_dict() for sr in self.scenario_results],
        }

        path.write_text(json.dumps(data, indent=2, default=json_default))
        logger.info("Results saved to %s", path)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Experiment: {self.dataset_name}",
            f"  Model AUC: {self.model_auc:.4f}",
            f"  n_ref={self.n_ref}, n_mon={self.n_mon}, p={self.n_features}",
            f"  Scenarios run: {len(self.scenario_results)}",
            f"  Total time: {self.total_time_seconds:.1f}s "
            f"(fit: {self.fit_time_seconds:.1f}s)",
            "",
        ]
        for sr in self.scenario_results:
            lines.append(
                f"  {sr.scenario} (mag={sr.magnitude:.2f}): "
                f"SWIFT_max={sr.swift_max:.6f}, SWIFT_mean={sr.swift_mean:.6f}, "
                f"drifted={len(sr.swift_drifted)}/{len(sr.swift_scores)}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def json_default(obj: Any) -> Any:
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
# Model training
# ---------------------------------------------------------------------------


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    categorical_features: list[str] | None = None,
    val_fraction: float = 0.2,
    random_state: int = 42,
    num_boost_round: int = 100,
    lgb_params: dict[str, Any] | None = None,
) -> tuple[lgb.Booster, float]:
    """Train a LightGBM binary classifier and return (model, validation_auc).

    Args:
        X: Feature DataFrame.
        y: Binary target Series.
        feature_names: Feature names to use.
        categorical_features: Names of categorical features.
        val_fraction: Fraction of data for validation.
        random_state: Random seed.
        num_boost_round: Number of boosting rounds.
        lgb_params: Override LightGBM parameters.

    Returns:
        (model, val_auc) -- trained Booster and AUC on validation set.
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    val_size = int(n * val_fraction)

    # Shuffle indices
    indices = rng.permutation(n)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_train = X.iloc[train_idx][feature_names]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx][feature_names]
    y_val = y.iloc[val_idx]

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "num_threads": 1,
        "seed": random_state,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }
    if lgb_params:
        params.update(lgb_params)

    cat_cols = categorical_features or []
    # Only include categoricals that are in feature_names
    cat_cols = [c for c in cat_cols if c in feature_names]

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feature_names,
        categorical_feature=cat_cols if cat_cols else "auto",
    )
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        feature_name=feature_names,
        categorical_feature=cat_cols if cat_cols else "auto",
        reference=train_data,
    )

    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(period=0)],  # suppress output
    )

    # Compute validation AUC
    y_pred = model.predict(X_val)
    from experiments.evaluation import compute_model_performance

    val_auc = compute_model_performance(y_val.values, y_pred)

    logger.info("Model trained: AUC=%.4f (val_size=%d)", val_auc, val_size)
    return model, val_auc


# ---------------------------------------------------------------------------
# Common data preparation
# ---------------------------------------------------------------------------


def prepare_experiment_data(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    categorical_features: list[str],
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,
           lgb.Booster, float, SWIFTMonitor, float, np.ndarray]:
    """Split data, train model, fit SWIFT monitor.

    Returns:
        (X_ref, y_ref, X_mon_clean, y_mon, model, model_auc,
         monitor, fit_time, shap_values_ref)
    """
    rng = np.random.default_rng(config.random_state)

    # Split into reference and monitoring
    n = len(X)
    n_ref = int(n * config.ref_fraction)
    indices = rng.permutation(n)
    ref_idx = indices[:n_ref]
    mon_idx = indices[n_ref:]

    X_ref = X.iloc[ref_idx].reset_index(drop=True)
    y_ref = y.iloc[ref_idx].reset_index(drop=True)
    X_mon_clean = X.iloc[mon_idx].reset_index(drop=True)
    y_mon = y.iloc[mon_idx].reset_index(drop=True)

    logger.info(
        "Data split: n_ref=%d, n_mon=%d (ref_fraction=%.2f)",
        len(X_ref), len(X_mon_clean), config.ref_fraction,
    )

    # Train model
    logger.info("Training LightGBM model...")
    model, model_auc = train_model(
        X_ref, y_ref, feature_names,
        categorical_features=categorical_features,
        random_state=config.random_state,
    )

    # Fit SWIFT
    logger.info("Fitting SWIFTMonitor...")
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
    logger.info("SWIFT fit complete in %.1fs", fit_time)

    shap_values_ref = monitor.shap_values_

    return (X_ref, y_ref, X_mon_clean, y_mon,
            model, model_auc, monitor, fit_time, shap_values_ref)
