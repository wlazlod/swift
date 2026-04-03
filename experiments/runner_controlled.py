"""Controlled experiment runner — runs scenarios S1-S9 with drift injection.

Typical usage:
    from experiments.runner_controlled import run_controlled_experiment
    from experiments.runner_base import ExperimentConfig

    cfg = ExperimentConfig(dataset_name="taiwan_credit", scenarios=["S1", "S2", "S9"])
    result = run_controlled_experiment(X, y, feature_names, numeric_features,
                                        categorical_features, cfg)
    result.save("results/taiwan_credit.json")
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import shap

from experiments.baselines import run_all_baselines
from experiments.drift import DriftConfig, inject_drift
from experiments.runner_base import (
    ExperimentConfig,
    ExperimentResult,
    ScenarioResult,
    _get_scenario_map,
    prepare_experiment_data,
)

logger = logging.getLogger(__name__)


def run_controlled_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    config: ExperimentConfig,
) -> ExperimentResult:
    """Run a full controlled experiment: train -> fit -> drift -> score -> evaluate.

    Steps:
        1. Split data into reference and monitoring sets
        2. Train LightGBM on reference data
        3. Fit SWIFTMonitor on reference data
        4. For each (scenario x magnitude):
            a. Inject drift into monitoring data
            b. Run SWIFT scoring + permutation test
            c. Run all baselines
            d. Record results
        5. Return ExperimentResult

    Args:
        X: Full feature DataFrame.
        y: Full target Series.
        feature_names: Feature names.
        numeric_features: Numeric feature names.
        categorical_features: Categorical feature names.
        config: Experiment configuration.

    Returns:
        ExperimentResult with all scenario results.
    """
    t_start = time.time()

    # Prepare data, model, monitor
    (X_ref, y_ref, X_mon_clean, y_mon,
     model, model_auc, monitor, fit_time, shap_values_ref) = prepare_experiment_data(
        X, y, feature_names, categorical_features, config,
    )

    scenario_map = _get_scenario_map()

    # Create a single RNG for the entire experiment loop so each scenario
    # gets a different (but reproducible) part of the random stream.
    test_rng = np.random.default_rng(config.random_state + 1000)

    scenario_results: list[ScenarioResult] = []

    for scenario_code in config.scenarios:
        drift_scenario = scenario_map[scenario_code]
        magnitudes_for_scenario = config.magnitudes_for(scenario_code)

        for magnitude in magnitudes_for_scenario:
            logger.info(
                "Running scenario %s (magnitude=%.2f)...",
                scenario_code, magnitude,
            )

            # Inject drift
            drift_config = DriftConfig(
                scenario=drift_scenario,
                magnitude=magnitude,
                n_features=config.n_features_to_drift,
                random_state=config.random_state,
            )

            # Build bucket_sets dict for S8
            bucket_sets_dict: dict[str, Any] | None = None
            if scenario_code == "S8" and monitor.bucket_sets_ is not None:
                bucket_sets_dict = dict(monitor.bucket_sets_)

            drift_result = inject_drift(
                X=X_mon_clean,
                config=drift_config,
                shap_values=shap_values_ref,
                feature_names=feature_names,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                y=y_mon,
                bucket_sets=bucket_sets_dict,
            )

            X_mon_drifted = drift_result.X_drifted[feature_names]

            # SWIFT scoring + test
            scenario_seed = test_rng.integers(0, 2**31)
            monitor.set_params(random_state=int(scenario_seed))
            swift_result = monitor.test(X_mon_drifted)

            swift_scores = {
                r.feature_name: r.swift_score
                for r in swift_result.feature_results
            }
            swift_pvalues = {
                r.feature_name: r.p_value
                for r in swift_result.feature_results
            }
            swift_drifted = list(swift_result.drifted_features)

            # Compute SHAP values on drifted monitoring data (for Decker baseline)
            explainer = shap.TreeExplainer(model)
            shap_values_mon = np.asarray(
                explainer.shap_values(X_mon_drifted)
            )

            # Run baselines
            baseline_scores = run_all_baselines(
                X_ref=X_ref[feature_names],
                X_mon=X_mon_drifted,
                feature_names=feature_names,
                shap_values=shap_values_ref,
                shap_values_mon=shap_values_mon,
                bucket_sets=monitor.bucket_sets_,
                model=model,
            )

            # Record results
            sr = ScenarioResult(
                scenario=scenario_code,
                magnitude=magnitude,
                swift_scores=swift_scores,
                swift_pvalues=swift_pvalues,
                swift_drifted=swift_drifted,
                swift_max=swift_result.swift_max,
                swift_mean=swift_result.swift_mean,
                baseline_scores=baseline_scores,
                drifted_features=drift_result.drifted_features,
                description=drift_result.description,
            )
            scenario_results.append(sr)

            logger.info(
                "  %s (mag=%.2f): SWIFT_max=%.6f, SWIFT_mean=%.6f, "
                "drifted=%d/%d. %s",
                scenario_code, magnitude,
                sr.swift_max, sr.swift_mean,
                len(sr.swift_drifted), len(feature_names),
                drift_result.description,
            )

    total_time = time.time() - t_start

    result = ExperimentResult(
        dataset_name=config.dataset_name,
        scenario_results=scenario_results,
        model_auc=model_auc,
        n_ref=len(X_ref),
        n_mon=len(X_mon_clean),
        n_features=len(feature_names),
        feature_names=feature_names,
        total_time_seconds=total_time,
        fit_time_seconds=fit_time,
        config=config,
    )

    logger.info("Experiment complete.\n%s", result.summary())
    return result
