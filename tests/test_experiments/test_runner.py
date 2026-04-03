"""Tests for experiment runner — orchestrates full experiment pipelines.

TDD: Red → Green → Refactor.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from experiments.runner_base import (
    ExperimentConfig,
    ExperimentResult,
    ScenarioResult,
    train_model,
)
from experiments.runner_controlled import run_controlled_experiment
from experiments.runner_gradual import (
    GradualDriftConfig2,
    GradualDriftExperimentResult,
    GradualDriftStepResult,
    compute_detection_delay,
    run_gradual_drift_experiment,
)


# ---------------------------------------------------------------------------
# Fixtures (small synthetic data for fast tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Small synthetic dataset for fast tests."""
    rng = np.random.default_rng(42)
    n = 500
    X = pd.DataFrame({
        "feat_important_1": rng.normal(0, 1, n),
        "feat_important_2": rng.normal(5, 2, n),
        "feat_unimportant_1": rng.normal(0, 1, n),
        "feat_unimportant_2": rng.normal(0, 1, n),
        "feat_cat": rng.choice([0, 1, 2], n),
    })
    # Target: depends on feat_important_1 and feat_important_2
    logit = 0.5 * X["feat_important_1"] + 0.3 * X["feat_important_2"] - 1.0
    prob = 1 / (1 + np.exp(-logit))
    y = pd.Series((rng.random(n) < prob).astype(int))
    return X, y


@pytest.fixture(scope="module")
def feature_names() -> list[str]:
    return [
        "feat_important_1",
        "feat_important_2",
        "feat_unimportant_1",
        "feat_unimportant_2",
        "feat_cat",
    ]


@pytest.fixture(scope="module")
def numeric_features() -> list[str]:
    return [
        "feat_important_1",
        "feat_important_2",
        "feat_unimportant_1",
        "feat_unimportant_2",
    ]


@pytest.fixture(scope="module")
def categorical_features() -> list[str]:
    return ["feat_cat"]


# ---------------------------------------------------------------------------
# Tests: train_model
# ---------------------------------------------------------------------------

class TestTrainModel:
    """Test the model training utility."""

    def test_returns_booster_and_auc(
        self, synthetic_data, feature_names,
    ):
        X, y = synthetic_data
        model, auc = train_model(X, y, feature_names)

        assert isinstance(model, lgb.Booster)
        assert isinstance(auc, float)
        assert 0.5 <= auc <= 1.0

    def test_auc_reasonable(
        self, synthetic_data, feature_names,
    ):
        """The model should learn something from the data."""
        X, y = synthetic_data
        model, auc = train_model(X, y, feature_names)

        # With 500 samples and 20% validation, AUC should be > 0.55
        assert auc > 0.55, f"AUC too low: {auc}"

    def test_reproducible(
        self, synthetic_data, feature_names,
    ):
        """Same random_state should give same model."""
        X, y = synthetic_data
        _, auc1 = train_model(X, y, feature_names, random_state=99)
        _, auc2 = train_model(X, y, feature_names, random_state=99)
        assert auc1 == auc2


# ---------------------------------------------------------------------------
# Tests: ExperimentConfig
# ---------------------------------------------------------------------------

class TestExperimentConfig:
    """Test experiment configuration."""

    def test_default_config(self):
        cfg = ExperimentConfig(dataset_name="test")
        assert cfg.dataset_name == "test"
        assert cfg.n_permutations > 0
        assert cfg.alpha == 0.05
        assert len(cfg.scenarios) > 0
        assert len(cfg.magnitudes) > 0

    def test_custom_config(self):
        cfg = ExperimentConfig(
            dataset_name="test",
            scenarios=["S1", "S9"],
            magnitudes=[0.5, 1.0],
            n_permutations=100,
            alpha=0.01,
        )
        assert cfg.scenarios == ["S1", "S9"]
        assert cfg.magnitudes == [0.5, 1.0]
        assert cfg.n_permutations == 100
        assert cfg.alpha == 0.01

    def test_s4_in_default_scenarios(self):
        """S4 should be included in DEFAULT_SCENARIOS."""
        from experiments.runner_base import DEFAULT_SCENARIOS
        assert "S4" in DEFAULT_SCENARIOS

    def test_s4_has_magnitude_overrides(self):
        """S4 should have per-scenario magnitude overrides."""
        from experiments.runner_base import DEFAULT_SCENARIO_MAGNITUDES
        assert "S4" in DEFAULT_SCENARIO_MAGNITUDES
        mags = DEFAULT_SCENARIO_MAGNITUDES["S4"]
        # All magnitudes should be in (0, 1] range (rotation fractions of π/4)
        for m in mags:
            assert 0.0 < m <= 1.0, f"S4 magnitude {m} out of expected range"

    def test_s4_magnitudes_via_config(self):
        """ExperimentConfig.magnitudes_for('S4') should use overrides."""
        cfg = ExperimentConfig(dataset_name="test")
        s4_mags = cfg.magnitudes_for("S4")
        # Should use the overrides, not the default [0.5, 1.0, 2.0]
        assert all(m <= 1.0 for m in s4_mags)


# ---------------------------------------------------------------------------
# Tests: ScenarioResult
# ---------------------------------------------------------------------------

class TestScenarioResult:

    def test_scenario_result_fields(self):
        sr = ScenarioResult(
            scenario="S1",
            magnitude=1.0,
            swift_scores={"f1": 0.1, "f2": 0.2},
            swift_pvalues={"f1": 0.01, "f2": 0.5},
            swift_drifted=["f1"],
            swift_max=0.2,
            swift_mean=0.15,
            baseline_scores={"PSI": {"f1": 0.05, "f2": 0.03}},
            drifted_features=["f1"],
            description="S1: test",
        )
        assert sr.scenario == "S1"
        assert sr.magnitude == 1.0
        assert sr.swift_max == 0.2
        assert len(sr.swift_drifted) == 1

    def test_to_dict(self):
        sr = ScenarioResult(
            scenario="S1",
            magnitude=1.0,
            swift_scores={"f1": 0.1},
            swift_pvalues={"f1": 0.01},
            swift_drifted=["f1"],
            swift_max=0.1,
            swift_mean=0.1,
            baseline_scores={"PSI": {"f1": 0.05}},
            drifted_features=["f1"],
            description="S1: test",
        )
        d = sr.to_dict()
        assert isinstance(d, dict)
        assert d["scenario"] == "S1"
        assert "swift_scores" in d
        assert "baseline_scores" in d


# ---------------------------------------------------------------------------
# Tests: run_controlled_experiment
# ---------------------------------------------------------------------------

class TestRunControlledExperiment:
    """Test the controlled experiment runner (main orchestrator)."""

    def test_runs_without_error(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        """The runner should complete without error on minimal config."""
        X, y = synthetic_data
        cfg = ExperimentConfig(
            dataset_name="synthetic_test",
            scenarios=["S1", "S9"],
            magnitudes=[1.0],
            n_permutations=50,  # small for speed
            n_features_to_drift=2,
            random_state=42,
        )

        result = run_controlled_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        assert isinstance(result, ExperimentResult)

    def test_returns_all_scenarios(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        """Should have results for each (scenario × magnitude) combo."""
        X, y = synthetic_data
        scenarios = ["S1", "S9"]
        magnitudes = [0.5, 1.0]
        cfg = ExperimentConfig(
            dataset_name="synthetic_test",
            scenarios=scenarios,
            magnitudes=magnitudes,
            n_permutations=50,
            n_features_to_drift=2,
            random_state=42,
        )

        result = run_controlled_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        # S9 is "no drift" so magnitude is irrelevant → len(magnitudes) combos for S1 + 1 for S9
        # Actually: S9 is run once regardless of magnitudes
        # But S1 is run for each magnitude
        # Expected: S1×0.5, S1×1.0, S9×0.0
        assert len(result.scenario_results) >= 3

    def test_swift_scores_present(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        """Each scenario result should have SWIFT scores for all features."""
        X, y = synthetic_data
        cfg = ExperimentConfig(
            dataset_name="synthetic_test",
            scenarios=["S1"],
            magnitudes=[1.0],
            n_permutations=50,
            n_features_to_drift=2,
            random_state=42,
        )

        result = run_controlled_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        for sr in result.scenario_results:
            assert len(sr.swift_scores) == len(feature_names)
            for fname in feature_names:
                assert fname in sr.swift_scores
                assert sr.swift_scores[fname] >= 0.0

    def test_baseline_scores_present(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        """Each scenario should have baseline scores."""
        X, y = synthetic_data
        cfg = ExperimentConfig(
            dataset_name="synthetic_test",
            scenarios=["S1"],
            magnitudes=[1.0],
            n_permutations=50,
            n_features_to_drift=2,
            random_state=42,
        )

        result = run_controlled_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        for sr in result.scenario_results:
            assert "PSI" in sr.baseline_scores
            assert "KS" in sr.baseline_scores
            assert "Raw_W1" in sr.baseline_scores

    def test_s1_detects_drift(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        """S1 with strong drift should be detected by SWIFT."""
        X, y = synthetic_data
        cfg = ExperimentConfig(
            dataset_name="synthetic_test",
            scenarios=["S1"],
            magnitudes=[3.0],  # Strong drift
            n_permutations=100,
            n_features_to_drift=2,
            random_state=42,
        )

        result = run_controlled_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        sr = result.scenario_results[0]
        # At least some features should have large SWIFT scores
        assert sr.swift_max > 0.0
        # Should detect at least one drifted feature
        assert len(sr.swift_drifted) >= 1

    def test_s9_no_drift(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        """S9 (no drift) should have small SWIFT scores."""
        X, y = synthetic_data
        cfg = ExperimentConfig(
            dataset_name="synthetic_test",
            scenarios=["S9"],
            magnitudes=[0.0],
            n_permutations=100,
            n_features_to_drift=2,
            random_state=42,
        )

        result = run_controlled_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        sr = result.scenario_results[0]
        # SWIFT scores should be small under null
        assert sr.swift_mean < 0.5  # Not exact zero due to sampling

    def test_model_auc_recorded(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        """The result should record the model's AUC."""
        X, y = synthetic_data
        cfg = ExperimentConfig(
            dataset_name="synthetic_test",
            scenarios=["S9"],
            magnitudes=[0.0],
            n_permutations=50,
            n_features_to_drift=2,
            random_state=42,
        )

        result = run_controlled_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        assert result.model_auc > 0.5

    def test_save_results(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        """Results should be serializable to JSON."""
        X, y = synthetic_data
        cfg = ExperimentConfig(
            dataset_name="synthetic_test",
            scenarios=["S9"],
            magnitudes=[0.0],
            n_permutations=50,
            n_features_to_drift=2,
            random_state=42,
        )

        result = run_controlled_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "results.json"
            result.save(out_path)

            assert out_path.exists()
            data = json.loads(out_path.read_text())
            assert data["dataset_name"] == "synthetic_test"
            assert "scenario_results" in data

    def test_ref_mon_split(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        """Runner should split data into ref and mon sets."""
        X, y = synthetic_data
        cfg = ExperimentConfig(
            dataset_name="synthetic_test",
            scenarios=["S9"],
            magnitudes=[0.0],
            n_permutations=50,
            ref_fraction=0.6,
            n_features_to_drift=2,
            random_state=42,
        )

        result = run_controlled_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        assert result.n_ref > 0
        assert result.n_mon > 0
        assert result.n_ref + result.n_mon == len(X)
        # With ref_fraction=0.6 and 500 obs: ref ≈ 300, mon ≈ 200
        assert abs(result.n_ref - 300) < 10

    def test_multiple_scenarios_and_magnitudes(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        """Full multi-scenario run should produce correct result count."""
        X, y = synthetic_data
        cfg = ExperimentConfig(
            dataset_name="synthetic_test",
            scenarios=["S1", "S2", "S3", "S9"],
            magnitudes=[0.5, 1.0, 2.0],
            n_permutations=50,
            n_features_to_drift=2,
            random_state=42,
        )

        result = run_controlled_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        # S1, S2, S3 each get 3 magnitudes, S9 gets 1 (no drift)
        expected = 3 * 3 + 1  # = 10
        assert len(result.scenario_results) == expected

    def test_timing_recorded(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        """Should record timing information."""
        X, y = synthetic_data
        cfg = ExperimentConfig(
            dataset_name="synthetic_test",
            scenarios=["S9"],
            magnitudes=[0.0],
            n_permutations=50,
            n_features_to_drift=2,
            random_state=42,
        )

        result = run_controlled_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        assert result.total_time_seconds > 0
        assert result.fit_time_seconds > 0

    def test_s4_covariate_rotation_runs(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        """S4 covariate rotation should run and produce results."""
        X, y = synthetic_data
        cfg = ExperimentConfig(
            dataset_name="synthetic_test",
            scenarios=["S4"],
            n_permutations=50,
            n_features_to_drift=2,
            random_state=42,
        )

        result = run_controlled_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        assert isinstance(result, ExperimentResult)
        # S4 uses DEFAULT_SCENARIO_MAGNITUDES, so multiple magnitudes
        assert len(result.scenario_results) >= 1

        for sr in result.scenario_results:
            assert sr.scenario == "S4"
            # S4 rotates exactly 2 features
            assert len(sr.drifted_features) == 2
            # SWIFT should produce scores for all features
            assert len(sr.swift_scores) == len(feature_names)
            # S4 changes joint distribution but preserves marginals approximately,
            # so SWIFT (marginal monitor) should show small scores
            # We don't assert a threshold here — just verify it runs

    def test_bbsd_psi_in_baseline_scores(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        """BBSD_PSI (Score-PSI) should be present in baseline_scores."""
        X, y = synthetic_data
        cfg = ExperimentConfig(
            dataset_name="synthetic_test",
            scenarios=["S1"],
            magnitudes=[1.0],
            n_permutations=50,
            n_features_to_drift=2,
            random_state=42,
        )

        result = run_controlled_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        sr = result.scenario_results[0]
        # BBSD_PSI should be computed as a baseline
        assert "BBSD_PSI" in sr.baseline_scores, (
            f"Missing BBSD_PSI. Available baselines: {list(sr.baseline_scores.keys())}"
        )
        # It should have a "_model_output" key (model-level score)
        assert "_model_output" in sr.baseline_scores["BBSD_PSI"]
        # Score should be non-negative
        assert sr.baseline_scores["BBSD_PSI"]["_model_output"] >= 0.0


# ---------------------------------------------------------------------------
# Tests: compute_detection_delay
# ---------------------------------------------------------------------------

class TestComputeDetectionDelay:
    """Test the detection delay helper."""

    def test_detects_at_first_exceedance(self):
        scores = [0.1, 0.2, 0.5, 0.8, 1.0]
        assert compute_detection_delay(scores, threshold=0.4) == 3  # step 3 (1-indexed)

    def test_detects_at_step_one(self):
        scores = [1.0, 2.0, 3.0]
        assert compute_detection_delay(scores, threshold=0.5) == 1

    def test_never_detected(self):
        scores = [0.1, 0.2, 0.3]
        assert compute_detection_delay(scores, threshold=1.0) is None

    def test_empty_scores(self):
        assert compute_detection_delay([], threshold=0.5) is None

    def test_equal_to_threshold_not_detected(self):
        """Score must be strictly greater than threshold."""
        scores = [0.5, 0.5, 0.5]
        assert compute_detection_delay(scores, threshold=0.5) is None


# ---------------------------------------------------------------------------
# Tests: run_gradual_drift_experiment (S10)
# ---------------------------------------------------------------------------

class TestRunGradualDriftExperiment:
    """Test the S10 gradual drift experiment runner."""

    def test_runs_without_error(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        """The S10 runner should complete without error on minimal config."""
        X, y = synthetic_data
        cfg = GradualDriftConfig2(
            dataset_name="synthetic_s10",
            n_steps=4,
            max_magnitude=2.0,
            n_features_to_drift=2,
            n_permutations=30,
            n_null_runs=5,
            random_state=42,
        )

        result = run_gradual_drift_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        assert isinstance(result, GradualDriftExperimentResult)

    def test_correct_number_of_steps(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        X, y = synthetic_data
        cfg = GradualDriftConfig2(
            dataset_name="synthetic_s10",
            n_steps=6,
            max_magnitude=2.0,
            n_features_to_drift=2,
            n_permutations=30,
            n_null_runs=3,
            random_state=42,
        )

        result = run_gradual_drift_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        assert result.n_steps == 6
        assert len(result.step_results) == 6

    def test_step_magnitudes_increase(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        X, y = synthetic_data
        cfg = GradualDriftConfig2(
            dataset_name="synthetic_s10",
            n_steps=4,
            max_magnitude=2.0,
            n_features_to_drift=2,
            n_permutations=30,
            n_null_runs=3,
            random_state=42,
        )

        result = run_gradual_drift_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        magnitudes = [sr.magnitude for sr in result.step_results]
        for i in range(1, len(magnitudes)):
            assert magnitudes[i] > magnitudes[i - 1]

    def test_swift_scores_increase_with_drift(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        """SWIFT_max should generally increase as drift magnitude grows."""
        X, y = synthetic_data
        cfg = GradualDriftConfig2(
            dataset_name="synthetic_s10",
            n_steps=4,
            max_magnitude=3.0,
            n_features_to_drift=2,
            n_permutations=50,
            n_null_runs=3,
            random_state=42,
        )

        result = run_gradual_drift_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        swift_maxes = [sr.swift_max for sr in result.step_results]
        # Last step score should be higher than first step
        assert swift_maxes[-1] > swift_maxes[0]

    def test_detection_delay_present(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        X, y = synthetic_data
        cfg = GradualDriftConfig2(
            dataset_name="synthetic_s10",
            n_steps=4,
            max_magnitude=2.0,
            n_features_to_drift=2,
            n_permutations=30,
            n_null_runs=5,
            random_state=42,
        )

        result = run_gradual_drift_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        assert "SWIFT_max" in result.detection_delay
        # Baseline methods should also have detection delay entries
        assert len(result.detection_delay) > 1

    def test_null_thresholds_computed(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        X, y = synthetic_data
        cfg = GradualDriftConfig2(
            dataset_name="synthetic_s10",
            n_steps=3,
            max_magnitude=2.0,
            n_features_to_drift=2,
            n_permutations=30,
            n_null_runs=5,
            random_state=42,
        )

        result = run_gradual_drift_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        assert "SWIFT_max" in result.null_threshold
        assert result.null_threshold["SWIFT_max"] > 0

    def test_save_and_load(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        X, y = synthetic_data
        cfg = GradualDriftConfig2(
            dataset_name="synthetic_s10",
            n_steps=3,
            max_magnitude=2.0,
            n_features_to_drift=2,
            n_permutations=30,
            n_null_runs=3,
            random_state=42,
        )

        result = run_gradual_drift_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "s10_results.json"
            result.save(out_path)

            assert out_path.exists()
            data = json.loads(out_path.read_text())
            assert data["dataset_name"] == "synthetic_s10"
            assert data["n_steps"] == 3
            assert "detection_delay" in data
            assert "step_results" in data
            assert len(data["step_results"]) == 3

    def test_summary_is_readable(
        self,
        synthetic_data,
        feature_names,
        numeric_features,
        categorical_features,
    ):
        X, y = synthetic_data
        cfg = GradualDriftConfig2(
            dataset_name="synthetic_s10",
            n_steps=3,
            max_magnitude=2.0,
            n_features_to_drift=2,
            n_permutations=30,
            n_null_runs=3,
            random_state=42,
        )

        result = run_gradual_drift_experiment(
            X=X,
            y=y,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=cfg,
        )

        summary = result.summary()
        assert "S10 Gradual Drift" in summary
        assert "detection" in summary.lower() or "Detection" in summary
        assert "SWIFT_max" in summary
