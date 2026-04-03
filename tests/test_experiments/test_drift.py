"""Tests for drift injection framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.drift import (
    DriftConfig,
    DriftResult,
    DriftScenario,
    GradualDriftConfig,
    GradualDriftResult,
    inject_drift,
    inject_gradual_drift,
)


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Create sample data for drift injection tests."""
    rng = np.random.default_rng(42)
    n = 1000
    X = pd.DataFrame({
        "feat_imp_1": rng.normal(10, 2, n),       # Important numeric
        "feat_imp_2": rng.normal(50, 10, n),      # Important numeric
        "feat_imp_3": rng.normal(0.5, 0.1, n),    # Important numeric
        "feat_unimp_1": rng.normal(100, 20, n),   # Unimportant numeric
        "feat_unimp_2": rng.normal(0, 1, n),      # Unimportant numeric
        "feat_unimp_3": rng.normal(5, 3, n),      # Unimportant numeric
        "cat_1": rng.choice([0, 1, 2, 3], n, p=[0.4, 0.3, 0.2, 0.1]),  # Categorical
        "cat_2": rng.choice([0, 1], n, p=[0.6, 0.4]),  # Binary categorical
    })

    # SHAP values: first 3 features are "important", last 3 are "unimportant"
    shap_values = np.zeros((n, 8))
    shap_values[:, 0] = rng.normal(0, 0.5, n)    # feat_imp_1: high SHAP
    shap_values[:, 1] = rng.normal(0, 0.4, n)    # feat_imp_2: high SHAP
    shap_values[:, 2] = rng.normal(0, 0.3, n)    # feat_imp_3: high SHAP
    shap_values[:, 3] = rng.normal(0, 0.01, n)   # feat_unimp_1: low SHAP
    shap_values[:, 4] = rng.normal(0, 0.01, n)   # feat_unimp_2: low SHAP
    shap_values[:, 5] = rng.normal(0, 0.01, n)   # feat_unimp_3: low SHAP
    shap_values[:, 6] = rng.normal(0, 0.02, n)   # cat_1: low SHAP
    shap_values[:, 7] = rng.normal(0, 0.02, n)   # cat_2: low SHAP

    return X, shap_values


@pytest.fixture
def feature_names() -> list[str]:
    return [
        "feat_imp_1", "feat_imp_2", "feat_imp_3",
        "feat_unimp_1", "feat_unimp_2", "feat_unimp_3",
        "cat_1", "cat_2",
    ]


@pytest.fixture
def numeric_features() -> list[str]:
    return [
        "feat_imp_1", "feat_imp_2", "feat_imp_3",
        "feat_unimp_1", "feat_unimp_2", "feat_unimp_3",
    ]


@pytest.fixture
def categorical_features() -> list[str]:
    return ["cat_1", "cat_2"]


class TestS1MeanShiftImportant:
    """S1: Mean shift on important features."""

    def test_shifts_important_features(self, sample_data, feature_names, numeric_features):
        X, shap_values = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S1_MEAN_SHIFT_IMPORTANT,
            magnitude=1.0,
            n_features=3,
        )
        result = inject_drift(
            X, config, shap_values=shap_values,
            feature_names=feature_names, numeric_features=numeric_features,
        )

        assert result.scenario == DriftScenario.S1_MEAN_SHIFT_IMPORTANT
        assert len(result.drifted_features) == 3

        # Important features should be shifted
        for col in result.drifted_features:
            original_mean = X[col].mean()
            drifted_mean = result.X_drifted[col].mean()
            sigma = X[col].std()
            # Shift should be approximately magnitude * sigma
            assert abs(drifted_mean - original_mean - 1.0 * sigma) < 0.1 * sigma

    def test_unimportant_features_unchanged(self, sample_data, feature_names, numeric_features):
        X, shap_values = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S1_MEAN_SHIFT_IMPORTANT,
            magnitude=1.0,
            n_features=3,
        )
        result = inject_drift(
            X, config, shap_values=shap_values,
            feature_names=feature_names, numeric_features=numeric_features,
        )

        # Unimportant features should be unchanged
        for col in numeric_features:
            if col not in result.drifted_features:
                pd.testing.assert_series_equal(X[col], result.X_drifted[col])

    def test_magnitude_scaling(self, sample_data, feature_names, numeric_features):
        X, shap_values = sample_data
        for magnitude in [0.5, 1.0, 2.0]:
            config = DriftConfig(
                scenario=DriftScenario.S1_MEAN_SHIFT_IMPORTANT,
                magnitude=magnitude,
                n_features=1,
            )
            result = inject_drift(
                X, config, shap_values=shap_values,
                feature_names=feature_names, numeric_features=numeric_features,
            )
            col = result.drifted_features[0]
            shift = result.X_drifted[col].mean() - X[col].mean()
            expected_shift = magnitude * X[col].std()
            assert abs(shift - expected_shift) < 0.1 * X[col].std()


class TestS2MeanShiftUnimportant:
    """S2: Mean shift on unimportant features."""

    def test_shifts_unimportant_features(self, sample_data, feature_names, numeric_features):
        X, shap_values = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S2_MEAN_SHIFT_UNIMPORTANT,
            magnitude=2.0,
            n_features=3,
        )
        result = inject_drift(
            X, config, shap_values=shap_values,
            feature_names=feature_names, numeric_features=numeric_features,
        )

        assert len(result.drifted_features) == 3
        # Should select least important features
        for col in result.drifted_features:
            assert col in {"feat_unimp_1", "feat_unimp_2", "feat_unimp_3"}


class TestS3VarianceChange:
    """S3: Variance change."""

    def test_doubles_variance(self, sample_data, feature_names, numeric_features):
        X, shap_values = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S3_VARIANCE_CHANGE,
            magnitude=2.0,
            n_features=2,
        )
        result = inject_drift(
            X, config, shap_values=shap_values,
            feature_names=feature_names, numeric_features=numeric_features,
        )

        for col in result.drifted_features:
            original_var = X[col].var()
            drifted_var = result.X_drifted[col].var()
            # Variance should be ~2x
            assert abs(drifted_var / original_var - 2.0) < 0.1

    def test_preserves_mean(self, sample_data, feature_names, numeric_features):
        X, shap_values = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S3_VARIANCE_CHANGE,
            magnitude=2.0,
            n_features=2,
        )
        result = inject_drift(
            X, config, shap_values=shap_values,
            feature_names=feature_names, numeric_features=numeric_features,
        )

        for col in result.drifted_features:
            # Mean should be preserved
            assert abs(X[col].mean() - result.X_drifted[col].mean()) < 0.01 * X[col].std()


class TestS4CovariateRotation:
    """S4: Covariate rotation."""

    def test_rotation_changes_correlation(self, sample_data, feature_names, numeric_features):
        X, _ = sample_data
        # Use specific target features
        config = DriftConfig(
            scenario=DriftScenario.S4_COVARIATE_ROTATION,
            magnitude=1.0,
            target_features=["feat_imp_1", "feat_imp_2"],
        )
        result = inject_drift(
            X, config, feature_names=feature_names,
            numeric_features=numeric_features,
        )

        # Correlation should change
        orig_corr = X[["feat_imp_1", "feat_imp_2"]].corr().iloc[0, 1]
        new_corr = result.X_drifted[["feat_imp_1", "feat_imp_2"]].corr().iloc[0, 1]
        # They should differ (rotation changes correlation)
        assert abs(orig_corr - new_corr) > 0.01 or abs(orig_corr) < 0.05

    def test_rotation_preserves_marginal_mean_approximately(
        self, sample_data, feature_names, numeric_features,
    ):
        """S4 rotation should approximately preserve per-feature means."""
        X, _ = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S4_COVARIATE_ROTATION,
            magnitude=0.5,
            target_features=["feat_imp_1", "feat_imp_2"],
        )
        result = inject_drift(
            X, config, feature_names=feature_names,
            numeric_features=numeric_features,
        )

        for col in ["feat_imp_1", "feat_imp_2"]:
            orig_mean = X[col].mean()
            new_mean = result.X_drifted[col].mean()
            sigma = X[col].std()
            # Mean should be approximately preserved (within ~0.5σ tolerance)
            assert abs(orig_mean - new_mean) < 0.5 * sigma, (
                f"Mean of {col} shifted too much: {orig_mean:.3f} -> {new_mean:.3f}"
            )

    def test_rotation_drifted_features_are_targets(
        self, sample_data, feature_names, numeric_features,
    ):
        """S4 should report exactly the 2 target features as drifted."""
        X, _ = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S4_COVARIATE_ROTATION,
            magnitude=1.0,
            target_features=["feat_imp_1", "feat_imp_2"],
        )
        result = inject_drift(
            X, config, feature_names=feature_names,
            numeric_features=numeric_features,
        )
        assert set(result.drifted_features) == {"feat_imp_1", "feat_imp_2"}

    def test_rotation_non_target_features_unchanged(
        self, sample_data, feature_names, numeric_features,
    ):
        """Non-target features should remain identical after S4 rotation."""
        X, _ = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S4_COVARIATE_ROTATION,
            magnitude=1.0,
            target_features=["feat_imp_1", "feat_imp_2"],
        )
        result = inject_drift(
            X, config, feature_names=feature_names,
            numeric_features=numeric_features,
        )
        for col in feature_names:
            if col not in ["feat_imp_1", "feat_imp_2"]:
                pd.testing.assert_series_equal(X[col], result.X_drifted[col])

    def test_rotation_magnitude_scaling(
        self, sample_data, feature_names, numeric_features,
    ):
        """Larger magnitude → larger rotation angle → larger correlation change."""
        X, _ = sample_data
        corr_changes = []
        for mag in [0.25, 0.50, 1.0]:
            config = DriftConfig(
                scenario=DriftScenario.S4_COVARIATE_ROTATION,
                magnitude=mag,
                target_features=["feat_imp_1", "feat_imp_2"],
            )
            result = inject_drift(
                X, config, feature_names=feature_names,
                numeric_features=numeric_features,
            )
            orig_corr = X[["feat_imp_1", "feat_imp_2"]].corr().iloc[0, 1]
            new_corr = result.X_drifted[["feat_imp_1", "feat_imp_2"]].corr().iloc[0, 1]
            corr_changes.append(abs(orig_corr - new_corr))

        # Larger magnitude should produce larger correlation change
        # (not strictly monotonic due to modular arithmetic, but in [0,1] range it should be)
        assert corr_changes[-1] >= corr_changes[0]


class TestS5SubpopulationShift:
    """S5: Subpopulation shift."""

    def test_replaces_fraction(self, sample_data, feature_names, numeric_features):
        X, _ = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S5_SUBPOPULATION_SHIFT,
            magnitude=0.10,  # 10% replacement
        )
        result = inject_drift(
            X, config, feature_names=feature_names,
            numeric_features=numeric_features,
        )

        # ~10% of data should differ
        changed = (result.X_drifted != X).any(axis=1).mean()
        assert 0.08 < changed < 0.12

    def test_shift_is_detectable(self, sample_data, feature_names, numeric_features):
        X, _ = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S5_SUBPOPULATION_SHIFT,
            magnitude=0.20,
        )
        result = inject_drift(
            X, config, feature_names=feature_names,
            numeric_features=numeric_features,
        )

        # Overall mean should shift somewhat
        for col in numeric_features[:3]:
            shift = abs(result.X_drifted[col].mean() - X[col].mean())
            assert shift > 0


class TestS6CategoryFreqShift:
    """S6: Category frequency shift."""

    def test_shifts_category_frequencies(
        self, sample_data, feature_names, categorical_features,
    ):
        X, _ = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S6_CATEGORY_FREQ_SHIFT,
            magnitude=0.5,  # Replace 50% of most frequent category
            target_features=["cat_1"],
        )
        result = inject_drift(
            X, config, feature_names=feature_names,
            categorical_features=categorical_features,
        )

        # Distribution of cat_1 should change
        orig_dist = X["cat_1"].value_counts(normalize=True)
        new_dist = result.X_drifted["cat_1"].value_counts(normalize=True)
        # Most frequent category should have fewer observations
        assert new_dist.iloc[0] < orig_dist.iloc[0] or orig_dist.index[0] != new_dist.index[0]

    def test_no_categorical_returns_empty(self, sample_data, feature_names):
        X, _ = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S6_CATEGORY_FREQ_SHIFT,
            magnitude=0.5,
        )
        result = inject_drift(
            X, config, feature_names=feature_names,
            categorical_features=[],  # No categorical features
        )
        assert result.drifted_features == []


class TestS7NullRateIncrease:
    """S7: Null rate increase."""

    def test_increases_null_rate(self, sample_data, feature_names, numeric_features):
        X, _ = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S7_NULL_RATE_INCREASE,
            magnitude=0.20,  # Target 20% null rate
            target_features=["feat_imp_1"],
        )
        result = inject_drift(
            X, config, feature_names=feature_names,
            numeric_features=numeric_features,
        )

        null_rate = result.X_drifted["feat_imp_1"].isnull().mean()
        assert 0.15 < null_rate < 0.25


class TestS8BenignDrift:
    """S8: Benign (virtual) drift — within-bucket jitter."""

    def test_jitters_values_within_buckets(self, sample_data, feature_names, numeric_features):
        X, shap_values = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S8_BENIGN_DRIFT,
            magnitude=0.5,
            target_features=["feat_imp_1", "feat_unimp_1"],
        )
        result = inject_drift(
            X, config, shap_values=shap_values,
            feature_names=feature_names, numeric_features=numeric_features,
        )

        # Values should change (jittered)
        for col in result.drifted_features:
            assert not np.allclose(X[col].values, result.X_drifted[col].values)

        # But distribution mean should remain approximately the same (within-bucket jitter)
        for col in result.drifted_features:
            mean_diff = abs(X[col].mean() - result.X_drifted[col].mean())
            # Jitter within buckets should NOT shift the overall mean much
            assert mean_diff < 0.5 * X[col].std(), (
                f"Mean shifted too much for {col}: {mean_diff:.4f} vs std={X[col].std():.4f}"
            )

    def test_all_numeric_features_drifted_by_default(
        self, sample_data, feature_names, numeric_features,
    ):
        X, shap_values = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S8_BENIGN_DRIFT,
            magnitude=0.5,
        )
        result = inject_drift(
            X, config, shap_values=shap_values,
            feature_names=feature_names, numeric_features=numeric_features,
        )

        # Default targets all numeric features
        for col in numeric_features:
            assert col in result.drifted_features

    def test_does_not_require_shap_values(
        self, sample_data, feature_names, numeric_features,
    ):
        """S8 no longer requires shap_values (it uses bucket_sets or quantile fallback)."""
        X, _ = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S8_BENIGN_DRIFT,
            magnitude=0.5,
            target_features=["feat_imp_1"],
        )
        # Should NOT raise — falls back to quantile-based jitter
        result = inject_drift(
            X, config, shap_values=None,
            feature_names=feature_names, numeric_features=numeric_features,
        )
        assert "feat_imp_1" in result.drifted_features


class TestS9NoDrift:
    """S9: No drift (null hypothesis)."""

    def test_returns_unchanged_data(self, sample_data, feature_names):
        X, _ = sample_data
        config = DriftConfig(scenario=DriftScenario.S9_NO_DRIFT)
        result = inject_drift(X, config, feature_names=feature_names)

        pd.testing.assert_frame_equal(X, result.X_drifted)
        assert result.drifted_features == []
        assert result.magnitude == 0.0


class TestDriftWithExplicitTargets:
    """Test that explicit target_features overrides auto-selection."""

    def test_explicit_targets(self, sample_data, feature_names, numeric_features):
        X, shap_values = sample_data
        targets = ["feat_unimp_1", "feat_unimp_2"]
        config = DriftConfig(
            scenario=DriftScenario.S1_MEAN_SHIFT_IMPORTANT,
            magnitude=1.0,
            target_features=targets,
        )
        result = inject_drift(
            X, config, shap_values=shap_values,
            feature_names=feature_names, numeric_features=numeric_features,
        )

        assert result.drifted_features == targets

    def test_description_is_informative(self, sample_data, feature_names, numeric_features):
        X, _ = sample_data
        config = DriftConfig(
            scenario=DriftScenario.S3_VARIANCE_CHANGE,
            magnitude=3.0,
            target_features=["feat_imp_1"],
        )
        result = inject_drift(
            X, config, feature_names=feature_names,
            numeric_features=numeric_features,
        )

        assert "3.0" in result.description
        assert "feat_imp_1" in result.description


class TestS10GradualDrift:
    """S10: Gradual drift — linear shift over multiple monitoring periods."""

    def test_returns_correct_number_of_steps(
        self, sample_data, feature_names, numeric_features,
    ):
        X, shap_values = sample_data
        config = GradualDriftConfig(
            n_steps=12,
            max_magnitude=3.0,
            n_features=3,
            random_state=42,
        )
        results = inject_gradual_drift(
            X, config,
            shap_values=shap_values,
            feature_names=feature_names,
            numeric_features=numeric_features,
        )

        assert isinstance(results, GradualDriftResult)
        assert len(results.steps) == 12

    def test_step_magnitudes_increase_linearly(
        self, sample_data, feature_names, numeric_features,
    ):
        X, shap_values = sample_data
        config = GradualDriftConfig(
            n_steps=12,
            max_magnitude=3.0,
        )
        results = inject_gradual_drift(
            X, config,
            shap_values=shap_values,
            feature_names=feature_names,
            numeric_features=numeric_features,
        )

        # Magnitudes should linearly increase from max_mag/n_steps to max_mag
        expected_mags = [(i + 1) / 12 * 3.0 for i in range(12)]
        actual_mags = [step.magnitude for step in results.steps]
        np.testing.assert_allclose(actual_mags, expected_mags, rtol=1e-10)

    def test_drift_increases_monotonically(
        self, sample_data, feature_names, numeric_features,
    ):
        X, shap_values = sample_data
        config = GradualDriftConfig(
            n_steps=6,
            max_magnitude=3.0,
            n_features=1,
        )
        results = inject_gradual_drift(
            X, config,
            shap_values=shap_values,
            feature_names=feature_names,
            numeric_features=numeric_features,
        )

        # Mean shift should increase monotonically
        col = results.steps[0].drifted_features[0]
        original_mean = X[col].mean()
        shifts = [
            abs(step.X_drifted[col].mean() - original_mean)
            for step in results.steps
        ]
        for i in range(1, len(shifts)):
            assert shifts[i] > shifts[i - 1], (
                f"Step {i} shift ({shifts[i]:.4f}) not greater than "
                f"step {i-1} shift ({shifts[i-1]:.4f})"
            )

    def test_first_step_is_small_shift(
        self, sample_data, feature_names, numeric_features,
    ):
        X, shap_values = sample_data
        config = GradualDriftConfig(
            n_steps=12,
            max_magnitude=3.0,
            n_features=1,
        )
        results = inject_gradual_drift(
            X, config,
            shap_values=shap_values,
            feature_names=feature_names,
            numeric_features=numeric_features,
        )

        # First step magnitude = 3.0/12 = 0.25σ — small but non-zero
        step0 = results.steps[0]
        col = step0.drifted_features[0]
        sigma = X[col].std()
        shift = abs(step0.X_drifted[col].mean() - X[col].mean())
        expected = 0.25 * sigma
        assert abs(shift - expected) < 0.1 * sigma

    def test_last_step_is_full_magnitude(
        self, sample_data, feature_names, numeric_features,
    ):
        X, shap_values = sample_data
        config = GradualDriftConfig(
            n_steps=12,
            max_magnitude=3.0,
            n_features=1,
        )
        results = inject_gradual_drift(
            X, config,
            shap_values=shap_values,
            feature_names=feature_names,
            numeric_features=numeric_features,
        )

        # Last step = full 3.0σ shift
        last = results.steps[-1]
        col = last.drifted_features[0]
        sigma = X[col].std()
        shift = abs(last.X_drifted[col].mean() - X[col].mean())
        expected = 3.0 * sigma
        assert abs(shift - expected) < 0.1 * sigma

    def test_all_steps_drift_same_features(
        self, sample_data, feature_names, numeric_features,
    ):
        X, shap_values = sample_data
        config = GradualDriftConfig(
            n_steps=4,
            max_magnitude=2.0,
            n_features=2,
        )
        results = inject_gradual_drift(
            X, config,
            shap_values=shap_values,
            feature_names=feature_names,
            numeric_features=numeric_features,
        )

        # All steps should drift the same set of features
        first_features = set(results.steps[0].drifted_features)
        for step in results.steps[1:]:
            assert set(step.drifted_features) == first_features

    def test_undrifted_features_unchanged_per_step(
        self, sample_data, feature_names, numeric_features,
    ):
        X, shap_values = sample_data
        config = GradualDriftConfig(
            n_steps=4,
            max_magnitude=2.0,
            n_features=1,
        )
        results = inject_gradual_drift(
            X, config,
            shap_values=shap_values,
            feature_names=feature_names,
            numeric_features=numeric_features,
        )

        drifted = set(results.steps[0].drifted_features)
        for step in results.steps:
            for col in numeric_features:
                if col not in drifted:
                    pd.testing.assert_series_equal(
                        X[col], step.X_drifted[col],
                        check_names=False,
                    )

    def test_explicit_target_features(
        self, sample_data, feature_names, numeric_features,
    ):
        X, shap_values = sample_data
        config = GradualDriftConfig(
            n_steps=4,
            max_magnitude=1.0,
            target_features=["feat_unimp_1"],
        )
        results = inject_gradual_drift(
            X, config,
            shap_values=shap_values,
            feature_names=feature_names,
            numeric_features=numeric_features,
        )

        for step in results.steps:
            assert step.drifted_features == ["feat_unimp_1"]

    def test_gradual_drift_result_has_metadata(
        self, sample_data, feature_names, numeric_features,
    ):
        X, shap_values = sample_data
        config = GradualDriftConfig(
            n_steps=6,
            max_magnitude=2.0,
        )
        results = inject_gradual_drift(
            X, config,
            shap_values=shap_values,
            feature_names=feature_names,
            numeric_features=numeric_features,
        )

        assert results.n_steps == 6
        assert results.max_magnitude == 2.0
        assert len(results.drifted_features) > 0
        # Step descriptions should mention the step number
        assert "step 1" in results.steps[0].description.lower()

    def test_scenario_enum_value(self):
        """S10 should have a DriftScenario enum value."""
        assert hasattr(DriftScenario, "S10_GRADUAL_DRIFT")
        assert DriftScenario.S10_GRADUAL_DRIFT.value == "S10"
