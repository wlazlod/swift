"""Tests for the SWIFT pipeline orchestrator.

Contract:
    SWIFTMonitor — Main user-facing class (sklearn-compatible):
        SWIFTMonitor(model=..., order=1, n_permutations=1000, alpha=0.05,
                     correction="benjamini-hochberg", n_synthetic=10,
                     max_samples=None, random_state=42)
        .fit(X)                       -> self
        .transform(X)                 -> pd.DataFrame of SHAP-transformed values
        .score(X, X_compare=None)     -> dict[str, float]
        .test(X, X_compare=None)      -> SWIFTResult
        .fit_transform(X)             -> pd.DataFrame (free from TransformerMixin)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from swift.pipeline import SWIFTMonitor
from swift.types import (
    CorrectionMethod,
    FeatureSWIFTResult,
    SWIFTResult,
    WassersteinOrder,
)


# ---------------------------------------------------------------------------
# TestSWIFTMonitorFit
# ---------------------------------------------------------------------------


class TestSWIFTMonitorFit:
    """Tests for SWIFTMonitor.fit()."""

    def test_fit_returns_self(self, trained_lgb_model, synthetic_data):
        """fit() should return self for chaining."""
        monitor = SWIFTMonitor(model=trained_lgb_model)
        result = monitor.fit(synthetic_data["X_ref"])
        assert result is monitor

    def test_fit_populates_bucket_sets(self, trained_lgb_model, synthetic_data):
        """After fit, bucket_sets should be populated for all features."""
        monitor = SWIFTMonitor(model=trained_lgb_model)
        monitor.fit(synthetic_data["X_ref"])

        assert monitor.bucket_sets_ is not None
        assert set(monitor.bucket_sets_.keys()) == set(
            synthetic_data["feature_names"]
        )

    def test_fit_populates_mean_shap(self, trained_lgb_model, synthetic_data):
        """After fit, all buckets should have mean_shap assigned."""
        monitor = SWIFTMonitor(model=trained_lgb_model)
        monitor.fit(synthetic_data["X_ref"])

        for fname, bs in monitor.bucket_sets_.items():
            for bucket in bs.buckets:
                assert bucket.mean_shap is not None, (
                    f"Feature {fname}, bucket {bucket.index}: mean_shap is None"
                )

    def test_fit_stores_reference(self, trained_lgb_model, synthetic_data):
        """fit() should store X_ref for permutation testing."""
        monitor = SWIFTMonitor(model=trained_lgb_model)
        monitor.fit(synthetic_data["X_ref"])

        assert monitor.X_ref_ is not None
        assert len(monitor.X_ref_) == len(synthetic_data["X_ref"])

    def test_fit_infers_feature_names(self, trained_lgb_model, synthetic_data):
        """fit() should infer feature_names_in_ from X.columns."""
        monitor = SWIFTMonitor(model=trained_lgb_model)
        monitor.fit(synthetic_data["X_ref"])

        np.testing.assert_array_equal(
            monitor.feature_names_in_,
            synthetic_data["feature_names"],
        )
        assert monitor.n_features_in_ == len(synthetic_data["feature_names"])

    def test_fit_stores_shap_values(self, trained_lgb_model, synthetic_data):
        """fit() should compute and store SHAP values."""
        monitor = SWIFTMonitor(model=trained_lgb_model)
        monitor.fit(synthetic_data["X_ref"])

        assert monitor.shap_values_ is not None
        assert monitor.shap_values_.shape == (
            len(synthetic_data["X_ref"]),
            len(synthetic_data["feature_names"]),
        )

    def test_fit_no_model_raises(self, synthetic_data):
        """fit() without a model should raise ValueError."""
        monitor = SWIFTMonitor()
        with pytest.raises(ValueError, match="requires a trained model"):
            monitor.fit(synthetic_data["X_ref"])


# ---------------------------------------------------------------------------
# TestSWIFTMonitorTransform
# ---------------------------------------------------------------------------


class TestSWIFTMonitorTransform:
    """Tests for SWIFTMonitor.transform()."""

    def test_transform_returns_dataframe(self, trained_lgb_model, synthetic_data):
        """transform() should return a DataFrame with same shape and columns."""
        monitor = SWIFTMonitor(model=trained_lgb_model)
        monitor.fit(synthetic_data["X_ref"])

        X_mon = synthetic_data["X_mon_drift"]
        result = monitor.transform(X_mon)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == X_mon.shape
        assert list(result.columns) == list(X_mon.columns)

    def test_transform_values_are_mean_shap(self, trained_lgb_model, synthetic_data):
        """Each transformed value should be a valid mean_shap from the bucket set."""
        monitor = SWIFTMonitor(model=trained_lgb_model)
        monitor.fit(synthetic_data["X_ref"])

        result = monitor.transform(synthetic_data["X_mon_drift"])

        for fname in synthetic_data["feature_names"]:
            bs = monitor.bucket_sets_[fname]
            valid_shaps = {
                b.mean_shap for b in bs.buckets if b.mean_shap is not None
            }
            # Every value in the column should be one of the bucket mean_shaps
            for val in result[fname].values:
                assert val in valid_shaps or np.isclose(
                    val, 0.0
                ), f"Unexpected SHAP value {val} for {fname}"

    def test_transform_not_fitted_raises(self, synthetic_data):
        """transform() before fit() should raise NotFittedError."""
        monitor = SWIFTMonitor()
        with pytest.raises(NotFittedError):
            monitor.transform(synthetic_data["X_mon_drift"])

    def test_transform_reference_matches_fit(self, trained_lgb_model, synthetic_data):
        """Transforming reference data should produce consistent values."""
        monitor = SWIFTMonitor(model=trained_lgb_model)
        monitor.fit(synthetic_data["X_ref"])

        result = monitor.transform(synthetic_data["X_ref"])
        assert result.shape == synthetic_data["X_ref"].shape
        # All values should be finite
        assert np.all(np.isfinite(result.values))


# ---------------------------------------------------------------------------
# TestSWIFTMonitorScore
# ---------------------------------------------------------------------------


class TestSWIFTMonitorScore:
    """Tests for SWIFTMonitor.score()."""

    def test_score_returns_per_feature_dict(
        self, trained_lgb_model, synthetic_data
    ):
        """score() should return a SWIFT score for each feature."""
        monitor = SWIFTMonitor(model=trained_lgb_model)
        monitor.fit(synthetic_data["X_ref"])
        scores = monitor.score(synthetic_data["X_mon_drift"])

        assert isinstance(scores, dict)
        assert set(scores.keys()) == set(synthetic_data["feature_names"])

    def test_score_drift_feature_highest(
        self, trained_lgb_model, synthetic_data
    ):
        """feature_0 was shifted — should have highest score."""
        monitor = SWIFTMonitor(model=trained_lgb_model)
        monitor.fit(synthetic_data["X_ref"])
        scores = monitor.score(synthetic_data["X_mon_drift"])

        max_feature = max(scores, key=scores.get)
        assert max_feature == "feature_0"

    def test_score_not_fitted_raises(self, synthetic_data):
        """score() before fit() should raise NotFittedError."""
        monitor = SWIFTMonitor()
        with pytest.raises(NotFittedError):
            monitor.score(synthetic_data["X_mon_drift"])

    def test_score_w2(self, trained_lgb_model, synthetic_data):
        """score() with order=2 should return valid scores."""
        monitor = SWIFTMonitor(model=trained_lgb_model, order=2)
        monitor.fit(synthetic_data["X_ref"])
        scores = monitor.score(synthetic_data["X_mon_drift"])

        for fname, s in scores.items():
            assert s >= 0.0
            assert np.isfinite(s)

    def test_score_sample_vs_sample(self, trained_lgb_model, synthetic_data):
        """score(X_a, X_compare=X_b) should compare two arbitrary samples."""
        monitor = SWIFTMonitor(model=trained_lgb_model)
        monitor.fit(synthetic_data["X_ref"])

        X_a = synthetic_data["X_mon_drift"]
        X_b = synthetic_data["X_mon_nodrift"]

        scores = monitor.score(X_a, X_compare=X_b)

        assert isinstance(scores, dict)
        assert set(scores.keys()) == set(synthetic_data["feature_names"])
        for s in scores.values():
            assert s >= 0.0
            assert np.isfinite(s)

    def test_score_sample_vs_sample_differs_from_ref(
        self, trained_lgb_model, synthetic_data
    ):
        """Sample-vs-sample scores differ from ref-vs-sample scores."""
        monitor = SWIFTMonitor(model=trained_lgb_model)
        monitor.fit(synthetic_data["X_ref"])

        X_a = synthetic_data["X_mon_drift"]
        X_b = synthetic_data["X_mon_nodrift"]

        scores_vs_ref = monitor.score(X_a)
        scores_vs_sample = monitor.score(X_a, X_compare=X_b)

        # At least one feature should have a different score
        diffs = [
            abs(scores_vs_ref[f] - scores_vs_sample[f])
            for f in synthetic_data["feature_names"]
        ]
        assert max(diffs) > 0.0, "Expected at least one score to differ"


# ---------------------------------------------------------------------------
# TestSWIFTMonitorTest
# ---------------------------------------------------------------------------


class TestSWIFTMonitorTest:
    """Tests for SWIFTMonitor.test() — the full pipeline."""

    def test_returns_swift_result(self, trained_lgb_model, synthetic_data):
        """test() should return a SWIFTResult dataclass."""
        monitor = SWIFTMonitor(
            model=trained_lgb_model,
            n_permutations=50,
            random_state=42,
        )
        monitor.fit(synthetic_data["X_ref"])
        result = monitor.test(synthetic_data["X_mon_drift"])

        assert isinstance(result, SWIFTResult)

    def test_feature_results_populated(
        self, trained_lgb_model, synthetic_data
    ):
        """Each feature should have a FeatureSWIFTResult."""
        monitor = SWIFTMonitor(
            model=trained_lgb_model,
            n_permutations=50,
            random_state=42,
        )
        monitor.fit(synthetic_data["X_ref"])
        result = monitor.test(synthetic_data["X_mon_drift"])

        assert result.num_features == len(synthetic_data["feature_names"])
        for fr in result.feature_results:
            assert isinstance(fr, FeatureSWIFTResult)
            assert fr.swift_score >= 0.0
            assert fr.p_value is not None
            assert 0.0 < fr.p_value <= 1.0
            assert fr.is_drifted is not None

    def test_drift_detected(self, trained_lgb_model, synthetic_data):
        """With a 1.5 sigma shift on feature_0, the pipeline should detect it."""
        monitor = SWIFTMonitor(
            model=trained_lgb_model,
            n_permutations=200,
            alpha=0.05,
            correction=CorrectionMethod.BH,
            random_state=42,
        )
        monitor.fit(synthetic_data["X_ref"])
        result = monitor.test(synthetic_data["X_mon_drift"])

        assert "feature_0" in result.drifted_features, (
            f"Expected feature_0 in drifted features, got {result.drifted_features}. "
            f"Feature results: {result.feature_results}"
        )

    def test_no_drift_not_detected(self, trained_lgb_model, synthetic_data):
        """With no drift, few/no features should be flagged."""
        monitor = SWIFTMonitor(
            model=trained_lgb_model,
            n_permutations=200,
            alpha=0.05,
            correction=CorrectionMethod.BH,
            random_state=42,
        )
        monitor.fit(synthetic_data["X_ref"])
        result = monitor.test(synthetic_data["X_mon_nodrift"])

        # Should flag at most 1 out of 5 (with BH at alpha=0.05)
        assert result.num_drifted <= 1, (
            f"Expected <=1 drifted feature with no drift, got {result.num_drifted}. "
            f"Drifted: {result.drifted_features}"
        )

    def test_aggregation_is_consistent(
        self, trained_lgb_model, synthetic_data
    ):
        """swift_max and swift_mean should be consistent with feature_results."""
        monitor = SWIFTMonitor(
            model=trained_lgb_model,
            n_permutations=50,
            random_state=42,
        )
        monitor.fit(synthetic_data["X_ref"])
        result = monitor.test(synthetic_data["X_mon_drift"])

        feature_scores = [fr.swift_score for fr in result.feature_results]
        assert result.swift_max == pytest.approx(max(feature_scores))
        assert result.swift_mean == pytest.approx(
            sum(feature_scores) / len(feature_scores)
        )

    def test_bonferroni_correction_works(
        self, trained_lgb_model, synthetic_data
    ):
        """Should work with Bonferroni correction."""
        monitor = SWIFTMonitor(
            model=trained_lgb_model,
            n_permutations=200,
            correction=CorrectionMethod.BONFERRONI,
            random_state=42,
        )
        monitor.fit(synthetic_data["X_ref"])
        result = monitor.test(synthetic_data["X_mon_drift"])

        assert result.correction_method == CorrectionMethod.BONFERRONI

    def test_not_fitted_raises(self, synthetic_data):
        """test() before fit() should raise NotFittedError."""
        monitor = SWIFTMonitor()
        with pytest.raises(NotFittedError):
            monitor.test(synthetic_data["X_mon_drift"])

    def test_reproducible(self, trained_lgb_model, synthetic_data):
        """Same seed should give identical results."""
        monitor = SWIFTMonitor(
            model=trained_lgb_model,
            n_permutations=50,
            random_state=99,
        )
        monitor.fit(synthetic_data["X_ref"])

        r1 = monitor.test(synthetic_data["X_mon_drift"])
        r2 = monitor.test(synthetic_data["X_mon_drift"])

        for fr1, fr2 in zip(r1.feature_results, r2.feature_results):
            assert fr1.swift_score == pytest.approx(fr2.swift_score)
            assert fr1.p_value == pytest.approx(fr2.p_value)

    def test_sample_vs_sample(self, trained_lgb_model, synthetic_data):
        """test(X_a, X_compare=X_b) should return valid SWIFTResult."""
        monitor = SWIFTMonitor(
            model=trained_lgb_model,
            n_permutations=50,
            random_state=42,
        )
        monitor.fit(synthetic_data["X_ref"])

        X_a = synthetic_data["X_mon_drift"]
        X_b = synthetic_data["X_mon_nodrift"]

        result = monitor.test(X_a, X_compare=X_b)

        assert isinstance(result, SWIFTResult)
        assert result.num_features == len(synthetic_data["feature_names"])

        for fr in result.feature_results:
            assert fr.swift_score >= 0.0
            assert fr.p_value is not None
            assert 0.0 < fr.p_value <= 1.0
            assert fr.is_drifted is not None

    def test_sample_vs_sample_detects_drift(
        self, trained_lgb_model, synthetic_data
    ):
        """Comparing drifted vs non-drifted samples should flag feature_0."""
        monitor = SWIFTMonitor(
            model=trained_lgb_model,
            n_permutations=200,
            alpha=0.05,
            correction=CorrectionMethod.BH,
            random_state=42,
        )
        monitor.fit(synthetic_data["X_ref"])

        X_a = synthetic_data["X_mon_drift"]
        X_b = synthetic_data["X_mon_nodrift"]

        result = monitor.test(X_a, X_compare=X_b)

        assert "feature_0" in result.drifted_features, (
            f"Expected feature_0 in drifted features, got "
            f"{result.drifted_features}"
        )


# ---------------------------------------------------------------------------
# TestSWIFTMonitorSklearn
# ---------------------------------------------------------------------------


class TestSWIFTMonitorSklearn:
    """Tests for sklearn integration (BaseEstimator + TransformerMixin)."""

    def test_get_params(self, trained_lgb_model):
        """get_params() should return all constructor parameters."""
        monitor = SWIFTMonitor(
            model=trained_lgb_model,
            order=2,
            n_permutations=500,
            alpha=0.01,
            correction="bonferroni",
            n_synthetic=5,
            max_samples=3000,
            random_state=123,
        )
        params = monitor.get_params()

        assert params["model"] is trained_lgb_model
        assert params["order"] == 2
        assert params["n_permutations"] == 500
        assert params["alpha"] == 0.01
        assert params["correction"] == "bonferroni"
        assert params["n_synthetic"] == 5
        assert params["max_samples"] == 3000
        assert params["random_state"] == 123

    def test_set_params(self, trained_lgb_model):
        """set_params() should update parameters and return self."""
        monitor = SWIFTMonitor(model=trained_lgb_model)
        result = monitor.set_params(random_state=99, n_permutations=500)

        assert result is monitor
        assert monitor.random_state == 99
        assert monitor.n_permutations == 500

    def test_fit_transform(self, trained_lgb_model, synthetic_data):
        """fit_transform() should produce the same result as fit() + transform()."""
        monitor1 = SWIFTMonitor(model=trained_lgb_model, random_state=42)
        monitor2 = SWIFTMonitor(model=trained_lgb_model, random_state=42)

        X_ref = synthetic_data["X_ref"]

        result_combined = monitor1.fit_transform(X_ref)
        monitor2.fit(X_ref)
        result_separate = monitor2.transform(X_ref)

        pd.testing.assert_frame_equal(result_combined, result_separate)

    def test_repr(self, trained_lgb_model):
        """__repr__ should return a readable string."""
        monitor = SWIFTMonitor(model=trained_lgb_model, n_permutations=200)
        repr_str = repr(monitor)

        assert "SWIFTMonitor" in repr_str
        assert "n_permutations=200" in repr_str
