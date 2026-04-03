"""Tests for ablation study implementations (A1-A6).

Each ablation removes or replaces one SWIFT component to isolate its contribution.
Tests verify correctness of each ablation variant's scoring function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import shap

from experiments.ablations import (
    AblationVariant,
    compute_a1_no_shap_normalization,
    compute_a2_no_model_buckets,
    compute_a3_psi_on_model_buckets,
    compute_a4_w2_instead_of_w1,
    compute_a5_importance_weighted,
    run_all_ablations,
)
from swift.bucketing import build_all_buckets
from swift.extraction import extract_decision_points_lgb
from swift.normalization import compute_bucket_shap
from swift.pipeline import SWIFTMonitor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_monitor(synthetic_data, trained_lgb_model):
    """Fit a SWIFTMonitor for ablation tests."""
    monitor = SWIFTMonitor(model=trained_lgb_model)
    monitor.fit(synthetic_data["X_ref"])
    return monitor


@pytest.fixture(scope="module")
def ref_shap(trained_lgb_model, synthetic_data):
    """Compute SHAP values on reference data."""
    explainer = shap.TreeExplainer(trained_lgb_model)
    sv = explainer.shap_values(synthetic_data["X_ref"])
    return np.asarray(sv)


# ---------------------------------------------------------------------------
# A1: No SHAP normalization — Wasserstein on raw values in model-aware buckets
# ---------------------------------------------------------------------------


class TestA1NoShapNormalization:
    """A1 computes W₁ on raw feature values (not SHAP-transformed) but
    still using model-aware bucket sets for the comparison."""

    def test_returns_dict_of_scores(self, synthetic_data, fitted_monitor):
        """A1 should return a dict of feature_name -> float score."""
        scores = compute_a1_no_shap_normalization(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            bucket_sets=fitted_monitor.bucket_sets_,
        )
        assert isinstance(scores, dict)
        assert set(scores.keys()) == set(synthetic_data["feature_names"])
        for v in scores.values():
            assert isinstance(v, float)
            assert v >= 0.0

    def test_detects_drift_without_shap(self, synthetic_data, fitted_monitor):
        """A1 should detect drift on feature_0 (shifted by 1.5σ) even
        without SHAP normalization — just based on raw value distribution."""
        scores = compute_a1_no_shap_normalization(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            bucket_sets=fitted_monitor.bucket_sets_,
        )
        # Feature 0 has drift — should have highest or near-highest score
        assert scores["feature_0"] > scores["feature_3"]
        assert scores["feature_0"] > scores["feature_4"]

    def test_no_drift_gives_low_scores(self, synthetic_data, fitted_monitor):
        """A1 with identical distributions should give low scores."""
        scores = compute_a1_no_shap_normalization(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_nodrift"],
            bucket_sets=fitted_monitor.bucket_sets_,
        )
        for v in scores.values():
            assert v < 0.5  # Raw W₁ on N(0,1) should be small

    def test_differs_from_full_swift(self, synthetic_data, fitted_monitor):
        """A1 scores should differ from full SWIFT (which uses SHAP normalization)."""
        a1_scores = compute_a1_no_shap_normalization(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            bucket_sets=fitted_monitor.bucket_sets_,
        )
        swift_scores = fitted_monitor.score(synthetic_data["X_mon_drift"])

        # They should not be identical (SHAP normalization changes the scale)
        differences = [abs(a1_scores[f] - swift_scores[f]) for f in a1_scores]
        assert max(differences) > 0.0


# ---------------------------------------------------------------------------
# A2: No model-aware buckets — SHAP normalization + W₁ on equal-frequency bins
# ---------------------------------------------------------------------------


class TestA2NoModelBuckets:
    """A2 uses SHAP normalization but with equal-frequency bins instead of
    model decision points."""

    def test_returns_dict_of_scores(
        self, synthetic_data, trained_lgb_model, ref_shap
    ):
        """A2 should return a dict of feature_name -> float score."""
        scores = compute_a2_no_model_buckets(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            shap_values=ref_shap,
            feature_names=synthetic_data["feature_names"],
            n_bins=10,
        )
        assert isinstance(scores, dict)
        assert set(scores.keys()) == set(synthetic_data["feature_names"])
        for v in scores.values():
            assert isinstance(v, float)
            assert v >= 0.0

    def test_detects_drift(self, synthetic_data, trained_lgb_model, ref_shap):
        """A2 should still detect drift on feature_0 even with equal-freq bins."""
        scores = compute_a2_no_model_buckets(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            shap_values=ref_shap,
            feature_names=synthetic_data["feature_names"],
            n_bins=10,
        )
        assert scores["feature_0"] > scores["feature_3"]

    def test_custom_n_bins(self, synthetic_data, trained_lgb_model, ref_shap):
        """A2 should work with different numbers of bins."""
        scores_5 = compute_a2_no_model_buckets(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            shap_values=ref_shap,
            feature_names=synthetic_data["feature_names"],
            n_bins=5,
        )
        scores_20 = compute_a2_no_model_buckets(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            shap_values=ref_shap,
            feature_names=synthetic_data["feature_names"],
            n_bins=20,
        )
        # Different bin counts should produce different scores
        assert scores_5["feature_0"] != scores_20["feature_0"]


# ---------------------------------------------------------------------------
# A3: PSI formula on model-aware buckets (already exists, test the wrapper)
# ---------------------------------------------------------------------------


class TestA3PsiOnModelBuckets:
    """A3 uses KL divergence (PSI formula) on model-aware buckets."""

    def test_returns_dict_of_scores(self, synthetic_data, fitted_monitor):
        """A3 should return a dict of feature_name -> float score."""
        scores = compute_a3_psi_on_model_buckets(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            bucket_sets=fitted_monitor.bucket_sets_,
        )
        assert isinstance(scores, dict)
        assert set(scores.keys()) == set(synthetic_data["feature_names"])
        for v in scores.values():
            assert isinstance(v, float)
            assert v >= 0.0

    def test_detects_drift(self, synthetic_data, fitted_monitor):
        """A3 should detect drift on feature_0."""
        scores = compute_a3_psi_on_model_buckets(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            bucket_sets=fitted_monitor.bucket_sets_,
        )
        assert scores["feature_0"] > scores["feature_3"]


# ---------------------------------------------------------------------------
# A4: W₂ instead of W₁
# ---------------------------------------------------------------------------


class TestA4W2InsteadOfW1:
    """A4 is full SWIFT but with Wasserstein order 2 instead of 1."""

    def test_returns_dict_of_scores(self, synthetic_data, fitted_monitor):
        """A4 should return a dict of feature_name -> float score."""
        scores = compute_a4_w2_instead_of_w1(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            bucket_sets=fitted_monitor.bucket_sets_,
        )
        assert isinstance(scores, dict)
        assert set(scores.keys()) == set(synthetic_data["feature_names"])
        for v in scores.values():
            assert isinstance(v, float)
            assert v >= 0.0

    def test_detects_drift(self, synthetic_data, fitted_monitor):
        """A4 (W₂) should also detect drift on feature_0."""
        scores = compute_a4_w2_instead_of_w1(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            bucket_sets=fitted_monitor.bucket_sets_,
        )
        assert scores["feature_0"] > scores["feature_3"]

    def test_differs_from_w1(self, synthetic_data, fitted_monitor):
        """W2 scores should differ from W1 scores (different metric)."""
        w1_scores = fitted_monitor.score(synthetic_data["X_mon_drift"])
        w2_scores = compute_a4_w2_instead_of_w1(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            bucket_sets=fitted_monitor.bucket_sets_,
        )
        # At least some features should have different scores
        diffs = [abs(w1_scores[f] - w2_scores[f]) for f in w1_scores]
        assert max(diffs) > 0.0


# ---------------------------------------------------------------------------
# A5: Importance-weighted aggregation
# ---------------------------------------------------------------------------


class TestA5ImportanceWeighted:
    """A5 uses SHAP-importance-weighted mean instead of equal-weight mean."""

    def test_returns_weighted_mean(self, synthetic_data, fitted_monitor, ref_shap):
        """A5 should return a dict with swift_weighted key."""
        result = compute_a5_importance_weighted(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            bucket_sets=fitted_monitor.bucket_sets_,
            shap_values=ref_shap,
            feature_names=synthetic_data["feature_names"],
        )
        assert "swift_weighted" in result
        assert "swift_max" in result
        assert "swift_mean" in result
        assert isinstance(result["swift_weighted"], float)
        assert result["swift_weighted"] >= 0.0

    def test_weighted_differs_from_mean(
        self, synthetic_data, fitted_monitor, ref_shap
    ):
        """Weighted mean should differ from unweighted mean when features
        have different importances."""
        result = compute_a5_importance_weighted(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            bucket_sets=fitted_monitor.bucket_sets_,
            shap_values=ref_shap,
            feature_names=synthetic_data["feature_names"],
        )
        # With drift on feature_0 (most important), weighted mean should
        # be higher than unweighted mean because feature_0 has higher weight
        # (This is expected because feature_0 has weight=1.0 in synthetic data)
        assert result["swift_weighted"] != result["swift_mean"]

    def test_drift_on_important_feature_increases_weighted(
        self, synthetic_data, fitted_monitor, ref_shap
    ):
        """When drift is on the most important feature, weighted score
        should be >= unweighted mean."""
        result_drift = compute_a5_importance_weighted(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            bucket_sets=fitted_monitor.bucket_sets_,
            shap_values=ref_shap,
            feature_names=synthetic_data["feature_names"],
        )
        # Drift on feature_0 (most important) -> weighted ≥ mean
        assert result_drift["swift_weighted"] >= result_drift["swift_mean"]


# ---------------------------------------------------------------------------
# run_all_ablations
# ---------------------------------------------------------------------------


class TestRunAllAblations:
    """Test the convenience function that runs all ablation variants."""

    def test_returns_all_ablation_scores(
        self, synthetic_data, fitted_monitor, ref_shap
    ):
        """run_all_ablations should return results for A1-A5."""
        results = run_all_ablations(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            bucket_sets=fitted_monitor.bucket_sets_,
            shap_values=ref_shap,
            feature_names=synthetic_data["feature_names"],
        )
        expected_keys = {"A1", "A2", "A3", "A4", "A5"}
        assert expected_keys.issubset(set(results.keys()))

    def test_each_ablation_has_max_and_mean(
        self, synthetic_data, fitted_monitor, ref_shap
    ):
        """Each ablation result should have max and mean scores."""
        results = run_all_ablations(
            X_ref=synthetic_data["X_ref"],
            X_mon=synthetic_data["X_mon_drift"],
            bucket_sets=fitted_monitor.bucket_sets_,
            shap_values=ref_shap,
            feature_names=synthetic_data["feature_names"],
        )
        for ablation_name, ablation_result in results.items():
            assert "max" in ablation_result, f"{ablation_name} missing 'max'"
            assert "mean" in ablation_result, f"{ablation_name} missing 'mean'"
            assert ablation_result["max"] >= 0.0
            assert ablation_result["mean"] >= 0.0

    def test_ablation_variant_enum(self):
        """AblationVariant enum should have all 6 ablations."""
        assert hasattr(AblationVariant, "A1_NO_SHAP_NORMALIZATION")
        assert hasattr(AblationVariant, "A2_NO_MODEL_BUCKETS")
        assert hasattr(AblationVariant, "A3_PSI_ON_MODEL_BUCKETS")
        assert hasattr(AblationVariant, "A4_W2_INSTEAD_OF_W1")
        assert hasattr(AblationVariant, "A5_IMPORTANCE_WEIGHTED")
        assert hasattr(AblationVariant, "A6_KERNEL_SHAP")
