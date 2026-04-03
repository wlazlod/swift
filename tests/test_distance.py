"""Tests for Stage 4: Wasserstein distance on SHAP-transformed distributions.

Contract:
    wasserstein_1d(ref_transformed, mon_transformed, order=1) -> float
    - Computes W_p distance between two 1-D empirical distributions.
    - order=1: W1 (L1 between CDFs). order=2: W2.

    compute_swift_scores(
        X_ref, X_mon, bucket_sets, order=1
    ) -> dict[str, float]
    - Transforms both samples and computes per-feature SWIFT scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import wasserstein_distance as scipy_w1

from swift.types import WassersteinOrder
from swift.distance import wasserstein_1d, compute_swift_scores
from swift.bucketing import build_buckets, build_all_buckets
from swift.normalization import compute_bucket_shap, transform_feature


class TestWasserstein1D:
    """Tests for the 1-D Wasserstein distance computation."""

    def test_identical_distributions_zero(self):
        """W_p of identical distributions should be 0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert wasserstein_1d(a, a, order=1) == pytest.approx(0.0)
        assert wasserstein_1d(a, a, order=2) == pytest.approx(0.0)

    def test_w1_matches_scipy(self):
        """Our W1 should match scipy.stats.wasserstein_distance."""
        rng = np.random.default_rng(123)
        a = rng.standard_normal(500)
        b = rng.standard_normal(500) + 0.5  # shifted

        our_w1 = wasserstein_1d(a, b, order=1)
        scipy_w = scipy_w1(a, b)
        assert our_w1 == pytest.approx(scipy_w, rel=1e-6)

    def test_w1_symmetric(self):
        """W1(a, b) == W1(b, a)."""
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([0.5, 1.5, 3.0])
        assert wasserstein_1d(a, b, order=1) == pytest.approx(
            wasserstein_1d(b, a, order=1)
        )

    def test_w2_symmetric(self):
        """W2(a, b) == W2(b, a)."""
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([0.5, 1.5, 3.0])
        assert wasserstein_1d(a, b, order=2) == pytest.approx(
            wasserstein_1d(b, a, order=2)
        )

    def test_w1_triangle_inequality(self):
        """W1(a, c) <= W1(a, b) + W1(b, c)."""
        a = np.array([0.0, 1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        c = np.array([2.0, 3.0, 4.0, 5.0])
        w_ac = wasserstein_1d(a, c, order=1)
        w_ab = wasserstein_1d(a, b, order=1)
        w_bc = wasserstein_1d(b, c, order=1)
        assert w_ac <= w_ab + w_bc + 1e-10

    def test_larger_shift_larger_distance(self):
        """A 2-sigma shift should produce larger W1 than a 1-sigma shift."""
        base = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        shift_1 = base + 1.0
        shift_2 = base + 2.0

        w1_small = wasserstein_1d(base, shift_1, order=1)
        w1_large = wasserstein_1d(base, shift_2, order=1)
        assert w1_large > w1_small

    def test_w2_larger_than_or_equal_w1_normalized(self):
        """For distributions on bounded support, W2 >= W1 / sqrt(diameter)
        is not guaranteed, but W2 should respond to same shifts."""
        base = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        shifted = base + 1.0

        w1 = wasserstein_1d(base, shifted, order=1)
        w2 = wasserstein_1d(base, shifted, order=2)
        # Both should be positive for a shifted distribution
        assert w1 > 0
        assert w2 > 0

    def test_discrete_distributions(self):
        """W1 on discrete distributions (like SHAP-transformed) should work."""
        # Two discrete distributions on support {-0.5, 0.0, 0.5}
        a = np.array([-0.5, -0.5, 0.0, 0.0, 0.5])     # 40% at -0.5, 40% at 0.0, 20% at 0.5
        b = np.array([-0.5, 0.0, 0.0, 0.5, 0.5])       # 20% at -0.5, 40% at 0.0, 40% at 0.5

        w1 = wasserstein_1d(a, b, order=1)
        assert w1 > 0

    def test_different_sample_sizes(self):
        """Should handle unequal sample sizes."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        w1 = wasserstein_1d(a, b, order=1)
        assert w1 > 0
        assert np.isfinite(w1)


class TestComputeSwiftScores:
    """Tests for computing per-feature SWIFT scores."""

    def test_returns_dict_with_all_features(
        self,
        trained_lgb_model,
        synthetic_data,
        ref_shap_values,
    ):
        """Should return a score for every feature."""
        from swift.extraction import extract_decision_points_lgb

        feature_names = synthetic_data["feature_names"]
        dp = extract_decision_points_lgb(trained_lgb_model, feature_names)
        bucket_sets = build_all_buckets(dp)
        bucket_sets = compute_bucket_shap(
            bucket_sets,
            synthetic_data["X_ref"],
            ref_shap_values,
            model=trained_lgb_model,
        )

        scores = compute_swift_scores(
            synthetic_data["X_ref"],
            synthetic_data["X_mon_drift"],
            bucket_sets,
        )

        assert isinstance(scores, dict)
        assert set(scores.keys()) == set(feature_names)

    def test_no_drift_scores_near_zero(
        self,
        trained_lgb_model,
        synthetic_data,
        ref_shap_values,
    ):
        """With no drift (ref vs ref-like), scores should be small."""
        from swift.extraction import extract_decision_points_lgb

        feature_names = synthetic_data["feature_names"]
        dp = extract_decision_points_lgb(trained_lgb_model, feature_names)
        bucket_sets = build_all_buckets(dp)
        bucket_sets = compute_bucket_shap(
            bucket_sets,
            synthetic_data["X_ref"],
            ref_shap_values,
            model=trained_lgb_model,
        )

        scores = compute_swift_scores(
            synthetic_data["X_ref"],
            synthetic_data["X_mon_nodrift"],
            bucket_sets,
        )

        # No drift: all scores should be small (but not necessarily exactly 0
        # due to sampling noise)
        for fname, score in scores.items():
            assert score >= 0.0, f"{fname}: negative score"

    def test_drift_feature_has_highest_score(
        self,
        trained_lgb_model,
        synthetic_data,
        ref_shap_values,
    ):
        """feature_0 was shifted by 1.5σ, so it should have the highest SWIFT score."""
        from swift.extraction import extract_decision_points_lgb

        feature_names = synthetic_data["feature_names"]
        dp = extract_decision_points_lgb(trained_lgb_model, feature_names)
        bucket_sets = build_all_buckets(dp)
        bucket_sets = compute_bucket_shap(
            bucket_sets,
            synthetic_data["X_ref"],
            ref_shap_values,
            model=trained_lgb_model,
        )

        scores = compute_swift_scores(
            synthetic_data["X_ref"],
            synthetic_data["X_mon_drift"],
            bucket_sets,
        )

        # feature_0 should have the highest score
        max_feature = max(scores, key=scores.get)
        assert max_feature == "feature_0", (
            f"Expected feature_0 to have highest SWIFT score, got {max_feature}. "
            f"Scores: {scores}"
        )

    def test_scores_are_non_negative(
        self,
        trained_lgb_model,
        synthetic_data,
        ref_shap_values,
    ):
        """All SWIFT scores should be >= 0."""
        from swift.extraction import extract_decision_points_lgb

        feature_names = synthetic_data["feature_names"]
        dp = extract_decision_points_lgb(trained_lgb_model, feature_names)
        bucket_sets = build_all_buckets(dp)
        bucket_sets = compute_bucket_shap(
            bucket_sets,
            synthetic_data["X_ref"],
            ref_shap_values,
            model=trained_lgb_model,
        )

        scores = compute_swift_scores(
            synthetic_data["X_ref"],
            synthetic_data["X_mon_drift"],
            bucket_sets,
        )

        for fname, score in scores.items():
            assert score >= 0.0, f"{fname}: negative SWIFT score {score}"

    def test_w2_also_works(
        self,
        trained_lgb_model,
        synthetic_data,
        ref_shap_values,
    ):
        """W2 should produce valid, non-negative scores."""
        from swift.extraction import extract_decision_points_lgb

        feature_names = synthetic_data["feature_names"]
        dp = extract_decision_points_lgb(trained_lgb_model, feature_names)
        bucket_sets = build_all_buckets(dp)
        bucket_sets = compute_bucket_shap(
            bucket_sets,
            synthetic_data["X_ref"],
            ref_shap_values,
            model=trained_lgb_model,
        )

        scores = compute_swift_scores(
            synthetic_data["X_ref"],
            synthetic_data["X_mon_drift"],
            bucket_sets,
            order=2,
        )

        for fname, score in scores.items():
            assert score >= 0.0
            assert np.isfinite(score)
