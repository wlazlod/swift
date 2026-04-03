"""Tests for Stage 5: Threshold calibration — permutation test, bootstrap, MTC.

Contract:
    permutation_test(
        X_ref, X_mon, bucket_sets, order=1, n_permutations=1000, rng=None
    ) -> dict[str, float]
    - Returns per-feature p-values from permutation test.
    - Pools ref+mon, draws random splits, applies pre-computed σ_j,
      computes SWIFT score per permutation.
    - p_j = (1 + #{b : SWIFT_j^(b) >= SWIFT_j^obs}) / (1 + B)

    bootstrap_threshold(
        X_ref, bucket_sets, n_mon, order=1, alpha=0.05,
        n_bootstrap=1000, rng=None
    ) -> dict[str, float]
    - Returns per-feature thresholds at (1-alpha) quantile.
    - Draws bootstrap samples of size n_mon from X_ref,
      computes SWIFT score for each.

    correct_pvalues(
        pvalues: dict[str, float], method: CorrectionMethod, alpha: float
    ) -> dict[str, bool]
    - Returns per-feature rejection decisions after MTC.
    - Bonferroni: reject if p_j < alpha / p
    - BH: sort p-values, find largest k s.t. p_(k) <= k*alpha/p
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swift.types import CorrectionMethod, WassersteinOrder
from swift.threshold import (
    permutation_test,
    bootstrap_threshold,
    correct_pvalues,
)
from swift.bucketing import build_all_buckets
from swift.normalization import compute_bucket_shap
from swift.distance import compute_swift_scores


# ===========================================================================
# Permutation test
# ===========================================================================
class TestPermutationTest:
    """Tests for permutation_test()."""

    def test_returns_pvalue_per_feature(
        self,
        trained_lgb_model,
        synthetic_data,
        ref_shap_values,
    ):
        """Should return a p-value for every feature in bucket_sets."""
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

        pvalues = permutation_test(
            synthetic_data["X_ref"],
            synthetic_data["X_mon_drift"],
            bucket_sets,
            n_permutations=100,
            rng=np.random.default_rng(42),
        )

        assert isinstance(pvalues, dict)
        assert set(pvalues.keys()) == set(feature_names)

    def test_pvalues_are_valid(
        self,
        trained_lgb_model,
        synthetic_data,
        ref_shap_values,
    ):
        """All p-values should be in (0, 1]."""
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

        pvalues = permutation_test(
            synthetic_data["X_ref"],
            synthetic_data["X_mon_drift"],
            bucket_sets,
            n_permutations=100,
            rng=np.random.default_rng(42),
        )

        for fname, pval in pvalues.items():
            assert 0.0 < pval <= 1.0, f"{fname}: p-value {pval} out of range"

    def test_drifted_feature_has_small_pvalue(
        self,
        trained_lgb_model,
        synthetic_data,
        ref_shap_values,
    ):
        """feature_0 was shifted by 1.5σ — its p-value should be small."""
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

        pvalues = permutation_test(
            synthetic_data["X_ref"],
            synthetic_data["X_mon_drift"],
            bucket_sets,
            n_permutations=200,
            rng=np.random.default_rng(42),
        )

        # With a 1.5σ shift and 200 permutations, p should be very small
        assert pvalues["feature_0"] < 0.05, (
            f"Expected feature_0 p-value < 0.05, got {pvalues['feature_0']}"
        )

    def test_no_drift_pvalues_not_significant(
        self,
        trained_lgb_model,
        synthetic_data,
        ref_shap_values,
    ):
        """With no drift, most p-values should be large (not significant)."""
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

        pvalues = permutation_test(
            synthetic_data["X_ref"],
            synthetic_data["X_mon_nodrift"],
            bucket_sets,
            n_permutations=200,
            rng=np.random.default_rng(42),
        )

        # At least 3 out of 5 features should have p > 0.05
        n_not_sig = sum(1 for p in pvalues.values() if p > 0.05)
        assert n_not_sig >= 3, (
            f"Expected at least 3 non-significant p-values, got {n_not_sig}. "
            f"P-values: {pvalues}"
        )

    def test_pvalue_formula_correct(self):
        """Verify the (1 + sum) / (1 + B) formula on a tiny example."""
        # Build a minimal synthetic example
        rng = np.random.default_rng(999)
        n = 100
        X_ref = pd.DataFrame({"x": rng.standard_normal(n)})
        X_mon = pd.DataFrame({"x": rng.standard_normal(n) + 5.0})  # massive drift

        from swift.bucketing import build_buckets

        dp = {"x": np.array([0.0])}
        bucket_sets = build_all_buckets(dp)

        # Manually set mean_shap so we don't need a model
        from dataclasses import replace

        bs = bucket_sets["x"]
        new_buckets = []
        for b in bs.buckets:
            new_buckets.append(replace(b, mean_shap=float(b.index) * 0.5))
        bs.buckets = new_buckets

        pvalues = permutation_test(
            X_ref, X_mon, bucket_sets,
            n_permutations=99,
            rng=np.random.default_rng(42),
        )

        # With a +5.0 shift, observed score should be larger than all
        # permuted scores → p = 1 / (1 + 99) = 0.01
        assert pvalues["x"] <= 0.05, (
            f"Expected p <= 0.05 for massive drift, got {pvalues['x']}"
        )

    def test_reproducible_with_same_seed(
        self,
        trained_lgb_model,
        synthetic_data,
        ref_shap_values,
    ):
        """Same rng seed should produce identical p-values."""
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

        pv1 = permutation_test(
            synthetic_data["X_ref"],
            synthetic_data["X_mon_drift"],
            bucket_sets,
            n_permutations=50,
            rng=np.random.default_rng(123),
        )
        pv2 = permutation_test(
            synthetic_data["X_ref"],
            synthetic_data["X_mon_drift"],
            bucket_sets,
            n_permutations=50,
            rng=np.random.default_rng(123),
        )

        for fname in feature_names:
            assert pv1[fname] == pytest.approx(pv2[fname])

    def test_w2_order_works(
        self,
        trained_lgb_model,
        synthetic_data,
        ref_shap_values,
    ):
        """Permutation test should also work with W2."""
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

        pvalues = permutation_test(
            synthetic_data["X_ref"],
            synthetic_data["X_mon_drift"],
            bucket_sets,
            order=2,
            n_permutations=50,
            rng=np.random.default_rng(42),
        )

        for fname, pval in pvalues.items():
            assert 0.0 < pval <= 1.0


# ===========================================================================
# Bootstrap threshold
# ===========================================================================
class TestBootstrapThreshold:
    """Tests for bootstrap_threshold()."""

    def test_returns_threshold_per_feature(
        self,
        trained_lgb_model,
        synthetic_data,
        ref_shap_values,
    ):
        """Should return a threshold for every feature."""
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

        thresholds = bootstrap_threshold(
            synthetic_data["X_ref"],
            bucket_sets,
            n_mon=len(synthetic_data["X_mon_drift"]),
            n_bootstrap=100,
            rng=np.random.default_rng(42),
        )

        assert isinstance(thresholds, dict)
        assert set(thresholds.keys()) == set(feature_names)

    def test_thresholds_are_non_negative(
        self,
        trained_lgb_model,
        synthetic_data,
        ref_shap_values,
    ):
        """Thresholds should be >= 0 (Wasserstein distances are non-negative)."""
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

        thresholds = bootstrap_threshold(
            synthetic_data["X_ref"],
            bucket_sets,
            n_mon=len(synthetic_data["X_mon_drift"]),
            n_bootstrap=100,
            rng=np.random.default_rng(42),
        )

        for fname, thresh in thresholds.items():
            assert thresh >= 0.0, f"{fname}: negative threshold {thresh}"
            assert np.isfinite(thresh), f"{fname}: non-finite threshold {thresh}"

    def test_drift_exceeds_threshold(
        self,
        trained_lgb_model,
        synthetic_data,
        ref_shap_values,
    ):
        """The drifted feature_0 should exceed its bootstrap threshold."""
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

        thresholds = bootstrap_threshold(
            synthetic_data["X_ref"],
            bucket_sets,
            n_mon=len(synthetic_data["X_mon_drift"]),
            alpha=0.05,
            n_bootstrap=200,
            rng=np.random.default_rng(42),
        )

        scores = compute_swift_scores(
            synthetic_data["X_ref"],
            synthetic_data["X_mon_drift"],
            bucket_sets,
        )

        # Drifted feature should exceed threshold
        assert scores["feature_0"] > thresholds["feature_0"], (
            f"Expected feature_0 score ({scores['feature_0']:.4f}) > "
            f"threshold ({thresholds['feature_0']:.4f})"
        )

    def test_stricter_alpha_gives_higher_threshold(
        self,
        trained_lgb_model,
        synthetic_data,
        ref_shap_values,
    ):
        """Lower alpha should produce higher thresholds."""
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

        thresh_05 = bootstrap_threshold(
            synthetic_data["X_ref"],
            bucket_sets,
            n_mon=len(synthetic_data["X_mon_drift"]),
            alpha=0.05,
            n_bootstrap=200,
            rng=np.random.default_rng(42),
        )
        thresh_01 = bootstrap_threshold(
            synthetic_data["X_ref"],
            bucket_sets,
            n_mon=len(synthetic_data["X_mon_drift"]),
            alpha=0.01,
            n_bootstrap=200,
            rng=np.random.default_rng(42),
        )

        # Stricter alpha → higher quantile → higher threshold
        for fname in feature_names:
            assert thresh_01[fname] >= thresh_05[fname], (
                f"{fname}: α=0.01 threshold ({thresh_01[fname]:.4f}) < "
                f"α=0.05 threshold ({thresh_05[fname]:.4f})"
            )


# ===========================================================================
# Multiple Testing Correction
# ===========================================================================
class TestCorrectPvalues:
    """Tests for correct_pvalues()."""

    def test_bonferroni_basic(self):
        """Bonferroni: reject if p_j < alpha / p."""
        pvalues = {"a": 0.001, "b": 0.03, "c": 0.5}
        alpha = 0.05

        decisions = correct_pvalues(pvalues, CorrectionMethod.BONFERRONI, alpha)

        # alpha / 3 = 0.0167
        assert decisions["a"] is True   # 0.001 < 0.0167
        assert decisions["b"] is False  # 0.03 > 0.0167
        assert decisions["c"] is False  # 0.5 > 0.0167

    def test_bonferroni_all_significant(self):
        """If all p-values are tiny, all should be rejected."""
        pvalues = {"x": 0.001, "y": 0.002, "z": 0.003}
        decisions = correct_pvalues(pvalues, CorrectionMethod.BONFERRONI, alpha=0.05)

        # alpha/3 ≈ 0.0167, all p < 0.0167
        assert all(decisions.values())

    def test_bonferroni_none_significant(self):
        """If all p-values are large, none should be rejected."""
        pvalues = {"x": 0.5, "y": 0.6, "z": 0.8}
        decisions = correct_pvalues(pvalues, CorrectionMethod.BONFERRONI, alpha=0.05)

        assert not any(decisions.values())

    def test_bh_basic(self):
        """BH: sort p-values, find largest k s.t. p_(k) <= k*alpha/p."""
        pvalues = {"a": 0.005, "b": 0.03, "c": 0.5}
        alpha = 0.05

        decisions = correct_pvalues(pvalues, CorrectionMethod.BH, alpha)

        # Sorted: a=0.005 (rank 1), b=0.03 (rank 2), c=0.5 (rank 3)
        # BH thresholds: 1*0.05/3=0.0167, 2*0.05/3=0.0333, 3*0.05/3=0.05
        # a: 0.005 <= 0.0167 ✓
        # b: 0.03 <= 0.0333 ✓
        # c: 0.5 > 0.05 ✗
        # Largest k=2, so reject features with rank <= 2
        assert decisions["a"] is True
        assert decisions["b"] is True
        assert decisions["c"] is False

    def test_bh_more_powerful_than_bonferroni(self):
        """BH should reject at least as many as Bonferroni."""
        pvalues = {"a": 0.005, "b": 0.02, "c": 0.04, "d": 0.5, "e": 0.8}
        alpha = 0.05

        bonf = correct_pvalues(pvalues, CorrectionMethod.BONFERRONI, alpha)
        bh = correct_pvalues(pvalues, CorrectionMethod.BH, alpha)

        n_bonf = sum(bonf.values())
        n_bh = sum(bh.values())
        assert n_bh >= n_bonf, (
            f"BH rejected {n_bh} but Bonferroni rejected {n_bonf}"
        )

    def test_bh_all_significant(self):
        """If all p-values are tiny, all should be rejected under BH."""
        pvalues = {"x": 0.001, "y": 0.002, "z": 0.003}
        decisions = correct_pvalues(pvalues, CorrectionMethod.BH, alpha=0.05)
        assert all(decisions.values())

    def test_bh_none_significant(self):
        """If all p-values are large, none should be rejected under BH."""
        pvalues = {"x": 0.5, "y": 0.6, "z": 0.8}
        decisions = correct_pvalues(pvalues, CorrectionMethod.BH, alpha=0.05)
        assert not any(decisions.values())

    def test_single_feature(self):
        """With 1 feature, Bonferroni and BH should agree."""
        pvalues = {"x": 0.03}
        alpha = 0.05

        bonf = correct_pvalues(pvalues, CorrectionMethod.BONFERRONI, alpha)
        bh = correct_pvalues(pvalues, CorrectionMethod.BH, alpha)

        # With 1 feature: Bonferroni threshold = 0.05/1 = 0.05
        # BH threshold for rank 1: 1*0.05/1 = 0.05
        # Both: 0.03 < 0.05 → reject
        assert bonf["x"] is True
        assert bh["x"] is True

    def test_returns_dict_with_same_keys(self):
        """Output should have same keys as input."""
        pvalues = {"a": 0.1, "b": 0.2, "c": 0.3}
        decisions = correct_pvalues(pvalues, CorrectionMethod.BONFERRONI, alpha=0.05)
        assert set(decisions.keys()) == set(pvalues.keys())

    def test_edge_case_pvalue_equals_threshold(self):
        """When p-value exactly equals the Bonferroni threshold, should NOT reject.

        Convention: reject if p < threshold (strict inequality).
        """
        # alpha / 2 = 0.025
        pvalues = {"x": 0.025, "y": 0.5}
        decisions = correct_pvalues(pvalues, CorrectionMethod.BONFERRONI, alpha=0.05)

        # 0.025 is NOT < 0.025 (strict), so should not reject
        assert decisions["x"] is False


# ===========================================================================
# Permutation test — max_samples subsampling
# ===========================================================================
class TestPermutationTestMaxSamples:
    """Tests for the max_samples parameter in permutation_test."""

    def test_max_samples_returns_valid_pvalues(self):
        """With max_samples set, p-values should still be valid."""
        rng = np.random.default_rng(42)
        n = 2000
        X_ref = pd.DataFrame({"x": rng.standard_normal(n)})
        X_mon = pd.DataFrame({"x": rng.standard_normal(n) + 3.0})

        from swift.bucketing import build_all_buckets

        dp = {"x": np.array([0.0, 1.0])}
        bucket_sets = build_all_buckets(dp)

        from dataclasses import replace

        bs = bucket_sets["x"]
        new_buckets = []
        for b in bs.buckets:
            new_buckets.append(replace(b, mean_shap=float(b.index) * 0.5))
        bs.buckets = new_buckets

        pvalues = permutation_test(
            X_ref,
            X_mon,
            bucket_sets,
            n_permutations=50,
            max_samples=500,
            rng=np.random.default_rng(42),
        )

        assert 0.0 < pvalues["x"] <= 1.0

    def test_max_samples_detects_drift(self):
        """Subsampled permutation test should still detect strong drift."""
        rng = np.random.default_rng(42)
        n = 3000
        X_ref = pd.DataFrame({"x": rng.standard_normal(n)})
        X_mon = pd.DataFrame({"x": rng.standard_normal(n) + 5.0})

        from swift.bucketing import build_all_buckets

        dp = {"x": np.array([0.0])}
        bucket_sets = build_all_buckets(dp)

        from dataclasses import replace

        bs = bucket_sets["x"]
        new_buckets = []
        for b in bs.buckets:
            new_buckets.append(replace(b, mean_shap=float(b.index) * 0.5))
        bs.buckets = new_buckets

        pvalues = permutation_test(
            X_ref,
            X_mon,
            bucket_sets,
            n_permutations=99,
            max_samples=500,
            rng=np.random.default_rng(42),
        )

        assert pvalues["x"] <= 0.05

    def test_max_samples_none_uses_full_data(
        self,
        trained_lgb_model,
        synthetic_data,
        ref_shap_values,
    ):
        """When max_samples is None, should produce identical results to default."""
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

        pv_default = permutation_test(
            synthetic_data["X_ref"],
            synthetic_data["X_mon_drift"],
            bucket_sets,
            n_permutations=50,
            rng=np.random.default_rng(42),
        )

        pv_none = permutation_test(
            synthetic_data["X_ref"],
            synthetic_data["X_mon_drift"],
            bucket_sets,
            n_permutations=50,
            max_samples=None,
            rng=np.random.default_rng(42),
        )

        for fname in feature_names:
            assert pv_default[fname] == pytest.approx(pv_none[fname])
