"""Tests for baseline drift detection methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.baselines import (
    compute_decker,
    compute_ks,
    compute_mmd,
    compute_psi,
    compute_raw_wasserstein,
    compute_ssi,
    run_all_baselines,
)


@pytest.fixture
def identical_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two identical DataFrames — no drift."""
    rng = np.random.default_rng(42)
    n = 1000
    X = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(10, 5, n),
        "c": rng.normal(100, 20, n),
    })
    return X, X.copy()


@pytest.fixture
def shifted_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two DataFrames with mean shift in feature 'a'."""
    rng = np.random.default_rng(42)
    n = 1000
    X_ref = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(10, 5, n),
        "c": rng.normal(100, 20, n),
    })
    rng2 = np.random.default_rng(123)
    X_mon = pd.DataFrame({
        "a": rng2.normal(2, 1, n),   # 2σ mean shift
        "b": rng2.normal(10, 5, n),  # No shift
        "c": rng2.normal(100, 20, n),  # No shift
    })
    return X_ref, X_mon


class TestPSI:
    """Tests for PSI computation."""

    def test_no_drift_low_psi(self, identical_data):
        X_ref, X_mon = identical_data
        psi = compute_psi(X_ref, X_mon)
        for fname, val in psi.items():
            assert val < 0.05, f"PSI for {fname} should be near 0 for identical data"

    def test_drift_high_psi(self, shifted_data):
        X_ref, X_mon = shifted_data
        psi = compute_psi(X_ref, X_mon)
        # Feature 'a' has 2σ shift → PSI should be large
        assert psi["a"] > 0.1
        # Features 'b', 'c' should have low PSI
        assert psi["b"] < 0.1
        assert psi["c"] < 0.1

    def test_psi_non_negative(self, shifted_data):
        X_ref, X_mon = shifted_data
        psi = compute_psi(X_ref, X_mon)
        for val in psi.values():
            assert val >= 0

    def test_psi_ordering(self, shifted_data):
        """PSI for drifted feature should be larger than for stable features."""
        X_ref, X_mon = shifted_data
        psi = compute_psi(X_ref, X_mon)
        # Feature 'a' has the most drift → highest PSI
        assert psi["a"] > psi["b"]
        assert psi["a"] > psi["c"]


class TestSSI:
    """Tests for SSI (importance-weighted PSI)."""

    def test_ssi_scales_by_importance(self, shifted_data):
        X_ref, X_mon = shifted_data
        n = len(X_ref)
        # Feature 'a' is "important", 'b' and 'c' are not
        shap_values = np.zeros((n, 3))
        shap_values[:, 0] = np.random.default_rng(42).normal(0, 1.0, n)  # a: high SHAP
        shap_values[:, 1] = np.random.default_rng(42).normal(0, 0.01, n)  # b: low SHAP
        shap_values[:, 2] = np.random.default_rng(42).normal(0, 0.01, n)  # c: low SHAP

        ssi = compute_ssi(X_ref, X_mon, shap_values, feature_names=["a", "b", "c"])
        psi = compute_psi(X_ref, X_mon, feature_names=["a", "b", "c"])

        # SSI for 'a' should be large (important + drifted)
        assert ssi["a"] > 0
        # SSI for 'b', 'c' should be near 0 (unimportant)
        assert ssi["b"] < ssi["a"]


class TestKS:
    """Tests for KS test."""

    def test_no_drift_low_statistic(self, identical_data):
        X_ref, X_mon = identical_data
        ks = compute_ks(X_ref, X_mon)
        for val in ks.values():
            assert val < 0.1

    def test_drift_high_statistic(self, shifted_data):
        X_ref, X_mon = shifted_data
        ks = compute_ks(X_ref, X_mon)
        assert ks["a"] > 0.3  # 2σ shift should give large KS statistic

    def test_returns_pvalues(self, shifted_data):
        X_ref, X_mon = shifted_data
        ks_pvalues = compute_ks(X_ref, X_mon, return_pvalues=True)
        assert ks_pvalues["a"] < 0.01  # Significant drift in 'a'


class TestRawWasserstein:
    """Tests for raw Wasserstein distance."""

    def test_no_drift_small_distance(self, identical_data):
        X_ref, X_mon = identical_data
        w = compute_raw_wasserstein(X_ref, X_mon)
        for val in w.values():
            assert val < 0.1

    def test_drift_large_distance(self, shifted_data):
        X_ref, X_mon = shifted_data
        w = compute_raw_wasserstein(X_ref, X_mon)
        # Feature 'a' shifted by 2 units → W₁ ≈ 2
        assert w["a"] > 1.0
        assert w["b"] < 1.0

    def test_w2_also_works(self, shifted_data):
        X_ref, X_mon = shifted_data
        w2 = compute_raw_wasserstein(X_ref, X_mon, order=2)
        assert w2["a"] > 1.0


class TestMMD:
    """Tests for MMD computation."""

    def test_no_drift_near_zero(self, identical_data):
        X_ref, X_mon = identical_data
        mmd = compute_mmd(X_ref, X_mon)
        for val in mmd.values():
            assert val < 0.1

    def test_drift_positive(self, shifted_data):
        X_ref, X_mon = shifted_data
        mmd = compute_mmd(X_ref, X_mon)
        assert mmd["a"] > 0

    def test_multivariate_mmd(self, shifted_data):
        X_ref, X_mon = shifted_data
        mmd = compute_mmd(X_ref, X_mon, per_feature=False)
        assert "_multivariate" in mmd
        assert mmd["_multivariate"] > 0


class TestRunAllBaselines:
    """Tests for the convenience function."""

    def test_returns_all_methods(self, shifted_data):
        X_ref, X_mon = shifted_data
        features = ["a", "b", "c"]
        shap_values = np.random.default_rng(42).normal(0, 1, (len(X_ref), 3))

        results = run_all_baselines(
            X_ref, X_mon, features, shap_values=shap_values,
        )

        assert "PSI" in results
        assert "KS" in results
        assert "Raw_W1" in results
        assert "MMD" in results
        assert "SSI" in results

        # Each method should have scores for all features
        for method, scores in results.items():
            assert set(scores.keys()) == set(features)

    def test_decker_included_when_shap_mon_provided(self, shifted_data):
        """Decker baseline should appear when both shap_values and shap_values_mon are given."""
        X_ref, X_mon = shifted_data
        features = ["a", "b", "c"]
        rng = np.random.default_rng(42)
        shap_ref = rng.normal(0, 1, (len(X_ref), 3))
        shap_mon = rng.normal(0, 1, (len(X_mon), 3))

        results = run_all_baselines(
            X_ref, X_mon, features,
            shap_values=shap_ref,
            shap_values_mon=shap_mon,
        )

        assert "Decker_KS" in results
        assert set(results["Decker_KS"].keys()) == set(features)

    def test_decker_excluded_when_shap_mon_missing(self, shifted_data):
        """Decker baseline should NOT appear when shap_values_mon is not provided."""
        X_ref, X_mon = shifted_data
        features = ["a", "b", "c"]
        shap_ref = np.random.default_rng(42).normal(0, 1, (len(X_ref), 3))

        results = run_all_baselines(
            X_ref, X_mon, features,
            shap_values=shap_ref,
        )

        assert "Decker_KS" not in results


class TestDecker:
    """Tests for Decker et al. (KS on SHAP value distributions)."""

    def test_identical_shap_low_score(self):
        """Identical SHAP distributions should yield small KS statistics."""
        rng = np.random.default_rng(42)
        shap_vals = rng.normal(0, 1, (500, 3))
        features = ["a", "b", "c"]

        scores = compute_decker(shap_vals, shap_vals.copy(), features)

        for fname, val in scores.items():
            assert val < 0.1, f"Decker KS for {fname} should be near 0 for identical data"

    def test_shifted_shap_high_score(self):
        """Shifted SHAP distribution on one feature should yield high KS statistic."""
        rng = np.random.default_rng(42)
        n = 1000
        shap_ref = rng.normal(0, 1, (n, 3))

        rng2 = np.random.default_rng(123)
        shap_mon = rng2.normal(0, 1, (n, 3))
        # Shift feature 'a' SHAP values by 2σ
        shap_mon[:, 0] += 2.0

        features = ["a", "b", "c"]
        scores = compute_decker(shap_ref, shap_mon, features)

        # Feature 'a' should have high KS
        assert scores["a"] > 0.3
        # Features 'b', 'c' should have low KS
        assert scores["b"] < 0.15
        assert scores["c"] < 0.15

    def test_returns_pvalues(self):
        """return_pvalues=True should return p-values instead of statistics."""
        rng = np.random.default_rng(42)
        n = 1000
        shap_ref = rng.normal(0, 1, (n, 3))

        rng2 = np.random.default_rng(123)
        shap_mon = rng2.normal(0, 1, (n, 3))
        shap_mon[:, 0] += 2.0  # Shift feature 'a'

        features = ["a", "b", "c"]
        pvals = compute_decker(shap_ref, shap_mon, features, return_pvalues=True)

        # Feature 'a' should have very small p-value
        assert pvals["a"] < 0.01
        # Features 'b', 'c' should have larger p-values
        assert pvals["b"] > 0.01
        assert pvals["c"] > 0.01

    def test_non_negative_scores(self):
        """All KS statistics should be non-negative."""
        rng = np.random.default_rng(42)
        shap_ref = rng.normal(0, 1, (200, 4))
        shap_mon = rng.normal(0.5, 1, (200, 4))
        features = ["f1", "f2", "f3", "f4"]

        scores = compute_decker(shap_ref, shap_mon, features)

        for val in scores.values():
            assert val >= 0

    def test_mismatched_columns_raises(self):
        """Mismatched column count should raise ValueError."""
        shap_ref = np.random.default_rng(42).normal(0, 1, (100, 3))
        shap_mon = np.random.default_rng(42).normal(0, 1, (100, 5))

        with pytest.raises(ValueError, match="expected 3"):
            compute_decker(shap_ref, shap_mon, ["a", "b", "c"])

    def test_mismatched_ref_columns_raises(self):
        """Mismatched ref column count should raise ValueError."""
        shap_ref = np.random.default_rng(42).normal(0, 1, (100, 5))
        shap_mon = np.random.default_rng(42).normal(0, 1, (100, 3))

        with pytest.raises(ValueError, match="expected 3"):
            compute_decker(shap_ref, shap_mon, ["a", "b", "c"])

    def test_ordering_matches_drifted_feature(self):
        """The drifted feature should have the highest KS statistic."""
        rng = np.random.default_rng(42)
        n = 1000
        shap_ref = rng.normal(0, 1, (n, 3))

        rng2 = np.random.default_rng(99)
        shap_mon = rng2.normal(0, 1, (n, 3))
        # Only feature 'b' has a large shift
        shap_mon[:, 1] += 3.0

        features = ["a", "b", "c"]
        scores = compute_decker(shap_ref, shap_mon, features)

        assert scores["b"] > scores["a"]
        assert scores["b"] > scores["c"]

    def test_empty_shap_returns_zero(self):
        """Empty SHAP arrays with NaNs should return 0."""
        shap_ref = np.full((100, 2), np.nan)
        shap_mon = np.random.default_rng(42).normal(0, 1, (100, 2))

        scores = compute_decker(shap_ref, shap_mon, ["a", "b"])

        for val in scores.values():
            assert val == 0.0
