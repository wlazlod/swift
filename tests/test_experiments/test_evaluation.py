"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.evaluation import (
    compute_auroc,
    compute_detection_metrics,
    compute_fpr_from_pvalues,
    compute_model_performance,
    compute_spearman_correlation,
    compute_temporal_drift_analysis,
    compute_tpr_at_fpr,
)


class TestTPRatFPR:
    """Tests for TPR at fixed FPR."""

    def test_perfect_separation(self):
        """When drift scores are always higher than null, TPR should be 1.0."""
        scores_drift = np.array([10, 11, 12, 13, 14])
        scores_null = np.array([0, 1, 2, 3, 4])
        tpr = compute_tpr_at_fpr(scores_drift, scores_null, target_fpr=0.05)
        assert tpr == 1.0

    def test_no_separation(self):
        """When distributions overlap completely, TPR should be low."""
        rng = np.random.default_rng(42)
        scores_drift = rng.normal(0, 1, 1000)
        scores_null = rng.normal(0, 1, 1000)
        tpr = compute_tpr_at_fpr(scores_drift, scores_null, target_fpr=0.05)
        # Should be near the FPR level
        assert tpr < 0.15

    def test_empty_arrays(self):
        tpr = compute_tpr_at_fpr(np.array([]), np.array([1, 2, 3]), target_fpr=0.05)
        assert tpr == 0.0


class TestAUROC:
    """Tests for AUROC computation."""

    def test_perfect_auroc(self):
        scores_drift = np.array([10, 11, 12])
        scores_null = np.array([0, 1, 2])
        auroc = compute_auroc(scores_drift, scores_null)
        assert auroc == 1.0

    def test_random_auroc(self):
        rng = np.random.default_rng(42)
        scores_drift = rng.normal(0, 1, 1000)
        scores_null = rng.normal(0, 1, 1000)
        auroc = compute_auroc(scores_drift, scores_null)
        assert 0.45 < auroc < 0.55

    def test_partial_separation(self):
        rng = np.random.default_rng(42)
        scores_drift = rng.normal(2, 1, 500)
        scores_null = rng.normal(0, 1, 500)
        auroc = compute_auroc(scores_drift, scores_null)
        assert 0.8 < auroc < 1.0


class TestDetectionMetrics:
    """Tests for comprehensive detection metrics."""

    def test_returns_all_fields(self):
        scores_drift = np.array([5, 6, 7, 8, 9])
        scores_null = np.array([0, 1, 2, 3, 4])
        metrics = compute_detection_metrics(scores_drift, scores_null)

        assert metrics.tpr_at_5fpr == 1.0
        assert metrics.tpr_at_1fpr == 1.0
        assert metrics.auroc == 1.0
        assert metrics.n_drift_trials == 5
        assert metrics.n_null_trials == 5


class TestSpearmanCorrelation:
    """Tests for Spearman rank correlation."""

    def test_perfect_positive_correlation(self):
        drift_scores = np.array([1, 2, 3, 4, 5])
        degradation = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        rho, pval = compute_spearman_correlation(drift_scores, degradation)
        assert rho > 0.99

    def test_perfect_negative_correlation(self):
        drift_scores = np.array([1, 2, 3, 4, 5])
        degradation = np.array([0.05, 0.04, 0.03, 0.02, 0.01])
        rho, pval = compute_spearman_correlation(drift_scores, degradation)
        assert rho < -0.99

    def test_too_few_observations(self):
        rho, pval = compute_spearman_correlation(np.array([1, 2]), np.array([3, 4]))
        assert rho == 0.0
        assert pval == 1.0


class TestFPRFromPValues:
    """Tests for FPR from p-values."""

    def test_uniform_pvalues(self):
        """Under H0, p-values are uniform → FPR ≈ α."""
        rng = np.random.default_rng(42)
        pvalues = rng.uniform(0, 1, 10000)
        fpr = compute_fpr_from_pvalues(pvalues, alpha=0.05)
        assert 0.04 < fpr < 0.06

    def test_all_significant(self):
        pvalues = np.array([0.001, 0.002, 0.003])
        fpr = compute_fpr_from_pvalues(pvalues, alpha=0.05)
        assert fpr == 1.0

    def test_none_significant(self):
        pvalues = np.array([0.5, 0.6, 0.7])
        fpr = compute_fpr_from_pvalues(pvalues, alpha=0.05)
        assert fpr == 0.0


class TestModelPerformance:
    """Tests for model AUC computation."""

    def test_perfect_model(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        auc = compute_model_performance(y_true, y_prob)
        assert auc == 1.0

    def test_random_model(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 1000)
        y_prob = rng.uniform(0, 1, 1000)
        auc = compute_model_performance(y_true, y_prob)
        assert 0.45 < auc < 0.55


class TestTemporalDriftAnalysis:
    """Tests for temporal drift analysis."""

    def test_computes_correlations(self):
        period_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        model_aucs = np.array([0.80, 0.78, 0.75, 0.70, 0.65])
        ref_auc = 0.82

        drift_scores = {
            "SWIFT": np.array([0.01, 0.03, 0.05, 0.10, 0.15]),
            "PSI": np.array([0.02, 0.08, 0.04, 0.12, 0.06]),
        }

        result = compute_temporal_drift_analysis(
            period_labels, drift_scores, model_aucs, ref_auc,
        )

        assert "SWIFT" in result.spearman_rho
        assert "PSI" in result.spearman_rho
        # SWIFT scores are monotonically increasing with degradation → high ρ
        assert result.spearman_rho["SWIFT"] > 0.9
        assert len(result.auc_degradation) == 5
        assert result.auc_degradation[0] == pytest.approx(0.02)  # 0.82 - 0.80
