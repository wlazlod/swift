"""Tests for Stage 6: Aggregation — model-level SWIFT scores.

Contract:
    aggregate_scores(
        scores: dict[str, float],
        weights: dict[str, float] | None = None,
    ) -> AggregatedScores

    AggregatedScores is a frozen dataclass with:
        swift_max: float
        swift_mean: float
        swift_weighted: float | None
        max_feature: str

    compute_importance_weights(
        shap_values: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, float]
    - Returns normalized mean |SHAP| weights: w_j = |φ̄_j| / Σ|φ̄_k|
"""

from __future__ import annotations

import numpy as np
import pytest

from swift.aggregation import (
    aggregate_scores,
    compute_importance_weights,
    AggregatedScores,
)


class TestAggregateScores:
    """Tests for aggregate_scores()."""

    def test_max_score(self):
        """swift_max should be the largest per-feature score."""
        scores = {"a": 0.1, "b": 0.5, "c": 0.3}
        agg = aggregate_scores(scores)
        assert agg.swift_max == pytest.approx(0.5)

    def test_max_feature(self):
        """max_feature should be the name of the highest-scoring feature."""
        scores = {"a": 0.1, "b": 0.5, "c": 0.3}
        agg = aggregate_scores(scores)
        assert agg.max_feature == "b"

    def test_mean_score(self):
        """swift_mean should be the unweighted average."""
        scores = {"a": 0.1, "b": 0.5, "c": 0.3}
        agg = aggregate_scores(scores)
        expected = (0.1 + 0.5 + 0.3) / 3.0
        assert agg.swift_mean == pytest.approx(expected)

    def test_no_weights_means_no_weighted(self):
        """Without weights, swift_weighted should be None."""
        scores = {"a": 0.1, "b": 0.5}
        agg = aggregate_scores(scores)
        assert agg.swift_weighted is None

    def test_weighted_score(self):
        """swift_weighted = Σ w_j * score_j."""
        scores = {"a": 0.2, "b": 0.4, "c": 0.6}
        weights = {"a": 0.5, "b": 0.3, "c": 0.2}
        agg = aggregate_scores(scores, weights=weights)

        expected = 0.5 * 0.2 + 0.3 * 0.4 + 0.2 * 0.6
        assert agg.swift_weighted == pytest.approx(expected)

    def test_uniform_weights_equals_mean(self):
        """With uniform weights (1/p), weighted should equal mean."""
        scores = {"a": 0.1, "b": 0.5, "c": 0.3}
        weights = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        agg = aggregate_scores(scores, weights=weights)

        assert agg.swift_weighted == pytest.approx(agg.swift_mean, rel=1e-10)

    def test_single_feature(self):
        """With 1 feature, max == mean == score."""
        scores = {"x": 0.42}
        agg = aggregate_scores(scores)
        assert agg.swift_max == pytest.approx(0.42)
        assert agg.swift_mean == pytest.approx(0.42)
        assert agg.max_feature == "x"

    def test_frozen_dataclass(self):
        """AggregatedScores should be immutable."""
        scores = {"a": 0.1}
        agg = aggregate_scores(scores)
        with pytest.raises(AttributeError):
            agg.swift_max = 999.0

    def test_all_zeros(self):
        """All-zero scores should produce all-zero aggregations."""
        scores = {"a": 0.0, "b": 0.0, "c": 0.0}
        agg = aggregate_scores(scores)
        assert agg.swift_max == pytest.approx(0.0)
        assert agg.swift_mean == pytest.approx(0.0)


class TestComputeImportanceWeights:
    """Tests for compute_importance_weights()."""

    def test_weights_sum_to_one(self):
        """Normalized importance weights should sum to 1."""
        shap_values = np.array([
            [0.1, -0.3, 0.2],
            [0.2, -0.4, 0.1],
            [0.3, -0.2, 0.3],
        ])
        weights = compute_importance_weights(
            shap_values, ["a", "b", "c"]
        )
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_returns_dict_with_feature_names(self):
        """Should return a weight for each feature name."""
        shap_values = np.array([[0.1, 0.2], [0.3, 0.4]])
        weights = compute_importance_weights(shap_values, ["x", "y"])
        assert set(weights.keys()) == {"x", "y"}

    def test_higher_shap_gets_higher_weight(self):
        """Feature with larger mean |SHAP| should have higher weight."""
        shap_values = np.array([
            [0.1, 1.0],
            [0.2, 0.8],
            [0.1, 1.2],
        ])
        weights = compute_importance_weights(shap_values, ["low", "high"])
        assert weights["high"] > weights["low"]

    def test_all_equal_shap_gives_uniform(self):
        """If all features have same mean |SHAP|, weights should be uniform."""
        shap_values = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ])
        weights = compute_importance_weights(
            shap_values, ["a", "b", "c"]
        )
        for w in weights.values():
            assert w == pytest.approx(1 / 3)

    def test_sign_does_not_matter(self):
        """Negative SHAP values should contribute equally (absolute value)."""
        shap_values = np.array([
            [0.5, -0.5],
            [0.5, -0.5],
        ])
        weights = compute_importance_weights(shap_values, ["pos", "neg"])
        assert weights["pos"] == pytest.approx(weights["neg"])

    def test_integration_with_real_data(
        self,
        synthetic_data,
        ref_shap_values,
    ):
        """Should work with real SHAP values from conftest."""
        weights = compute_importance_weights(
            ref_shap_values,
            synthetic_data["feature_names"],
        )
        assert len(weights) == len(synthetic_data["feature_names"])
        assert sum(weights.values()) == pytest.approx(1.0)
        assert all(w >= 0 for w in weights.values())
