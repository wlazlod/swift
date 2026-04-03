"""Tests for Stage 1: Decision point extraction from LightGBM models.

Contract:
    extract_decision_points(model, feature_names) -> dict[str, np.ndarray]
    - Returns sorted, unique split thresholds per feature.
    - Features not used in any split get an empty array.
    - Works with LightGBM Booster objects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb

from swift.extraction import extract_decision_points_lgb


class TestExtractDecisionPointsLGB:
    """Tests for LightGBM decision point extraction."""

    def test_returns_dict_with_all_features(
        self, trained_lgb_model: lgb.Booster, synthetic_data: dict
    ):
        """Should return a dict keyed by every feature name."""
        feature_names = synthetic_data["feature_names"]
        result = extract_decision_points_lgb(trained_lgb_model, feature_names)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(feature_names)

    def test_values_are_sorted_numpy_arrays(
        self, trained_lgb_model: lgb.Booster, synthetic_data: dict
    ):
        """Each value should be a sorted 1-D numpy array of unique floats."""
        feature_names = synthetic_data["feature_names"]
        result = extract_decision_points_lgb(trained_lgb_model, feature_names)

        for fname, thresholds in result.items():
            assert isinstance(thresholds, np.ndarray), f"{fname}: not ndarray"
            assert thresholds.ndim == 1, f"{fname}: not 1-D"
            # Sorted ascending
            if len(thresholds) > 1:
                assert np.all(
                    thresholds[:-1] < thresholds[1:]
                ), f"{fname}: not strictly sorted"
            # Unique
            assert len(thresholds) == len(np.unique(thresholds)), f"{fname}: not unique"

    def test_important_features_have_splits(
        self, trained_lgb_model: lgb.Booster, synthetic_data: dict
    ):
        """Features 0, 1, 2 (non-zero weights) should have at least 1 split point."""
        feature_names = synthetic_data["feature_names"]
        result = extract_decision_points_lgb(trained_lgb_model, feature_names)

        for i in [0, 1, 2]:
            fname = feature_names[i]
            assert len(result[fname]) > 0, (
                f"{fname} should have splits (weight={synthetic_data['weights'][i]})"
            )

    def test_noise_features_may_have_fewer_splits(
        self, trained_lgb_model: lgb.Booster, synthetic_data: dict
    ):
        """Noise features (3, 4) should have fewer splits than feature_0."""
        feature_names = synthetic_data["feature_names"]
        result = extract_decision_points_lgb(trained_lgb_model, feature_names)

        n_splits_0 = len(result[feature_names[0]])
        for i in [3, 4]:
            n_splits_noise = len(result[feature_names[i]])
            # Not a hard requirement — noise features CAN have splits via
            # random selection, but should generally have fewer.
            assert n_splits_noise <= n_splits_0 or n_splits_noise == 0

    def test_split_values_are_finite(
        self, trained_lgb_model: lgb.Booster, synthetic_data: dict
    ):
        """All split thresholds should be finite real numbers."""
        feature_names = synthetic_data["feature_names"]
        result = extract_decision_points_lgb(trained_lgb_model, feature_names)

        for fname, thresholds in result.items():
            assert np.all(np.isfinite(thresholds)), f"{fname}: non-finite threshold"

    def test_deterministic(
        self, trained_lgb_model: lgb.Booster, synthetic_data: dict
    ):
        """Calling twice should produce identical results."""
        feature_names = synthetic_data["feature_names"]
        r1 = extract_decision_points_lgb(trained_lgb_model, feature_names)
        r2 = extract_decision_points_lgb(trained_lgb_model, feature_names)

        for fname in feature_names:
            np.testing.assert_array_equal(r1[fname], r2[fname])


class TestExtractDecisionPointsLGBEdgeCases:
    """Edge-case tests for LightGBM extraction."""

    def test_single_tree_model(self, synthetic_data: dict):
        """Should work with a model containing a single tree."""
        X = synthetic_data["X_train"]
        y = synthetic_data["y_train"]
        train_data = lgb.Dataset(X, label=y)
        params = {
            "objective": "binary",
            "num_leaves": 4,
            "verbose": -1,
            "seed": 42,
        }
        model = lgb.train(params, train_data, num_boost_round=1)
        feature_names = synthetic_data["feature_names"]

        result = extract_decision_points_lgb(model, feature_names)
        assert isinstance(result, dict)
        total_splits = sum(len(v) for v in result.values())
        assert total_splits > 0, "Single-tree model should have at least 1 split"

    def test_deep_model_has_more_splits(self, synthetic_data: dict):
        """A deeper model should produce more (or equal) unique splits."""
        X = synthetic_data["X_train"]
        y = synthetic_data["y_train"]
        feature_names = synthetic_data["feature_names"]

        # Shallow model
        ds = lgb.Dataset(X, label=y)
        shallow = lgb.train(
            {"objective": "binary", "num_leaves": 4, "verbose": -1, "seed": 42},
            ds,
            num_boost_round=10,
        )
        r_shallow = extract_decision_points_lgb(shallow, feature_names)

        # Deep model
        deep = lgb.train(
            {"objective": "binary", "num_leaves": 31, "verbose": -1, "seed": 42},
            ds,
            num_boost_round=50,
        )
        r_deep = extract_decision_points_lgb(deep, feature_names)

        total_shallow = sum(len(v) for v in r_shallow.values())
        total_deep = sum(len(v) for v in r_deep.values())
        assert total_deep >= total_shallow
