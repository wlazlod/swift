"""Tests for Stage 2: Bucket construction from decision points.

Contract:
    build_buckets(decision_points, feature_name) -> BucketSet
    - Creates buckets from sorted split thresholds: (-inf, t1), [t1, t2), ..., [tm, inf)
    - Always includes a null bucket (index 0).
    - Total buckets = len(decision_points) + 2 (including null).

    build_all_buckets(decision_points_dict) -> dict[str, BucketSet]
    - Applies build_buckets to all features.
"""

from __future__ import annotations

import numpy as np
import pytest

from swift.types import Bucket, BucketSet, BucketType
from swift.bucketing import build_buckets, build_all_buckets


class TestBuildBuckets:
    """Tests for single-feature bucket construction."""

    def test_three_splits_produce_five_buckets(self):
        """Splits at [1, 2, 3] -> null + (-inf,1) + [1,2) + [2,3) + [3,inf) = 5."""
        decision_points = np.array([1.0, 2.0, 3.0])
        bs = build_buckets(decision_points, "feature_a")

        assert isinstance(bs, BucketSet)
        assert bs.feature_name == "feature_a"
        assert bs.num_buckets == 5  # null + 4 numeric

    def test_null_bucket_is_always_present(self):
        """Even with zero decision points, there should be a null bucket."""
        bs_empty = build_buckets(np.array([]), "f")
        null_buckets = [b for b in bs_empty.buckets if b.bucket_type == BucketType.NULL]
        assert len(null_buckets) == 1

        bs_nonempty = build_buckets(np.array([5.0]), "f")
        null_buckets2 = [b for b in bs_nonempty.buckets if b.bucket_type == BucketType.NULL]
        assert len(null_buckets2) == 1

    def test_zero_splits_produce_two_buckets(self):
        """No decision points -> null + (-inf, inf) = 2 buckets."""
        bs = build_buckets(np.array([]), "f")
        assert bs.num_buckets == 2  # null + one catch-all

    def test_one_split_produce_three_buckets(self):
        """Split at [5.0] -> null + (-inf,5) + [5,inf) = 3 buckets."""
        bs = build_buckets(np.array([5.0]), "f")
        assert bs.num_buckets == 3

    def test_bucket_boundaries_are_correct(self):
        """Verify exact boundaries for splits at [1.0, 3.0]."""
        bs = build_buckets(np.array([1.0, 3.0]), "f")

        # Filter out null bucket
        numeric = [b for b in bs.buckets if b.bucket_type == BucketType.NUMERIC]
        numeric.sort(key=lambda b: b.lower)

        assert len(numeric) == 3
        # Bucket 0: (-inf, 1.0)
        assert np.isneginf(numeric[0].lower)
        assert numeric[0].upper == 1.0
        # Bucket 1: [1.0, 3.0)
        assert numeric[1].lower == 1.0
        assert numeric[1].upper == 3.0
        # Bucket 2: [3.0, inf)
        assert numeric[2].lower == 3.0
        assert np.isposinf(numeric[2].upper)

    def test_assign_bucket_numeric(self):
        """Values should be assigned to the correct bucket."""
        bs = build_buckets(np.array([1.0, 3.0]), "f")

        # Value < 1.0 -> first numeric bucket
        idx_low = bs.assign_bucket(-5.0)
        b_low = bs.buckets[idx_low]
        assert b_low.bucket_type == BucketType.NUMERIC
        assert np.isneginf(b_low.lower)

        # Value = 1.0 -> second numeric bucket [1.0, 3.0)
        idx_mid = bs.assign_bucket(1.0)
        b_mid = bs.buckets[idx_mid]
        assert b_mid.lower == 1.0
        assert b_mid.upper == 3.0

        # Value = 2.99 -> still second numeric bucket
        assert bs.assign_bucket(2.99) == idx_mid

        # Value = 3.0 -> third numeric bucket [3.0, inf)
        idx_high = bs.assign_bucket(3.0)
        b_high = bs.buckets[idx_high]
        assert b_high.lower == 3.0

    def test_assign_bucket_null(self):
        """None and NaN should be assigned to the null bucket."""
        bs = build_buckets(np.array([1.0]), "f")

        idx_none = bs.assign_bucket(None)
        assert bs.buckets[idx_none].bucket_type == BucketType.NULL

        idx_nan = bs.assign_bucket(float("nan"))
        assert bs.buckets[idx_nan].bucket_type == BucketType.NULL

    def test_decision_points_stored(self):
        """The BucketSet should store the original decision points."""
        dp = np.array([2.0, 5.0, 8.0])
        bs = build_buckets(dp, "f")
        np.testing.assert_array_equal(bs.decision_points, dp)

    def test_bucket_indices_are_unique(self):
        """All bucket indices should be unique."""
        bs = build_buckets(np.array([1.0, 2.0, 3.0, 4.0]), "f")
        indices = [b.index for b in bs.buckets]
        assert len(indices) == len(set(indices))


class TestBuildAllBuckets:
    """Tests for batch bucket construction across multiple features."""

    def test_returns_dict_keyed_by_feature(self):
        """Should return a dict with one BucketSet per feature."""
        dp = {
            "f1": np.array([1.0, 2.0]),
            "f2": np.array([5.0]),
            "f3": np.array([]),
        }
        result = build_all_buckets(dp)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"f1", "f2", "f3"}
        for name, bs in result.items():
            assert isinstance(bs, BucketSet)
            assert bs.feature_name == name

    def test_bucket_counts_are_correct(self):
        """Each feature should have len(dp) + 2 buckets."""
        dp = {
            "f1": np.array([1.0, 2.0]),  # 4 buckets
            "f2": np.array([5.0]),  # 3 buckets
            "f3": np.array([]),  # 2 buckets
        }
        result = build_all_buckets(dp)

        assert result["f1"].num_buckets == 4
        assert result["f2"].num_buckets == 3
        assert result["f3"].num_buckets == 2

    def test_integration_with_extraction(
        self, trained_lgb_model, synthetic_data
    ):
        """Buckets should build correctly from extracted decision points."""
        from swift.extraction import extract_decision_points_lgb

        dp = extract_decision_points_lgb(
            trained_lgb_model, synthetic_data["feature_names"]
        )
        result = build_all_buckets(dp)

        for fname in synthetic_data["feature_names"]:
            bs = result[fname]
            expected = len(dp[fname]) + 2
            assert bs.num_buckets == expected, (
                f"{fname}: expected {expected} buckets, got {bs.num_buckets}"
            )
