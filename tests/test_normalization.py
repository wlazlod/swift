"""Tests for Stage 3: SHAP normalization.

Contract:
    compute_bucket_shap(
        bucket_sets: dict[str, BucketSet],
        X_ref: pd.DataFrame,
        shap_values: np.ndarray,
        model: lgb.Booster | None = None,
        n_synthetic: int = 10,
    ) -> dict[str, BucketSet]
    - Computes mean SHAP per bucket and writes it into each Bucket's mean_shap.
    - For empty buckets: creates synthetic observations to compute mean SHAP.
    - Returns the same BucketSet objects but with mean_shap populated.

    transform_feature(
        values: np.ndarray | pd.Series,
        bucket_set: BucketSet,
    ) -> np.ndarray
    - Maps each value to its bucket's mean SHAP value.
    - Returns 1-D array of transformed values (same length as input).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb

from swift.types import BucketSet, BucketType
from swift.bucketing import build_buckets, build_all_buckets
from swift.extraction import extract_decision_points_lgb
from swift.normalization import compute_bucket_shap, transform_feature


@pytest.fixture
def simple_bucket_set() -> BucketSet:
    """BucketSet with splits at [1.0, 3.0] -> 4 buckets (null + 3 numeric)."""
    return build_buckets(np.array([1.0, 3.0]), "f")


class TestComputeBucketShap:
    """Tests for computing mean SHAP per bucket."""

    def test_all_buckets_get_mean_shap(
        self,
        trained_lgb_model: lgb.Booster,
        synthetic_data: dict,
        ref_shap_values: np.ndarray,
    ):
        """Every bucket that has observations should get a non-None mean_shap."""
        dp = extract_decision_points_lgb(
            trained_lgb_model, synthetic_data["feature_names"]
        )
        bucket_sets = build_all_buckets(dp)
        bucket_sets = compute_bucket_shap(
            bucket_sets,
            synthetic_data["X_ref"],
            ref_shap_values,
            model=trained_lgb_model,
        )

        for fname, bs in bucket_sets.items():
            for bucket in bs.buckets:
                # Null bucket might be empty (no nulls in data) — that's OK,
                # but numeric buckets should have mean_shap populated.
                if bucket.bucket_type == BucketType.NUMERIC:
                    assert bucket.mean_shap is not None, (
                        f"{fname}, bucket {bucket.index}: mean_shap is None"
                    )

    def test_mean_shap_is_finite(
        self,
        trained_lgb_model: lgb.Booster,
        synthetic_data: dict,
        ref_shap_values: np.ndarray,
    ):
        """All mean_shap values should be finite real numbers."""
        dp = extract_decision_points_lgb(
            trained_lgb_model, synthetic_data["feature_names"]
        )
        bucket_sets = build_all_buckets(dp)
        bucket_sets = compute_bucket_shap(
            bucket_sets,
            synthetic_data["X_ref"],
            ref_shap_values,
            model=trained_lgb_model,
        )

        for fname, bs in bucket_sets.items():
            for bucket in bs.buckets:
                if bucket.mean_shap is not None:
                    assert np.isfinite(bucket.mean_shap), (
                        f"{fname}, bucket {bucket.index}: non-finite mean_shap"
                    )

    def test_mean_shap_is_mean_of_shap_values(self):
        """For known data, mean_shap should equal the arithmetic mean of SHAP values
        for observations in that bucket."""
        # Simple case: 2 decision points [2, 4], 6 observations
        dp = np.array([2.0, 4.0])
        bs_dict = {"f": build_buckets(dp, "f")}

        X_ref = pd.DataFrame({"f": [0.5, 1.5, 2.5, 3.0, 4.5, 6.0]})
        # SHAP values for these 6 observations
        shap_vals = np.array([[0.1], [0.2], [0.5], [0.6], [0.9], [1.0]])
        # Bucket assignments:
        #   0.5 -> (-inf, 2) -> bucket 1, SHAP = 0.1
        #   1.5 -> (-inf, 2) -> bucket 1, SHAP = 0.2
        #   2.5 -> [2, 4)    -> bucket 2, SHAP = 0.5
        #   3.0 -> [2, 4)    -> bucket 2, SHAP = 0.6
        #   4.5 -> [4, inf)  -> bucket 3, SHAP = 0.9
        #   6.0 -> [4, inf)  -> bucket 3, SHAP = 1.0

        result = compute_bucket_shap(bs_dict, X_ref, shap_vals)

        bs = result["f"]
        # Find numeric buckets by index
        b1 = next(b for b in bs.buckets if b.index == 1)  # (-inf, 2)
        b2 = next(b for b in bs.buckets if b.index == 2)  # [2, 4)
        b3 = next(b for b in bs.buckets if b.index == 3)  # [4, inf)

        assert b1.mean_shap == pytest.approx(0.15, abs=1e-10)   # (0.1+0.2)/2
        assert b2.mean_shap == pytest.approx(0.55, abs=1e-10)   # (0.5+0.6)/2
        assert b3.mean_shap == pytest.approx(0.95, abs=1e-10)   # (0.9+1.0)/2

    def test_null_bucket_with_nulls(
        self,
        trained_lgb_model_with_nulls: lgb.Booster,
        synthetic_data_with_nulls: dict,
    ):
        """When data has nulls, the null bucket should get a mean_shap."""
        import shap

        X_ref = synthetic_data_with_nulls["X_ref"]
        feature_names = synthetic_data_with_nulls["feature_names"]

        explainer = shap.TreeExplainer(trained_lgb_model_with_nulls)
        shap_vals = np.asarray(explainer.shap_values(X_ref))

        dp = extract_decision_points_lgb(trained_lgb_model_with_nulls, feature_names)
        bucket_sets = build_all_buckets(dp)
        bucket_sets = compute_bucket_shap(
            bucket_sets, X_ref, shap_vals, model=trained_lgb_model_with_nulls
        )

        # feature_0 has nulls, so its null bucket should have a mean_shap
        bs_f0 = bucket_sets["feature_0"]
        null_bucket = next(b for b in bs_f0.buckets if b.bucket_type == BucketType.NULL)
        assert null_bucket.mean_shap is not None

    def test_empty_bucket_gets_synthetic_fill(self):
        """An empty bucket (no observations in D_ref) should get mean_shap
        via synthetic observation creation when a model is provided."""
        # Create a bucket set with a threshold that no observation falls into
        dp = np.array([0.0, 100.0])  # bucket [0, 100) will have obs; [100, inf) empty
        bs_dict = {"f": build_buckets(dp, "f")}

        X_ref = pd.DataFrame({"f": [0.5, 1.0, 50.0]})  # all in [0, 100)
        shap_vals = np.array([[0.3], [0.4], [0.5]])

        # Without model: empty bucket should get 0.0
        result = compute_bucket_shap(bs_dict, X_ref, shap_vals, model=None)
        bs = result["f"]
        empty_bucket = next(b for b in bs.buckets if b.index == 3)  # [100, inf)
        assert empty_bucket.mean_shap == pytest.approx(0.0)


class TestTransformFeature:
    """Tests for mapping feature values to bucket mean SHAP."""

    def test_output_shape(self, simple_bucket_set: BucketSet):
        """Output should have same length as input."""
        # Manually set mean_shap on the buckets
        bs = _set_bucket_shaps(simple_bucket_set, {0: -0.5, 1: -0.2, 2: 0.3, 3: 0.8})

        values = np.array([0.0, 0.5, 2.0, 5.0])
        result = transform_feature(values, bs)
        assert result.shape == (4,)

    def test_values_mapped_correctly(self, simple_bucket_set: BucketSet):
        """Each value should be mapped to the correct bucket's mean_shap."""
        bs = _set_bucket_shaps(simple_bucket_set, {0: -0.5, 1: -0.2, 2: 0.3, 3: 0.8})

        values = np.array([0.0, 1.5, 5.0])
        result = transform_feature(values, bs)
        # 0.0 -> bucket (-inf, 1.0) -> mean_shap = -0.2
        # 1.5 -> bucket [1.0, 3.0) -> mean_shap = 0.3
        # 5.0 -> bucket [3.0, inf) -> mean_shap = 0.8
        np.testing.assert_array_almost_equal(result, [-0.2, 0.3, 0.8])

    def test_null_values_mapped(self, simple_bucket_set: BucketSet):
        """NaN values should be mapped to the null bucket's mean_shap."""
        bs = _set_bucket_shaps(simple_bucket_set, {0: -0.5, 1: -0.2, 2: 0.3, 3: 0.8})

        values = pd.Series([np.nan, 2.0, np.nan])
        result = transform_feature(values, bs)
        # NaN -> null bucket -> mean_shap = -0.5
        # 2.0 -> bucket [1.0, 3.0) -> mean_shap = 0.3
        np.testing.assert_array_almost_equal(result, [-0.5, 0.3, -0.5])

    def test_piecewise_constant(self, simple_bucket_set: BucketSet):
        """Values within the same bucket should map to the same SHAP value."""
        bs = _set_bucket_shaps(simple_bucket_set, {0: -0.5, 1: -0.2, 2: 0.3, 3: 0.8})

        # All in bucket [1.0, 3.0)
        values = np.array([1.0, 1.5, 2.0, 2.99])
        result = transform_feature(values, bs)
        assert np.all(result == 0.3)

    def test_integration_end_to_end(
        self,
        trained_lgb_model: lgb.Booster,
        synthetic_data: dict,
        ref_shap_values: np.ndarray,
    ):
        """Full chain: extract -> bucket -> compute SHAP -> transform."""
        feature_names = synthetic_data["feature_names"]
        dp = extract_decision_points_lgb(trained_lgb_model, feature_names)
        bucket_sets = build_all_buckets(dp)
        bucket_sets = compute_bucket_shap(
            bucket_sets,
            synthetic_data["X_ref"],
            ref_shap_values,
            model=trained_lgb_model,
        )

        # Transform reference and monitoring features
        for j, fname in enumerate(feature_names):
            ref_transformed = transform_feature(
                synthetic_data["X_ref"][fname], bucket_sets[fname]
            )
            mon_transformed = transform_feature(
                synthetic_data["X_mon_drift"][fname], bucket_sets[fname]
            )

            assert ref_transformed.shape == (1000,)
            assert mon_transformed.shape == (1000,)
            assert np.all(np.isfinite(ref_transformed))
            assert np.all(np.isfinite(mon_transformed))


def _set_bucket_shaps(bs: BucketSet, shap_map: dict[int, float]) -> BucketSet:
    """Helper: create a new BucketSet with mean_shap values set."""
    from dataclasses import replace

    new_buckets = tuple(
        replace(b, mean_shap=shap_map[b.index]) if b.index in shap_map else b
        for b in bs.buckets
    )
    return BucketSet(
        feature_name=bs.feature_name,
        buckets=new_buckets,
        decision_points=bs.decision_points,
    )
