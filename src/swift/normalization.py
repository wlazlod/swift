"""Stage 3: SHAP normalization — bucket-level mean SHAP and feature transformation.

Computes the mean SHAP value per bucket on the reference sample, then
provides a transformation function that maps any feature value to its
bucket's mean SHAP (the "SHAP normalization" step).

For empty buckets (no reference observations): creates synthetic observations
by sampling real rows and placing the feature value inside the empty bucket,
then computes SHAP on those synthetic observations.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

from swift.types import BucketSet, BucketType

logger = logging.getLogger(__name__)


def compute_bucket_shap(
    bucket_sets: dict[str, BucketSet],
    X_ref: pd.DataFrame,
    shap_values: np.ndarray,
    model: object | None = None,
    n_synthetic: int = 10,
    rng: np.random.Generator | None = None,
) -> dict[str, BucketSet]:
    """Compute mean SHAP per bucket for all features.

    For each feature j and bucket k, computes:
        mean_shap_j^k = mean(shap_j(x_i) for all i where x_ij in bucket k)

    If a bucket has zero observations in X_ref:
        - If model is provided: create n_synthetic observations by sampling
          real rows and setting the feature value to fall in the empty bucket,
          then compute SHAP on those synthetic observations.
        - If model is None: assign mean_shap = 0.0 with a warning.

    Args:
        bucket_sets: Dict of feature_name -> BucketSet (from build_all_buckets).
        X_ref: Reference DataFrame (n_ref x p).
        shap_values: SHAP values array of shape (n_ref, p).
        model: Trained model for computing SHAP on synthetic observations.
            If None, empty buckets get mean_shap = 0.0.
        n_synthetic: Number of synthetic observations to create for empty buckets.
        rng: Random number generator for reproducibility.

    Returns:
        Dict of feature_name -> BucketSet with mean_shap populated on each Bucket.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    feature_names = list(bucket_sets.keys())
    shap_values = np.asarray(shap_values)
    n_ref = len(X_ref)

    result: dict[str, BucketSet] = {}

    for j, fname in enumerate(feature_names):
        bs = bucket_sets[fname]
        feature_col = X_ref[fname].values
        shap_col = shap_values[:, j]

        new_buckets = []
        for bucket in bs.buckets:
            # Find observations that fall into this bucket
            mask = _make_bucket_mask(feature_col, bucket)
            count = int(np.sum(mask))

            if count > 0:
                mean_shap = float(np.mean(shap_col[mask]))
                new_buckets.append(replace(bucket, mean_shap=mean_shap))
            else:
                # Empty bucket — try synthetic fill
                mean_shap = _fill_empty_bucket(
                    bucket, bs, fname, j, X_ref, model, n_synthetic, rng
                )
                new_buckets.append(replace(bucket, mean_shap=mean_shap))

        bs.buckets = new_buckets
        result[fname] = bs

    return result


def transform_feature(
    values: np.ndarray | pd.Series,
    bucket_set: BucketSet,
) -> np.ndarray:
    """Map feature values to their bucket's mean SHAP value.

    This is the SHAP transformation sigma_j defined in the paper:
        sigma_j(x_ij) = mean_shap_j^{bucket(x_ij)}

    Uses vectorized numpy operations for performance on large arrays:
      - Identifies NaN positions and maps them to the null bucket.
      - For numeric values, uses np.searchsorted on sorted decision
        points to assign bucket indices in O(n log k) time.
      - Falls back to element-wise assignment for categorical buckets.

    Args:
        values: 1-D array or Series of feature values (may contain NaN).
        bucket_set: BucketSet with mean_shap populated on each bucket.

    Returns:
        1-D numpy array of transformed values (same length as input).
    """
    values_arr = np.asarray(values, dtype=np.float64)
    n = len(values_arr)
    result = np.empty(n, dtype=np.float64)

    # Build a lookup array: bucket_index -> mean_shap
    max_idx = max(b.index for b in bucket_set.buckets)
    shap_lookup = np.zeros(max_idx + 1, dtype=np.float64)
    for b in bucket_set.buckets:
        shap_lookup[b.index] = b.mean_shap if b.mean_shap is not None else 0.0

    # Identify null bucket and categorical buckets
    null_bucket = None
    has_categoricals = False
    for b in bucket_set.buckets:
        if b.bucket_type == BucketType.NULL:
            null_bucket = b
        elif b.bucket_type == BucketType.CATEGORICAL:
            has_categoricals = True

    if has_categoricals:
        # Fall back to element-wise for categorical features
        return _transform_feature_elementwise(values_arr, bucket_set)

    # ── Vectorized path for numeric features ─────────────────────────
    nan_mask = np.isnan(values_arr)

    # Assign numeric buckets using searchsorted on decision points.
    # BucketSet has sorted decision_points (the split thresholds).
    # Bucket layout: null_bucket (index 0), then numeric buckets
    # in order of intervals:
    #   bucket 1: (-inf, dp[0])
    #   bucket 2: [dp[0], dp[1])
    #   ...
    #   bucket k+1: [dp[k-1], inf)
    #
    # np.searchsorted(dp, val, side='right') gives the number of
    # decision points <= val, which maps to bucket index offset by 1
    # (since bucket 0 is null).

    dp = bucket_set.decision_points
    if len(dp) > 0:
        # searchsorted with side='right' gives index i such that
        # dp[i-1] <= val < dp[i], which is bucket (i + 1) if null=0
        bucket_indices = np.searchsorted(dp, values_arr, side="right")
        # Offset by 1 because bucket 0 is the null bucket
        bucket_indices = bucket_indices + 1
    else:
        # No decision points: everything goes to bucket 1
        bucket_indices = np.ones(n, dtype=np.intp)

    # Handle NaN → null bucket
    if null_bucket is not None:
        bucket_indices[nan_mask] = null_bucket.index

    # Clip to valid range (safety)
    np.clip(bucket_indices, 0, max_idx, out=bucket_indices)

    # Vectorized lookup
    result = shap_lookup[bucket_indices]

    return result


def _transform_feature_elementwise(
    values_arr: np.ndarray,
    bucket_set: BucketSet,
) -> np.ndarray:
    """Fallback element-wise transform for categorical features.

    Args:
        values_arr: 1-D float64 array of feature values.
        bucket_set: BucketSet with mean_shap populated.

    Returns:
        1-D numpy array of transformed values.
    """
    n = len(values_arr)
    result = np.empty(n, dtype=np.float64)

    for i in range(n):
        val = values_arr[i]
        if np.isnan(val):
            bucket_idx = bucket_set.assign_bucket(None)
        else:
            bucket_idx = bucket_set.assign_bucket(float(val))
        result[i] = bucket_set.get_mean_shap(bucket_idx)

    return result


def _make_bucket_mask(
    values: np.ndarray,
    bucket: object,
) -> np.ndarray:
    """Create a boolean mask for observations falling into the given bucket.

    Args:
        values: 1-D array of feature values (may contain NaN).
        bucket: A Bucket object.

    Returns:
        Boolean mask of shape (n,).
    """
    n = len(values)
    mask = np.zeros(n, dtype=bool)

    if bucket.bucket_type == BucketType.NULL:
        # Match NaN / None values
        mask = pd.isna(values)
        return np.asarray(mask, dtype=bool)

    if bucket.bucket_type == BucketType.CATEGORICAL:
        if bucket.categories is not None:
            for i in range(n):
                if values[i] in bucket.categories:
                    mask[i] = True
        return mask

    # Numeric bucket: [lower, upper)
    # Special cases for -inf and +inf boundaries
    lower = bucket.lower
    upper = bucket.upper

    not_nan = ~pd.isna(values)
    vals = np.where(not_nan, values, 0.0)  # replace NaN for comparison

    if np.isneginf(lower):
        # (-inf, upper)
        mask = not_nan & (vals < upper)
    elif np.isposinf(upper):
        # [lower, inf)
        mask = not_nan & (vals >= lower)
    else:
        # [lower, upper)
        mask = not_nan & (vals >= lower) & (vals < upper)

    return np.asarray(mask, dtype=bool)


def _fill_empty_bucket(
    bucket: object,
    bucket_set: BucketSet,
    feature_name: str,
    feature_idx: int,
    X_ref: pd.DataFrame,
    model: object | None,
    n_synthetic: int,
    rng: np.random.Generator,
) -> float:
    """Fill an empty bucket by creating synthetic observations and computing SHAP.

    If model is None, returns 0.0 with a warning.

    Args:
        bucket: The empty Bucket.
        bucket_set: Parent BucketSet.
        feature_name: Name of the feature.
        feature_idx: Column index of the feature.
        X_ref: Reference DataFrame.
        model: Trained model (or None).
        n_synthetic: Number of synthetic observations to create.
        rng: Random number generator.

    Returns:
        Mean SHAP value for the synthetic observations.
    """
    if model is None:
        logger.warning(
            "Feature '%s', bucket %d: no observations and no model provided. "
            "Assigning mean_shap = 0.0.",
            feature_name,
            bucket.index,
        )
        return 0.0

    # Create synthetic observations
    n_ref = len(X_ref)
    sample_indices = rng.integers(0, n_ref, size=n_synthetic)
    X_synthetic = X_ref.iloc[sample_indices].copy()

    # Set the feature value to fall inside the bucket
    if bucket.bucket_type == BucketType.NULL:
        X_synthetic[feature_name] = np.nan
    elif bucket.bucket_type == BucketType.NUMERIC:
        synthetic_val = _sample_value_in_bucket(bucket, rng)
        X_synthetic[feature_name] = synthetic_val
    else:
        # Categorical — pick first category
        if bucket.categories:
            X_synthetic[feature_name] = next(iter(bucket.categories))

    # Compute SHAP for synthetic observations
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        synthetic_shap = explainer.shap_values(X_synthetic)
        synthetic_shap = np.asarray(synthetic_shap)
        mean_shap = float(np.mean(synthetic_shap[:, feature_idx]))

        logger.info(
            "Feature '%s', bucket %d: filled with %d synthetic obs, "
            "mean_shap = %.6f.",
            feature_name,
            bucket.index,
            n_synthetic,
            mean_shap,
        )
        return mean_shap

    except Exception as e:
        logger.warning(
            "Feature '%s', bucket %d: SHAP computation on synthetic obs failed: %s. "
            "Assigning mean_shap = 0.0.",
            feature_name,
            bucket.index,
            e,
        )
        return 0.0


def _sample_value_in_bucket(
    bucket: object,
    rng: np.random.Generator,
) -> float:
    """Sample a representative value inside a numeric bucket.

    For bounded buckets: midpoint.
    For (-inf, upper): upper - 1.0
    For [lower, inf): lower + 1.0

    Args:
        bucket: A numeric Bucket.
        rng: Random number generator (unused for now, but available).

    Returns:
        A float value inside the bucket.
    """
    lower = bucket.lower
    upper = bucket.upper

    if np.isneginf(lower) and np.isposinf(upper):
        return 0.0
    elif np.isneginf(lower):
        return upper - 1.0
    elif np.isposinf(upper):
        return lower + 1.0
    else:
        return (lower + upper) / 2.0
