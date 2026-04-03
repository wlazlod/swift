"""Stage 2: Bucket construction from decision points.

Given sorted, unique decision points (split thresholds) per feature,
constructs a BucketSet with:
    - A null bucket for missing values (index 0)
    - Numeric buckets: (-inf, t1), [t1, t2), ..., [tm, inf)
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from swift.types import Bucket, BucketSet, BucketType

logger = logging.getLogger(__name__)


def build_buckets(
    decision_points: np.ndarray,
    feature_name: str,
) -> BucketSet:
    """Construct a BucketSet from sorted decision points for a single feature.

    Args:
        decision_points: Sorted 1-D array of unique split thresholds.
        feature_name: Name of the feature.

    Returns:
        A BucketSet containing:
            - Bucket 0: null bucket (for missing values)
            - Buckets 1..m+1: numeric interval buckets
        Total: len(decision_points) + 2 buckets.
    """
    dp = np.asarray(decision_points, dtype=np.float64)
    m = len(dp)
    buckets: list[Bucket] = []

    # Index 0: null bucket
    buckets.append(
        Bucket(
            bucket_type=BucketType.NULL,
            index=0,
            lower=float("nan"),
            upper=float("nan"),
        )
    )

    if m == 0:
        # No decision points: single catch-all bucket (-inf, inf)
        buckets.append(
            Bucket(
                bucket_type=BucketType.NUMERIC,
                index=1,
                lower=float("-inf"),
                upper=float("inf"),
            )
        )
    else:
        # First bucket: (-inf, t1)
        buckets.append(
            Bucket(
                bucket_type=BucketType.NUMERIC,
                index=1,
                lower=float("-inf"),
                upper=float(dp[0]),
            )
        )

        # Middle buckets: [t_k, t_{k+1})  for k = 0..m-2
        for k in range(m - 1):
            buckets.append(
                Bucket(
                    bucket_type=BucketType.NUMERIC,
                    index=k + 2,
                    lower=float(dp[k]),
                    upper=float(dp[k + 1]),
                )
            )

        # Last bucket: [t_m, inf)
        buckets.append(
            Bucket(
                bucket_type=BucketType.NUMERIC,
                index=m + 1,
                lower=float(dp[-1]),
                upper=float("inf"),
            )
        )

    bucket_set = BucketSet(
        feature_name=feature_name,
        buckets=buckets,
        decision_points=dp,
    )

    logger.debug(
        "Feature '%s': %d decision points -> %d buckets.",
        feature_name,
        m,
        bucket_set.num_buckets,
    )

    return bucket_set


def build_all_buckets(
    decision_points_dict: dict[str, np.ndarray],
) -> dict[str, BucketSet]:
    """Construct BucketSets for all features.

    Args:
        decision_points_dict: Dict mapping feature name -> sorted decision points.

    Returns:
        Dict mapping feature name -> BucketSet.
    """
    result: dict[str, BucketSet] = {}
    for fname, dp in decision_points_dict.items():
        result[fname] = build_buckets(dp, fname)
    return result
