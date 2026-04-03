"""Data types and structures for the SWIFT pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class BucketType(Enum):
    """Type of bucket."""

    NULL = "null"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class WassersteinOrder(Enum):
    """Order of Wasserstein distance."""

    W1 = 1
    W2 = 2


class CorrectionMethod(Enum):
    """Multiple testing correction method.

    Supports construction from strings via ``CorrectionMethod.resolve()``,
    so users can write ``correction="benjamini-hochberg"`` instead of
    importing the enum.
    """

    BONFERRONI = "bonferroni"
    BH = "benjamini-hochberg"

    @classmethod
    def resolve(cls, value: CorrectionMethod | str) -> CorrectionMethod:
        """Return a ``CorrectionMethod`` from a member, its value, or an alias.

        Parameters
        ----------
        value : CorrectionMethod or str
            Enum member, canonical string (``"bonferroni"``,
            ``"benjamini-hochberg"``), or alias (``"bh"``, ``"fdr"``,
            ``"bonf"``).  Case-insensitive.

        Returns
        -------
        CorrectionMethod

        Raises
        ------
        ValueError
            If *value* cannot be resolved.
        """
        if isinstance(value, cls):
            return value
        key = value.strip().lower()
        # Direct enum-value lookup.
        for member in cls:
            if member.value == key:
                return member
        # Alias lookup.
        canonical = _CORRECTION_ALIASES.get(key)
        if canonical is not None:
            for member in cls:
                if member.value == canonical:
                    return member
        valid = sorted(
            {m.value for m in cls} | set(_CORRECTION_ALIASES.keys())
        )
        raise ValueError(
            f"Unknown correction method {value!r}. "
            f"Valid values: {valid}"
        )


# Aliases kept outside the Enum to avoid creating spurious members.
_CORRECTION_ALIASES: dict[str, str] = {
    "bh": "benjamini-hochberg",
    "fdr": "benjamini-hochberg",
    "bonf": "bonferroni",
}


@dataclass(frozen=True)
class Bucket:
    """A single bucket for a feature.

    For numeric features: defined by [lower, upper) interval.
    For null bucket: lower=upper=NaN.
    For categorical: categories is a frozenset of category values.
    """

    bucket_type: BucketType
    index: int
    lower: float = float("-inf")
    upper: float = float("inf")
    categories: Optional[frozenset] = None
    mean_shap: Optional[float] = None

    def contains(self, value: float | str | None) -> bool:
        """Check if a value falls into this bucket."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return self.bucket_type == BucketType.NULL

        if self.bucket_type == BucketType.NULL:
            return False

        if self.bucket_type == BucketType.CATEGORICAL:
            return self.categories is not None and value in self.categories

        # Numeric: [lower, upper) except for the first bucket which is (-inf, upper)
        if np.isneginf(self.lower):
            return value < self.upper
        if np.isposinf(self.upper):
            return value >= self.lower
        return self.lower <= value < self.upper


@dataclass(frozen=True)
class BucketSet:
    """Collection of buckets for a single feature.

    Attributes:
        feature_name: Name of the feature.
        buckets: List of Bucket objects (including null bucket).
        decision_points: Sorted array of decision points (split thresholds).
    """

    feature_name: str
    buckets: tuple[Bucket, ...] = field(default_factory=tuple)
    decision_points: np.ndarray = field(
        default_factory=lambda: np.array([]), hash=False, compare=False
    )

    def assign_bucket(self, value: float | str | None) -> int:
        """Return the index of the bucket that contains the given value.

        Raises:
            ValueError: If the value does not fall into any bucket.
        """
        for bucket in self.buckets:
            if bucket.contains(value):
                return bucket.index
        raise ValueError(
            f"Value {value!r} does not fall into any bucket for feature '{self.feature_name}'"
        )

    def get_mean_shap(self, bucket_index: int) -> float:
        """Return the mean SHAP value for the given bucket index.

        Raises:
            KeyError: If the bucket index does not exist.
        """
        for bucket in self.buckets:
            if bucket.index == bucket_index:
                if bucket.mean_shap is None:
                    logger.warning(
                        "Bucket %d of feature '%s' has no mean SHAP value assigned.",
                        bucket_index,
                        self.feature_name,
                    )
                    return 0.0
                return bucket.mean_shap
        raise KeyError(
            f"Bucket index {bucket_index} not found for feature '{self.feature_name}'"
        )

    @property
    def num_buckets(self) -> int:
        """Total number of buckets including null."""
        return len(self.buckets)


@dataclass(frozen=True)
class FeatureSWIFTResult:
    """SWIFT result for a single feature.

    Attributes:
        feature_name: Name of the feature.
        swift_score: SWIFT score (Wasserstein distance on SHAP-transformed distributions).
        wasserstein_order: Which Wasserstein order was used (W1 or W2).
        p_value: p-value from permutation or bootstrap test (None if not computed).
        is_drifted: Whether the feature is flagged as drifted after correction.
        num_buckets: Number of buckets used.
    """

    feature_name: str
    swift_score: float
    wasserstein_order: WassersteinOrder = WassersteinOrder.W1
    p_value: Optional[float] = None
    is_drifted: Optional[bool] = None
    num_buckets: int = 0


@dataclass(frozen=True)
class SWIFTResult:
    """Aggregate SWIFT result across all features.

    Attributes:
        feature_results: Per-feature results.
        swift_max: Maximum SWIFT score across features.
        swift_mean: Mean SWIFT score across features.
        swift_weighted: Importance-weighted SWIFT score (optional).
        alpha: Significance level used for testing.
        correction_method: Multiple testing correction method used.
    """

    feature_results: tuple[FeatureSWIFTResult, ...]
    swift_max: float
    swift_mean: float
    swift_weighted: Optional[float] = None
    alpha: float = 0.05
    correction_method: Optional[CorrectionMethod] = None

    @property
    def num_features(self) -> int:
        """Number of features monitored."""
        return len(self.feature_results)

    @property
    def num_drifted(self) -> int:
        """Number of features flagged as drifted."""
        return sum(1 for r in self.feature_results if r.is_drifted is True)

    @property
    def drifted_features(self) -> tuple[str, ...]:
        """Names of features flagged as drifted."""
        return tuple(r.feature_name for r in self.feature_results if r.is_drifted is True)
