"""Stage 6: Aggregation — feature-level to model-level SWIFT scores.

Provides three aggregation strategies from the paper (§3.6):
    - Maximum SWIFT: most-drifted feature  (SWIFT_max)
    - Mean SWIFT: average feature drift    (SWIFT_mean)
    - Weighted SWIFT: importance-weighted   (SWIFT_weighted)

Functions:
    aggregate_scores:          Compute model-level summary statistics.
    compute_importance_weights: Normalized mean |SHAP| weights w_j.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AggregatedScores:
    """Model-level SWIFT aggregation.

    Attributes:
        swift_max: Maximum per-feature SWIFT score.
        swift_mean: Unweighted mean of per-feature scores.
        swift_weighted: Importance-weighted score (None if no weights).
        max_feature: Name of the feature with the highest score.
    """

    swift_max: float
    swift_mean: float
    max_feature: str
    swift_weighted: Optional[float] = None


def aggregate_scores(
    scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> AggregatedScores:
    """Aggregate per-feature SWIFT scores to model level.

    Args:
        scores: Dict of feature_name → SWIFT score.
        weights: Optional dict of feature_name → importance weight.
            If provided, weights should sum to 1 (or be normalizable).
            If None, swift_weighted is set to None.

    Returns:
        AggregatedScores with max, mean, and optionally weighted score.
    """
    if not scores:
        raise ValueError("scores dict must not be empty.")

    names = list(scores.keys())
    vals = np.array([scores[n] for n in names])

    swift_max = float(np.max(vals))
    swift_mean = float(np.mean(vals))
    max_feature = names[int(np.argmax(vals))]

    swift_weighted: Optional[float] = None
    if weights is not None:
        weighted_sum = sum(weights[n] * scores[n] for n in names)
        swift_weighted = float(weighted_sum)

    return AggregatedScores(
        swift_max=swift_max,
        swift_mean=swift_mean,
        max_feature=max_feature,
        swift_weighted=swift_weighted,
    )


def compute_importance_weights(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Compute normalized mean absolute SHAP importance weights.

    w_j = |φ̄_j| / Σ_k |φ̄_k|

    where φ̄_j = mean(|SHAP_j(x_i)|) across all reference observations.

    This generalizes SSI's IV-based weights with model-aware
    SHAP-based feature importance.

    Args:
        shap_values: SHAP values array of shape (n, p).
        feature_names: List of p feature names.

    Returns:
        Dict of feature_name → weight (non-negative, sums to 1).
    """
    shap_values = np.asarray(shap_values)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    total = np.sum(mean_abs_shap)
    if total == 0.0:
        # Uniform weights if all SHAP values are zero
        p = len(feature_names)
        return {name: 1.0 / p for name in feature_names}

    weights = mean_abs_shap / total
    return {name: float(weights[j]) for j, name in enumerate(feature_names)}
