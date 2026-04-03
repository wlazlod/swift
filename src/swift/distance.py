"""Stage 4: Wasserstein distance on SHAP-transformed distributions.

Computes the Wasserstein distance (W₁ or W₂) between two 1-D empirical
distributions of SHAP-transformed feature values.  The transformation
σ_j is computed ONCE from D_ref (Stage 3) and applied identically to
both reference and monitoring samples.

Functions:
    wasserstein_1d:  W_p distance between two 1-D arrays.
    compute_swift_scores: Per-feature SWIFT scores for ref vs mon.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance as _scipy_w1

from swift.normalization import transform_feature
from swift.types import BucketSet

logger = logging.getLogger(__name__)


def wasserstein_1d(
    u: np.ndarray,
    v: np.ndarray,
    order: int = 1,
) -> float:
    """Compute the p-th Wasserstein distance between two 1-D empirical distributions.

    For order=1 this is the Earth Mover's Distance (L₁ area between CDFs).
    For order=2 this is the root of the integral of squared CDF differences.

    Both are computed via the sorted-quantile formula:
        W_p^p = (1/N) Σ |F⁻¹_u(i/N) - F⁻¹_v(i/N)|^p
    using linear interpolation of quantile functions on a common grid
    so that unequal sample sizes are handled correctly.

    Args:
        u: 1-D array of samples from distribution P.
        v: 1-D array of samples from distribution Q.
        order: Wasserstein order (1 or 2).

    Returns:
        Non-negative float W_p(P, Q).

    Raises:
        ValueError: If order is not 1 or 2, or arrays are empty.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    if u.size == 0 or v.size == 0:
        raise ValueError("Input arrays must not be empty.")

    u_sorted = np.sort(u)
    v_sorted = np.sort(v)

    if order == 1 and u.size == v.size:
        # Fast path: equal-size W₁ is just mean absolute difference of
        # sorted values (equivalent to scipy's implementation).
        return float(np.mean(np.abs(u_sorted - v_sorted)))

    # General case: interpolate quantile functions on a common grid.
    # Build the combined set of quantile evaluation points.
    n_u = u.size
    n_v = v.size

    # CDF values at each sorted observation
    cdf_u = np.linspace(1.0 / n_u, 1.0, n_u)
    cdf_v = np.linspace(1.0 / n_v, 1.0, n_v)

    # Merge CDF grids and evaluate quantile functions at all grid points
    all_cdf = np.sort(np.concatenate([cdf_u, cdf_v]))
    all_cdf = np.unique(all_cdf)

    # Quantile function (inverse CDF): for a given probability p, the
    # quantile is the smallest x such that CDF(x) >= p.  With sorted
    # samples, np.searchsorted gives the correct index.
    q_u = u_sorted[np.clip(np.searchsorted(cdf_u, all_cdf, side="left"), 0, n_u - 1)]
    q_v = v_sorted[np.clip(np.searchsorted(cdf_v, all_cdf, side="left"), 0, n_v - 1)]

    # Compute spacing weights so we integrate over the [0,1] interval
    diffs = np.diff(np.concatenate([[0.0], all_cdf]))

    if order == 1:
        result = float(np.sum(np.abs(q_u - q_v) * diffs))
    else:  # order == 2
        result = float(np.sqrt(np.sum((q_u - q_v) ** 2 * diffs)))

    return result


def compute_swift_scores(
    X_ref: pd.DataFrame,
    X_mon: pd.DataFrame,
    bucket_sets: dict[str, BucketSet],
    order: int = 1,
) -> dict[str, float]:
    """Compute per-feature SWIFT scores: W_p on SHAP-transformed distributions.

    For each feature j:
        1. Apply σ_j (from bucket_sets) to both ref and mon columns.
        2. Compute W_p between the two transformed arrays.

    The transformation σ_j was fitted on D_ref (Stage 3) and is *not*
    recomputed here — it is applied identically to both samples.

    Args:
        X_ref: Reference DataFrame (n_ref × p).
        X_mon: Monitoring DataFrame (n_mon × p).
        bucket_sets: Dict of feature_name → BucketSet with mean_shap
            already computed (output of compute_bucket_shap).
        order: Wasserstein order (1 or 2).

    Returns:
        Dict of feature_name → SWIFT score (non-negative float).
    """
    scores: dict[str, float] = {}

    for fname, bs in bucket_sets.items():
        # Transform reference values
        ref_transformed = transform_feature(X_ref[fname], bs)

        # Transform monitoring values using the SAME σ_j
        mon_transformed = transform_feature(X_mon[fname], bs)

        # Compute Wasserstein distance
        score = wasserstein_1d(ref_transformed, mon_transformed, order=order)

        logger.debug(
            "Feature '%s': W%d = %.6f (ref_n=%d, mon_n=%d)",
            fname,
            order,
            score,
            len(ref_transformed),
            len(mon_transformed),
        )

        scores[fname] = score

    return scores
