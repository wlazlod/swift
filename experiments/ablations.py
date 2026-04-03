"""Ablation study implementations (A1-A6) for SWIFT paper.

Each ablation removes or replaces one SWIFT pipeline component to
isolate its contribution to overall performance.

Ablations:
    A1: No SHAP normalization — W₁ on raw feature values in model-aware buckets
    A2: No model-aware buckets — SHAP normalization + W₁ on equal-frequency bins
    A3: PSI on model-aware buckets — KL divergence instead of Wasserstein
    A4: W₂ instead of W₁ — full SWIFT with Wasserstein order 2
    A5: Importance-weighted aggregation — SHAP-weighted mean
    A6: KernelSHAP instead of TreeSHAP (separate due to computational cost)

All ablation scoring functions return per-feature dicts for consistency
with the main SWIFT pipeline.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from swift.aggregation import aggregate_scores, compute_importance_weights
from swift.distance import compute_swift_scores, wasserstein_1d
from experiments.baselines import (
    _vectorized_assign_buckets,
    compute_psi_on_model_buckets,
)
from swift.normalization import transform_feature
from swift.types import BucketSet, BucketType

logger = logging.getLogger(__name__)


class AblationVariant(enum.Enum):
    """Enum for all ablation variants."""

    A1_NO_SHAP_NORMALIZATION = "A1"
    A2_NO_MODEL_BUCKETS = "A2"
    A3_PSI_ON_MODEL_BUCKETS = "A3"
    A4_W2_INSTEAD_OF_W1 = "A4"
    A5_IMPORTANCE_WEIGHTED = "A5"
    A6_KERNEL_SHAP = "A6"


# ---------------------------------------------------------------------------
# A1: No SHAP normalization — Wasserstein on raw values in model-aware buckets
# ---------------------------------------------------------------------------


def compute_a1_no_shap_normalization(
    X_ref: pd.DataFrame,
    X_mon: pd.DataFrame,
    bucket_sets: dict[str, BucketSet],
    order: int = 1,
) -> dict[str, float]:
    """A1: Wasserstein on raw feature values (no SHAP transformation).

    Uses model-aware buckets from SWIFT Stage 1-2 to bin values,
    then computes W₁ directly on the raw feature values within each
    bucket's population, NOT on SHAP-transformed values.

    This isolates the contribution of SHAP normalization by showing
    what happens when we only have model-aware bucketing.

    Args:
        X_ref: Reference DataFrame.
        X_mon: Monitoring DataFrame.
        bucket_sets: Model-aware bucket sets from SWIFTMonitor.fit().
        order: Wasserstein order (1 or 2).

    Returns:
        Dict of feature_name → W_p score on raw feature values.
    """
    scores: dict[str, float] = {}

    for fname, bs in bucket_sets.items():
        ref_vals = X_ref[fname].dropna().values.astype(np.float64)
        mon_vals = X_mon[fname].dropna().values.astype(np.float64)

        if len(ref_vals) == 0 or len(mon_vals) == 0:
            scores[fname] = 0.0
            continue

        scores[fname] = wasserstein_1d(ref_vals, mon_vals, order=order)

    return scores


# ---------------------------------------------------------------------------
# A2: No model-aware buckets — SHAP normalization with equal-frequency bins
# ---------------------------------------------------------------------------


def _build_equal_freq_bucket_sets(
    X_ref: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list[str],
    n_bins: int = 10,
) -> dict[str, BucketSet]:
    """Build bucket sets using equal-frequency quantile bins.

    For each feature, creates n_bins equal-frequency bins from reference
    data, then computes mean SHAP per bin (same as SWIFT Stage 3 but
    with quantile-based boundaries instead of decision points).

    Args:
        X_ref: Reference DataFrame.
        shap_values: SHAP values (n_ref, p).
        feature_names: Feature names.
        n_bins: Number of equal-frequency bins.

    Returns:
        Dict of feature_name → BucketSet with mean_shap populated.
    """
    from swift.types import Bucket

    bucket_sets: dict[str, BucketSet] = {}
    shap_values = np.asarray(shap_values)

    for j, fname in enumerate(feature_names):
        col = X_ref[fname].values.astype(np.float64)
        shap_col = shap_values[:, j]
        not_nan = ~np.isnan(col)

        # Compute quantile bin edges from non-NaN reference data
        quantiles = np.linspace(0, 100, n_bins + 1)
        col_clean = col[not_nan]
        if len(col_clean) == 0:
            # All NaN — create a single null bucket
            null_bucket = Bucket(
                index=0,
                bucket_type=BucketType.NULL,
                lower=float("-inf"),
                upper=float("inf"),
                mean_shap=float(np.mean(shap_col)) if len(shap_col) > 0 else 0.0,
            )
            bucket_sets[fname] = BucketSet(
                feature_name=fname,
                decision_points=[],
                buckets=[null_bucket],
            )
            continue

        edges = np.unique(np.percentile(col_clean, quantiles))
        # edges are [min_val, q10, q20, ..., max_val]
        # Convert to decision points (internal edges)
        decision_points = list(edges[1:-1])  # drop min and max

        # Build buckets: null bucket + numeric buckets
        buckets: list[Bucket] = []

        # Null bucket (index 0)
        nan_mask = np.isnan(col)
        null_count = int(np.sum(nan_mask))
        null_shap = float(np.mean(shap_col[nan_mask])) if null_count > 0 else 0.0
        buckets.append(Bucket(
            index=0,
            bucket_type=BucketType.NULL,
            lower=float("-inf"),
            upper=float("inf"),
            mean_shap=null_shap,
        ))

        # Numeric buckets
        all_edges = [float("-inf")] + list(edges[1:-1]) + [float("inf")]
        for k in range(len(all_edges) - 1):
            lower = all_edges[k]
            upper = all_edges[k + 1]

            # Mask for this bucket
            if np.isneginf(lower):
                mask = not_nan & (col < upper)
            elif np.isposinf(upper):
                mask = not_nan & (col >= lower)
            else:
                mask = not_nan & (col >= lower) & (col < upper)

            count = int(np.sum(mask))
            mean_shap = float(np.mean(shap_col[mask])) if count > 0 else 0.0

            buckets.append(Bucket(
                index=k + 1,
                bucket_type=BucketType.NUMERIC,
                lower=lower,
                upper=upper,
                mean_shap=mean_shap,
            ))

        bucket_sets[fname] = BucketSet(
            feature_name=fname,
            decision_points=sorted(decision_points),
            buckets=buckets,
        )

    return bucket_sets


def compute_a2_no_model_buckets(
    X_ref: pd.DataFrame,
    X_mon: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list[str],
    n_bins: int = 10,
) -> dict[str, float]:
    """A2: SHAP normalization + W₁ but with equal-frequency bins.

    Replaces model-aware bucketing (SWIFT Stage 1-2) with standard
    equal-frequency quantile bins, then applies SHAP normalization
    and Wasserstein distance as usual.

    This isolates the contribution of model-aware bucketing.

    Args:
        X_ref: Reference DataFrame.
        X_mon: Monitoring DataFrame.
        shap_values: SHAP values (n_ref, p).
        feature_names: Feature names.
        n_bins: Number of equal-frequency bins.

    Returns:
        Dict of feature_name → SWIFT score with equal-freq bins.
    """
    # Build equal-frequency bucket sets with SHAP normalization
    eq_bucket_sets = _build_equal_freq_bucket_sets(
        X_ref, shap_values, feature_names, n_bins=n_bins,
    )

    # Compute SWIFT scores using these bucket sets
    return compute_swift_scores(X_ref, X_mon, eq_bucket_sets, order=1)


# ---------------------------------------------------------------------------
# A3: PSI on model-aware buckets (wrapper for existing baseline)
# ---------------------------------------------------------------------------


def compute_a3_psi_on_model_buckets(
    X_ref: pd.DataFrame,
    X_mon: pd.DataFrame,
    bucket_sets: dict[str, BucketSet],
    epsilon: float = 1e-4,
) -> dict[str, float]:
    """A3: KL divergence (PSI formula) on model-aware buckets.

    Uses the same model-aware buckets as SWIFT but replaces Wasserstein
    distance with KL divergence (PSI formula). This isolates the
    contribution of using Wasserstein over KL divergence.

    This is a thin wrapper around compute_psi_on_model_buckets.

    Args:
        X_ref: Reference DataFrame.
        X_mon: Monitoring DataFrame.
        bucket_sets: Model-aware bucket sets.
        epsilon: Smoothing constant for PSI.

    Returns:
        Dict of feature_name → PSI score on model-aware buckets.
    """
    return compute_psi_on_model_buckets(
        X_ref, X_mon, bucket_sets, epsilon=epsilon,
    )


# ---------------------------------------------------------------------------
# A4: W₂ instead of W₁
# ---------------------------------------------------------------------------


def compute_a4_w2_instead_of_w1(
    X_ref: pd.DataFrame,
    X_mon: pd.DataFrame,
    bucket_sets: dict[str, BucketSet],
) -> dict[str, float]:
    """A4: Full SWIFT with Wasserstein order 2 instead of order 1.

    The main SWIFT pipeline uses W₁ (Earth Mover's Distance). This
    ablation tests whether W₂ (root-mean-square CDF difference) would
    be more or less effective.

    Args:
        X_ref: Reference DataFrame.
        X_mon: Monitoring DataFrame.
        bucket_sets: Model-aware bucket sets with SHAP normalization.

    Returns:
        Dict of feature_name → W₂-based SWIFT score.
    """
    return compute_swift_scores(X_ref, X_mon, bucket_sets, order=2)


# ---------------------------------------------------------------------------
# A5: Importance-weighted aggregation
# ---------------------------------------------------------------------------


def compute_a5_importance_weighted(
    X_ref: pd.DataFrame,
    X_mon: pd.DataFrame,
    bucket_sets: dict[str, BucketSet],
    shap_values: np.ndarray,
    feature_names: list[str],
    order: int = 1,
) -> dict[str, float]:
    """A5: SWIFT with importance-weighted aggregation.

    Uses full SWIFT scoring but aggregates with SHAP-importance
    weights instead of equal-weight mean/max.

    Returns a dict with keys: per_feature, swift_max, swift_mean,
    swift_weighted, weights.

    Args:
        X_ref: Reference DataFrame.
        X_mon: Monitoring DataFrame.
        bucket_sets: Model-aware bucket sets.
        shap_values: SHAP values for computing importance weights.
        feature_names: Feature names.
        order: Wasserstein order.

    Returns:
        Dict with aggregation results (max, mean, weighted).
    """
    # Compute per-feature SWIFT scores (same as main pipeline)
    scores = compute_swift_scores(X_ref, X_mon, bucket_sets, order=order)

    # Compute importance weights
    weights = compute_importance_weights(shap_values, feature_names)

    # Aggregate with weights
    agg = aggregate_scores(scores, weights=weights)

    return {
        "per_feature": scores,
        "swift_max": agg.swift_max,
        "swift_mean": agg.swift_mean,
        "swift_weighted": agg.swift_weighted,
        "weights": weights,
        "max": agg.swift_max,
        "mean": agg.swift_mean,
    }


# ---------------------------------------------------------------------------
# run_all_ablations
# ---------------------------------------------------------------------------


def run_all_ablations(
    X_ref: pd.DataFrame,
    X_mon: pd.DataFrame,
    bucket_sets: dict[str, BucketSet],
    shap_values: np.ndarray,
    feature_names: list[str],
    n_bins: int = 10,
) -> dict[str, dict[str, Any]]:
    """Run all ablation variants A1-A5 and return summarized results.

    A6 (KernelSHAP) is excluded because it requires re-fitting with
    a different SHAP method, which is computationally expensive and
    handled separately.

    Args:
        X_ref: Reference DataFrame.
        X_mon: Monitoring DataFrame.
        bucket_sets: Model-aware bucket sets (from fitted SWIFTMonitor).
        shap_values: SHAP values for reference data.
        feature_names: Feature names.
        n_bins: Number of bins for A2 (equal-frequency).

    Returns:
        Dict of ablation_code → result dict with 'max', 'mean',
        and optionally 'per_feature' scores.
    """
    results: dict[str, dict[str, Any]] = {}

    # A1: No SHAP normalization
    logger.info("Running A1: No SHAP normalization...")
    a1_scores = compute_a1_no_shap_normalization(X_ref, X_mon, bucket_sets)
    a1_agg = aggregate_scores(a1_scores)
    results["A1"] = {
        "per_feature": a1_scores,
        "max": a1_agg.swift_max,
        "mean": a1_agg.swift_mean,
    }

    # A2: No model-aware buckets
    logger.info("Running A2: No model-aware buckets (n_bins=%d)...", n_bins)
    a2_scores = compute_a2_no_model_buckets(
        X_ref, X_mon, shap_values, feature_names, n_bins=n_bins,
    )
    a2_agg = aggregate_scores(a2_scores)
    results["A2"] = {
        "per_feature": a2_scores,
        "max": a2_agg.swift_max,
        "mean": a2_agg.swift_mean,
    }

    # A3: PSI on model-aware buckets
    logger.info("Running A3: PSI on model-aware buckets...")
    a3_scores = compute_a3_psi_on_model_buckets(X_ref, X_mon, bucket_sets)
    a3_agg = aggregate_scores(a3_scores)
    results["A3"] = {
        "per_feature": a3_scores,
        "max": a3_agg.swift_max,
        "mean": a3_agg.swift_mean,
    }

    # A4: W₂ instead of W₁
    logger.info("Running A4: W₂ instead of W₁...")
    a4_scores = compute_a4_w2_instead_of_w1(X_ref, X_mon, bucket_sets)
    a4_agg = aggregate_scores(a4_scores)
    results["A4"] = {
        "per_feature": a4_scores,
        "max": a4_agg.swift_max,
        "mean": a4_agg.swift_mean,
    }

    # A5: Importance-weighted aggregation
    logger.info("Running A5: Importance-weighted aggregation...")
    a5_result = compute_a5_importance_weighted(
        X_ref, X_mon, bucket_sets, shap_values, feature_names,
    )
    results["A5"] = a5_result

    return results
