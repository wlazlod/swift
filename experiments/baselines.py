"""Baseline drift detection methods for comparison with SWIFT.

Implements:
    - PSI (Population Stability Index) / CSI (Characteristic Stability Index)
    - SSI (Stability of SHAP Index — importance-weighted CSI)
    - KS test (Kolmogorov-Smirnov two-sample test)
    - Raw W₁ (Wasserstein distance on original features, no SHAP)
    - MMD (Maximum Mean Discrepancy)
    - BBSD (Black-Box Shift Detection — Rabanser et al., 2019)

All baselines return per-feature scores in a consistent format:
    Dict[str, float] mapping feature_name → score.

BBSD is a model-level test (not per-feature), so its results are stored
under the key ``_model_output``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PSI / CSI
# ---------------------------------------------------------------------------


def _compute_psi_1d(
    ref: np.ndarray,
    mon: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Compute PSI for a single feature using equal-frequency bins.

    PSI = Σ (p_i - q_i) × ln(p_i / q_i)

    where p_i = proportion in bin i for monitoring, q_i = for reference.
    Uses Laplace-style smoothing (epsilon) to avoid log(0).

    Args:
        ref: Reference sample for this feature.
        mon: Monitoring sample for this feature.
        n_bins: Number of equal-frequency bins (default 10).
        epsilon: Smoothing constant for empty bins.

    Returns:
        PSI value (≥ 0).
    """
    ref_clean = ref[~np.isnan(ref)]
    mon_clean = mon[~np.isnan(mon)]

    if len(ref_clean) == 0 or len(mon_clean) == 0:
        return 0.0

    # Create bins using reference quantiles
    quantiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(ref_clean, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    # Deduplicate bin edges
    bins = np.unique(bins)

    ref_counts = np.histogram(ref_clean, bins=bins)[0].astype(float)
    mon_counts = np.histogram(mon_clean, bins=bins)[0].astype(float)

    # Proportions with smoothing
    ref_props = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * len(ref_counts))
    mon_props = (mon_counts + epsilon) / (mon_counts.sum() + epsilon * len(mon_counts))

    psi = np.sum((mon_props - ref_props) * np.log(mon_props / ref_props))
    return float(psi)


def compute_psi(
    X_ref: pd.DataFrame,
    X_mon: pd.DataFrame,
    feature_names: list[str] | None = None,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute PSI (CSI) for all features.

    Args:
        X_ref: Reference DataFrame.
        X_mon: Monitoring DataFrame.
        feature_names: Features to compute PSI for. Defaults to all columns.
        n_bins: Number of equal-frequency bins.

    Returns:
        Dict of feature_name → PSI score.
    """
    if feature_names is None:
        feature_names = list(X_ref.columns)

    results = {}
    for fname in feature_names:
        ref_vals = X_ref[fname].values.astype(float)
        mon_vals = X_mon[fname].values.astype(float)
        results[fname] = _compute_psi_1d(ref_vals, mon_vals, n_bins=n_bins)

    return results


# ---------------------------------------------------------------------------
# SSI (Stability of SHAP Index — importance-weighted CSI)
# ---------------------------------------------------------------------------


def compute_ssi(
    X_ref: pd.DataFrame,
    X_mon: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list[str] | None = None,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute SSI (importance-weighted PSI/CSI).

    SSI_j = w_j × CSI_j, where w_j = mean|SHAP_j| / Σ mean|SHAP_k|

    Args:
        X_ref: Reference DataFrame.
        X_mon: Monitoring DataFrame.
        shap_values: SHAP values array (n_ref × p).
        feature_names: Feature names. Defaults to X_ref.columns.
        n_bins: Number of bins for PSI computation.

    Returns:
        Dict of feature_name → SSI score (importance-weighted PSI).
    """
    if feature_names is None:
        feature_names = list(X_ref.columns)

    # Compute PSI per feature
    psi_scores = compute_psi(X_ref, X_mon, feature_names, n_bins=n_bins)

    # Compute importance weights from SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total_importance = mean_abs_shap.sum()
    if total_importance == 0:
        weights = np.ones(len(feature_names)) / len(feature_names)
    else:
        weights = mean_abs_shap / total_importance

    results = {}
    for i, fname in enumerate(feature_names):
        results[fname] = weights[i] * psi_scores[fname]

    return results


# ---------------------------------------------------------------------------
# KS Test
# ---------------------------------------------------------------------------


def compute_ks(
    X_ref: pd.DataFrame,
    X_mon: pd.DataFrame,
    feature_names: list[str] | None = None,
    return_pvalues: bool = False,
) -> dict[str, float]:
    """Compute KS test statistic for all features.

    Args:
        X_ref: Reference DataFrame.
        X_mon: Monitoring DataFrame.
        feature_names: Features to test. Defaults to all columns.
        return_pvalues: If True, return p-values instead of statistics.

    Returns:
        Dict of feature_name → KS statistic (or p-value).
    """
    if feature_names is None:
        feature_names = list(X_ref.columns)

    results = {}
    for fname in feature_names:
        ref_vals = X_ref[fname].dropna().values.astype(float)
        mon_vals = X_mon[fname].dropna().values.astype(float)

        if len(ref_vals) == 0 or len(mon_vals) == 0:
            results[fname] = 0.0
            continue

        ks_stat, p_value = stats.ks_2samp(ref_vals, mon_vals)
        results[fname] = float(p_value if return_pvalues else ks_stat)

    return results


# ---------------------------------------------------------------------------
# Raw Wasserstein (on original features, no SHAP normalization)
# ---------------------------------------------------------------------------


def compute_raw_wasserstein(
    X_ref: pd.DataFrame,
    X_mon: pd.DataFrame,
    feature_names: list[str] | None = None,
    order: int = 1,
) -> dict[str, float]:
    """Compute Wasserstein distance on original (non-SHAP-transformed) features.

    This is the ablation baseline: Wasserstein without SHAP normalization.

    Args:
        X_ref: Reference DataFrame.
        X_mon: Monitoring DataFrame.
        feature_names: Features to compute. Defaults to all columns.
        order: Wasserstein order (1 or 2).

    Returns:
        Dict of feature_name → raw Wasserstein distance.
    """
    if feature_names is None:
        feature_names = list(X_ref.columns)

    results = {}
    for fname in feature_names:
        ref_vals = X_ref[fname].dropna().values.astype(float)
        mon_vals = X_mon[fname].dropna().values.astype(float)

        if len(ref_vals) == 0 or len(mon_vals) == 0:
            results[fname] = 0.0
            continue

        if order == 1:
            w = stats.wasserstein_distance(ref_vals, mon_vals)
        else:
            # W2: use energy_distance / 2 approximation or compute from sorted quantiles
            ref_sorted = np.sort(ref_vals)
            mon_sorted = np.sort(mon_vals)
            # Interpolate to common grid for unequal sizes
            n = max(len(ref_sorted), len(mon_sorted))
            ref_q = np.interp(
                np.linspace(0, 1, n), np.linspace(0, 1, len(ref_sorted)), ref_sorted,
            )
            mon_q = np.interp(
                np.linspace(0, 1, n), np.linspace(0, 1, len(mon_sorted)), mon_sorted,
            )
            w = np.sqrt(np.mean((ref_q - mon_q) ** 2))

        results[fname] = float(w)

    return results


# ---------------------------------------------------------------------------
# MMD (Maximum Mean Discrepancy)
# ---------------------------------------------------------------------------


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """Compute RBF kernel matrix between X and Y.

    K(x, y) = exp(-gamma * ||x - y||²)
    """
    # Use squared Euclidean distances
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    distances = XX + YY.T - 2.0 * X @ Y.T
    return np.exp(-gamma * distances)


def _subsample(
    arr: np.ndarray,
    max_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Subsample array to max_n rows if it exceeds that size."""
    if len(arr) <= max_n:
        return arr
    idx = rng.choice(len(arr), size=max_n, replace=False)
    return arr[idx]


def compute_mmd(
    X_ref: pd.DataFrame,
    X_mon: pd.DataFrame,
    feature_names: list[str] | None = None,
    gamma: float | None = None,
    per_feature: bool = True,
    max_samples: int = 2000,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute MMD (Maximum Mean Discrepancy) with RBF kernel.

    MMD²(P, Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]

    For large datasets, subsamples to max_samples to keep O(n²) kernel
    computation tractable.

    Args:
        X_ref: Reference DataFrame.
        X_mon: Monitoring DataFrame.
        feature_names: Features to compute. Defaults to all columns.
        gamma: RBF kernel bandwidth. If None, uses median heuristic.
        per_feature: If True, compute MMD per feature; if False, multivariate.
        max_samples: Maximum number of samples per distribution for kernel
            computation. Larger datasets are subsampled. Default 2000.
        random_state: Random seed for subsampling.

    Returns:
        Dict of feature_name → MMD² score.
    """
    rng = np.random.default_rng(random_state)

    if feature_names is None:
        feature_names = list(X_ref.columns)

    if not per_feature:
        # Multivariate MMD on all features
        ref_vals = X_ref[feature_names].dropna().values.astype(float)
        mon_vals = X_mon[feature_names].dropna().values.astype(float)

        ref_vals = _subsample(ref_vals, max_samples, rng)
        mon_vals = _subsample(mon_vals, max_samples, rng)

        if gamma is None:
            # Median heuristic
            combined = np.vstack([ref_vals[:500], mon_vals[:500]])
            dists = np.sum((combined[:, None] - combined[None, :]) ** 2, axis=-1)
            median_dist = np.median(dists[dists > 0])
            gamma = 1.0 / median_dist if median_dist > 0 else 1.0

        mmd2 = _mmd2_unbiased(ref_vals, mon_vals, gamma)
        return {"_multivariate": float(mmd2)}

    # Per-feature MMD
    results = {}
    for fname in feature_names:
        ref_vals = X_ref[fname].dropna().values.astype(float).reshape(-1, 1)
        mon_vals = X_mon[fname].dropna().values.astype(float).reshape(-1, 1)

        if len(ref_vals) == 0 or len(mon_vals) == 0:
            results[fname] = 0.0
            continue

        # Subsample for speed (O(n²) kernel computation)
        ref_vals = _subsample(ref_vals, max_samples, rng)
        mon_vals = _subsample(mon_vals, max_samples, rng)

        if gamma is None:
            # Median heuristic per feature
            combined = np.vstack([ref_vals[:500], mon_vals[:500]])
            dists = (combined[:, None] - combined[None, :]) ** 2
            dists = dists.squeeze()
            median_dist = np.median(dists[dists > 0])
            g = 1.0 / median_dist if median_dist > 0 else 1.0
        else:
            g = gamma

        mmd2 = _mmd2_unbiased(ref_vals, mon_vals, g)
        results[fname] = float(max(0.0, mmd2))  # Unbiased can be slightly negative

    return results


def _mmd2_unbiased(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float,
) -> float:
    """Compute unbiased MMD² estimate.

    Uses the U-statistic estimator:
    MMD²_u = 1/(m(m-1)) Σ_{i≠j} k(x_i, x_j)
           - 2/(mn) Σ_{i,j} k(x_i, y_j)
           + 1/(n(n-1)) Σ_{i≠j} k(y_i, y_j)
    """
    m, n = len(X), len(Y)
    if m < 2 or n < 2:
        return 0.0

    Kxx = _rbf_kernel(X, X, gamma)
    Kyy = _rbf_kernel(Y, Y, gamma)
    Kxy = _rbf_kernel(X, Y, gamma)

    # Remove diagonal for unbiased estimate
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    mmd2 = (
        Kxx.sum() / (m * (m - 1))
        - 2 * Kxy.sum() / (m * n)
        + Kyy.sum() / (n * (n - 1))
    )
    return float(mmd2)


# ---------------------------------------------------------------------------
# BBSD (Black-Box Shift Detection — Rabanser et al., 2019)
# ---------------------------------------------------------------------------


def compute_bbsd(
    X_ref: pd.DataFrame,
    X_mon: pd.DataFrame,
    model: object,
    feature_names: list[str] | None = None,
    test: str = "ks",
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> dict[str, float]:
    """Compute BBSD: two-sample test on model prediction distributions.

    Black-Box Shift Detection (Rabanser et al., 2019) tests whether
    the model's output distribution has shifted between reference and
    monitoring data.  Unlike per-feature tests, this is a *model-level*
    indicator — if the score distribution has changed, the shift is
    likely decision-relevant.

    Two test variants are supported:

    * ``"ks"`` — Kolmogorov-Smirnov two-sample statistic on predicted
      probabilities.  Returns the KS statistic (higher = more shift).
    * ``"psi"`` — PSI on predicted probability bins (score-PSI).
      Uses equal-frequency bins on the reference predictions.

    Because BBSD is inherently a model-level (not per-feature) test,
    the result dict contains a single key ``"_model_output"`` with the
    aggregate score.

    Args:
        X_ref: Reference DataFrame.
        X_mon: Monitoring DataFrame.
        model: Trained model with a ``.predict()`` method that returns
            predicted probabilities (e.g., LightGBM Booster).
        feature_names: Feature names to use for prediction.
            Defaults to X_ref.columns.
        test: Two-sample test to apply on predictions.
            ``"ks"`` for KS statistic (default), ``"psi"`` for score-PSI.
        n_bins: Number of bins for PSI variant (ignored for ``"ks"``).
        epsilon: Smoothing constant for PSI variant (ignored for ``"ks"``).

    Returns:
        Dict with key ``"_model_output"`` → BBSD score.

    Raises:
        ValueError: If ``test`` is not ``"ks"`` or ``"psi"``.
    """
    if feature_names is None:
        feature_names = list(X_ref.columns)

    # Obtain predicted probabilities from the model
    ref_preds = np.asarray(model.predict(X_ref[feature_names]), dtype=np.float64)
    mon_preds = np.asarray(model.predict(X_mon[feature_names]), dtype=np.float64)

    # Handle multi-output models (e.g., multiclass softmax)
    # For binary classifiers, predict() returns a 1-D array of P(y=1).
    if ref_preds.ndim > 1:
        # Multiclass: take the max-class probability per observation
        ref_preds = ref_preds.max(axis=1)
        mon_preds = mon_preds.max(axis=1)

    if test == "ks":
        ks_stat, _ = stats.ks_2samp(ref_preds, mon_preds)
        score = float(ks_stat)
    elif test == "psi":
        score = _compute_psi_1d(ref_preds, mon_preds, n_bins=n_bins, epsilon=epsilon)
    else:
        raise ValueError(
            f"BBSD test must be 'ks' or 'psi', got '{test}'"
        )

    return {"_model_output": score}


# ---------------------------------------------------------------------------
# SWIFT ablation: PSI on model-aware buckets
# ---------------------------------------------------------------------------


def compute_psi_on_model_buckets(
    X_ref: pd.DataFrame,
    X_mon: pd.DataFrame,
    bucket_sets: dict,
    epsilon: float = 1e-4,
) -> dict[str, float]:
    """Compute PSI using model-aware buckets instead of equal-frequency bins.

    Ablation A3: isolates the contribution of Wasserstein vs KL divergence
    by using model-aware buckets (same as SWIFT) but PSI formula instead of W₁.

    Args:
        X_ref: Reference DataFrame.
        X_mon: Monitoring DataFrame.
        bucket_sets: Dict of feature_name → BucketSet (from SWIFT fit).
        epsilon: Smoothing constant.

    Returns:
        Dict of feature_name → PSI score on model-aware buckets.
    """
    results = {}
    for fname, bs in bucket_sets.items():
        n_buckets = bs.num_buckets

        # Vectorized bucket assignment using searchsorted
        ref_buckets = _vectorized_assign_buckets(X_ref[fname].values, bs)
        mon_buckets = _vectorized_assign_buckets(X_mon[fname].values, bs)

        # Count proportions per bucket
        ref_counts = np.zeros(n_buckets)
        mon_counts = np.zeros(n_buckets)
        for b in range(n_buckets):
            ref_counts[b] = (ref_buckets == b).sum()
            mon_counts[b] = (mon_buckets == b).sum()

        ref_props = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * n_buckets)
        mon_props = (mon_counts + epsilon) / (mon_counts.sum() + epsilon * n_buckets)

        psi = np.sum((mon_props - ref_props) * np.log(mon_props / ref_props))
        results[fname] = float(psi)

    return results


def _vectorized_assign_buckets(
    values: np.ndarray,
    bucket_set: object,
) -> np.ndarray:
    """Assign bucket indices to all values using vectorized operations.

    Uses the same logic as the vectorized transform_feature: for numeric
    features, uses np.searchsorted on decision points; NaN values map to
    the null bucket (index 0).

    Args:
        values: 1-D array of feature values (may contain NaN).
        bucket_set: BucketSet object with decision_points attribute.

    Returns:
        1-D integer array of bucket indices.
    """
    from swift.types import BucketType

    values_arr = np.asarray(values, dtype=np.float64)
    n = len(values_arr)

    # Check for categorical buckets — fall back to element-wise
    for b in bucket_set.buckets:
        if b.bucket_type == BucketType.CATEGORICAL:
            return np.array([bucket_set.assign_bucket(
                None if np.isnan(v) else float(v)
            ) for v in values_arr])

    nan_mask = np.isnan(values_arr)
    dp = bucket_set.decision_points

    if len(dp) > 0:
        bucket_indices = np.searchsorted(dp, values_arr, side="right") + 1
    else:
        bucket_indices = np.ones(n, dtype=np.intp)

    # NaN → null bucket (index 0)
    bucket_indices[nan_mask] = 0

    max_idx = max(b.index for b in bucket_set.buckets)
    np.clip(bucket_indices, 0, max_idx, out=bucket_indices)

    return bucket_indices


# ---------------------------------------------------------------------------
# Decker et al. (2024) — KS test on SHAP value distributions
# ---------------------------------------------------------------------------


def compute_decker(
    shap_values_ref: np.ndarray,
    shap_values_mon: np.ndarray,
    feature_names: list[str],
    return_pvalues: bool = False,
) -> dict[str, float]:
    """Compute Decker et al. drift test: KS on per-feature SHAP distributions.

    Decker et al. (2024) propose testing for distribution shift by comparing
    the SHAP value distributions between reference and monitoring data on a
    per-feature basis.  The intuition is that if the feature's contribution
    to the model changes, the SHAP value distribution will shift — even if
    the raw feature distribution does not.

    For each feature j, we apply a two-sample KS test to
    SHAP_ref[:, j] vs SHAP_mon[:, j].

    References:
        Decker, T., Gross, R., Lebacher, M., & Pfahlberg, A. (2024).
        A SHAP-based approach for model monitoring.

    Args:
        shap_values_ref: SHAP values for reference data (n_ref x p).
        shap_values_mon: SHAP values for monitoring data (n_mon x p).
        feature_names: Feature names corresponding to columns.
        return_pvalues: If True, return p-values instead of KS statistics.

    Returns:
        Dict of feature_name -> KS statistic (or p-value).
    """
    n_features = len(feature_names)
    if shap_values_ref.shape[1] != n_features:
        raise ValueError(
            f"shap_values_ref has {shap_values_ref.shape[1]} columns, "
            f"expected {n_features}"
        )
    if shap_values_mon.shape[1] != n_features:
        raise ValueError(
            f"shap_values_mon has {shap_values_mon.shape[1]} columns, "
            f"expected {n_features}"
        )

    results: dict[str, float] = {}
    for j, fname in enumerate(feature_names):
        ref_shap = shap_values_ref[:, j]
        mon_shap = shap_values_mon[:, j]

        # Drop NaN (shouldn't happen for SHAP, but be safe)
        ref_clean = ref_shap[~np.isnan(ref_shap)]
        mon_clean = mon_shap[~np.isnan(mon_shap)]

        if len(ref_clean) == 0 or len(mon_clean) == 0:
            results[fname] = 0.0
            continue

        ks_stat, p_value = stats.ks_2samp(ref_clean, mon_clean)
        results[fname] = float(p_value if return_pvalues else ks_stat)

    return results


# ---------------------------------------------------------------------------
# Convenience: run all baselines
# ---------------------------------------------------------------------------


def run_all_baselines(
    X_ref: pd.DataFrame,
    X_mon: pd.DataFrame,
    feature_names: list[str],
    shap_values: Optional[np.ndarray] = None,
    shap_values_mon: Optional[np.ndarray] = None,
    bucket_sets: Optional[dict] = None,
    model: Optional[object] = None,
    n_bins: int = 10,
) -> dict[str, dict[str, float]]:
    """Run all baseline methods and return results.

    Args:
        X_ref: Reference DataFrame.
        X_mon: Monitoring DataFrame.
        feature_names: Feature names to compute.
        shap_values: Reference SHAP values (needed for SSI and Decker).
        shap_values_mon: Monitoring SHAP values (needed for Decker).
        bucket_sets: SWIFT bucket sets (needed for PSI-on-model-buckets ablation).
        model: Trained model with ``.predict()`` (needed for BBSD).
        n_bins: Number of bins for PSI.

    Returns:
        Dict of method_name → Dict of feature_name → score.
    """
    results = {}

    logger.info("Computing PSI...")
    results["PSI"] = compute_psi(X_ref, X_mon, feature_names, n_bins=n_bins)

    logger.info("Computing KS...")
    results["KS"] = compute_ks(X_ref, X_mon, feature_names)

    logger.info("Computing Raw W₁...")
    results["Raw_W1"] = compute_raw_wasserstein(X_ref, X_mon, feature_names, order=1)

    logger.info("Computing MMD...")
    results["MMD"] = compute_mmd(X_ref, X_mon, feature_names, per_feature=True)

    if shap_values is not None:
        logger.info("Computing SSI...")
        results["SSI"] = compute_ssi(
            X_ref, X_mon, shap_values, feature_names, n_bins=n_bins,
        )

    if bucket_sets is not None:
        logger.info("Computing PSI on model-aware buckets...")
        results["PSI_model_buckets"] = compute_psi_on_model_buckets(
            X_ref, X_mon, bucket_sets,
        )

    if model is not None:
        logger.info("Computing BBSD (KS on model predictions)...")
        results["BBSD_KS"] = compute_bbsd(
            X_ref, X_mon, model, feature_names, test="ks",
        )
        logger.info("Computing BBSD (PSI on model predictions)...")
        results["BBSD_PSI"] = compute_bbsd(
            X_ref, X_mon, model, feature_names, test="psi",
            n_bins=n_bins,
        )

    if shap_values is not None and shap_values_mon is not None:
        logger.info("Computing Decker (KS on SHAP distributions)...")
        results["Decker_KS"] = compute_decker(
            shap_values, shap_values_mon, feature_names,
        )

    return results
