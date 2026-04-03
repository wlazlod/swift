"""Stage 5: Threshold calibration — permutation test, bootstrap, multiple testing correction.

Provides three functions:
    permutation_test:   Per-feature p-values via permutation test.
    bootstrap_threshold: Per-feature absolute thresholds via bootstrap.
    correct_pvalues:    Multiple testing correction (Bonferroni / BH).

Key invariant: the SHAP transformation σ_j is computed ONCE from D_ref
(Stage 3) and reused for every permutation / bootstrap draw.  SHAP is
never recomputed inside the permutation loop.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from swift.distance import compute_swift_scores, wasserstein_1d
from swift.normalization import transform_feature
from swift.types import BucketSet, CorrectionMethod

logger = logging.getLogger(__name__)


def permutation_test(
    X_ref: pd.DataFrame,
    X_mon: pd.DataFrame,
    bucket_sets: dict[str, BucketSet],
    order: int = 1,
    n_permutations: int = 1000,
    max_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Compute per-feature p-values via permutation test.

    Under H₀ (no drift), reference and monitoring samples are
    exchangeable.  We pool them, draw random splits, and compute
    SWIFT scores under the null.

    The pre-computed SHAP transformation σ_j (stored in bucket_sets)
    is applied identically to every permutation — it is NOT recomputed.

    The p-value uses the conservative formula:
        p_j = (1 + #{b : SWIFT_j^(b) ≥ SWIFT_j^obs}) / (1 + B)

    When ``max_samples`` is set and the pooled data exceeds it, both
    reference and monitoring data are randomly subsampled (preserving
    the ref/mon ratio) before running the permutation loop.  This
    provides a significant speedup on large datasets with negligible
    impact on statistical power.

    Args:
        X_ref: Reference DataFrame (n_ref × p).
        X_mon: Monitoring DataFrame (n_mon × p).
        bucket_sets: Dict of feature → BucketSet with mean_shap populated.
        order: Wasserstein order (1 or 2).
        n_permutations: Number of permutations B.
        max_samples: Maximum total pool size.  If the pool (n_ref + n_mon)
            exceeds this, subsample proportionally.  None = no limit.
        rng: Random number generator for reproducibility.

    Returns:
        Dict of feature_name → p-value ∈ (0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_ref = len(X_ref)
    n_mon = len(X_mon)
    n_pool = n_ref + n_mon

    # ── Optional subsampling for performance ─────────────────────────
    if max_samples is not None and n_pool > max_samples:
        ref_ratio = n_ref / n_pool
        n_ref_sub = max(1, int(max_samples * ref_ratio))
        n_mon_sub = max(1, max_samples - n_ref_sub)

        ref_idx = rng.choice(n_ref, size=n_ref_sub, replace=False)
        mon_idx = rng.choice(n_mon, size=n_mon_sub, replace=False)

        X_ref = X_ref.iloc[ref_idx].reset_index(drop=True)
        X_mon = X_mon.iloc[mon_idx].reset_index(drop=True)

        logger.info(
            "Permutation test: subsampled pool from %d to %d "
            "(ref: %d→%d, mon: %d→%d)",
            n_pool, n_ref_sub + n_mon_sub,
            n_ref, n_ref_sub, n_mon, n_mon_sub,
        )

        n_ref = n_ref_sub
        n_mon = n_mon_sub
        n_pool = n_ref + n_mon

    feature_names = list(bucket_sets.keys())

    # ── Observed SWIFT scores ──────────────────────────────────────────
    observed_scores = compute_swift_scores(X_ref, X_mon, bucket_sets, order=order)

    # ── Pre-transform the pooled data per feature ─────────────────────
    # This avoids redundant transform_feature calls inside the loop.
    pool_transformed: dict[str, np.ndarray] = {}
    for fname, bs in bucket_sets.items():
        pool_vals = pd.concat(
            [X_ref[fname], X_mon[fname]], ignore_index=True
        )
        pool_transformed[fname] = transform_feature(pool_vals, bs)

    # ── Permutation loop ──────────────────────────────────────────────
    count_ge: dict[str, int] = {fname: 0 for fname in feature_names}

    for b in range(n_permutations):
        # Random split of pooled indices
        perm = rng.permutation(n_pool)
        idx_ref = perm[:n_ref]
        idx_mon = perm[n_ref:]

        for fname in feature_names:
            ref_t = pool_transformed[fname][idx_ref]
            mon_t = pool_transformed[fname][idx_mon]
            perm_score = wasserstein_1d(ref_t, mon_t, order=order)

            if perm_score >= observed_scores[fname]:
                count_ge[fname] += 1

    # ── p-values: (1 + count) / (1 + B) ──────────────────────────────
    pvalues: dict[str, float] = {}
    for fname in feature_names:
        pvalues[fname] = (1 + count_ge[fname]) / (1 + n_permutations)

    logger.info(
        "Permutation test (B=%d, order=%d): p-values = %s",
        n_permutations,
        order,
        {k: f"{v:.4f}" for k, v in pvalues.items()},
    )

    return pvalues


def bootstrap_threshold(
    X_ref: pd.DataFrame,
    bucket_sets: dict[str, BucketSet],
    n_mon: int,
    order: int = 1,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Compute per-feature absolute thresholds via bootstrap.

    Draws bootstrap samples of size n_mon from X_ref, computes SWIFT
    scores against X_ref, and returns the (1 − α) quantile as the
    threshold for each feature.

    This provides a principled alternative to PSI's ad-hoc 0.10 / 0.25.

    Args:
        X_ref: Reference DataFrame (n_ref × p).
        bucket_sets: Dict of feature → BucketSet with mean_shap populated.
        n_mon: Size of monitoring sample (bootstrap sample size).
        order: Wasserstein order (1 or 2).
        alpha: Significance level (threshold is (1-α) quantile).
        n_bootstrap: Number of bootstrap iterations B.
        rng: Random number generator for reproducibility.

    Returns:
        Dict of feature_name → threshold (non-negative float).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_ref = len(X_ref)
    feature_names = list(bucket_sets.keys())

    # Pre-transform reference data
    ref_transformed: dict[str, np.ndarray] = {}
    for fname, bs in bucket_sets.items():
        ref_transformed[fname] = transform_feature(X_ref[fname], bs)

    # Bootstrap loop
    boot_scores: dict[str, list[float]] = {fname: [] for fname in feature_names}

    for b in range(n_bootstrap):
        # Draw bootstrap sample indices from reference
        boot_idx = rng.integers(0, n_ref, size=n_mon)

        for fname in feature_names:
            boot_t = ref_transformed[fname][boot_idx]
            score = wasserstein_1d(ref_transformed[fname], boot_t, order=order)
            boot_scores[fname].append(score)

    # Threshold at (1 - alpha) quantile
    quantile = 1.0 - alpha
    thresholds: dict[str, float] = {}
    for fname in feature_names:
        thresholds[fname] = float(np.quantile(boot_scores[fname], quantile))

    logger.info(
        "Bootstrap thresholds (B=%d, α=%.3f, order=%d): %s",
        n_bootstrap,
        alpha,
        order,
        {k: f"{v:.6f}" for k, v in thresholds.items()},
    )

    return thresholds


def correct_pvalues(
    pvalues: dict[str, float],
    method: CorrectionMethod,
    alpha: float = 0.05,
) -> dict[str, bool]:
    """Apply multiple testing correction and return rejection decisions.

    Bonferroni: reject if p_j < α / p  (controls FWER).
    Benjamini-Hochberg: sort p-values, find largest k s.t.
        p_(k) ≤ k·α / p, then reject all with rank ≤ k  (controls FDR).

    Uses strict inequality (p < threshold) for both methods.

    Args:
        pvalues: Dict of feature_name → p-value.
        method: CorrectionMethod.BONFERRONI or CorrectionMethod.BH.
        alpha: Significance level.

    Returns:
        Dict of feature_name → bool (True = reject H₀ / flagged as drifted).
    """
    p = len(pvalues)
    if p == 0:
        return {}

    if method == CorrectionMethod.BONFERRONI:
        threshold = alpha / p
        return {fname: pval < threshold for fname, pval in pvalues.items()}

    if method == CorrectionMethod.BH:
        # Sort p-values ascending
        sorted_items = sorted(pvalues.items(), key=lambda x: x[1])
        sorted_names = [name for name, _ in sorted_items]
        sorted_pvals = [pval for _, pval in sorted_items]

        # Find largest k such that p_(k) <= k * alpha / p
        max_k = 0
        for k_idx in range(p):
            rank = k_idx + 1  # 1-indexed
            bh_threshold = rank * alpha / p
            if sorted_pvals[k_idx] <= bh_threshold:
                max_k = rank

        # Reject all features with rank <= max_k
        rejected_names = set(sorted_names[:max_k])
        return {fname: fname in rejected_names for fname in pvalues}

    raise ValueError(f"Unknown correction method: {method}")
