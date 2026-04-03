"""Evaluation metrics for drift detection experiments.

Metrics:
    - TPR @ fixed FPR (detection power)
    - AUROC of drift detector (drift present/absent classification)
    - Spearman ρ (correlation with model performance degradation)
    - False Positive Rate under null
    - Detection delay (for gradual drift)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionMetrics:
    """Aggregate detection performance metrics.

    Attributes:
        tpr_at_5fpr: True positive rate at 5% FPR.
        tpr_at_1fpr: True positive rate at 1% FPR.
        auroc: AUROC of the drift detector.
        fpr_under_null: False positive rate when no drift is present.
        n_drift_trials: Number of drift trials.
        n_null_trials: Number of null (no-drift) trials.
    """

    tpr_at_5fpr: float
    tpr_at_1fpr: float
    auroc: float
    fpr_under_null: float
    n_drift_trials: int
    n_null_trials: int


def compute_tpr_at_fpr(
    scores_drift: np.ndarray,
    scores_null: np.ndarray,
    target_fpr: float = 0.05,
) -> float:
    """Compute TPR at a given FPR level.

    The threshold is set such that FPR = target_fpr on the null scores,
    then TPR is measured on the drift scores.

    Args:
        scores_drift: Scores under drift (higher = more drift).
        scores_null: Scores under null (no drift).
        target_fpr: Target false positive rate.

    Returns:
        TPR at the given FPR level.
    """
    if len(scores_null) == 0 or len(scores_drift) == 0:
        return 0.0

    # Threshold: (1 - target_fpr) quantile of null scores
    threshold = np.quantile(scores_null, 1 - target_fpr)
    tpr = (scores_drift > threshold).mean()
    return float(tpr)


def compute_auroc(
    scores_drift: np.ndarray,
    scores_null: np.ndarray,
) -> float:
    """Compute AUROC for the drift detector.

    Treats drift detection as binary classification:
        - Positive class: drift present (scores_drift)
        - Negative class: no drift (scores_null)

    Args:
        scores_drift: Scores under drift.
        scores_null: Scores under null.

    Returns:
        AUROC value.
    """
    if len(scores_drift) == 0 or len(scores_null) == 0:
        return 0.5

    labels = np.concatenate([
        np.ones(len(scores_drift)),
        np.zeros(len(scores_null)),
    ])
    scores = np.concatenate([scores_drift, scores_null])

    # Simple AUROC via Mann-Whitney U
    # AUROC = U / (n1 * n0)
    n1 = len(scores_drift)
    n0 = len(scores_null)

    # Count pairs where drift score > null score
    u_stat = 0.0
    for s in scores_drift:
        u_stat += (s > scores_null).sum() + 0.5 * (s == scores_null).sum()

    auroc = u_stat / (n1 * n0)
    return float(auroc)


def compute_detection_metrics(
    scores_drift: np.ndarray,
    scores_null: np.ndarray,
) -> DetectionMetrics:
    """Compute comprehensive detection metrics.

    Args:
        scores_drift: Array of drift scores when drift IS present.
        scores_null: Array of drift scores when NO drift (null hypothesis).

    Returns:
        DetectionMetrics with TPR@5%FPR, TPR@1%FPR, AUROC, FPR.
    """
    tpr_5 = compute_tpr_at_fpr(scores_drift, scores_null, target_fpr=0.05)
    tpr_1 = compute_tpr_at_fpr(scores_drift, scores_null, target_fpr=0.01)
    auroc = compute_auroc(scores_drift, scores_null)

    # FPR under null at α=0.05 using the drift threshold
    # (This measures calibration: should be ≈ 0.05 for well-calibrated methods)
    fpr = 0.0  # Computed separately when we have p-values

    return DetectionMetrics(
        tpr_at_5fpr=tpr_5,
        tpr_at_1fpr=tpr_1,
        auroc=auroc,
        fpr_under_null=fpr,
        n_drift_trials=len(scores_drift),
        n_null_trials=len(scores_null),
    )


def compute_spearman_correlation(
    drift_scores: np.ndarray,
    performance_degradation: np.ndarray,
) -> tuple[float, float]:
    """Compute Spearman rank correlation between drift scores and model degradation.

    This tests the model-relevance of the drift signal: does higher drift score
    correspond to larger AUC degradation?

    Args:
        drift_scores: Array of drift scores per time window.
        performance_degradation: Array of performance degradation (e.g., ΔAUC) per window.
            Positive values = degradation (AUC went down).

    Returns:
        (rho, p_value) — Spearman correlation coefficient and its p-value.
    """
    if len(drift_scores) < 3:
        logger.warning("Too few observations for Spearman correlation: %d", len(drift_scores))
        return 0.0, 1.0

    rho, p_value = sp_stats.spearmanr(drift_scores, performance_degradation)
    return float(rho), float(p_value)


def compute_fpr_from_pvalues(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Compute empirical FPR from p-values under null hypothesis.

    Under H0, p-values should be uniform → rejection rate should ≈ α.

    Args:
        pvalues: Array of p-values (one per null trial).
        alpha: Significance level.

    Returns:
        Empirical FPR (fraction of p-values < α).
    """
    if len(pvalues) == 0:
        return 0.0
    return float((pvalues < alpha).mean())


def compute_model_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Compute AUC-ROC for a model's predictions.

    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted probabilities.

    Returns:
        AUC-ROC score.
    """
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true — AUC undefined.")
        return 0.5

    # Compute AUC via Mann-Whitney U
    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5

    u_stat = 0.0
    for s in pos_scores:
        u_stat += (s > neg_scores).sum() + 0.5 * (s == neg_scores).sum()

    auc = u_stat / (len(pos_scores) * len(neg_scores))
    return float(auc)


@dataclass(frozen=True)
class TemporalDriftResult:
    """Result of temporal drift analysis.

    Attributes:
        period_labels: Labels for each time window.
        drift_scores: Drift score per window (per method).
        model_aucs: Model AUC per window.
        auc_degradation: AUC degradation per window (ref AUC - window AUC).
        spearman_rho: Spearman ρ between drift scores and AUC degradation.
        spearman_pvalue: P-value for the Spearman test.
    """

    period_labels: list[str]
    drift_scores: dict[str, np.ndarray]  # method_name → scores per period
    model_aucs: np.ndarray
    auc_degradation: np.ndarray
    spearman_rho: dict[str, float]  # method_name → ρ
    spearman_pvalue: dict[str, float]  # method_name → p-value


def compute_temporal_drift_analysis(
    period_labels: list[str],
    drift_scores_by_method: dict[str, np.ndarray],
    model_aucs: np.ndarray,
    ref_auc: float,
) -> TemporalDriftResult:
    """Analyze temporal drift scores vs model performance degradation.

    Args:
        period_labels: Labels for each time period.
        drift_scores_by_method: Dict of method_name → array of drift scores per period.
        model_aucs: Model AUC for each period.
        ref_auc: Reference period AUC (for computing degradation).

    Returns:
        TemporalDriftResult with correlations per method.
    """
    auc_degradation = ref_auc - model_aucs  # positive = degradation

    spearman_rho = {}
    spearman_pvalue = {}

    for method_name, scores in drift_scores_by_method.items():
        rho, pval = compute_spearman_correlation(scores, auc_degradation)
        spearman_rho[method_name] = rho
        spearman_pvalue[method_name] = pval
        logger.info(
            "Spearman ρ(%s, AUC degradation) = %.3f (p=%.4f)",
            method_name, rho, pval,
        )

    return TemporalDriftResult(
        period_labels=period_labels,
        drift_scores=drift_scores_by_method,
        model_aucs=model_aucs,
        auc_degradation=auc_degradation,
        spearman_rho=spearman_rho,
        spearman_pvalue=spearman_pvalue,
    )
