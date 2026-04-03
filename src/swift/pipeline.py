"""End-to-end SWIFT pipeline orchestrator.

Provides SWIFTMonitor — the user-facing class that ties all 5 stages
together into a scikit-learn-style fit / transform / score / test API.

Typical usage::

    monitor = SWIFTMonitor(model=lgb_model, n_permutations=200)
    monitor.fit(X_ref)

    # SHAP-transformed reference (same shape as X_ref)
    X_transformed = monitor.transform(X_ref)

    # Quick scores (no significance testing)
    scores = monitor.score(X_mon)

    # Full pipeline with permutation test + MTC
    result = monitor.test(X_mon)
    print(result.drifted_features)
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from swift.aggregation import aggregate_scores, compute_importance_weights
from swift.bucketing import build_all_buckets
from swift.distance import compute_swift_scores
from swift.extraction import extract_decision_points
from swift.normalization import compute_bucket_shap, transform_feature
from swift.plotting import plot_bucket_profile, plot_feature_swift_scores
from swift.threshold import correct_pvalues, permutation_test
from swift.types import (
    BucketSet,
    CorrectionMethod,
    FeatureSWIFTResult,
    SWIFTResult,
    WassersteinOrder,
)

logger = logging.getLogger(__name__)


class SWIFTMonitor(BaseEstimator, TransformerMixin):
    """SHAP-Weighted Impact Feature Testing monitor.

    Orchestrates the 5-stage SWIFT pipeline:

        1. Extract decision points from trained model
        2. Build buckets from decision points
        3. Compute bucket-level mean SHAP (SHAP normalization σ_j)
        4. Compute Wasserstein distance on SHAP-transformed distributions
        5. Permutation-based significance testing with MTC

    The transformation σ_j is computed ONCE during ``fit()`` and applied
    identically to all monitoring samples via ``transform()``.

    Parameters
    ----------
    model : object
        Trained tree-ensemble model (LightGBM ``Booster`` or XGBoost
        ``Booster``).  Required — passed as a constructor dependency
        (analogous to ``sklearn.feature_selection.SelectFromModel``).
    order : int, default=1
        Wasserstein order (1 → W₁, 2 → W₂).
    n_permutations : int, default=1000
        Number of permutations for p-value estimation in ``test()``.
    alpha : float, default=0.05
        Significance level for multiple testing correction.
    correction : CorrectionMethod or str, default="benjamini-hochberg"
        Multiple testing correction method.  Accepts enum members or
        strings (``"bonferroni"``, ``"benjamini-hochberg"``, ``"bh"``).
    n_synthetic : int, default=10
        Number of synthetic observations to create for empty buckets
        during ``fit()``.
    max_samples : int or None, default=None
        Maximum total pool size (n_ref + n_mon) for the permutation test.
        If exceeded, subsample proportionally.  ``None`` = no limit.
    random_state : int, default=42
        Seed for the random number generator, ensuring reproducibility.

    Attributes
    ----------
    bucket_sets_ : dict[str, BucketSet]
        Per-feature bucket sets with ``mean_shap`` populated (set by ``fit``).
    X_ref_ : pd.DataFrame
        Copy of the reference DataFrame (stored for permutation testing).
    shap_values_ : np.ndarray
        SHAP values computed on the reference data.
    feature_names_in_ : np.ndarray
        Feature names inferred from ``X.columns`` during ``fit``.
    n_features_in_ : int
        Number of features seen during ``fit``.

    Examples
    --------
    >>> monitor = SWIFTMonitor(model=lgb_model, n_permutations=200)
    >>> monitor.fit(X_ref)
    SWIFTMonitor(...)
    >>> result = monitor.test(X_mon)
    >>> result.drifted_features
    ('feature_3',)
    """

    def __init__(
        self,
        model: object = None,
        order: int = 1,
        n_permutations: int = 1000,
        alpha: float = 0.05,
        correction: Union[CorrectionMethod, str] = CorrectionMethod.BH,
        n_synthetic: int = 10,
        max_samples: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.order = order
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.correction = correction
        self.n_synthetic = n_synthetic
        self.max_samples = max_samples
        self.random_state = random_state

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: None = None,
    ) -> SWIFTMonitor:
        """Fit the SWIFT monitor on reference data.

        Executes stages 1–3: extraction → bucketing → SHAP normalization.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_ref, n_features)
            Reference DataFrame.  Feature names are inferred from
            ``X.columns``.
        y : ignored
            Not used; present for API compatibility.

        Returns
        -------
        self
        """
        if self.model is None:
            raise ValueError(
                "SWIFTMonitor requires a trained model.  "
                "Pass it via the constructor: SWIFTMonitor(model=my_model)."
            )

        rng = np.random.default_rng(self.random_state)

        # Infer feature names (sklearn convention).
        self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        feature_names = list(self.feature_names_in_)

        self.X_ref_ = X.copy()

        # Stage 1: Extract decision points
        logger.info("Stage 1: Extracting decision points...")
        decision_points = extract_decision_points(self.model, feature_names)

        # Stage 2: Build buckets
        logger.info("Stage 2: Building buckets...")
        bucket_sets = build_all_buckets(decision_points)

        # Compute SHAP values on reference data
        logger.info("Computing SHAP values on reference data...")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        shap_values = np.asarray(shap_values)
        self.shap_values_ = shap_values

        # Stage 3: SHAP normalization
        logger.info("Stage 3: Computing bucket-level mean SHAP...")
        self.bucket_sets_ = compute_bucket_shap(
            bucket_sets,
            X,
            shap_values,
            model=self.model,
            n_synthetic=self.n_synthetic,
            rng=rng,
        )

        n_buckets_total = sum(
            bs.num_buckets for bs in self.bucket_sets_.values()
        )
        logger.info(
            "SWIFT monitor fitted: %d features, %d total buckets.",
            len(feature_names),
            n_buckets_total,
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the SHAP transformation σ_j to every feature.

        Each value ``x_ij`` is mapped to the mean SHAP of its bucket:
        ``σ_j(x_ij) = mean_shap_j^{bucket(x_ij)}``.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        pd.DataFrame
            SHAP-transformed DataFrame (same shape and column names).
        """
        check_is_fitted(self)
        result = pd.DataFrame(index=X.index)
        for fname in self.feature_names_in_:
            result[fname] = transform_feature(
                X[fname].values, self.bucket_sets_[fname]
            )
        return result

    def score(
        self,
        X: pd.DataFrame,
        X_compare: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute per-feature SWIFT scores (stage 4 only, no testing).

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Monitoring DataFrame.  When *X_compare* is ``None``, this is
            compared against the fitted reference ``X_ref_``.
        X_compare : pd.DataFrame or None
            Optional second sample.  When provided, SWIFT scores are
            computed between *X* and *X_compare* instead of between
            ``X_ref_`` and *X*.  The SHAP transformation σ_j is always
            the one fitted on ``X_ref_``.

        Returns
        -------
        dict[str, float]
            Feature name → SWIFT score (Wasserstein distance on
            SHAP-transformed distributions).
        """
        check_is_fitted(self)

        if X_compare is not None:
            return compute_swift_scores(
                X, X_compare, self.bucket_sets_, order=self.order
            )

        return compute_swift_scores(
            self.X_ref_, X, self.bucket_sets_, order=self.order
        )

    def test(
        self,
        X: pd.DataFrame,
        X_compare: pd.DataFrame | None = None,
    ) -> SWIFTResult:
        """Run the full SWIFT pipeline: score + test + aggregate.

        Stages 4–5 + aggregation:

        - Compute per-feature SWIFT scores (Wasserstein on SHAP-transformed).
        - Permutation test for p-values.
        - Multiple testing correction.
        - Model-level aggregation.

        All hyperparameters (``order``, ``n_permutations``, ``alpha``,
        ``correction``, ``max_samples``) are taken from instance attributes
        set in the constructor.  Use ``set_params()`` to override for
        individual calls (e.g. different ``random_state`` per experiment
        repetition).

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Monitoring DataFrame.  When *X_compare* is ``None``, this is
            compared against the fitted reference ``X_ref_``.
        X_compare : pd.DataFrame or None
            Optional second sample.  When provided, the test compares
            *X* against *X_compare* instead of ``X_ref_`` against *X*.
            The SHAP transformation σ_j is always the one fitted on
            ``X_ref_``.

        Returns
        -------
        SWIFTResult
            Per-feature and model-level results.
        """
        check_is_fitted(self)

        rng = np.random.default_rng(self.random_state)
        correction = CorrectionMethod.resolve(self.correction)

        # Determine the two samples to compare
        if X_compare is not None:
            X_a, X_b = X, X_compare
        else:
            X_a, X_b = self.X_ref_, X

        # Stage 4: Compute SWIFT scores
        logger.info("Stage 4: Computing SWIFT scores...")
        scores = compute_swift_scores(
            X_a, X_b, self.bucket_sets_, order=self.order
        )

        # Stage 5: Permutation test + MTC
        logger.info(
            "Stage 5: Permutation test (B=%d)...", self.n_permutations
        )
        pvalues = permutation_test(
            X_a,
            X_b,
            self.bucket_sets_,
            order=self.order,
            n_permutations=self.n_permutations,
            max_samples=self.max_samples,
            rng=rng,
        )

        decisions = correct_pvalues(pvalues, correction, self.alpha)

        # Build per-feature results
        w_order = (
            WassersteinOrder.W1 if self.order == 1 else WassersteinOrder.W2
        )
        feature_results: list[FeatureSWIFTResult] = []
        for fname in self.feature_names_in_:
            feature_results.append(
                FeatureSWIFTResult(
                    feature_name=fname,
                    swift_score=scores[fname],
                    wasserstein_order=w_order,
                    p_value=pvalues[fname],
                    is_drifted=decisions[fname],
                    num_buckets=self.bucket_sets_[fname].num_buckets,
                )
            )

        # Aggregation
        agg = aggregate_scores(scores)

        result = SWIFTResult(
            feature_results=tuple(feature_results),
            swift_max=agg.swift_max,
            swift_mean=agg.swift_mean,
            alpha=self.alpha,
            correction_method=correction,
        )

        logger.info(
            "SWIFT test complete: %d/%d features drifted (α=%.3f, %s). "
            "SWIFT_max=%.6f, SWIFT_mean=%.6f",
            result.num_drifted,
            result.num_features,
            self.alpha,
            correction.value,
            result.swift_max,
            result.swift_mean,
        )

        return result

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_buckets(
        self,
        feature_name: str,
        X: pd.DataFrame | None = None,
        X_compare: pd.DataFrame | None = None,
        labels: tuple[str, str] = ("Reference", "Comparison"),
        figsize: tuple[float, float] = (10, 5),
        title: str | None = None,
        max_label_buckets: int = 20,
        x_axis: str = "bucket",
    ) -> tuple:
        """Plot the bucketing profile for a single feature.

        Shows mean SHAP per bucket (line + 95 % error band) on the left
        y-axis and observation density (filled line) on the right y-axis.

        Parameters
        ----------
        feature_name : str
            Feature to visualise.  Must be in ``feature_names_in_``.
        X : pd.DataFrame or None
            Sample whose density is shown as the *primary* line.  When
            ``None`` (default), the fitted reference ``X_ref_`` is used.
        X_compare : pd.DataFrame or None
            Optional second sample for density comparison.
            Must contain *feature_name* as a column.  Each sample's
            density is normalised to 1.0 independently.
        labels : tuple of str
            Legend labels ``(primary_label, comparison_label)``.
        figsize : tuple, default (10, 5)
            Figure size in inches.
        title : str or None
            Custom title.  Defaults to
            ``"Bucketing Profile: {feature_name}"``.
        max_label_buckets : int, default 20
            Use compact index labels (``B0, B1, …``) when the number
            of buckets exceeds this threshold.
        x_axis : {"bucket", "natural"}
            ``"bucket"`` (default) uses integer bucket indices.
            ``"natural"`` uses actual feature-value positions (bucket
            midpoints).

        Returns
        -------
        (Figure, Axes)
            Matplotlib figure and primary (SHAP) axes.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the monitor has not been fitted.
        ValueError
            If *feature_name* is not in ``feature_names_in_`` or
            *X_compare* / *X* do not contain the column.
        """
        check_is_fitted(self)

        feature_list = list(self.feature_names_in_)
        if feature_name not in feature_list:
            raise ValueError(
                f"Unknown feature '{feature_name}'.  "
                f"Available: {feature_list}"
            )

        feat_idx = feature_list.index(feature_name)

        # Primary density values (None → use ref)
        primary_values = None
        if X is not None:
            if feature_name not in X.columns:
                raise ValueError(
                    f"X is missing column '{feature_name}'."
                )
            primary_values = X[feature_name].to_numpy()

        # Comparison density values
        compare_values = None
        if X_compare is not None:
            if feature_name not in X_compare.columns:
                raise ValueError(
                    f"X_compare is missing column '{feature_name}'."
                )
            compare_values = X_compare[feature_name].to_numpy()

        return plot_bucket_profile(
            bucket_set=self.bucket_sets_[feature_name],
            feature_values=self.X_ref_[feature_name].to_numpy(),
            shap_values=self.shap_values_[:, feat_idx],
            compare_values=compare_values,
            primary_values=primary_values,
            labels=labels,
            figsize=figsize,
            title=title,
            max_label_buckets=max_label_buckets,
            x_axis=x_axis,
        )

    def plot_swift_scores(
        self,
        result: SWIFTResult,
        result_compare: SWIFTResult | None = None,
        labels: tuple[str, str] = ("Result A", "Result B"),
        threshold: float | None = None,
        sort_by: str = "score",
        figsize: tuple[float, float] = (12, 5),
        title: str | None = None,
    ) -> tuple:
        """Plot SWIFT scores per feature from a ``test()`` result.

        Draws one bar per feature, colored red (drifted) or blue (not
        drifted), with horizontal reference lines for ``SWIFT_max``,
        ``SWIFT_mean``, and an optional user-provided *threshold*.

        In comparison mode (when *result_compare* is given), draws
        grouped side-by-side bars with neutral coloring.

        Parameters
        ----------
        result : SWIFTResult
            Primary result from ``test()``.
        result_compare : SWIFTResult or None
            Optional second result for grouped comparison.
        labels : tuple of str
            Legend labels ``(result_label, compare_label)``.
        threshold : float or None
            Optional detection threshold drawn as a dotted black line.
        sort_by : {"score", "name", "original"}
            Feature ordering on the x-axis.  ``"original"`` preserves
            the order from ``feature_names_in_``.
        figsize : tuple, default (12, 5)
            Figure size in inches.
        title : str or None
            Custom title.

        Returns
        -------
        (Figure, Axes)

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the monitor has not been fitted.
        """
        check_is_fitted(self)
        return plot_feature_swift_scores(
            result=result,
            result_compare=result_compare,
            labels=labels,
            threshold=threshold,
            sort_by=sort_by,
            feature_order=list(self.feature_names_in_),
            figsize=figsize,
            title=title,
        )
