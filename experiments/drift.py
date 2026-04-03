"""Drift injection framework for controlled experiments.

Implements scenarios S1-S9 from the paper design:
    S1: Mean shift (important features)
    S2: Mean shift (unimportant features)
    S3: Variance change
    S4: Covariate rotation (correlation shift)
    S5: Subpopulation shift
    S6: Category frequency shift
    S7: Null rate increase
    S8: Benign drift (virtual — shift feature but preserve P(y|X))
    S9: No drift (null hypothesis)

Each scenario function takes a DataFrame and returns a drifted copy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DriftScenario(Enum):
    """Drift scenario identifier."""

    S1_MEAN_SHIFT_IMPORTANT = "S1"
    S2_MEAN_SHIFT_UNIMPORTANT = "S2"
    S3_VARIANCE_CHANGE = "S3"
    S4_COVARIATE_ROTATION = "S4"
    S5_SUBPOPULATION_SHIFT = "S5"
    S6_CATEGORY_FREQ_SHIFT = "S6"
    S7_NULL_RATE_INCREASE = "S7"
    S8_BENIGN_DRIFT = "S8"
    S9_NO_DRIFT = "S9"
    S10_GRADUAL_DRIFT = "S10"


@dataclass(frozen=True)
class DriftConfig:
    """Configuration for a drift injection.

    Attributes:
        scenario: Which drift scenario.
        magnitude: Drift strength (meaning depends on scenario).
        target_features: Feature names to apply drift to.
            If None, auto-selected based on SHAP importance.
        n_features: Number of features to drift (if target_features is None).
        random_state: Seed for reproducibility.
    """

    scenario: DriftScenario
    magnitude: float = 1.0
    target_features: Optional[list[str]] = None
    n_features: int = 3
    random_state: int = 42


@dataclass
class DriftResult:
    """Result of drift injection.

    Attributes:
        X_drifted: DataFrame with drift injected.
        scenario: Which scenario was applied.
        drifted_features: Names of features that were actually drifted.
        magnitude: Drift magnitude applied.
        description: Human-readable description of what was done.
    """

    X_drifted: pd.DataFrame
    scenario: DriftScenario
    drifted_features: list[str]
    magnitude: float
    description: str


@dataclass(frozen=True)
class GradualDriftConfig:
    """Configuration for S10 gradual drift injection.

    Produces a time series of monitoring datasets with linearly
    increasing mean shift on important features.

    Attributes:
        n_steps: Number of monitoring periods (default 12).
        max_magnitude: Magnitude (in σ) at the final step.
        target_features: Explicit feature names to drift.
            If None, auto-selects top-N by SHAP importance.
        n_features: Number of features to drift (if target_features is None).
        random_state: Seed for reproducibility.
    """

    n_steps: int = 12
    max_magnitude: float = 3.0
    target_features: Optional[list[str]] = None
    n_features: int = 3
    random_state: int = 42


@dataclass
class GradualDriftResult:
    """Result of S10 gradual drift injection.

    Contains one DriftResult per monitoring step, plus metadata.

    Attributes:
        steps: List of DriftResult, one per step (increasing magnitude).
        n_steps: Number of steps.
        max_magnitude: Maximum magnitude at the last step.
        drifted_features: Feature names that are drifted across all steps.
    """

    steps: list[DriftResult]
    n_steps: int
    max_magnitude: float
    drifted_features: list[str]


def _select_features_by_importance(
    feature_names: list[str],
    shap_values: np.ndarray,
    n: int,
    important: bool = True,
    numeric_features: list[str] | None = None,
) -> list[str]:
    """Select top-N or bottom-N features by mean |SHAP|.

    Args:
        feature_names: All feature names.
        shap_values: SHAP values array (n_obs × n_features).
        n: Number of features to select.
        important: If True, select most important; if False, least important.
        numeric_features: If provided, restrict selection to these features.

    Returns:
        List of selected feature names.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=feature_names)

    if numeric_features is not None:
        importance = importance[importance.index.isin(numeric_features)]

    if important:
        return importance.nlargest(n).index.tolist()
    else:
        return importance.nsmallest(n).index.tolist()


def inject_drift(
    X: pd.DataFrame,
    config: DriftConfig,
    shap_values: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    y: pd.Series | None = None,
    bucket_sets: dict[str, Any] | None = None,
) -> DriftResult:
    """Inject drift into a dataset according to the specified scenario.

    Args:
        X: Original feature DataFrame (n × p).
        config: Drift configuration.
        shap_values: SHAP values for feature importance ranking (n × p).
            Required for S1, S2 when target_features is None.
        feature_names: Feature names. Defaults to X.columns.
        numeric_features: Names of numeric features. Defaults to all.
        categorical_features: Names of categorical features. Defaults to none.
        y: Target variable.
        bucket_sets: Dict of feature_name → BucketSet (needed for S8 to
            jitter within model bucket boundaries).

    Returns:
        DriftResult with drifted data and metadata.
    """
    rng = np.random.default_rng(config.random_state)
    feature_names = feature_names or list(X.columns)
    numeric_features = numeric_features or list(X.columns)
    categorical_features = categorical_features or []

    scenario_fn = _SCENARIO_DISPATCH[config.scenario]
    return scenario_fn(
        X=X.copy(),
        config=config,
        shap_values=shap_values,
        feature_names=feature_names,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        y=y,
        rng=rng,
        bucket_sets=bucket_sets,
    )


def _s1_mean_shift_important(
    X: pd.DataFrame,
    config: DriftConfig,
    shap_values: np.ndarray | None,
    feature_names: list[str],
    numeric_features: list[str],
    **kwargs,
) -> DriftResult:
    """S1: Shift mean of most important features by magnitude × σ."""
    targets = config.target_features
    if targets is None:
        if shap_values is None:
            raise ValueError("S1 requires shap_values when target_features is None")
        targets = _select_features_by_importance(
            feature_names, shap_values, config.n_features,
            important=True, numeric_features=numeric_features,
        )

    X_d = X.copy()
    for col in targets:
        if col in numeric_features:
            sigma = X_d[col].std()
            X_d[col] = X_d[col] + config.magnitude * sigma

    return DriftResult(
        X_drifted=X_d,
        scenario=config.scenario,
        drifted_features=targets,
        magnitude=config.magnitude,
        description=(
            f"S1: Mean shift of {config.magnitude}σ on top-{len(targets)} "
            f"important features: {targets}"
        ),
    )


def _s2_mean_shift_unimportant(
    X: pd.DataFrame,
    config: DriftConfig,
    shap_values: np.ndarray | None,
    feature_names: list[str],
    numeric_features: list[str],
    **kwargs,
) -> DriftResult:
    """S2: Shift mean of least important features by magnitude × σ."""
    targets = config.target_features
    if targets is None:
        if shap_values is None:
            raise ValueError("S2 requires shap_values when target_features is None")
        targets = _select_features_by_importance(
            feature_names, shap_values, config.n_features,
            important=False, numeric_features=numeric_features,
        )

    X_d = X.copy()
    for col in targets:
        if col in numeric_features:
            sigma = X_d[col].std()
            X_d[col] = X_d[col] + config.magnitude * sigma

    return DriftResult(
        X_drifted=X_d,
        scenario=config.scenario,
        drifted_features=targets,
        magnitude=config.magnitude,
        description=(
            f"S2: Mean shift of {config.magnitude}σ on bottom-{len(targets)} "
            f"unimportant features: {targets}"
        ),
    )


def _s3_variance_change(
    X: pd.DataFrame,
    config: DriftConfig,
    shap_values: np.ndarray | None,
    feature_names: list[str],
    numeric_features: list[str],
    **kwargs,
) -> DriftResult:
    """S3: Multiply variance of target features by magnitude.

    magnitude=2.0 means double the variance (scale std by sqrt(2)).
    """
    rng = kwargs.get("rng", np.random.default_rng(config.random_state))
    targets = config.target_features
    if targets is None:
        if shap_values is None:
            targets = numeric_features[: config.n_features]
        else:
            targets = _select_features_by_importance(
                feature_names, shap_values, config.n_features,
                important=True, numeric_features=numeric_features,
            )

    X_d = X.copy()
    scale_factor = np.sqrt(config.magnitude)  # magnitude = variance multiplier

    for col in targets:
        if col in numeric_features:
            mean = X_d[col].mean()
            X_d[col] = mean + (X_d[col] - mean) * scale_factor

    return DriftResult(
        X_drifted=X_d,
        scenario=config.scenario,
        drifted_features=targets,
        magnitude=config.magnitude,
        description=(
            f"S3: Variance multiplied by {config.magnitude}x "
            f"(std scaled by {scale_factor:.2f}x) on features: {targets}"
        ),
    )


def _s4_covariate_rotation(
    X: pd.DataFrame,
    config: DriftConfig,
    feature_names: list[str],
    numeric_features: list[str],
    **kwargs,
) -> DriftResult:
    """S4: Rotate two correlated features to change their correlation.

    Picks the two most correlated numeric features and applies a rotation
    matrix to them, preserving marginal distributions approximately
    but changing the joint distribution.

    magnitude controls the rotation angle: magnitude * π/4 radians.
    """
    targets = config.target_features
    if targets is None:
        # Find the two most correlated numeric features
        num_X = X[numeric_features].select_dtypes(include=[np.number])
        if num_X.shape[1] < 2:
            raise ValueError("S4 requires at least 2 numeric features")
        corr = num_X.corr().abs()
        corr_arr = corr.to_numpy().copy()
        np.fill_diagonal(corr_arr, 0)
        idx = np.unravel_index(corr_arr.argmax(), corr_arr.shape)
        targets = [corr.index[idx[0]], corr.columns[idx[1]]]

    if len(targets) != 2:
        raise ValueError(f"S4 requires exactly 2 target features, got {len(targets)}")

    X_d = X.copy()
    col_a, col_b = targets

    # Standardize
    mean_a, std_a = X_d[col_a].mean(), X_d[col_a].std()
    mean_b, std_b = X_d[col_b].mean(), X_d[col_b].std()
    z_a = (X_d[col_a] - mean_a) / std_a
    z_b = (X_d[col_b] - mean_b) / std_b

    # Rotation
    theta = config.magnitude * np.pi / 4  # magnitude=1 → 45° rotation
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    z_a_new = cos_t * z_a - sin_t * z_b
    z_b_new = sin_t * z_a + cos_t * z_b

    # De-standardize
    X_d[col_a] = z_a_new * std_a + mean_a
    X_d[col_b] = z_b_new * std_b + mean_b

    return DriftResult(
        X_drifted=X_d,
        scenario=config.scenario,
        drifted_features=targets,
        magnitude=config.magnitude,
        description=(
            f"S4: Covariate rotation of {np.degrees(theta):.1f}° "
            f"between {targets[0]} and {targets[1]}"
        ),
    )


def _s5_subpopulation_shift(
    X: pd.DataFrame,
    config: DriftConfig,
    feature_names: list[str],
    numeric_features: list[str],
    **kwargs,
) -> DriftResult:
    """S5: Add a subpopulation with shifted feature profiles.

    Replaces magnitude*100% of the data with a shifted subpopulation.
    The shifted subpopulation has all numeric features shifted by 2σ.
    Default magnitude=0.10 → replace 10% of data.
    """
    rng = kwargs.get("rng", np.random.default_rng(config.random_state))
    frac = config.magnitude  # fraction of data to replace

    n_replace = int(len(X) * frac)
    if n_replace < 1:
        n_replace = 1

    X_d = X.copy()

    # Upcast integer numeric columns to float so iloc += float works
    targets = config.target_features or numeric_features
    for col in targets:
        if col in numeric_features and X_d[col].dtype.kind == "i":
            X_d[col] = X_d[col].astype(float)

    # Sample rows to replace (cap at dataset size; allow duplicates when frac > 1)
    if n_replace > len(X_d):
        n_replace = len(X_d)
    replace_idx = rng.choice(len(X_d), size=n_replace, replace=False)

    # Shift all numeric features for the subpopulation
    for col in targets:
        if col in numeric_features:
            sigma = X_d[col].std()
            X_d.iloc[replace_idx, X_d.columns.get_loc(col)] += 2.0 * sigma

    return DriftResult(
        X_drifted=X_d,
        scenario=config.scenario,
        drifted_features=targets,
        magnitude=config.magnitude,
        description=(
            f"S5: {frac*100:.0f}% subpopulation replacement with +2σ shift "
            f"on {len(targets)} features"
        ),
    )


def _s6_category_freq_shift(
    X: pd.DataFrame,
    config: DriftConfig,
    categorical_features: list[str],
    **kwargs,
) -> DriftResult:
    """S6: Shift category frequencies of categorical features.

    For each target categorical feature, the most frequent category's
    observations are partially replaced with the least frequent category.
    magnitude controls the fraction of the most frequent category to replace.
    """
    rng = kwargs.get("rng", np.random.default_rng(config.random_state))
    targets = config.target_features
    if targets is None:
        targets = categorical_features[: config.n_features]

    if not targets:
        return DriftResult(
            X_drifted=X.copy(),
            scenario=config.scenario,
            drifted_features=[],
            magnitude=config.magnitude,
            description="S6: No categorical features to shift",
        )

    X_d = X.copy()
    for col in targets:
        if col not in X_d.columns:
            continue
        counts = X_d[col].value_counts()
        if len(counts) < 2:
            continue
        most_freq = counts.index[0]
        least_freq = counts.index[-1]

        # Replace magnitude fraction of most_freq with least_freq
        most_freq_mask = X_d[col] == most_freq
        n_available = int(most_freq_mask.sum())
        n_replace = min(int(n_available * config.magnitude), n_available)
        if n_replace < 1:
            continue
        replace_idx = rng.choice(
            X_d.index[most_freq_mask], size=n_replace, replace=False,
        )
        X_d.loc[replace_idx, col] = least_freq

    return DriftResult(
        X_drifted=X_d,
        scenario=config.scenario,
        drifted_features=targets,
        magnitude=config.magnitude,
        description=(
            f"S6: Category frequency shift on {targets} "
            f"(replace {config.magnitude*100:.0f}% of most frequent with least frequent)"
        ),
    )


def _s7_null_rate_increase(
    X: pd.DataFrame,
    config: DriftConfig,
    feature_names: list[str],
    numeric_features: list[str],
    **kwargs,
) -> DriftResult:
    """S7: Increase null rate from current level to magnitude fraction.

    magnitude = target null rate (e.g., 0.20 = 20% nulls).
    Injects NaN randomly into target features.
    """
    rng = kwargs.get("rng", np.random.default_rng(config.random_state))
    targets = config.target_features
    if targets is None:
        targets = numeric_features[: config.n_features]

    X_d = X.copy()
    for col in targets:
        current_null_rate = X_d[col].isnull().mean()
        additional_null_rate = config.magnitude - current_null_rate
        if additional_null_rate <= 0:
            continue
        non_null_mask = X_d[col].notna()
        n_to_nullify = int(non_null_mask.sum() * additional_null_rate / (1 - current_null_rate))
        if n_to_nullify < 1:
            continue
        nullify_idx = rng.choice(
            X_d.index[non_null_mask], size=min(n_to_nullify, non_null_mask.sum()),
            replace=False,
        )
        X_d.loc[nullify_idx, col] = np.nan

    return DriftResult(
        X_drifted=X_d,
        scenario=config.scenario,
        drifted_features=targets,
        magnitude=config.magnitude,
        description=(
            f"S7: Null rate increased to {config.magnitude*100:.0f}% on features: {targets}"
        ),
    )


def _s8_benign_drift(
    X: pd.DataFrame,
    config: DriftConfig,
    shap_values: np.ndarray | None,
    feature_names: list[str],
    numeric_features: list[str],
    y: pd.Series | None = None,
    **kwargs,
) -> DriftResult:
    """S8: Benign (virtual) drift — shift features but P(y|X) unchanged.

    Adds noise *within* model bucket boundaries so observations stay in the
    same decision regions.  The marginal feature distribution changes, but
    the model output is (approximately) unchanged because every observation
    maps to the same SHAP-normalised value before and after the drift.

    This is fundamentally different from S2 (mean shift on unimportant
    features) which translates the entire distribution and CAN change bucket
    assignments.

    When ``bucket_sets`` is available (passed via ``kwargs``), the per-bucket
    boundaries are used.  Otherwise falls back to within-quantile jittering
    using 20 equal-frequency bins (still keeps most observations in the same
    region, just with coarser granularity).

    ``magnitude`` controls the jitter strength as a fraction of each bucket's
    width (e.g., 0.5 → uniform noise in ±0.25 × bucket_width).
    """
    rng = kwargs.get("rng", np.random.default_rng(config.random_state))
    bucket_sets = kwargs.get("bucket_sets")

    targets = config.target_features
    if targets is None:
        # Default: all numeric features (drift every feature, but benignly)
        targets = list(numeric_features)

    X_d = X.copy()
    actually_drifted: list[str] = []

    for col in targets:
        if col not in numeric_features:
            continue

        values = X_d[col].values.astype(float).copy()
        non_null_mask = ~np.isnan(values)
        if non_null_mask.sum() == 0:
            continue

        # Determine bucket edges for this feature
        edges: np.ndarray | None = None
        if bucket_sets is not None and col in bucket_sets:
            bs = bucket_sets[col]
            if len(bs.decision_points) > 0:
                edges = np.sort(bs.decision_points)

        if edges is None or len(edges) == 0:
            # Fallback: 20 equal-frequency bins
            non_null_vals = values[non_null_mask]
            edges = np.unique(np.percentile(non_null_vals, np.linspace(0, 100, 21)))
            if len(edges) <= 1:
                continue  # constant feature — skip
            # Remove the endpoints; use inner edges
            edges = edges[1:-1]

        # For each non-null value, find its bucket and jitter within bounds
        non_null_vals = values[non_null_mask]
        bucket_idx = np.searchsorted(edges, non_null_vals, side="right")

        lower = np.full_like(non_null_vals, -np.inf)
        upper = np.full_like(non_null_vals, np.inf)

        mask_not_first = bucket_idx > 0
        lower[mask_not_first] = edges[np.clip(bucket_idx[mask_not_first] - 1, 0, len(edges) - 1)]
        mask_not_last = bucket_idx < len(edges)
        upper[mask_not_last] = edges[np.clip(bucket_idx[mask_not_last], 0, len(edges) - 1)]

        # Compute bucket width; for unbounded buckets, use the global std
        sigma = np.nanstd(non_null_vals)
        width = upper - lower
        # Replace inf widths with 2 * sigma (reasonable default for open buckets)
        inf_mask = np.isinf(width)
        width[inf_mask] = 2.0 * sigma

        # Add uniform noise within the bucket: U(-half, +half) where half = magnitude * width / 2
        half_range = config.magnitude * width / 2.0
        noise = rng.uniform(-1.0, 1.0, size=len(non_null_vals)) * half_range

        # Clamp to bucket boundaries to guarantee no bucket crossing
        jittered = non_null_vals + noise
        has_lower = ~np.isneginf(lower)
        has_upper = ~np.isposinf(upper)
        jittered = np.where(has_lower, np.maximum(jittered, lower), jittered)
        jittered = np.where(has_upper, np.minimum(jittered, upper - 1e-12), jittered)

        values[non_null_mask] = jittered
        X_d[col] = values
        actually_drifted.append(col)

    return DriftResult(
        X_drifted=X_d,
        scenario=config.scenario,
        drifted_features=actually_drifted,
        magnitude=config.magnitude,
        description=(
            f"S8: Benign within-bucket jitter (magnitude={config.magnitude}) "
            f"on {len(actually_drifted)} features (preserves model decision regions)"
        ),
    )


def _s9_no_drift(
    X: pd.DataFrame,
    config: DriftConfig,
    **kwargs,
) -> DriftResult:
    """S9: No drift (null hypothesis). Returns data unchanged."""
    return DriftResult(
        X_drifted=X.copy(),
        scenario=config.scenario,
        drifted_features=[],
        magnitude=0.0,
        description="S9: No drift applied (null hypothesis)",
    )


# Dispatch table
_SCENARIO_DISPATCH = {
    DriftScenario.S1_MEAN_SHIFT_IMPORTANT: _s1_mean_shift_important,
    DriftScenario.S2_MEAN_SHIFT_UNIMPORTANT: _s2_mean_shift_unimportant,
    DriftScenario.S3_VARIANCE_CHANGE: _s3_variance_change,
    DriftScenario.S4_COVARIATE_ROTATION: _s4_covariate_rotation,
    DriftScenario.S5_SUBPOPULATION_SHIFT: _s5_subpopulation_shift,
    DriftScenario.S6_CATEGORY_FREQ_SHIFT: _s6_category_freq_shift,
    DriftScenario.S7_NULL_RATE_INCREASE: _s7_null_rate_increase,
    DriftScenario.S8_BENIGN_DRIFT: _s8_benign_drift,
    DriftScenario.S9_NO_DRIFT: _s9_no_drift,
    DriftScenario.S10_GRADUAL_DRIFT: _s1_mean_shift_important,  # Same shift logic, different magnitude per step
}


def inject_gradual_drift(
    X: pd.DataFrame,
    config: GradualDriftConfig,
    shap_values: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> GradualDriftResult:
    """Inject S10 gradual drift: linearly increasing mean shift over N steps.

    At step i (1-indexed), the magnitude is ``(i / n_steps) * max_magnitude``,
    applied as a mean shift of ``magnitude × σ`` on the target features.

    Each step produces an independent drifted copy of ``X`` — steps are NOT
    cumulative on the same DataFrame.

    Args:
        X: Original feature DataFrame (n × p).
        config: Gradual drift configuration.
        shap_values: SHAP values for auto-selecting important features.
        feature_names: Feature names. Defaults to X.columns.
        numeric_features: Names of numeric features. Defaults to all.
        categorical_features: Names of categorical features. Defaults to none.

    Returns:
        GradualDriftResult with one DriftResult per step.
    """
    feature_names = feature_names or list(X.columns)
    numeric_features = numeric_features or list(X.columns)
    categorical_features = categorical_features or []

    # Determine target features (once, shared across all steps)
    targets = config.target_features
    if targets is None:
        if shap_values is None:
            raise ValueError(
                "S10 requires shap_values when target_features is None"
            )
        targets = _select_features_by_importance(
            feature_names, shap_values, config.n_features,
            important=True, numeric_features=numeric_features,
        )

    # Generate one DriftResult per step
    steps: list[DriftResult] = []
    for step_idx in range(config.n_steps):
        step_num = step_idx + 1  # 1-indexed
        step_magnitude = (step_num / config.n_steps) * config.max_magnitude

        drift_config = DriftConfig(
            scenario=DriftScenario.S10_GRADUAL_DRIFT,
            magnitude=step_magnitude,
            target_features=targets,
            n_features=config.n_features,
            random_state=config.random_state,
        )
        result = inject_drift(
            X=X,
            config=drift_config,
            shap_values=shap_values,
            feature_names=feature_names,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        # Override the description to include step info
        result.description = (
            f"S10: Gradual drift step {step_num}/{config.n_steps} — "
            f"mean shift of {step_magnitude:.3f}σ on features: {targets}"
        )
        steps.append(result)

    return GradualDriftResult(
        steps=steps,
        n_steps=config.n_steps,
        max_magnitude=config.max_magnitude,
        drifted_features=targets,
    )
