"""Visualization utilities for SWIFT monitoring results.

Provides two public plotting functions:

- ``plot_bucket_profile`` — SHAP response curve + density per bucket for a feature.
- ``plot_feature_swift_scores`` — Bar chart of SWIFT scores per feature.

Both return ``(Figure, Axes)`` tuples for further customization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from swift.types import BucketSet, SWIFTResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assign_buckets_vectorized(
    values: np.ndarray,
    bucket_set: BucketSet,
) -> np.ndarray:
    """Assign bucket indices to an array of feature values.

    Returns an integer array of the same length as *values*, where each
    element is the bucket index (0 = NULL, 1..m+1 = numeric buckets).
    """
    decision_points = bucket_set.decision_points
    n = len(values)
    indices = np.empty(n, dtype=int)

    # Identify NaN (-> null bucket 0)
    nan_mask = np.isnan(values.astype(float))

    if len(decision_points) == 0:
        # Single catch-all bucket at index 1
        indices[:] = 1
    else:
        # np.searchsorted gives the position in decision_points,
        # offset by +1 because bucket 0 is NULL.
        numeric_indices = np.searchsorted(decision_points, values, side="right") + 1
        # Clip to valid range [1, num_buckets - 1]
        max_idx = len(decision_points) + 1
        numeric_indices = np.clip(numeric_indices, 1, max_idx)
        indices[:] = numeric_indices

    indices[nan_mask] = 0
    return indices


def _compute_bucket_stats(
    bucket_set: BucketSet,
    feature_values: np.ndarray,
    shap_values: np.ndarray | None = None,
) -> dict:
    """Compute per-bucket density, mean_shap, and shap_std.

    Parameters
    ----------
    bucket_set : BucketSet
        Fitted bucket set for the feature.
    feature_values : np.ndarray
        Raw feature values.
    shap_values : np.ndarray or None
        SHAP values for this feature (same length as feature_values).
        If provided, computes per-bucket SHAP std.

    Returns
    -------
    dict with keys:
        - bucket_indices: list[int] (bucket index for each bucket)
        - densities: np.ndarray (fraction per bucket, sums to ~1.0)
        - mean_shaps: np.ndarray (mean SHAP per bucket from BucketSet)
        - shap_stds: np.ndarray (std of SHAP values per bucket; zeros if
          shap_values is None)
        - counts: np.ndarray (raw count per bucket)
    """
    num_buckets = bucket_set.num_buckets
    bucket_indices = list(range(num_buckets))

    assignments = _assign_buckets_vectorized(feature_values, bucket_set)
    n_total = len(feature_values)

    counts = np.zeros(num_buckets, dtype=int)
    mean_shaps = np.zeros(num_buckets)
    shap_stds = np.zeros(num_buckets)

    for k in bucket_indices:
        mask = assignments == k
        counts[k] = mask.sum()
        mean_shaps[k] = bucket_set.get_mean_shap(k)

        if shap_values is not None and counts[k] > 0:
            shap_stds[k] = np.std(shap_values[mask])

    densities = counts / max(n_total, 1)

    return {
        "bucket_indices": bucket_indices,
        "densities": densities,
        "mean_shaps": mean_shaps,
        "shap_stds": shap_stds,
        "counts": counts,
    }


def _compute_sample_densities(
    bucket_set: BucketSet,
    feature_values: np.ndarray,
) -> np.ndarray:
    """Compute per-bucket density for an arbitrary sample (normalized to 1.0)."""
    num_buckets = bucket_set.num_buckets
    assignments = _assign_buckets_vectorized(feature_values, bucket_set)
    n_total = len(feature_values)

    densities = np.zeros(num_buckets)
    for k in range(num_buckets):
        densities[k] = (assignments == k).sum()
    densities = densities / max(n_total, 1)

    return densities


def _format_bucket_labels(
    bucket_set: BucketSet,
    max_label_buckets: int = 20,
) -> list[str]:
    """Generate bucket labels: interval notation or B0/B1/... index form."""
    num_buckets = bucket_set.num_buckets

    if num_buckets > max_label_buckets:
        return [f"B{k}" for k in range(num_buckets)]

    labels: list[str] = []
    for bucket in bucket_set.buckets:
        if bucket.index == 0:
            labels.append("NULL")
        elif np.isneginf(bucket.lower):
            labels.append(f"< {bucket.upper:.2g}")
        elif np.isposinf(bucket.upper):
            labels.append(f"\u2265 {bucket.lower:.2g}")
        else:
            labels.append(f"[{bucket.lower:.2g}, {bucket.upper:.2g})")

    return labels


def _compute_natural_x_positions(
    bucket_set: BucketSet,
    feature_values: np.ndarray,
) -> np.ndarray:
    """Compute x-positions on the natural feature-value scale.

    Bucket midpoints are used for finite intervals.  Edge buckets use
    the observed min/max of *feature_values*.  The NULL bucket is placed
    to the left of the numeric range with a gap of 1.5× the median
    bucket width.

    Returns
    -------
    np.ndarray
        One x-position per bucket (length = ``bucket_set.num_buckets``).
    """
    dp = bucket_set.decision_points
    num_buckets = bucket_set.num_buckets

    # Observed data range (ignoring NaN)
    finite_vals = feature_values[np.isfinite(feature_values.astype(float))]
    if len(finite_vals) == 0:
        # Fallback: evenly spaced
        return np.arange(num_buckets, dtype=float)
    data_min = float(np.min(finite_vals))
    data_max = float(np.max(finite_vals))

    positions = np.zeros(num_buckets, dtype=float)

    # Compute positions for numeric buckets (index >= 1)
    for bucket in bucket_set.buckets:
        idx = bucket.index
        if idx == 0:
            continue  # NULL — handled below

        lo = bucket.lower
        hi = bucket.upper

        if np.isneginf(lo) and np.isposinf(hi):
            # Single catch-all bucket
            positions[idx] = (data_min + data_max) / 2
        elif np.isneginf(lo):
            positions[idx] = data_min
        elif np.isposinf(hi):
            positions[idx] = data_max
        else:
            positions[idx] = (lo + hi) / 2

    # Compute median bucket width for the gap
    if len(dp) >= 2:
        widths = np.diff(dp)
        median_width = float(np.median(widths))
    elif len(dp) == 1:
        median_width = max(abs(data_max - data_min) * 0.1, 1.0)
    else:
        median_width = max(abs(data_max - data_min) * 0.1, 1.0)

    # NULL bucket: place to the left with a gap
    numeric_min = positions[1] if num_buckets > 1 else data_min
    positions[0] = numeric_min - 1.5 * median_width

    return positions


# ---------------------------------------------------------------------------
# Public standalone functions
# ---------------------------------------------------------------------------


def plot_bucket_profile(
    bucket_set: BucketSet,
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    compare_values: np.ndarray | None = None,
    primary_values: np.ndarray | None = None,
    labels: tuple[str, str] = ("Reference", "Comparison"),
    figsize: tuple[float, float] = (10, 5),
    title: str | None = None,
    max_label_buckets: int = 20,
    x_axis: str = "bucket",
) -> tuple[Figure, Axes]:
    """Plot the bucketing profile for a single feature.

    Shows mean SHAP per bucket (line + error band) and observation density
    (filled line).  Optionally overlays a comparison sample's density.

    Parameters
    ----------
    bucket_set : BucketSet
        Fitted bucket set for the feature.
    feature_values : np.ndarray
        Raw feature values from the reference sample (used for the SHAP
        curve and, when *primary_values* is ``None``, for the primary
        density).
    shap_values : np.ndarray
        SHAP values for this feature on the reference sample.
    compare_values : np.ndarray or None
        Raw feature values from a comparison sample.  If provided, shows
        a second density line.
    primary_values : np.ndarray or None
        If provided, these values are used for the primary density
        instead of *feature_values*.  Useful for showing density of an
        arbitrary sample while the SHAP curve stays anchored to the
        reference.
    labels : tuple of str
        Legend labels for (primary, comparison) densities.
    figsize : tuple, default (10, 5)
        Figure size.
    title : str or None
        Custom title.  Defaults to ``"Bucketing Profile: {feature_name}"``.
    max_label_buckets : int, default 20
        Switch from interval notation to bucket indices if exceeded.
    x_axis : {"bucket", "natural"}
        ``"bucket"`` (default) uses integer bucket indices on the x-axis.
        ``"natural"`` uses actual feature-value positions (bucket
        midpoints, data min/max for edge buckets, NULL placed at the
        left with a gap).

    Returns
    -------
    (Figure, Axes)
    """
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette()

    # Determine which values to use for primary density
    density_values = primary_values if primary_values is not None else feature_values

    stats = _compute_bucket_stats(bucket_set, feature_values, shap_values)
    primary_densities = _compute_sample_densities(bucket_set, density_values)
    bucket_labels = _format_bucket_labels(bucket_set, max_label_buckets)
    num_buckets = bucket_set.num_buckets

    # -- X positions --
    use_natural = x_axis == "natural"
    if use_natural:
        x_positions = _compute_natural_x_positions(bucket_set, feature_values)
    else:
        x_positions = np.arange(num_buckets, dtype=float)

    fig, ax_shap = plt.subplots(figsize=figsize)
    ax_density = ax_shap.twinx()

    # -- SHAP = 0 reference line (visible behind everything) --
    ax_shap.axhline(
        y=0,
        color="#888888",
        linewidth=1.4,
        linestyle="--",
        alpha=0.7,
        zorder=1.5,
        label="_nolegend_",
    )

    # -- Density lines + filled area (right y-axis) --
    comparison_mode = compare_values is not None

    # Primary density
    ax_density.plot(
        x_positions,
        primary_densities,
        color=palette[0],
        linewidth=1.5,
        alpha=0.8,
        zorder=1,
        label=labels[0] if comparison_mode else "Observation density",
    )
    ax_density.fill_between(
        x_positions,
        0,
        primary_densities,
        color=palette[0],
        alpha=0.15,
        zorder=0.9,
    )

    if comparison_mode:
        compare_densities = _compute_sample_densities(
            bucket_set, compare_values,
        )
        ax_density.plot(
            x_positions,
            compare_densities,
            color=palette[1],
            linewidth=1.5,
            alpha=0.8,
            zorder=1,
            label=labels[1],
        )
        ax_density.fill_between(
            x_positions,
            0,
            compare_densities,
            color=palette[1],
            alpha=0.15,
            zorder=0.9,
        )

    ax_density.set_ylabel("Density")

    # -- SHAP line (left y-axis) --
    shap_color = palette[2]
    mean_shaps = stats["mean_shaps"]
    shap_stds = stats["shap_stds"]

    ax_shap.plot(
        x_positions,
        mean_shaps,
        color=shap_color,
        marker="o",
        linewidth=2,
        markersize=6,
        zorder=3,
        label="Mean SHAP \u00b1 2 std",
    )

    # Error band (only where count > 0)
    upper = mean_shaps + 2 * shap_stds
    lower = mean_shaps - 2 * shap_stds
    mask_nonzero = stats["counts"] > 0
    ax_shap.fill_between(
        x_positions,
        lower,
        upper,
        where=mask_nonzero,
        alpha=0.2,
        color=shap_color,
        zorder=2,
    )

    ax_shap.set_ylabel("SHAP Value")

    # -- X-axis --
    if use_natural:
        # Natural axis: let matplotlib auto-format, but annotate NULL
        ax_shap.set_xlabel(bucket_set.feature_name)

        # Add a vertical dotted line to separate NULL from numeric range
        if num_buckets > 1:
            sep_x = (x_positions[0] + x_positions[1]) / 2
            ax_shap.axvline(
                sep_x,
                color="#aaaaaa",
                linewidth=1.0,
                linestyle=":",
                alpha=0.6,
                zorder=0.5,
            )
            # Label the NULL point
            ax_shap.annotate(
                "NULL",
                xy=(x_positions[0], 0),
                xytext=(x_positions[0], 0),
                fontsize=8,
                ha="center",
                va="bottom",
                color="#666666",
            )

        # Use auto-ticks for the numeric part, but add NULL tick
        # We set minor ticks off and let matplotlib handle major ticks
        ax_shap.xaxis.set_major_locator(mticker.AutoLocator())
    else:
        # Bucket-index mode: explicit tick labels
        ax_shap.set_xlabel("Bucket")
        ax_shap.set_xticks(x_positions)

        # Adaptive label formatting to avoid overlap
        if num_buckets > 10:
            fontsize = 7
            rotation = 90
        elif num_buckets > 3:
            fontsize = 8
            rotation = 60
        else:
            fontsize = 9
            rotation = 0

        ax_shap.set_xticklabels(
            bucket_labels,
            rotation=rotation,
            ha="right" if rotation > 0 else "center",
            fontsize=fontsize,
        )

    # -- Title --
    plot_title = title or f"Bucketing Profile: {bucket_set.feature_name}"
    ax_shap.set_title(plot_title)

    # -- Legend (combine both axes) --
    lines_shap, labels_shap = ax_shap.get_legend_handles_labels()
    lines_density, labels_density = ax_density.get_legend_handles_labels()
    ax_shap.legend(
        lines_shap + lines_density,
        labels_shap + labels_density,
        loc="upper right",
    )

    # Ensure SHAP line is drawn on top of density
    ax_shap.set_zorder(ax_density.get_zorder() + 1)
    ax_shap.patch.set_visible(False)

    fig.tight_layout()
    return fig, ax_shap


def plot_feature_swift_scores(
    result: SWIFTResult,
    result_compare: SWIFTResult | None = None,
    labels: tuple[str, str] = ("Result A", "Result B"),
    threshold: float | None = None,
    sort_by: str = "score",
    feature_order: list[str] | None = None,
    figsize: tuple[float, float] = (12, 5),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot SWIFT scores per feature as a bar chart with reference lines.

    Optionally compare two ``SWIFTResult`` objects side by side.

    Parameters
    ----------
    result : SWIFTResult
        Primary result from ``SWIFTMonitor.test()``.
    result_compare : SWIFTResult or None
        Optional second result for side-by-side comparison.
    labels : tuple of str
        Legend labels for the two results in comparison mode.
    threshold : float or None
        Optional detection threshold horizontal line.
    sort_by : {"score", "name", "original"}
        Feature ordering on x-axis.  ``"original"`` uses *feature_order*.
    feature_order : list[str] or None
        Original feature order (from ``monitor.feature_names_in_``).
        Required when ``sort_by="original"``.
    figsize : tuple, default (12, 5)
        Figure size.
    title : str or None
        Custom title.

    Returns
    -------
    (Figure, Axes)
    """
    sns.set_theme(style="whitegrid")

    comparison_mode = result_compare is not None

    # Build dicts: feature_name -> score / is_drifted
    scores_a = {fr.feature_name: fr.swift_score for fr in result.feature_results}
    drifted_a = {fr.feature_name: fr.is_drifted for fr in result.feature_results}

    if comparison_mode:
        scores_b = {
            fr.feature_name: fr.swift_score
            for fr in result_compare.feature_results
        }

    # Determine feature order
    feature_names = list(scores_a.keys())
    if sort_by == "score":
        feature_names = sorted(feature_names, key=lambda f: scores_a[f], reverse=True)
    elif sort_by == "name":
        feature_names = sorted(feature_names)
    elif sort_by == "original" and feature_order is not None:
        feature_names = [f for f in feature_order if f in scores_a]
    # else: keep dict order

    n_features = len(feature_names)
    x_positions = np.arange(n_features)

    fig, ax = plt.subplots(figsize=figsize)

    if comparison_mode:
        # -- Comparison mode: grouped bars --
        palette = sns.color_palette()
        bar_width = 0.35

        vals_a = [scores_a[f] for f in feature_names]
        vals_b = [scores_b.get(f, 0.0) for f in feature_names]

        ax.bar(
            x_positions - bar_width / 2,
            vals_a,
            width=bar_width,
            color=palette[0],
            edgecolor=_darken(palette[0]),
            label=labels[0],
        )
        ax.bar(
            x_positions + bar_width / 2,
            vals_b,
            width=bar_width,
            color=palette[1],
            edgecolor=_darken(palette[1]),
            label=labels[1],
        )

        # Only threshold line in comparison mode
        if threshold is not None:
            ax.axhline(
                threshold,
                linestyle=":",
                color="black",
                linewidth=1.5,
                label=f"Threshold = {threshold:.4f}",
            )

        plot_title = title or "SWIFT Scores Comparison"

    else:
        # -- Single result mode: drift-colored bars --
        color_drifted = "#e74c3c"
        color_ok = "#3498db"
        edge_drifted = _darken_hex(color_drifted)
        edge_ok = _darken_hex(color_ok)

        vals = [scores_a[f] for f in feature_names]
        colors = [
            color_drifted if drifted_a.get(f) else color_ok
            for f in feature_names
        ]
        edges = [
            edge_drifted if drifted_a.get(f) else edge_ok
            for f in feature_names
        ]

        bars = ax.bar(
            x_positions,
            vals,
            color=colors,
            edgecolor=edges,
            width=0.6,
        )

        # Proxy artists for legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=color_drifted, edgecolor=edge_drifted, label="Drifted"),
            Patch(facecolor=color_ok, edgecolor=edge_ok, label="Not drifted"),
        ]

        # Horizontal lines
        ax.axhline(
            result.swift_max,
            linestyle="--",
            color=color_drifted,
            linewidth=1.2,
            label=f"SWIFT max = {result.swift_max:.4f}",
        )
        ax.axhline(
            result.swift_mean,
            linestyle="--",
            color=color_ok,
            linewidth=1.2,
            label=f"SWIFT mean = {result.swift_mean:.4f}",
        )

        if threshold is not None:
            ax.axhline(
                threshold,
                linestyle=":",
                color="black",
                linewidth=1.5,
                label=f"Threshold = {threshold:.4f}",
            )

        legend_elements.extend(ax.get_legend_handles_labels()[0])
        # Reset and rebuild legend
        ax.legend(handles=legend_elements, loc="upper right")

        plot_title = title or "SWIFT Scores per Feature"

    # -- Axes --
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        feature_names,
        rotation=45 if n_features > 5 else 0,
        ha="right" if n_features > 5 else "center",
    )
    ax.set_ylabel("SWIFT Score")
    ax.set_title(plot_title)

    if comparison_mode:
        ax.legend(loc="upper right")

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def _darken(color: tuple, factor: float = 0.7) -> tuple:
    """Darken an RGB(A) tuple by *factor*."""
    return tuple(c * factor for c in color[:3])


def _darken_hex(hex_color: str, factor: float = 0.7) -> str:
    """Darken a hex color string by *factor*."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"
