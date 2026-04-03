# Visualization

SWIFT provides two main visualization methods, both accessible directly from the `SWIFTMonitor` instance. All plots return `(Figure, Axes)` tuples for further customization.

## Bucket Profile

The bucket profile visualizes how a feature's SHAP response and observation density are distributed across buckets.

### Basic Usage

```python
# Reference density only
fig, ax = monitor.plot_buckets("feature_0")
```

<figure markdown="span">
  ![Bucket profile — single](../assets/images/bucket_profile_single.png){ width="100%" }
  <figcaption>Bucket profile with reference density only</figcaption>
</figure>

The plot shows:

- **Left y-axis**: Mean SHAP value per bucket (line + markers) with a shaded band showing mean +/- 2 standard deviations
- **Right y-axis**: Observation density (fraction of samples in each bucket)
- **X-axis**: Bucket intervals
- **Dashed grey line**: SHAP = 0 reference

### Comparing Distributions

Pass `X_compare` to overlay a second distribution (e.g., monitoring vs. reference):

```python
fig, ax = monitor.plot_buckets(
    "feature_0",
    X_compare=X_drifted,
    labels=("Reference", "Drifted"),
)
```

<figure markdown="span">
  ![Bucket profile — comparison](../assets/images/bucket_profile_comparison.png){ width="100%" }
  <figcaption>Reference vs. drifted density comparison — the density shift is clearly visible</figcaption>
</figure>

### Custom Density Source

By default, the density is computed from the reference data used during `fit()`. Pass `X` to use a different data source:

```python
# Show monitoring density instead of reference
fig, ax = monitor.plot_buckets("feature_0", X=X_mon)
```

### Natural X-axis

For a more interpretable x-axis, use `x_axis="natural"` to show bucket midpoints on the feature-value scale instead of bucket indices:

```python
fig, ax = monitor.plot_buckets("feature_0", x_axis="natural")
```

<figure markdown="span">
  ![Bucket profile — natural x-axis](../assets/images/bucket_profile_natural.png){ width="100%" }
  <figcaption>Natural x-axis mode — bucket midpoints on the feature-value scale</figcaption>
</figure>

When there are more than 20 buckets, the default mode automatically switches to compact labels (`B0, B1, ...`) for readability.

## SWIFT Score Plot

The SWIFT score plot provides an overview of drift across all features.

### Single Result

```python
fig, ax = monitor.plot_swift_scores(result, threshold=0.01)
```

<figure markdown="span">
  ![SWIFT scores — drifted](../assets/images/swift_scores_drifted.png){ width="100%" }
  <figcaption>SWIFT scores with drift-colored bars — red indicates drifted features</figcaption>
</figure>

Shows:

- **Bars**: One per feature, colored red (drifted) or blue (not drifted)
- **Horizontal lines**: SWIFT max, SWIFT mean, and optional threshold

### Comparing Two Results

```python
fig, ax = monitor.plot_swift_scores(
    result_clean,
    result_compare=result_drifted,
    labels=("Clean", "Drifted"),
)
```

<figure markdown="span">
  ![SWIFT scores — comparison](../assets/images/swift_scores_comparison.png){ width="100%" }
  <figcaption>Side-by-side comparison of SWIFT scores between clean and drifted data</figcaption>
</figure>

Shows grouped side-by-side bars with neutral coloring for direct comparison.

## Customization

Both methods return matplotlib `(Figure, Axes)` tuples, so you can customize the plots further:

```python
fig, ax = monitor.plot_swift_scores(result)
ax.set_title("My Custom Title")
ax.set_ylabel("SWIFT Score")
fig.savefig("swift_scores.png", dpi=300, bbox_inches="tight")
```

## Standalone Functions

The plotting functions are also available as standalone functions for advanced use:

```python
from swift.plotting import plot_bucket_profile, plot_feature_swift_scores
```

See the [Plotting API Reference](../api/plotting.md) for full details.
