# SWIFT: SHAP-Weighted Impact Feature Testing

Model-aware distribution monitoring for production ML systems.

SWIFT detects feature drift by comparing SHAP-transformed distributions between reference and monitoring data. Unlike traditional drift detection that treats features independently of the model, SWIFT weights distribution changes by their impact on model predictions — flagging only the shifts that affect model behavior.

## Installation

```bash
# From source (editable)
uv pip install -e ".[dev]"

# With experiment dependencies
uv pip install -e ".[dev,experiments]"
```

Requires Python >= 3.11.

## Quick Start

```python
from swift import SWIFTMonitor

# Create monitor with a trained tree-ensemble model
monitor = SWIFTMonitor(model=lgb_model, n_permutations=200)

# Fit on reference data (stages 1-3)
monitor.fit(X_ref)

# Test monitoring data for drift (stages 4-5)
result = monitor.test(X_mon)
print(result.drifted_features)
```

## The SWIFT Pipeline

SWIFT executes a 5-stage pipeline:

| Stage | Operation | Description |
|-------|-----------|-------------|
| 1 | **Extraction** | Extract decision points (split thresholds) from the trained model |
| 2 | **Bucketing** | Partition each feature's domain into buckets based on decision points |
| 3 | **SHAP Normalization** | Compute bucket-level mean SHAP values on reference data |
| 4 | **Wasserstein Distance** | Measure distance between SHAP-transformed distributions |
| 5 | **Permutation Test + MTC** | Estimate p-values and apply multiple testing correction |

Stages 1-3 run during `fit()`. Stages 4-5 run during `test()`.

## API Reference

### `SWIFTMonitor`

```python
SWIFTMonitor(
    model,                          # Trained tree-ensemble (LightGBM/XGBoost)
    order=1,                        # Wasserstein order (1 or 2)
    n_permutations=1000,            # Permutations for p-value estimation
    alpha=0.05,                     # Significance level
    correction="benjamini-hochberg",# MTC method ("bonferroni", "bh", "fdr")
    n_synthetic=10,                 # Synthetic observations for empty buckets
    max_samples=None,               # Max pool size for permutation test
    random_state=42,                # RNG seed
)
```

### Core Methods

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `fit(X)` | `DataFrame` | `self` | Learn reference distribution (stages 1-3) |
| `transform(X)` | `DataFrame` | `DataFrame` | Map values to bucket-level mean SHAP |
| `fit_transform(X)` | `DataFrame` | `DataFrame` | Fit and transform in one call |
| `score(X)` | `DataFrame` | `dict[str, float]` | Per-feature Wasserstein distances vs. reference (stage 4) |
| `score(X, X_compare=Y)` | `DataFrame, DataFrame` | `dict[str, float]` | Per-feature distances: X vs. Y (SHAP transform from reference) |
| `test(X)` | `DataFrame` | `SWIFTResult` | Full drift test vs. reference with p-values and MTC |
| `test(X, X_compare=Y)` | `DataFrame, DataFrame` | `SWIFTResult` | Full drift test: X vs. Y (SHAP transform from reference) |

### Visualization Methods

| Method | Description |
|--------|-------------|
| `plot_buckets(feature_name)` | SHAP response curve + reference density per bucket |
| `plot_buckets(feature_name, X=...)` | Custom density source (e.g. monitoring sample instead of reference) |
| `plot_buckets(feature_name, X_compare=...)` | Side-by-side density comparison |
| `plot_buckets(feature_name, x_axis="natural")` | Feature-value scale on x-axis (instead of bucket indices) |
| `plot_swift_scores(result)` | Bar chart of SWIFT scores, colored by drift status |
| `plot_swift_scores(result, result_compare=...)` | Grouped bars comparing two test results |

Both return `(Figure, Axes)` tuples for further customization.

### Result Objects

**`SWIFTResult`** — returned by `test()`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `feature_results` | `tuple[FeatureSWIFTResult, ...]` | Per-feature results |
| `swift_max` | `float` | Maximum SWIFT score across features |
| `swift_mean` | `float` | Mean SWIFT score across features |
| `num_drifted` | `int` | Number of drifted features |
| `drifted_features` | `tuple[str, ...]` | Names of drifted features |

**`FeatureSWIFTResult`** — per-feature detail:

| Attribute | Type | Description |
|-----------|------|-------------|
| `feature_name` | `str` | Feature name |
| `swift_score` | `float` | SWIFT score (Wasserstein distance) |
| `p_value` | `float` | Permutation test p-value |
| `is_drifted` | `bool` | Drift flag after MTC |
| `num_buckets` | `int` | Number of buckets used |

## Visualization

### Bucketing Profile

```python
# Single sample — reference density only
fig, ax = monitor.plot_buckets("feature_0")

# Comparison — reference vs. drifted density side-by-side
fig, ax = monitor.plot_buckets(
    "feature_0",
    X_compare=X_drifted,
    labels=("Reference", "Drifted"),
)

# Natural x-axis — bucket midpoints on the feature-value scale
fig, ax = monitor.plot_buckets("feature_0", x_axis="natural")

# Custom density source — show monitoring density instead of reference
fig, ax = monitor.plot_buckets("feature_0", X=X_mon)
```

The bucketing profile shows:
- **Left y-axis**: Mean SHAP per bucket (line + markers) with a shaded 95% band (mean +/- 2 std)
- **Right y-axis**: Observation density (line + fill, fraction of samples in each bucket)
- **X-axis**: Bucket intervals (default), compact labels `B0, B1, ...` for >20 buckets, or feature-value scale with `x_axis="natural"`
- **SHAP=0 reference**: Dashed grey horizontal line

### SWIFT Scores

```python
# Single result — bars colored by drift status
fig, ax = monitor.plot_swift_scores(result, threshold=0.01)

# Comparison — grouped bars from two test results
fig, ax = monitor.plot_swift_scores(
    result_clean,
    result_compare=result_drifted,
    labels=("Clean", "Drifted"),
)
```

The score plot shows:
- **Single mode**: Bars colored red (drifted) or blue (not drifted), with horizontal lines for SWIFT max, SWIFT mean, and an optional threshold
- **Comparison mode**: Grouped side-by-side bars with neutral coloring

## scikit-learn Integration

`SWIFTMonitor` inherits from `BaseEstimator` and `TransformerMixin`:

```python
# Parameter inspection and modification
monitor.get_params()
monitor.set_params(alpha=0.01, correction="bonferroni")

# fit_transform shorthand
X_transformed = monitor.fit_transform(X_ref)

# Clone with same parameters
from sklearn.base import clone
monitor2 = clone(monitor)
```

## Sample vs. Sample Comparison

By default `score()` and `test()` compare monitoring data against the fitted reference. Pass `X_compare` to compare two arbitrary samples instead — the SHAP transformation (buckets + mean SHAP) is always the one learned from the reference:

```python
# Score: monitoring vs. drifted
scores = monitor.score(X_mon, X_compare=X_drifted)

# Full test: monitoring vs. drifted
result = monitor.test(X_mon, X_compare=X_drifted)
print(result.drifted_features)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | (required) | Trained LightGBM or XGBoost model |
| `order` | `1` | Wasserstein order: 1 (W1) or 2 (W2, more sensitive to variance) |
| `n_permutations` | `1000` | Number of permutations for p-value estimation |
| `alpha` | `0.05` | Significance level for MTC |
| `correction` | `"benjamini-hochberg"` | MTC method. Aliases: `"bh"`, `"fdr"`, `"bonf"` |
| `n_synthetic` | `10` | Synthetic observations for empty buckets during fit |
| `max_samples` | `None` | Max pool size for permutation test (subsamples if exceeded) |
| `random_state` | `42` | RNG seed for reproducibility |

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=swift --cov-report=term-missing

# Lint
ruff check src/ tests/
```

## Project Structure

```
swift/
├── src/swift/
│   ├── __init__.py          # Public API exports
│   ├── pipeline.py          # SWIFTMonitor (main entry point)
│   ├── plotting.py          # Visualization functions
│   ├── extraction.py        # Stage 1: Decision point extraction
│   ├── bucketing.py         # Stage 2: Bucket construction
│   ├── normalization.py     # Stage 3: SHAP normalization
│   ├── distance.py          # Stage 4: Wasserstein distance
│   ├── threshold.py         # Stage 5: Permutation test + MTC
│   ├── aggregation.py       # Model-level score aggregation
│   └── types.py             # Data types (BucketSet, SWIFTResult, etc.)
├── tests/                   # Test suite
├── experiments/             # Experiment runners (ablations, power analysis, etc.)
├── notebooks/
│   └── getting_started.ipynb
└── pyproject.toml
```

## Citation

```bibtex
@article{swift2025,
  title={SWIFT: SHAP-Weighted Impact Feature Testing for Model-Aware Distribution Monitoring},
  year={2025}
}
```
