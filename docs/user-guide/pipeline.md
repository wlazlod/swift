# The SWIFT Pipeline

SWIFT executes a 5-stage pipeline to detect model-relevant feature drift. Stages 1-3 run during `fit()` and learn the reference distribution. Stages 4-5 run during `test()` and evaluate monitoring data for drift.

## Stage 1: Decision Point Extraction

**Module:** [`swift.extraction`][swift.extraction]

The pipeline starts by extracting **decision points** (split thresholds) from the trained tree-ensemble model. These are the feature values where the model's trees make split decisions — they represent the boundaries that the model considers meaningful.

For a LightGBM model with 100 trees and a feature that appears in 50 splits, the extraction step collects all 50 unique threshold values for that feature.

```python
from swift.extraction import extract_decision_points

# Extract decision points per feature
decision_points = extract_decision_points(model, feature_names)
# Returns: dict[str, ndarray] — feature name → sorted threshold array
```

## Stage 2: Bucketing

**Module:** [`swift.bucketing`][swift.bucketing]

Decision points partition each feature's domain into **buckets**. Each bucket is a contiguous interval bounded by adjacent decision points, with the first bucket extending to $-\infty$ and the last to $+\infty$.

For example, if a feature has decision points `[0.5, 1.2, 3.0]`, the resulting buckets are:

| Bucket | Interval |
|--------|----------|
| B0 | $(-\infty, 0.5]$ |
| B1 | $(0.5, 1.2]$ |
| B2 | $(1.2, 3.0]$ |
| B3 | $(3.0, +\infty)$ |

Each bucket knows its boundaries and type (left-infinite, interior, or right-infinite).

```python
from swift.bucketing import build_all_buckets

# Build bucket sets for all features
bucket_sets = build_all_buckets(decision_points)
# Returns: dict[str, BucketSet]
```

## Stage 3: SHAP Normalization

**Module:** [`swift.normalization`][swift.normalization]

For each bucket, SWIFT computes the **mean SHAP value** of all reference observations falling in that bucket. This transforms the feature space from raw values to SHAP-impact space — observations in regions where the model relies heavily on a feature get higher weight.

Empty buckets (no reference observations) are filled with synthetic observations drawn uniformly within the bucket bounds, controlled by the `n_synthetic` parameter.

```python
from swift.normalization import compute_bucket_shap

# Compute mean SHAP per bucket
bucket_shap = compute_bucket_shap(
    X_ref, shap_values, bucket_sets, n_synthetic=10
)
```

After this stage, the reference distribution is fully characterized: each feature has a set of buckets with associated mean SHAP values and observation counts.

## Stage 4: Wasserstein Distance

**Module:** [`swift.distance`][swift.distance]

To detect drift, SWIFT computes the **Wasserstein distance** between the SHAP-transformed reference distribution and the monitoring distribution. The Wasserstein distance (earth mover's distance) measures the minimum "work" needed to transform one distribution into another.

SWIFT supports two orders:

- **W1** (order=1): The standard earth mover's distance. Robust and interpretable.
- **W2** (order=2): More sensitive to variance changes. Useful when the spread of the distribution matters.

```python
from swift.distance import compute_swift_scores

# Compute per-feature SWIFT scores
scores = compute_swift_scores(
    X_ref_transformed, X_mon_transformed, bucket_sets, order=1
)
# Returns: dict[str, float]
```

## Stage 5: Permutation Test + MTC

**Module:** [`swift.threshold`][swift.threshold]

Raw Wasserstein distances don't have a natural threshold for "significant drift." SWIFT uses a **permutation test** to estimate p-values:

1. Pool the reference and monitoring observations
2. Randomly split the pool into two groups of the original sizes
3. Compute the Wasserstein distance between the random groups
4. Repeat `n_permutations` times to build a null distribution
5. The p-value is the fraction of permutation distances >= the observed distance

Since SWIFT tests multiple features simultaneously, **multiple testing correction** (MTC) is applied to control the false discovery rate:

| Method | Description |
|--------|-------------|
| `"benjamini-hochberg"` / `"bh"` / `"fdr"` | Controls FDR — recommended default |
| `"bonferroni"` / `"bonf"` | Controls FWER — more conservative |

```python
from swift.threshold import permutation_test, correct_pvalues

# Permutation test for a single feature
p_value = permutation_test(
    ref_transformed, mon_transformed, n_permutations=1000
)

# Apply MTC across all features
corrected = correct_pvalues(p_values, method="benjamini-hochberg")
```

## Putting It All Together

The `SWIFTMonitor` class orchestrates all five stages:

```python
from swift import SWIFTMonitor

monitor = SWIFTMonitor(
    model=lgb_model,
    order=1,
    n_permutations=1000,
    alpha=0.05,
    correction="benjamini-hochberg",
)

# Stages 1-3
monitor.fit(X_ref)

# Stages 4-5
result = monitor.test(X_mon)
print(result.drifted_features)
```

See the [API Reference](../api/pipeline.md) for the full `SWIFTMonitor` documentation.
