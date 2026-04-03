# Configuration

All `SWIFTMonitor` parameters can be set at construction time or modified later via `set_params()`.

## Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | LightGBM / XGBoost | *(required)* | Trained tree-ensemble model. Must expose tree structure (e.g., `model.booster_.dump_model()` for LightGBM). |
| `order` | `int` | `1` | Wasserstein order. `1` for W1 (earth mover's distance), `2` for W2 (more sensitive to variance). |
| `n_permutations` | `int` | `1000` | Number of permutations for the permutation test. More permutations = more precise p-values but slower. |
| `alpha` | `float` | `0.05` | Significance level for the multiple testing correction. Features with corrected p-value < alpha are flagged as drifted. |
| `correction` | `str` | `"benjamini-hochberg"` | Multiple testing correction method. See [MTC Methods](#mtc-methods) below. |
| `n_synthetic` | `int` | `10` | Number of synthetic observations to generate for empty buckets during `fit()`. |
| `max_samples` | `int \| None` | `None` | Maximum pool size for the permutation test. If the pooled data exceeds this size, it is subsampled. `None` means no limit. |
| `random_state` | `int` | `42` | Random number generator seed for reproducibility (permutation test and synthetic observations). |

## MTC Methods

The `correction` parameter accepts the following values:

| Value | Aliases | Controls | Description |
|-------|---------|----------|-------------|
| `"benjamini-hochberg"` | `"bh"`, `"fdr"` | FDR | Controls the false discovery rate. Recommended default — good balance between power and false positive control. |
| `"bonferroni"` | `"bonf"` | FWER | Controls the family-wise error rate. More conservative — use when false positives are costly. |

## Wasserstein Order

The `order` parameter controls how the Wasserstein distance is computed:

- **W1 (order=1)**: Measures the mean absolute difference. Robust to outliers and easy to interpret as the average "shift" in SHAP space.
- **W2 (order=2)**: Measures the root-mean-square difference. More sensitive to variance changes and large deviations.

For most use cases, W1 is recommended. Use W2 when you care specifically about changes in the spread of the distribution.

## Permutation Test Tuning

The `n_permutations` parameter controls the precision of the p-value estimate:

- **Low (100-200)**: Fast, suitable for exploratory analysis. P-value resolution: ~0.005-0.01.
- **Medium (1000)**: Default. Good balance of speed and precision. P-value resolution: ~0.001.
- **High (5000-10000)**: High precision. Use for final results or when distinguishing between similar p-values matters.

!!! tip "Performance"
    The permutation test is the most compute-intensive part of the pipeline. If runtime is a concern, start with `n_permutations=200` for fast iteration, then increase for final results.

The `max_samples` parameter can limit the pool size for the permutation test. This is useful when reference and monitoring datasets are very large:

```python
monitor = SWIFTMonitor(
    model=model,
    n_permutations=1000,
    max_samples=5000,  # Subsample if pool > 5000
)
```

## Modifying Parameters

Since `SWIFTMonitor` follows the scikit-learn API, parameters can be inspected and modified:

```python
# Inspect current parameters
params = monitor.get_params()

# Modify parameters (requires re-fitting if changing model or n_synthetic)
monitor.set_params(alpha=0.01, correction="bonferroni")
```

!!! warning
    Changing `model`, `n_synthetic`, or `order` after `fit()` invalidates the fitted state. Call `fit()` again after modifying these parameters.
