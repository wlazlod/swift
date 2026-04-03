# scikit-learn Integration

`SWIFTMonitor` inherits from `sklearn.base.BaseEstimator` and `sklearn.base.TransformerMixin`, making it a first-class scikit-learn citizen.

## Estimator API

### Parameter Management

```python
from swift import SWIFTMonitor

monitor = SWIFTMonitor(model=model, n_permutations=200, alpha=0.05)

# Inspect all parameters
monitor.get_params()
# {'model': ..., 'order': 1, 'n_permutations': 200, 'alpha': 0.05,
#  'correction': 'benjamini-hochberg', 'n_synthetic': 10,
#  'max_samples': None, 'random_state': 42}

# Modify parameters
monitor.set_params(alpha=0.01, correction="bonferroni")
```

### Cloning

Cloning creates a new unfitted estimator with identical parameters:

```python
from sklearn.base import clone

monitor2 = clone(monitor)
# monitor2 is unfitted, with same parameters as monitor
```

## Transformer API

### fit_transform

The standard `fit_transform` shorthand works as expected:

```python
# These are equivalent:
X_transformed = monitor.fit_transform(X_ref)

# and:
monitor.fit(X_ref)
X_transformed = monitor.transform(X_ref)
```

`transform()` maps each observation to its bucket's mean SHAP value. The output has the same shape and column names as the input.

## Sample vs. Sample Comparison

By default, `score()` and `test()` compare monitoring data against the fitted reference. Pass `X_compare` to compare two arbitrary samples instead — the SHAP transformation (buckets + mean SHAP) is always the one learned from the reference:

```python
# Score: monitoring vs. drifted
scores = monitor.score(X_mon, X_compare=X_drifted)

# Full test: monitoring vs. drifted
result = monitor.test(X_mon, X_compare=X_drifted)
print(result.drifted_features)
```

This is useful when you want to compare two time windows of production data against each other, rather than against the original reference.
