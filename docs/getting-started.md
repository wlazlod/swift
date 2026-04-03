# Getting Started

## Requirements

- Python >= 3.11
- A trained tree-ensemble model (LightGBM or XGBoost)

## Installation

### From Source (recommended for development)

```bash
# Clone the repository
git clone https://github.com/wlazlod/swift.git
cd swift

# Install with dev dependencies
uv pip install -e ".[dev]"

# With experiment dependencies
uv pip install -e ".[dev,experiments]"
```

### From PyPI

```bash
pip install swift-monitoring
```

## Quick Start

### 1. Train a Model

SWIFT requires a trained tree-ensemble model. Here's a minimal example with LightGBM:

```python
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

# Prepare your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a LightGBM model
model = lgb.LGBMClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

### 2. Create a SWIFT Monitor

```python
from swift import SWIFTMonitor

monitor = SWIFTMonitor(
    model=model,
    n_permutations=200,
    alpha=0.05,
    correction="benjamini-hochberg",
)
```

### 3. Fit on Reference Data

The `fit()` method runs stages 1-3 of the pipeline: extracting decision points, building buckets, and computing SHAP normalization on the reference data.

```python
monitor.fit(X_ref)
```

### 4. Test for Drift

The `test()` method runs stages 4-5: computing Wasserstein distances and running the permutation test with multiple testing correction.

```python
result = monitor.test(X_mon)

# Check overall results
print(f"Drifted features: {result.drifted_features}")
print(f"Number drifted: {result.num_drifted}")
print(f"Max SWIFT score: {result.swift_max:.4f}")
print(f"Mean SWIFT score: {result.swift_mean:.4f}")
```

### 5. Inspect Per-Feature Results

```python
for fr in result.feature_results:
    status = "DRIFTED" if fr.is_drifted else "ok"
    print(f"  {fr.feature_name}: score={fr.swift_score:.4f}, "
          f"p={fr.p_value:.4f} [{status}]")
```

### 6. Visualize

```python
# Bucket profile for a specific feature
fig, ax = monitor.plot_buckets("feature_0")

# SWIFT scores overview
fig, ax = monitor.plot_swift_scores(result)
```

## What's Next?

- Learn about the [SWIFT Pipeline](user-guide/pipeline.md) in detail
- Explore [Visualization](user-guide/visualization.md) options
- See all [Configuration](user-guide/configuration.md) parameters
- Browse the [API Reference](api/index.md)
