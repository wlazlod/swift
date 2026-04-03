# API Reference

Full auto-generated API documentation from source code docstrings.

## Core

| Module | Description |
|--------|-------------|
| [`SWIFTMonitor`](pipeline.md) | Main entry point — the scikit-learn compatible monitor |
| [`Types`](types.md) | Data types: `Bucket`, `BucketSet`, `SWIFTResult`, enums |

## Pipeline Stages

| Module | Stage | Description |
|--------|-------|-------------|
| [`Extraction`](extraction.md) | 1 | Decision point extraction from tree models |
| [`Bucketing`](bucketing.md) | 2 | Bucket construction from decision points |
| [`Normalization`](normalization.md) | 3 | SHAP normalization per bucket |
| [`Distance`](distance.md) | 4 | Wasserstein distance computation |
| [`Threshold`](threshold.md) | 5 | Permutation test and multiple testing correction |

## Utilities

| Module | Description |
|--------|-------------|
| [`Aggregation`](aggregation.md) | Model-level score aggregation |
| [`Plotting`](plotting.md) | Visualization functions |
