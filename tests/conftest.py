"""Shared pytest fixtures for SWIFT tests.

Provides a trained LightGBM model, reference/monitoring datasets,
and pre-computed SHAP values for all test modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb
import shap


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def synthetic_data(rng: np.random.Generator) -> dict:
    """Generate a synthetic credit-scoring-like dataset.

    Returns a dict with keys:
        X_train, y_train  — training data (2000 obs, 5 features)
        X_ref, y_ref      — reference data (1000 obs, no drift)
        X_mon_drift, y_mon_drift — monitoring data with drift on feature_0
        X_mon_nodrift     — monitoring data with no drift
        feature_names     — list of feature names
    """
    n_train = 2000
    n_ref = 1000
    n_mon = 1000
    p = 5

    feature_names = [f"feature_{i}" for i in range(p)]

    # Training data: 5 continuous features ~ N(0, 1)
    X_train = rng.standard_normal((n_train, p))
    # Target: logistic function of weighted sum + noise
    weights = np.array([1.0, 0.5, -0.3, 0.0, 0.0])  # features 3,4 are noise
    logits = X_train @ weights + rng.standard_normal(n_train) * 0.5
    y_train = (logits > 0).astype(int)

    # Reference data (same distribution as training)
    X_ref = rng.standard_normal((n_ref, p))
    logits_ref = X_ref @ weights + rng.standard_normal(n_ref) * 0.5
    y_ref = (logits_ref > 0).astype(int)

    # Monitoring data WITH drift: shift feature_0 mean by +1.5 sigma
    X_mon_drift = rng.standard_normal((n_mon, p))
    X_mon_drift[:, 0] += 1.5  # <-- drift on the MOST important feature

    # Monitoring data WITHOUT drift (same distribution)
    X_mon_nodrift = rng.standard_normal((n_mon, p))

    return {
        "X_train": pd.DataFrame(X_train, columns=feature_names),
        "y_train": pd.Series(y_train, name="target"),
        "X_ref": pd.DataFrame(X_ref, columns=feature_names),
        "y_ref": pd.Series(y_ref, name="target"),
        "X_mon_drift": pd.DataFrame(X_mon_drift, columns=feature_names),
        "X_mon_nodrift": pd.DataFrame(X_mon_nodrift, columns=feature_names),
        "feature_names": feature_names,
        "weights": weights,
    }


@pytest.fixture(scope="session")
def synthetic_data_with_nulls(synthetic_data: dict, rng: np.random.Generator) -> dict:
    """Same as synthetic_data but with ~5% nulls injected in feature_0 and feature_2."""
    data = {k: v.copy() if hasattr(v, "copy") else v for k, v in synthetic_data.items()}

    for key in ["X_train", "X_ref", "X_mon_drift", "X_mon_nodrift"]:
        df = data[key].copy()
        n = len(df)
        null_mask_0 = rng.random(n) < 0.05
        null_mask_2 = rng.random(n) < 0.05
        df.loc[null_mask_0, "feature_0"] = np.nan
        df.loc[null_mask_2, "feature_2"] = np.nan
        data[key] = df

    return data


@pytest.fixture(scope="session")
def trained_lgb_model(synthetic_data: dict) -> lgb.Booster:
    """Train a LightGBM model on the synthetic data."""
    X_train = synthetic_data["X_train"]
    y_train = synthetic_data["y_train"]

    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 15,
        "learning_rate": 0.1,
        "n_estimators": 50,
        "verbose": -1,
        "seed": 42,
        "min_child_samples": 20,
    }
    model = lgb.train(
        params,
        train_data,
        num_boost_round=50,
    )
    return model


@pytest.fixture(scope="session")
def trained_lgb_model_with_nulls(synthetic_data_with_nulls: dict) -> lgb.Booster:
    """Train a LightGBM model on data with nulls."""
    X_train = synthetic_data_with_nulls["X_train"]
    y_train = synthetic_data_with_nulls["y_train"]

    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 15,
        "learning_rate": 0.1,
        "verbose": -1,
        "seed": 42,
        "min_child_samples": 20,
    }
    model = lgb.train(
        params,
        train_data,
        num_boost_round=50,
    )
    return model


@pytest.fixture(scope="session")
def ref_shap_values(trained_lgb_model: lgb.Booster, synthetic_data: dict) -> np.ndarray:
    """Compute SHAP values for the reference set using TreeSHAP.

    Returns array of shape (n_ref, p).
    """
    explainer = shap.TreeExplainer(trained_lgb_model)
    shap_values = explainer.shap_values(synthetic_data["X_ref"])
    return np.asarray(shap_values)
