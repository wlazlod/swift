"""Stage 1: Decision point extraction from trained models.

Extracts the set of unique split thresholds (decision points) per feature
from a trained model. These decision points define the bucket boundaries
for the SWIFT pipeline.

Supported model types:
    - LightGBM Booster (extract_decision_points_lgb)
    - XGBoost Booster (extract_decision_points_xgb)
"""

from __future__ import annotations

import json
import logging
from typing import Sequence

import numpy as np
import lightgbm as lgb

logger = logging.getLogger(__name__)


def extract_decision_points(
    model: object,
    feature_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Auto-dispatch extraction based on model type.

    Args:
        model: A trained LightGBM Booster or XGBoost Booster.
        feature_names: List of feature names (must match model's feature order).

    Returns:
        Dict mapping feature name -> sorted 1-D array of unique split thresholds.

    Raises:
        TypeError: If the model type is not supported.
    """
    if isinstance(model, lgb.Booster):
        return extract_decision_points_lgb(model, feature_names)

    # XGBoost Booster: avoid importing xgboost at module level
    try:
        import xgboost as xgb

        if isinstance(model, xgb.Booster):
            return extract_decision_points_xgb(model, feature_names)
    except ImportError:
        pass

    raise TypeError(
        f"Unsupported model type: {type(model).__name__}. "
        "SWIFT supports LightGBM Booster and XGBoost Booster."
    )


def extract_decision_points_lgb(
    model: lgb.Booster,
    feature_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Extract unique, sorted split thresholds per feature from a LightGBM Booster.

    For each feature, collects every split threshold used across all trees
    in the ensemble, deduplicates, and returns them in ascending order.

    Args:
        model: A trained LightGBM Booster.
        feature_names: List of feature names (must match model's feature order).

    Returns:
        Dict mapping feature name -> sorted 1-D array of unique split thresholds.
        Features never used in any split get an empty array.
    """
    model_dump = model.dump_model()
    trees = model_dump["tree_info"]

    # Build feature index -> name mapping
    n_features = len(feature_names)
    splits_per_feature: dict[str, list[float]] = {name: [] for name in feature_names}

    for tree_info in trees:
        tree = tree_info["tree_structure"]
        _collect_splits_recursive(tree, feature_names, n_features, splits_per_feature)

    # Sort and deduplicate
    result: dict[str, np.ndarray] = {}
    for fname in feature_names:
        raw = splits_per_feature[fname]
        if raw:
            unique_sorted = np.unique(np.array(raw, dtype=np.float64))
            result[fname] = unique_sorted
        else:
            result[fname] = np.array([], dtype=np.float64)

    n_total = sum(len(v) for v in result.values())
    logger.info(
        "Extracted %d unique split points across %d features from %d trees.",
        n_total,
        n_features,
        len(trees),
    )

    return result


def extract_decision_points_xgb(
    model: object,
    feature_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Extract unique, sorted split thresholds per feature from an XGBoost Booster.

    Uses ``get_dump(dump_format='json')`` to obtain JSON-serialised trees,
    then recursively collects every numeric split threshold per feature.

    Args:
        model: A trained ``xgboost.Booster``.
        feature_names: List of feature names (must match model's feature order).

    Returns:
        Dict mapping feature name -> sorted 1-D array of unique split thresholds.
        Features never used in any split get an empty array.
    """
    import xgboost as xgb  # local import — XGBoost is optional

    if not isinstance(model, xgb.Booster):
        raise TypeError(
            f"Expected xgboost.Booster, got {type(model).__name__}."
        )

    # Build feature-index-to-name mapping.
    # XGBoost uses "f0", "f1", ... by default, or user-set feature names.
    feat_name_set = set(feature_names)
    idx_to_name: dict[str, str] = {}
    for i, name in enumerate(feature_names):
        idx_to_name[f"f{i}"] = name
        idx_to_name[name] = name  # model may already use real names

    tree_dumps = model.get_dump(dump_format="json")

    splits_per_feature: dict[str, list[float]] = {n: [] for n in feature_names}

    for tree_json_str in tree_dumps:
        tree_node = json.loads(tree_json_str)
        _collect_xgb_splits_recursive(tree_node, idx_to_name, feat_name_set, splits_per_feature)

    # Sort and deduplicate
    result: dict[str, np.ndarray] = {}
    for fname in feature_names:
        raw = splits_per_feature[fname]
        if raw:
            result[fname] = np.unique(np.array(raw, dtype=np.float64))
        else:
            result[fname] = np.array([], dtype=np.float64)

    n_total = sum(len(v) for v in result.values())
    logger.info(
        "Extracted %d unique split points across %d features from %d XGBoost trees.",
        n_total,
        len(feature_names),
        len(tree_dumps),
    )

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _collect_splits_recursive(
    node: dict,
    feature_names: Sequence[str],
    n_features: int,
    splits_per_feature: dict[str, list[float]],
) -> None:
    """Recursively traverse a LightGBM tree node and collect split thresholds.

    Args:
        node: A tree node dict from LightGBM's dump_model().
        feature_names: List of feature names.
        n_features: Total number of features.
        splits_per_feature: Accumulator dict (mutated in place).
    """
    # Leaf node — no split
    if "leaf_index" in node:
        return

    # Internal node — has a split
    split_feature_idx = node["split_feature"]
    threshold = node["threshold"]

    # Only collect numeric splits (threshold is a float).
    # Categorical splits have threshold as a string like "0||1||3".
    if isinstance(threshold, (int, float)) and 0 <= split_feature_idx < n_features:
        fname = feature_names[split_feature_idx]
        splits_per_feature[fname].append(float(threshold))

    # Recurse into children
    if "left_child" in node:
        _collect_splits_recursive(
            node["left_child"], feature_names, n_features, splits_per_feature
        )
    if "right_child" in node:
        _collect_splits_recursive(
            node["right_child"], feature_names, n_features, splits_per_feature
        )


def _collect_xgb_splits_recursive(
    node: dict,
    idx_to_name: dict[str, str],
    feat_name_set: set[str],
    splits_per_feature: dict[str, list[float]],
) -> None:
    """Recursively traverse an XGBoost JSON tree node and collect split thresholds.

    Args:
        node: A tree node dict from ``xgb.Booster.get_dump(dump_format='json')``.
        idx_to_name: Mapping from XGBoost split identifiers ("f0", "f1", ...
            or real feature names) to canonical feature names.
        feat_name_set: Set of valid feature names.
        splits_per_feature: Accumulator dict (mutated in place).
    """
    # Leaf node — has "leaf" key, no "split"
    if "leaf" in node:
        return

    # Internal node — has "split" (feature name/id) and "split_condition" (threshold)
    split_id = node.get("split", "")
    threshold = node.get("split_condition")

    # Resolve feature name
    fname = idx_to_name.get(split_id, split_id)
    if fname in feat_name_set and threshold is not None:
        splits_per_feature[fname].append(float(threshold))

    # Recurse into children
    for child in node.get("children", []):
        _collect_xgb_splits_recursive(child, idx_to_name, feat_name_set, splits_per_feature)
