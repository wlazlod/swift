"""SWIFT: SHAP-Weighted Impact Feature Testing for Model-Aware Distribution Monitoring.

Usage::

    from swift import SWIFTMonitor

    monitor = SWIFTMonitor(model=lgb_model, n_permutations=200)
    monitor.fit(X_ref)
    result = monitor.test(X_mon)
    print(result.drifted_features)
"""

from swift.aggregation import AggregatedScores
from swift.pipeline import SWIFTMonitor
from swift.plotting import plot_bucket_profile, plot_feature_swift_scores
from swift.types import (
    Bucket,
    BucketSet,
    CorrectionMethod,
    FeatureSWIFTResult,
    SWIFTResult,
    WassersteinOrder,
)

__all__ = [
    "AggregatedScores",
    "Bucket",
    "BucketSet",
    "CorrectionMethod",
    "FeatureSWIFTResult",
    "SWIFTMonitor",
    "SWIFTResult",
    "WassersteinOrder",
    "plot_bucket_profile",
    "plot_feature_swift_scores",
]
