"""Smoke test: verify the package imports and fixtures work."""

from swift.types import Bucket, BucketType


def test_package_imports():
    """The swift package should be importable."""
    import swift

    assert hasattr(swift, "Bucket")
    assert hasattr(swift, "BucketSet")


def test_bucket_contains_null():
    """Null bucket should match None and NaN values."""
    null_bucket = Bucket(bucket_type=BucketType.NULL, index=0)
    assert null_bucket.contains(None)
    assert not null_bucket.contains(1.0)


def test_fixtures_load(synthetic_data, trained_lgb_model):
    """Fixtures should produce valid data and a trained model."""
    assert synthetic_data["X_train"].shape == (2000, 5)
    assert trained_lgb_model is not None
