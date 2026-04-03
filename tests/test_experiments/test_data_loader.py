"""Tests for experiment data loaders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.data_loader import (
    DatasetBundle,
    create_temporal_splits,
    load_bank_marketing,
    load_home_credit,
    load_lending_club,
    load_taiwan_credit,
)


class TestDatasetBundle:
    """Tests for the DatasetBundle dataclass."""

    def test_bundle_properties(self):
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        bundle = DatasetBundle(
            name="test",
            X=X,
            y=y,
            feature_names=["a", "b"],
            numeric_features=["a", "b"],
            categorical_features=[],
        )
        assert bundle.n == 3
        assert bundle.p == 2
        assert "test" in repr(bundle)

    def test_bundle_with_temporal(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        temporal = pd.Series(pd.to_datetime(["2020-01", "2020-02", "2020-03"]))
        bundle = DatasetBundle(
            name="test",
            X=X,
            y=y,
            feature_names=["a"],
            numeric_features=["a"],
            categorical_features=[],
            temporal_column="date",
            temporal_values=temporal,
        )
        assert bundle.temporal_column == "date"
        assert len(bundle.temporal_values) == 3


class TestTaiwanCredit:
    """Tests for Taiwan Credit dataset loading."""

    @pytest.fixture(scope="class")
    def bundle(self) -> DatasetBundle:
        return load_taiwan_credit()

    def test_shape(self, bundle):
        """Should have 30K rows and 23 features."""
        assert bundle.n == 30000
        assert bundle.p == 23

    def test_target_binary(self, bundle):
        """Target should be binary 0/1."""
        assert set(bundle.y.unique()) == {0, 1}

    def test_target_rate(self, bundle):
        """Default rate should be ~22%."""
        rate = bundle.y.mean()
        assert 0.15 < rate < 0.30

    def test_feature_types(self, bundle):
        """Should have 3 categorical and 20 numeric features."""
        assert len(bundle.categorical_features) == 3
        assert len(bundle.numeric_features) == 20
        assert set(bundle.categorical_features) == {"SEX", "EDUCATION", "MARRIAGE"}

    def test_no_nulls(self, bundle):
        """Taiwan Credit should have no missing values."""
        assert bundle.X.isnull().sum().sum() == 0

    def test_education_cleaned(self, bundle):
        """EDUCATION should not have undocumented values (0, 5, 6)."""
        edu_values = set(bundle.X["EDUCATION"].unique())
        assert edu_values.issubset({1, 2, 3, 4})

    def test_metadata(self, bundle):
        """Should have proper metadata."""
        assert "UCI" in bundle.metadata["source"]
        assert bundle.name == "taiwan_credit"


class TestBankMarketing:
    """Tests for Bank Marketing dataset loading."""

    @pytest.fixture(scope="class")
    def bundle(self) -> DatasetBundle:
        return load_bank_marketing()

    def test_shape(self, bundle):
        """Should have ~41K-45K rows."""
        assert 40000 < bundle.n < 46000

    def test_no_duration(self, bundle):
        """Duration column should be removed (data leakage)."""
        assert "duration" not in bundle.feature_names

    def test_target_binary(self, bundle):
        """Target should be binary 0/1."""
        assert set(bundle.y.unique()) == {0, 1}

    def test_target_rate(self, bundle):
        """Subscription rate should be ~11%."""
        rate = bundle.y.mean()
        assert 0.08 < rate < 0.15

    def test_has_categorical(self, bundle):
        """Should have categorical features (job, marital, etc.)."""
        assert len(bundle.categorical_features) > 0

    def test_categoricals_encoded(self, bundle):
        """Categorical features should be integer-encoded."""
        for col in bundle.categorical_features:
            assert bundle.X[col].dtype in [np.int64, np.int32, int]

    def test_temporal_column(self, bundle):
        """Should have temporal ordering."""
        assert bundle.temporal_column is not None
        assert bundle.temporal_values is not None
        assert len(bundle.temporal_values) == bundle.n

    def test_metadata(self, bundle):
        """Should have proper metadata."""
        assert "UCI" in bundle.metadata["source"]
        assert bundle.name == "bank_marketing"


class TestLendingClub:
    """Tests for Lending Club dataset loading (sampled for speed)."""

    @pytest.fixture(scope="class")
    def bundle(self) -> DatasetBundle:
        return load_lending_club(sample_frac=0.005, random_state=42)

    def test_shape_sampled(self, bundle):
        """Sampled dataset should have reasonable size and many features."""
        assert bundle.n > 5000  # ~0.5% of ~2.2M filtered to completed
        assert bundle.p > 30  # Many features after cleanup

    def test_target_binary(self, bundle):
        """Target should be binary 0/1."""
        assert set(bundle.y.unique()) == {0, 1}

    def test_target_rate(self, bundle):
        """Default rate should be ~20%."""
        rate = bundle.y.mean()
        assert 0.10 < rate < 0.30

    def test_name(self, bundle):
        """Bundle name should be lending_club."""
        assert bundle.name == "lending_club"

    def test_term_is_numeric(self, bundle):
        """Term should be numeric (36 or 60), not string."""
        if "term" in bundle.feature_names:
            assert bundle.X["term"].dtype in [np.float64, float]
            assert set(bundle.X["term"].dropna().unique()).issubset({36.0, 60.0})

    def test_emp_length_is_numeric(self, bundle):
        """emp_length should be numeric 0-10, not string."""
        if "emp_length" in bundle.feature_names:
            assert bundle.X["emp_length"].dtype in [np.float64, float]
            valid = bundle.X["emp_length"].dropna()
            assert valid.min() >= 0
            assert valid.max() <= 10

    def test_has_categorical(self, bundle):
        """Should have categorical features (grade, purpose, etc.)."""
        assert len(bundle.categorical_features) > 0

    def test_categoricals_encoded(self, bundle):
        """Categorical features should be integer-encoded."""
        for col in bundle.categorical_features:
            assert bundle.X[col].dtype in [np.int64, np.int32, int]

    def test_temporal_column(self, bundle):
        """Should have temporal info (issue_d)."""
        assert bundle.temporal_column == "issue_d"
        assert bundle.temporal_values is not None
        assert len(bundle.temporal_values) == bundle.n

    def test_temporal_is_datetime(self, bundle):
        """Temporal values should be datetime."""
        assert pd.api.types.is_datetime64_any_dtype(bundle.temporal_values)
        # Range should span 2007–2018
        min_date = bundle.temporal_values.min()
        max_date = bundle.temporal_values.max()
        assert min_date.year <= 2010
        assert max_date.year >= 2017

    def test_no_leakage_columns(self, bundle):
        """Post-origination leakage columns should be excluded."""
        leakage = {
            "total_pymnt", "recoveries", "collection_recovery_fee",
            "last_pymnt_amnt", "out_prncp",
        }
        for col in leakage:
            assert col not in bundle.feature_names

    def test_no_id_columns(self, bundle):
        """ID columns should be excluded."""
        for col in ["id", "member_id", "url"]:
            assert col not in bundle.feature_names

    def test_metadata(self, bundle):
        """Should have Kaggle source metadata."""
        assert "Kaggle" in bundle.metadata["source"]

    def test_no_all_null_features(self, bundle):
        """No feature should be 100% null."""
        null_frac = bundle.X.isnull().mean()
        assert (null_frac < 1.0).all()


class TestHomeCredit:
    """Tests for Home Credit dataset loading."""

    @pytest.fixture(scope="class")
    def bundle(self) -> DatasetBundle:
        return load_home_credit()

    def test_shape(self, bundle):
        """Should have ~307K rows and many features."""
        assert 300000 < bundle.n < 310000
        assert bundle.p > 50  # Many features

    def test_target_binary(self, bundle):
        """Target should be binary 0/1."""
        assert set(bundle.y.unique()) == {0, 1}

    def test_target_rate(self, bundle):
        """Default rate should be ~8%."""
        rate = bundle.y.mean()
        assert 0.05 < rate < 0.12

    def test_name(self, bundle):
        """Bundle name should be home_credit."""
        assert bundle.name == "home_credit"

    def test_no_id_column(self, bundle):
        """SK_ID_CURR should not be in features."""
        assert "SK_ID_CURR" not in bundle.feature_names

    def test_no_target_column(self, bundle):
        """TARGET should not be in features."""
        assert "TARGET" not in bundle.feature_names

    def test_days_employed_cleaned(self, bundle):
        """DAYS_EMPLOYED anomaly (365243) should be replaced with NaN."""
        if "DAYS_EMPLOYED" in bundle.feature_names:
            assert (bundle.X["DAYS_EMPLOYED"] == 365243).sum() == 0

    def test_has_categorical(self, bundle):
        """Should have categorical features."""
        assert len(bundle.categorical_features) > 0

    def test_categoricals_encoded(self, bundle):
        """Categorical features should be integer-encoded."""
        for col in bundle.categorical_features:
            assert bundle.X[col].dtype in [np.int64, np.int32, int]

    def test_no_temporal(self, bundle):
        """Home Credit has no temporal column (cross-sectional)."""
        assert bundle.temporal_column is None

    def test_metadata(self, bundle):
        """Should have Kaggle source metadata."""
        assert "Kaggle" in bundle.metadata["source"]

    def test_no_all_null_features(self, bundle):
        """No feature should be >95% null after cleanup."""
        null_frac = bundle.X.isnull().mean()
        assert (null_frac < 0.96).all()


class TestTemporalSplits:
    """Tests for create_temporal_splits (using Lending Club)."""

    @pytest.fixture(scope="class")
    def bundle(self) -> DatasetBundle:
        return load_lending_club(sample_frac=0.01, random_state=42)

    def test_quarterly_splits(self, bundle):
        """Should produce multiple quarterly windows."""
        splits = create_temporal_splits(bundle, period="Q", min_window_size=50)
        assert len(splits) > 10  # At least 10 quarters in 2007-2018

    def test_split_labels_are_strings(self, bundle):
        """Period labels should be strings like '2015Q4'."""
        splits = create_temporal_splits(bundle, period="Q", min_window_size=50)
        for label, X_w, y_w in splits:
            assert isinstance(label, str)
            assert "Q" in label or "M" in label or label[0].isdigit()

    def test_splits_have_data(self, bundle):
        """Each split should have X and y with matching lengths."""
        splits = create_temporal_splits(bundle, period="Q", min_window_size=50)
        for label, X_w, y_w in splits:
            assert len(X_w) == len(y_w)
            assert len(X_w) >= 50

    def test_min_window_filters(self, bundle):
        """Higher min_window_size should produce fewer windows."""
        splits_small = create_temporal_splits(bundle, period="Q", min_window_size=10)
        splits_large = create_temporal_splits(bundle, period="Q", min_window_size=500)
        assert len(splits_small) >= len(splits_large)
