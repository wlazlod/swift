"""Dataset loading and preprocessing for SWIFT experiments.

Supports:
    - Taiwan Credit (UCI id=350): Controlled drift experiments
    - Bank Marketing (UCI id=222): Cross-domain temporal experiments
    - Lending Club (Kaggle): Flagship temporal drift experiments
    - Home Credit (Kaggle): Scalability experiments

Each loader returns a standardized DatasetBundle dataclass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


@dataclass
class DatasetBundle:
    """Standardized output from a dataset loader.

    Attributes:
        name: Dataset identifier (e.g., "taiwan_credit").
        X: Feature matrix (n × p DataFrame).
        y: Binary target (n-length Series, 0/1).
        feature_names: List of feature names used.
        numeric_features: Features treated as numeric.
        categorical_features: Features treated as categorical.
        temporal_column: Column used for temporal ordering (if any).
        temporal_values: Temporal values per row (if any).
        metadata: Additional metadata (source, size, etc.).
    """

    name: str
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    temporal_column: Optional[str] = None
    temporal_values: Optional[pd.Series] = None
    metadata: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Number of observations."""
        return len(self.X)

    @property
    def p(self) -> int:
        """Number of features."""
        return len(self.feature_names)

    def __repr__(self) -> str:
        return (
            f"DatasetBundle(name={self.name!r}, n={self.n}, p={self.p}, "
            f"numeric={len(self.numeric_features)}, "
            f"categorical={len(self.categorical_features)}, "
            f"target_rate={self.y.mean():.3f})"
        )


def load_taiwan_credit() -> DatasetBundle:
    """Load and preprocess the Taiwan Credit dataset (UCI id=350).

    Target: default.payment.next.month (binary 0/1)
    Features: 23 features (LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
              PAY_0-PAY_6, BILL_AMT1-BILL_AMT6, PAY_AMT1-PAY_AMT6)

    Preprocessing:
        - SEX, EDUCATION, MARRIAGE treated as categorical (label-encoded for LightGBM)
        - PAY_0 through PAY_6 treated as ordinal numeric
        - All others treated as numeric
    """
    from ucimlrepo import fetch_ucirepo

    logger.info("Fetching Taiwan Credit dataset from UCI (id=350)...")
    dataset = fetch_ucirepo(id=350)

    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()

    # Rename columns from X1-X23 to descriptive names
    # (UCI repo returns generic Xn names; descriptions map to original names)
    col_rename = {
        "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE",
        "X5": "AGE", "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3",
        "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
        "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3",
        "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6",
        "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3",
        "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6",
    }
    # Only rename columns that exist (handles both Xn and named formats)
    rename_map = {k: v for k, v in col_rename.items() if k in X.columns}
    if rename_map:
        X = X.rename(columns=rename_map)

    # Ensure target is 0/1 integer
    y = y.astype(int)

    # Define feature types
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Clean EDUCATION: values 0, 5, 6 are undocumented → merge into "other" = 4
    X["EDUCATION"] = X["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # Ensure categorical columns are integer-coded for LightGBM native categorical
    for col in categorical_cols:
        X[col] = X[col].astype(int)

    # Ensure numeric columns are float
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

    feature_names = list(X.columns)

    bundle = DatasetBundle(
        name="taiwan_credit",
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        feature_names=feature_names,
        numeric_features=numeric_cols,
        categorical_features=categorical_cols,
        metadata={
            "source": "UCI ML Repository (id=350)",
            "url": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
            "original_size": len(X),
            "target_column": "default.payment.next.month",
            "license": "CC BY 4.0",
        },
    )

    logger.info("Taiwan Credit loaded: %s", bundle)
    return bundle


def load_bank_marketing() -> DatasetBundle:
    """Load and preprocess the Bank Marketing dataset (UCI id=222).

    Target: y (binary: yes=1, no=0 — subscribed to term deposit)
    Features: 20 features (after excluding 'duration' per standard practice)

    Preprocessing:
        - Remove 'duration' (data leakage — only known after call)
        - Binary encode target: yes → 1, no → 0
        - Categorical features are label-encoded for LightGBM
        - Data is ordered temporally (May 2008 – Nov 2010)
        - 'month' column used as temporal indicator
    """
    from ucimlrepo import fetch_ucirepo

    logger.info("Fetching Bank Marketing dataset from UCI (id=222)...")
    dataset = fetch_ucirepo(id=222)

    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()

    # Binary encode target
    y = (y == "yes").astype(int)

    # Remove 'duration' (data leakage per original paper)
    if "duration" in X.columns:
        X = X.drop(columns=["duration"])

    # Preserve month for temporal ordering before encoding
    # The dataset is ordered by date (May 2008 – Nov 2010 per UCI docs)
    month_order = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    # Create a row index as temporal proxy (data is time-ordered)
    temporal_values = pd.Series(range(len(X)), name="row_order")

    # Identify feature types
    categorical_cols = X.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Label-encode categorical features for LightGBM
    for col in categorical_cols:
        X[col] = X[col].astype("category").cat.codes.astype(int)

    feature_names = list(X.columns)

    bundle = DatasetBundle(
        name="bank_marketing",
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        feature_names=feature_names,
        numeric_features=numeric_cols,
        categorical_features=categorical_cols,
        temporal_column="row_order",
        temporal_values=temporal_values.reset_index(drop=True),
        metadata={
            "source": "UCI ML Repository (id=222)",
            "url": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
            "original_size": len(X),
            "target_column": "y",
            "license": "CC BY 4.0",
            "note": "duration removed (data leakage). Data ordered May 2008 – Nov 2010.",
        },
    )

    logger.info("Bank Marketing loaded: %s", bundle)
    return bundle


def load_lending_club(
    data_path: Path | str | None = None,
    sample_frac: float | None = None,
    random_state: int = 42,
) -> DatasetBundle:
    """Load and preprocess the Lending Club dataset.

    This dataset must be downloaded manually from Kaggle:
        kaggle datasets download -d wordsforthewise/lending-club

    Unzip into DWArticles/swift/data/lending_club/

    Args:
        data_path: Path to the CSV file. If None, looks in data/lending_club/.
        sample_frac: If set, sample this fraction of data (for development).
        random_state: Random seed for sampling.

    Target: loan_status → binary (Fully Paid=0, Charged Off/Default=1)
    Temporal: issue_d (loan issue date, YYYY-MM format)
    """
    if data_path is None:
        data_dir = DATA_DIR / "lending_club"
        # The Kaggle download creates a subdirectory structure:
        # lending_club/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv
        # Try nested CSVs first, then top-level CSVs, then gzipped files
        candidates = sorted(data_dir.glob("**/*.csv"))
        # Filter to actual files (not directories named *.csv)
        candidates = [c for c in candidates if c.is_file()]
        if not candidates:
            # Try gzipped
            candidates = sorted(data_dir.glob("*.csv.gz"))
        if not candidates:
            raise FileNotFoundError(
                f"No CSV files found in {data_dir}. Download from Kaggle:\n"
                "  kaggle datasets download -d wordsforthewise/lending-club\n"
                f"  unzip -d {data_dir} lending-club.zip"
            )
        # Prefer the 'accepted' file (not 'rejected')
        accepted = [c for c in candidates if "accepted" in c.name.lower()]
        data_path = accepted[0] if accepted else candidates[0]

    data_path = Path(data_path)
    logger.info("Loading Lending Club from %s...", data_path)

    # Load with low_memory=False to avoid dtype warnings
    df = pd.read_csv(data_path, low_memory=False)

    # Drop footer rows that are all NaN (some Kaggle CSV files have them)
    df = df.dropna(how="all")

    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=random_state)

    # Filter to completed loans only
    valid_statuses = {"Fully Paid", "Charged Off", "Default"}
    mask = df["loan_status"].isin(valid_statuses)
    df = df[mask].copy()

    # Binary target: 0 = Fully Paid, 1 = Charged Off / Default
    y = (df["loan_status"].isin({"Charged Off", "Default"})).astype(int)

    # Parse issue_d to datetime for temporal ordering (format: "Dec-2015")
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    temporal_values = df["issue_d"].copy()

    # Select features (exclude target, ID, date, text, and leakage columns)
    exclude_cols = {
        # Target and ID columns
        "loan_status", "id", "member_id",
        # Date columns (used separately or leakage)
        "issue_d", "earliest_cr_line", "last_pymnt_d", "next_pymnt_d",
        "last_credit_pull_d",
        # Text / high-cardinality columns
        "url", "desc", "title", "emp_title", "zip_code", "addr_state",
        # Post-origination / leakage columns
        "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
        "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
        "recoveries", "collection_recovery_fee", "last_pymnt_amnt",
        # Constant or near-constant columns
        "policy_code", "pymnt_plan",
        # Joint application columns (>98% null)
        "application_type", "annual_inc_joint", "dti_joint",
        "verification_status_joint", "revol_bal_joint",
        # Hardship columns (>99% null)
        "hardship_flag", "hardship_type", "hardship_reason",
        "hardship_status", "hardship_start_date", "hardship_end_date",
        "hardship_length", "hardship_dpd", "hardship_loan_status",
        "hardship_amount", "hardship_payoff_balance_amount",
        "hardship_last_payment_amount", "deferral_term",
        "payment_plan_start_date",
        "orig_projected_additional_accrued_interest",
        # Secondary applicant columns (>99% null)
        "sec_app_fico_range_low", "sec_app_fico_range_high",
        "sec_app_earliest_cr_line", "sec_app_inq_last_6mths",
        "sec_app_mort_acc", "sec_app_open_acc", "sec_app_revol_util",
        "sec_app_open_act_il", "sec_app_num_rev_accts",
        "sec_app_chargeoff_within_12_mths",
        "sec_app_collections_12_mths_ex_med",
        "sec_app_mths_since_last_major_derog",
        # Settlement columns (>97% null)
        "debt_settlement_flag", "debt_settlement_flag_date",
        "settlement_status", "settlement_date", "settlement_amount",
        "settlement_percentage", "settlement_term",
        # Disbursement method
        "disbursement_method",
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()

    # Clean 'term': strip whitespace and convert to numeric months
    if "term" in X.columns:
        X["term"] = X["term"].astype(str).str.strip().str.replace(" months", "", regex=False)
        X["term"] = pd.to_numeric(X["term"], errors="coerce")

    # Clean 'emp_length': convert to numeric years
    if "emp_length" in X.columns:
        emp_map = {
            "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
            "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
            "8 years": 8, "9 years": 9, "10+ years": 10,
        }
        X["emp_length"] = X["emp_length"].map(emp_map)  # NaN stays NaN

    # Clean 'int_rate': may have '%' suffix in some versions
    if "int_rate" in X.columns and X["int_rate"].dtype == object:
        X["int_rate"] = X["int_rate"].astype(str).str.replace("%", "", regex=False)
        X["int_rate"] = pd.to_numeric(X["int_rate"], errors="coerce")

    # Identify feature types
    categorical_cols = X.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Label-encode categorical features
    for col in categorical_cols:
        X[col] = X[col].astype("category").cat.codes.astype(int)
        # -1 from cat.codes means NaN → replace with -1 (LightGBM handles this)

    # Ensure numeric columns are float
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

    # Drop columns that are >95% null (not useful)
    null_frac = X.isnull().mean()
    drop_cols = null_frac[null_frac > 0.95].index.tolist()
    if drop_cols:
        logger.info("Dropping %d columns with >95%% nulls: %s", len(drop_cols), drop_cols[:5])
        X = X.drop(columns=drop_cols)
        categorical_cols = [c for c in categorical_cols if c not in drop_cols]
        numeric_cols = [c for c in numeric_cols if c not in drop_cols]

    feature_names = list(X.columns)

    bundle = DatasetBundle(
        name="lending_club",
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        feature_names=feature_names,
        numeric_features=numeric_cols,
        categorical_features=categorical_cols,
        temporal_column="issue_d",
        temporal_values=temporal_values.reset_index(drop=True),
        metadata={
            "source": "Kaggle (wordsforthewise/lending-club)",
            "url": "https://www.kaggle.com/datasets/wordsforthewise/lending-club",
            "original_size": len(X),
            "target_column": "loan_status",
            "dropped_high_null": drop_cols,
        },
    )

    logger.info("Lending Club loaded: %s", bundle)
    return bundle


def load_home_credit(
    data_path: Path | str | None = None,
) -> DatasetBundle:
    """Load and preprocess the Home Credit Default Risk dataset.

    This dataset must be downloaded manually from Kaggle:
        kaggle datasets download -d home-credit-default-risk
        unzip -d data/home_credit/ home-credit-default-risk.zip

    Uses only application_train.csv (main table). Relational tables
    (bureau, previous_application, etc.) are not used.

    Args:
        data_path: Path to application_train.csv. If None, looks in data/home_credit/.

    Target: TARGET (binary 0/1, 1 = default)
    """
    if data_path is None:
        data_dir = DATA_DIR / "home_credit"
        data_path = data_dir / "application_train.csv"
        if not data_path.exists():
            raise FileNotFoundError(
                f"File not found: {data_path}. Download from Kaggle:\n"
                "  kaggle competitions download -c home-credit-default-risk\n"
                f"  unzip -d {data_dir} home-credit-default-risk.zip"
            )

    data_path = Path(data_path)
    logger.info("Loading Home Credit from %s...", data_path)

    df = pd.read_csv(data_path)

    # Target
    y = df["TARGET"].astype(int)

    # Exclude ID and target columns
    exclude_cols = {"SK_ID_CURR", "TARGET"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()

    # Clean DAYS_EMPLOYED anomaly: 365243 = pensioner/not employed → NaN
    if "DAYS_EMPLOYED" in X.columns:
        anomaly_mask = X["DAYS_EMPLOYED"] == 365243
        n_anomaly = anomaly_mask.sum()
        if n_anomaly > 0:
            logger.info(
                "Replacing %d DAYS_EMPLOYED anomalies (365243) with NaN",
                n_anomaly,
            )
            X.loc[anomaly_mask, "DAYS_EMPLOYED"] = np.nan

    # Identify feature types
    categorical_cols = X.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Label-encode categorical features for LightGBM
    for col in categorical_cols:
        X[col] = X[col].astype("category").cat.codes.astype(int)
        # -1 from cat.codes means NaN → LightGBM handles -1 as missing

    # Ensure numeric columns are float
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

    # Drop columns that are >95% null (not useful)
    null_frac = X.isnull().mean()
    drop_cols = null_frac[null_frac > 0.95].index.tolist()
    if drop_cols:
        logger.info("Dropping %d columns with >95%% nulls: %s", len(drop_cols), drop_cols[:5])
        X = X.drop(columns=drop_cols)
        categorical_cols = [c for c in categorical_cols if c not in drop_cols]
        numeric_cols = [c for c in numeric_cols if c not in drop_cols]

    feature_names = list(X.columns)

    bundle = DatasetBundle(
        name="home_credit",
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        feature_names=feature_names,
        numeric_features=numeric_cols,
        categorical_features=categorical_cols,
        metadata={
            "source": "Kaggle (home-credit-default-risk)",
            "url": "https://www.kaggle.com/c/home-credit-default-risk",
            "original_size": len(X),
            "target_column": "TARGET",
            "dropped_high_null": drop_cols,
        },
    )

    logger.info("Home Credit loaded: %s", bundle)
    return bundle


def create_temporal_splits(
    bundle: DatasetBundle,
    period: str = "Q",
    min_window_size: int = 1000,
) -> list[tuple[str, pd.DataFrame, pd.Series]]:
    """Split a dataset into temporal windows.

    Args:
        bundle: DatasetBundle with temporal_values set.
        period: Pandas period string ('Q' for quarterly, 'M' for monthly).
        min_window_size: Minimum observations per window.

    Returns:
        List of (period_label, X_window, y_window) tuples.
    """
    if bundle.temporal_values is None:
        raise ValueError(f"Dataset {bundle.name} has no temporal column.")

    temporal = bundle.temporal_values.copy()
    if not pd.api.types.is_datetime64_any_dtype(temporal):
        temporal = pd.to_datetime(temporal, errors="coerce")

    periods = temporal.dt.to_period(period)
    windows = []

    for p in sorted(periods.unique()):
        mask = periods == p
        if mask.sum() >= min_window_size:
            X_w = bundle.X.loc[mask].copy()
            y_w = bundle.y.loc[mask].copy()
            windows.append((str(p), X_w, y_w))
        else:
            logger.debug(
                "Skipping period %s: only %d observations (min=%d)",
                p, mask.sum(), min_window_size,
            )

    logger.info(
        "Created %d temporal windows from %s (period=%s, min_size=%d)",
        len(windows), bundle.name, period, min_window_size,
    )
    return windows
