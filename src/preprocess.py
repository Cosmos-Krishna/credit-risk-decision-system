# ============================================================
# preprocess.py — Data Cleaning
# ============================================================
# WHY: Raw data has noise, leakage columns, and missing values.
# Cleaning MUST happen before any modelling step and be shared
# across train/test to avoid leakage.

import pandas as pd
import numpy as np
from src.config import DROP_COLS, TARGET_COL, OUTLIER_COLS


# ── 1. Drop leakage / low-value columns ──────────────────────
def drop_leakage_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: issue_d is post-loan info (data leakage).
    emp_title / title are high-cardinality free text with little
    signal and huge encoding cost.
    address leaks geographic patterns that don't generalise.
    """
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"[Preprocess] Dropped columns: {cols_to_drop}")
    return df


# ── 2. Clean term column ──────────────────────────────────────
def clean_term(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: 'term' comes as ' 36 months' (with space+text).
    We convert to integer months for numeric modelling.
    """
    df["term"] = df["term"].str.extract(r"(\d+)").astype(int)
    return df


# ── 3. Clean emp_length ───────────────────────────────────────
def clean_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: Employment length signals income stability.
    Convert to numeric years; fill NaN with median.
    """
    mapping = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
        "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
        "8 years": 8, "9 years": 9, "10+ years": 10
    }
    df["emp_length"] = df["emp_length"].map(mapping)
    df["emp_length"] = df["emp_length"].fillna(df["emp_length"].median())
    return df


# ── 4. Handle missing values ──────────────────────────────────
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: Most ML models cannot handle NaN.
    - mort_acc (mortgage accounts) correlates strongly with total_acc.
      Group-median imputation preserves this relationship.
    - Other numerics: median is robust to outliers.
    - Categoricals: mode (most common value).
    """
    # mort_acc: group-based imputation
    if "mort_acc" in df.columns and "total_acc" in df.columns:
        group_median = df.groupby("total_acc")["mort_acc"].transform("median")
        df["mort_acc"] = df["mort_acc"].fillna(group_median)
        df["mort_acc"] = df["mort_acc"].fillna(df["mort_acc"].median())

    # Numeric: median
    num_cols = df.select_dtypes("number").columns.tolist()
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical: mode
    cat_cols = df.select_dtypes("object").columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    remaining_nulls = df.isnull().sum().sum()
    print(f"[Preprocess] Imputation done. Remaining nulls: {remaining_nulls}")
    return df


# ── 5. Cap outliers ───────────────────────────────────────────
def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: Extreme outliers in annual_inc (max $8M), dti (9999),
    revol_bal distort distance-based models and tree splits.
    Percentile capping is simple and preserves the column.
    """
    for col, (lo_pct, hi_pct) in OUTLIER_COLS.items():
        if col not in df.columns:
            continue
        lo = df[col].quantile(lo_pct)
        hi = df[col].quantile(hi_pct)
        df[col] = df[col].clip(lo, hi)
        print(f"[Preprocess] Capped {col}: [{lo:.1f}, {hi:.1f}]")
    return df


# ── Master function ───────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Preprocess] Starting data cleaning …")
    df = drop_leakage_cols(df)
    df = clean_term(df)
    df = clean_emp_length(df)
    df = impute_missing(df)
    df = cap_outliers(df)
    print(f"[Preprocess] Clean shape: {df.shape}")
    return df


if __name__ == "__main__":
    from src.data_loader import load_data
    raw = load_data()
    clean = clean_data(raw)
    print(clean.dtypes)
