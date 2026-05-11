# ============================================================
# feature_engineering.py — Domain-Driven Feature Creation
# ============================================================
# WHY: Raw features rarely capture the economic signal directly.
# We create ratio / interaction features based on credit risk
# domain knowledge, then encode categoricals.

import pandas as pd
import numpy as np
from src.config import CURRENT_YEAR, DTI_BINS, DTI_LABELS, TARGET_COL


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    All new features are domain-motivated:

    loan_to_income      to Higher LTI = bigger burden relative to income.
                          Classic underwriting ratio.
    installment_ratio   to Monthly payment as fraction of income.
                          Predicts payment stress.
    credit_utilization  to How much revolving credit is in use.
                          High utilisation to stretched finances.
    account_age         to Older credit history to more stable borrower.
    dti_bucket          to Non-linear binning of DTI; models handle
                          buckets better than raw DTI skew.
    """
    print("\n[FeatEng] Adding engineered features …")

    # Guard against zero annual_inc (capped to small positive in preprocess)
    annual_inc_safe = df["annual_inc"].replace(0, 1)

    df["loan_to_income"] = df["loan_amnt"] / annual_inc_safe
    df["installment_ratio"] = df["installment"] / annual_inc_safe
    df["credit_utilization"] = df["revol_bal"] / (df["total_acc"] + 1)

    # Extract year from 'earliest_cr_line' (format: 'Jan-2000')
    df["earliest_cr_year"] = pd.to_datetime(
        df["earliest_cr_line"], format="%b-%Y", errors="coerce"
    ).dt.year
    df["account_age"] = CURRENT_YEAR - df["earliest_cr_year"]
    df["account_age"] = df["account_age"].fillna(df["account_age"].median())
    df = df.drop(columns=["earliest_cr_line", "earliest_cr_year"], errors="ignore")

    # DTI bucket (ordered categorical to ordinal or OHE later)
    df["dti_bucket"] = pd.cut(
        df["dti"], bins=DTI_BINS, labels=DTI_LABELS, right=False
    ).astype(str)

    print(f"[FeatEng] New features added: loan_to_income, installment_ratio, "
          f"credit_utilization, account_age, dti_bucket")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: Tree models (XGBoost, RF) need numeric inputs.
    One-hot encoding is simple, transparent, and avoids ordinal
    assumptions. We drop_first=True to avoid dummy variable trap.
    """
    cat_cols = df.select_dtypes("object").columns.tolist()
    # Remove target if it crept back as object
    cat_cols = [c for c in cat_cols if c != TARGET_COL]

    print(f"[FeatEng] One-hot encoding: {cat_cols}")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # Ensure bool columns to int (XGBoost compatibility)
    bool_cols = df.select_dtypes("bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    print(f"[FeatEng] Post-encoding shape: {df.shape}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_engineered_features(df)
    df = encode_categoricals(df)
    return df


if __name__ == "__main__":
    from data_loader import load_data
    from preprocess import clean_data
    raw = load_data()
    clean = clean_data(raw)
    feat = build_features(clean)
    print(feat.head(2).to_string())
