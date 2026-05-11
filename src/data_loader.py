# ============================================================
# data_loader.py — Load & Basic Sanity Checks
# ============================================================
# WHY: Centralise all I/O here so the rest of the pipeline
# never touches file paths or target encoding directly.

import pandas as pd
from src.config import DATA_PATH, TARGET_COL, DEFAULT_LABEL, PAID_LABEL


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw CSV and encode the binary target."""
    print(f"[DataLoader] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"[DataLoader] Raw shape: {df.shape}")

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"[DataLoader] Removing {duplicates} duplicate rows")
        df = df.drop_duplicates()

    # ── Encode target ────────────────────────────────────────
    # Keep only the two relevant statuses (no 'Current', 'Late' etc.)
    valid_statuses = {DEFAULT_LABEL, PAID_LABEL}
    before = len(df)
    df = df[df[TARGET_COL].isin(valid_statuses)].copy()
    print(f"[DataLoader] Rows after target filter: {len(df)} "
          f"(removed {before - len(df)} rows with other statuses)")

    df[TARGET_COL] = (df[TARGET_COL] == DEFAULT_LABEL).astype(int)
    default_rate = df[TARGET_COL].mean()
    print(f"[DataLoader] Default rate: {default_rate:.2%}  "
          f"(1=default, 0=paid)")

    return df


def quick_summary(df: pd.DataFrame) -> None:
    """Print a concise data summary for EDA awareness."""
    print("\n=== QUICK SUMMARY ===")
    print(f"Shape          : {df.shape}")
    print(f"Numeric cols   : {df.select_dtypes('number').shape[1]}")
    print(f"Categorical cols: {df.select_dtypes('object').shape[1]}")
    print("\nMissing values (non-zero only):")
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    for col, cnt in miss.items():
        print(f"  {col:30s}: {cnt:>6d}  ({cnt/len(df):.1%})")
    print(f"\nTarget distribution:\n{df[TARGET_COL].value_counts()}")


if __name__ == "__main__":
    df = load_data()
    quick_summary(df)
