# ============================================================
# eda.py — Exploratory Data Analysis
# ============================================================
# WHY: Before modelling, we must understand the data distribution,
# class imbalance, relationships between features and the target,
# and decide which features to keep / engineer.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from src.config import TARGET_COL, OUTPUT_DIR

PALETTE = {"Paid": "#2ecc71", "Default": "#e74c3c"}


def _save(fig, name: str):
    path = f"{OUTPUT_DIR}/{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved to {path}")


def plot_target_distribution(df: pd.DataFrame):
    """
    WHY: Shows class imbalance. ~20 % default rate to imbalanced
    dataset to need class weights / SMOTE in modelling.
    """
    counts = df[TARGET_COL].value_counts()
    labels = ["Fully Paid", "Charged Off"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, counts.values,
                color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.2)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")
    axes[0].set_title("Loan Status Count", fontsize=13)
    axes[0].set_ylabel("Count")

    axes[1].pie(counts.values, labels=labels, autopct="%1.1f%%",
                colors=["#2ecc71", "#e74c3c"], startangle=90,
                wedgeprops={"edgecolor": "white"})
    axes[1].set_title("Class Split", fontsize=13)

    fig.suptitle("Target Distribution (1 = Default)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "01_target_distribution")


def plot_interest_vs_default(df: pd.DataFrame):
    """
    WHY: Higher interest rates to lender perceives more risk to expect
    higher default rate. Confirms 'int_rate' is a strong signal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # KDE distribution
    for val, label, color in [(0, "Fully Paid", "#2ecc71"),
                               (1, "Charged Off", "#e74c3c")]:
        subset = df[df[TARGET_COL] == val]["int_rate"]
        subset.plot.kde(ax=axes[0], label=label, color=color, linewidth=2)
    axes[0].set_title("Interest Rate Distribution by Loan Status")
    axes[0].set_xlabel("Interest Rate (%)")
    axes[0].legend()

    # Mean int_rate by grade
    grade_default = df.groupby("grade")["int_rate"].mean().sort_index()
    grade_dr = df.groupby("grade")[TARGET_COL].mean().sort_index() * 100
    ax2 = axes[1]
    ax2b = ax2.twinx()
    ax2.bar(grade_default.index, grade_default.values,
            color="#3498db", alpha=0.6, label="Avg Interest Rate")
    ax2b.plot(grade_dr.index, grade_dr.values,
              color="#e74c3c", marker="o", linewidth=2, label="Default Rate %")
    ax2.set_title("Avg Interest Rate & Default Rate by Grade")
    ax2.set_xlabel("Loan Grade")
    ax2.set_ylabel("Interest Rate (%)")
    ax2b.set_ylabel("Default Rate (%)")
    ax2.legend(loc="upper left")
    ax2b.legend(loc="upper right")

    fig.tight_layout()
    _save(fig, "02_interest_vs_default")


def plot_grade_vs_default(df: pd.DataFrame):
    """
    WHY: Grade is LendingClub's own risk bucket. Validates that our
    model should perform at least as well as grade-based segmentation.
    """
    grade_stats = df.groupby("grade").agg(
        total=(TARGET_COL, "count"),
        defaults=(TARGET_COL, "sum")
    )
    grade_stats["default_rate"] = grade_stats["defaults"] / grade_stats["total"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(grade_stats)))
    axes[0].bar(grade_stats.index, grade_stats["default_rate"], color=colors)
    for i, (idx, row) in enumerate(grade_stats.iterrows()):
        axes[0].text(i, row["default_rate"] + 0.3,
                     f"{row['default_rate']:.1f}%", ha="center", fontsize=9)
    axes[0].set_title("Default Rate by Loan Grade")
    axes[0].set_xlabel("Grade")
    axes[0].set_ylabel("Default Rate (%)")

    axes[1].bar(grade_stats.index, grade_stats["total"], color="#3498db",
                alpha=0.7, label="Total")
    axes[1].bar(grade_stats.index, grade_stats["defaults"], color="#e74c3c",
                alpha=0.9, label="Defaults")
    axes[1].set_title("Volume & Defaults by Grade")
    axes[1].set_xlabel("Grade")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.tight_layout()
    _save(fig, "03_grade_vs_default")


def plot_correlation_heatmap(df: pd.DataFrame):
    """
    WHY: Spot highly-correlated predictors (multicollinearity risk for
    Logistic Regression) and see which features correlate with target.
    """
    num_df = df.select_dtypes("number").drop(columns=[TARGET_COL], errors="ignore")
    corr = num_df.corrwith(df[TARGET_COL]).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Correlation with target
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in corr.values]
    axes[0].barh(corr.index, corr.values, color=colors)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Feature Correlation with Target (default=1)")
    axes[0].set_xlabel("Pearson r")

    # Full heatmap (numeric only, top columns)
    top_cols = num_df.columns[:12]
    sns.heatmap(num_df[top_cols].corr(), ax=axes[1], cmap="coolwarm",
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.3)
    axes[1].set_title("Feature Correlation Heatmap")

    fig.tight_layout()
    _save(fig, "04_correlation_heatmap")


def plot_loan_amount_distribution(df: pd.DataFrame):
    """
    WHY: Loan amount is a key driver of loss severity (LGD).
    Understanding its distribution helps set portfolio limits.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for val, label, color in [(0, "Fully Paid", "#2ecc71"),
                               (1, "Charged Off", "#e74c3c")]:
        df[df[TARGET_COL] == val]["loan_amnt"].plot.hist(
            ax=axes[0], bins=40, alpha=0.6, color=color, label=label, density=True)
    axes[0].set_title("Loan Amount Distribution by Status")
    axes[0].set_xlabel("Loan Amount ($)")
    axes[0].legend()

    # Box-plot by term
    df.boxplot(column="loan_amnt", by=TARGET_COL, ax=axes[1],
               boxprops=dict(color="#3498db"),
               medianprops=dict(color="#e74c3c"))
    axes[1].set_title("Loan Amount by Default Status")
    axes[1].set_xlabel("Default (0=Paid, 1=Default)")
    axes[1].set_ylabel("Loan Amount ($)")
    fig.suptitle("")

    fig.tight_layout()
    _save(fig, "05_loan_amount_distribution")


def run_eda(df: pd.DataFrame):
    print("\n[EDA] Running exploratory data analysis …")
    plot_target_distribution(df)
    plot_interest_vs_default(df)
    plot_grade_vs_default(df)
    plot_correlation_heatmap(df)
    plot_loan_amount_distribution(df)
    print("[EDA] All plots saved.")


if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    run_eda(df)
