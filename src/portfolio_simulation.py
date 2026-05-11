# ============================================================
# portfolio_simulation.py — Portfolio Simulation & Business Impact
# ============================================================
# WHY: The model only has value if it produces better business
# outcomes than naive lending. We simulate the portfolio and
# compare system decisions vs. approving everyone (baseline).

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import OUTPUT_DIR, RECOVERY_RATE


def _save(fig, name):
    path = f"{OUTPUT_DIR}/{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[Portfolio] Saved to {path}")


def summarise_portfolio(decisions_df: pd.DataFrame, y_true: pd.Series = None) -> dict:
    """
    decisions_df: output of decision_engine.run_decision_engine()
    y_true:       actual loan_status (0/1) if available (test set)
    """
    total     = len(decisions_df)
    approved  = (decisions_df["decision"] == "APPROVE").sum()
    reviewed  = (decisions_df["decision"] == "REVIEW").sum()
    rejected  = (decisions_df["decision"] == "REJECT").sum()
    exp_profit_approved = decisions_df.loc[
        decisions_df["decision"] == "APPROVE", "expected_profit"
    ].sum()

    summary = {
        "total_loans"          : total,
        "approved"             : approved,
        "reviewed"             : reviewed,
        "rejected"             : rejected,
        "approval_rate"        : approved / total,
        "expected_profit_total": exp_profit_approved,
        "avg_profit_per_loan"  : exp_profit_approved / max(approved, 1),
    }

    if y_true is not None:
        y_true = y_true.reset_index(drop=True)
        approved_mask = decisions_df["decision"] == "APPROVE"
        actual_defaults_in_approved = y_true[approved_mask].sum()
        summary["actual_defaults_in_approved"] = int(actual_defaults_in_approved)
        summary["default_rate_in_approved"]    = (
            actual_defaults_in_approved / max(approved, 1)
        )

    print("\n[Portfolio] === PORTFOLIO SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:35s}: {v:>12.2f}")
        else:
            print(f"  {k:35s}: {v:>12,}")

    return summary


def plot_risk_segmentation(decisions_df: pd.DataFrame):
    """
    WHY: Visualise how the model segments borrowers into risk tiers.
    This is the core output for a credit risk manager.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Decision pie
    counts = decisions_df["decision"].value_counts()
    colors = {"APPROVE": "#2ecc71", "REVIEW": "#f39c12", "REJECT": "#e74c3c"}
    c_list = [colors.get(k, "#999") for k in counts.index]
    axes[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=c_list, wedgeprops={"edgecolor": "white"})
    axes[0].set_title("Decision Distribution")

    # PD distribution by risk level
    for level, color in [("Low", "#2ecc71"), ("Medium", "#f39c12"), ("High", "#e74c3c")]:
        subset = decisions_df[decisions_df["risk_level"] == level]["pd_score"]
        if len(subset):
            subset.plot.kde(ax=axes[1], label=level, color=color, linewidth=2)
    axes[1].set_title("PD Distribution by Risk Level")
    axes[1].set_xlabel("Probability of Default")
    axes[1].legend()

    # Expected profit by decision
    approved_df = decisions_df[decisions_df["decision"] == "APPROVE"]
    axes[2].hist(approved_df["expected_profit"], bins=50,
                 color="#3498db", alpha=0.7, edgecolor="white")
    axes[2].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[2].set_title("Expected Profit Distribution (Approved Loans)")
    axes[2].set_xlabel("Expected Profit ($)")
    axes[2].set_ylabel("Count")

    fig.suptitle("Risk Segmentation — Credit Decision System", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "09_risk_segmentation")


def plot_approval_profit_curve(decisions_df: pd.DataFrame):
    """
    WHY: Shows the tradeoff between approval rate and portfolio quality.
    As we approve more loans (lower threshold), profit eventually falls
    because we're taking on higher-risk borrowers.
    A credit manager uses this curve to set business-appropriate thresholds.
    """
    sorted_df = decisions_df.sort_values("pd_score")
    n = len(sorted_df)
    approval_rates = []
    cumulative_profits = []

    for pct in range(5, 101, 5):
        top_n = int(pct / 100 * n)
        subset = sorted_df.iloc[:top_n]
        approval_rates.append(pct / 100)
        cumulative_profits.append(subset["expected_profit"].sum())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(approval_rates, cumulative_profits,
            color="#3498db", linewidth=2.5, marker="o", markersize=4)
    ax.fill_between(approval_rates, cumulative_profits, alpha=0.1, color="#3498db")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Approval Rate (ordered by PD ↑)")
    ax.set_ylabel("Cumulative Expected Profit ($)")
    ax.set_title("Approval Rate vs Portfolio Expected Profit")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "10_approval_profit_curve")


def run_simulation(decisions_df: pd.DataFrame, y_true: pd.Series = None) -> dict:
    summary = summarise_portfolio(decisions_df, y_true)
    plot_risk_segmentation(decisions_df)
    plot_approval_profit_curve(decisions_df)
    return summary
