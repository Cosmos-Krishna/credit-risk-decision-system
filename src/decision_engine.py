# ============================================================
# decision_engine.py — Cost-Based Decision Engine
# ============================================================
# WHY: A probability alone is not a business decision.
# We combine PD with expected profit to produce an actionable
# outcome: APPROVE / REVIEW / REJECT.
#
# Expected Profit =
#   (1 - PD) × (loan_amnt × int_rate × term_years)   ← interest earned if paid
#   − PD × loan_amnt × (1 - recovery_rate)            ← loss given default
#
# Rules (from config, tunable):
#   PD < 0.10  AND profit > 0  to APPROVE
#   0.10 ≤ PD ≤ 0.25           to REVIEW
#   PD > 0.25  OR  profit < 0  to REJECT

import numpy as np
import pandas as pd
from src.config import PD_APPROVE, PD_REVIEW, RECOVERY_RATE, LGD

def compute_expected_loss(
    loan_amnt: np.ndarray,
    pd_score: np.ndarray,
    lgd: float = LGD,
) -> np.ndarray:
    """
    Expected Loss = PD × Exposure × LGD
    """
    return pd_score * loan_amnt * lgd


def compute_expected_profit(
    loan_amnt: np.ndarray,
    int_rate: np.ndarray,
    term: np.ndarray,
    pd_score: np.ndarray,
    recovery_rate: float = RECOVERY_RATE,
) -> np.ndarray:
    """
    Expected profit per loan:
      term is in months to divide by 12 for annualised interest approximation.
    """
    term_years   = term / 12.0
    interest_rev = (1 - pd_score) * loan_amnt * (int_rate / 100) * term_years
    default_loss = compute_expected_loss(
    loan_amnt,
    pd_score,
    LGD,
)
    return interest_rev - default_loss


def make_decision(
    pd_score: np.ndarray,
    expected_profit: np.ndarray,
    pd_approve: float = PD_APPROVE,
    pd_review:  float = PD_REVIEW,
) -> np.ndarray:
    """
    Vectorised decision rules.
    Returns array of strings: 'APPROVE' / 'REVIEW' / 'REJECT'.
    """
    decisions = []

    for pd_val, profit in zip(pd_score, expected_profit):

        # APPROVE
        if (pd_val < pd_approve) and (profit > 0):
            decisions.append("APPROVE")

        # REVIEW
        elif (pd_val < pd_review) and (profit > -500):
            decisions.append("REVIEW")

        # REJECT
        else:
            decisions.append("REJECT")

    return np.array(decisions)


def get_risk_level(pd_score: np.ndarray) -> np.ndarray:
    """
    Human-readable risk tier for reporting.
    """
    return np.where(
        pd_score < PD_APPROVE, "Low",
        np.where(pd_score <= PD_REVIEW, "Medium", "High")
    )


def run_decision_engine(
    df_raw: pd.DataFrame,
    pd_scores: np.ndarray,
) -> pd.DataFrame:
    """
    Merge PD scores with raw loan features to produce per-loan
    decisions, risk levels, and expected profit.

    df_raw must contain: loan_amnt, int_rate, term (numeric months)
    """
    out = df_raw[["loan_amnt", "int_rate", "term"]].copy().reset_index(drop=True)
    out["pd_score"]        = pd_scores
    out["risk_level"]      = get_risk_level(pd_scores)
    out["expected_profit"] = compute_expected_profit(
        out["loan_amnt"].values,
        out["int_rate"].values,
        out["term"].values,
        pd_scores,
    )
    out["decision"] = make_decision(pd_scores, out["expected_profit"].values)

    # Summary
    counts = pd.Series(out["decision"]).value_counts()
    print("\n[DecisionEngine] Decision Distribution:")
    for d, c in counts.items():
        print(f"  {d:8s}: {c:6,d}  ({c/len(out):.1%})")

    return out


if __name__ == "__main__":
    # Quick smoke test
    pd_scores = np.array([0.05, 0.15, 0.30, 0.08, 0.22])
    profits   = np.array([500, 200, -50, 800, 100])
    print(make_decision(pd_scores, profits))
    print(get_risk_level(pd_scores))
