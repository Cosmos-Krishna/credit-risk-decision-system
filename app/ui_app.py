# ============================================================
# ui_app.py  —  LendingClub Credit Risk Decision System
# ============================================================
# BACKEND:  Unchanged — loads model_artifact.pkl, calls
#           decision_engine.py exactly as before.
# FRONTEND: Rewritten with clean Streamlit components,
#           minimal CSS, fintech-style layout.
#
# Run:  streamlit run ui_app.py
# ============================================================

import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st

from src.decision_engine import compute_expected_profit, make_decision, get_risk_level


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk · Decision System",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────
# MINIMAL CSS — only what Streamlit components can't do alone
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { max-width: 780px; padding-top: 2rem; padding-bottom: 3rem; }
    #MainMenu, footer { visibility: hidden; }

    /* Decision badge */
    .decision-badge {
        display: inline-block;
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        padding: 0.45rem 1.6rem;
        border-radius: 8px;
        border: 2px solid;
    }
    .badge-approve { color: #16a34a; background: #f0fdf4; border-color: #86efac; }
    .badge-review  { color: #b45309; background: #fffbeb; border-color: #fcd34d; }
    .badge-reject  { color: #dc2626; background: #fef2f2; border-color: #fca5a5; }

    /* Risk pill */
    .risk-pill {
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 700;
        padding: 0.2rem 0.9rem;
        border-radius: 999px;
    }
    .pill-low    { color: #15803d; background: #dcfce7; }
    .pill-medium { color: #92400e; background: #fef3c7; }
    .pill-high   { color: #b91c1c; background: #fee2e2; }

    /* Large metric values */
    .big-metric { font-size: 2rem; font-weight: 800; line-height: 1.1; color: #1e293b; }
    .metric-label {
        font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.08em; text-transform: uppercase;
        color: #94a3b8; margin-top: 0.2rem;
    }
    .profit-pos { color: #16a34a; }
    .profit-neg { color: #dc2626; }

    /* Driver rows */
    .driver-label { font-size: 0.82rem; color: #475569; margin-bottom: 0.12rem; }
    .dir-up   { color: #dc2626; font-weight: 600; font-size: 0.75rem; }
    .dir-down { color: #16a34a; font-weight: 600; font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
ARTIFACT_PATH = ROOT_DIR / "models" / "model_artifact.pkl"

DECISION_CONFIG = {
    "APPROVE": {"badge_cls": "badge-approve", "icon": "✅"},
    "REVIEW":  {"badge_cls": "badge-review",  "icon": "⚠️"},
    "REJECT":  {"badge_cls": "badge-reject",  "icon": "❌"},
}
RISK_CONFIG = {
    "Low":    {"pill_cls": "pill-low",    "icon": "🟢"},
    "Medium": {"pill_cls": "pill-medium", "icon": "🟡"},
    "High":   {"pill_cls": "pill-high",   "icon": "🔴"},
}


# ─────────────────────────────────────────────────────────────
# BACKEND — logic unchanged from original ui_app.py
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def load_artifact(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_single_row(
    loan_amnt, annual_inc, dti, int_rate,
    revol_bal, total_acc,
    term=36, feature_cols=None,
) -> pd.DataFrame:
    """Reconstruct a single-row DataFrame matching the 50-feature training schema."""
    annual_inc_safe = max(annual_inc, 1)

    loan_to_income     = loan_amnt / annual_inc_safe
    installment        = loan_amnt * (int_rate / 100 / 12) / (
        1 - (1 + int_rate / 100 / 12) ** -term
    )
    installment_ratio  = installment / annual_inc_safe
    credit_utilization = revol_bal / (total_acc + 1)
    account_age        = 15  # neutral median assumption

    from src.config import DTI_BINS, DTI_LABELS
    dti_bucket = str(pd.cut([dti], bins=DTI_BINS, labels=DTI_LABELS, right=False)[0])

    row = {
        "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate,
        "installment": installment, "emp_length": 5.0,
        "annual_inc": annual_inc, "dti": dti,
        "open_acc": 11.0, "pub_rec": 0.0,
        "revol_bal": revol_bal, "revol_util": 54.8,
        "total_acc": total_acc, "mort_acc": 1.0,
        "pub_rec_bankruptcies": 0.0,
        "loan_to_income": loan_to_income,
        "installment_ratio": installment_ratio,
        "credit_utilization": credit_utilization,
        "account_age": account_age,
    }

    ohe_defaults = {
        "grade_B": 0, "grade_C": 0, "grade_D": 0,
        "grade_E": 0, "grade_F": 0, "grade_G": 0,
        "home_ownership_OTHER": 0, "home_ownership_OWN": 0,
        "home_ownership_RENT": 1,
        "verification_status_Source Verified": 0,
        "verification_status_Verified": 0,
        "purpose_credit_card": 0, "purpose_debt_consolidation": 1,
        "purpose_educational": 0, "purpose_home_improvement": 0,
        "purpose_house": 0, "purpose_major_purchase": 0,
        "purpose_medical": 0, "purpose_moving": 0,
        "purpose_other": 0, "purpose_renewable_energy": 0,
        "purpose_small_business": 0, "purpose_vacation": 0,
        "purpose_wedding": 0,
        "initial_list_status_w": 1,
        "application_type_Joint App": 0,
        "dti_bucket_High": 0, "dti_bucket_Low": 0,
        "dti_bucket_Medium": 0, "dti_bucket_Very_High": 0,
        "dti_bucket_Very_Low": 0, "dti_bucket_nan": 0,
    }

    for k in list(ohe_defaults):
        if k.startswith("dti_bucket_"):
            ohe_defaults[k] = 0
    dti_col = f"dti_bucket_{dti_bucket}"
    if dti_col in ohe_defaults:
        ohe_defaults[dti_col] = 1

    row.update(ohe_defaults)
    df_row = pd.DataFrame([row])

    if feature_cols is not None:
        for col in feature_cols:
            if col not in df_row.columns:
                df_row[col] = 0
        df_row = df_row[feature_cols]

    return df_row


def _get_shap_drivers(model, df_row: pd.DataFrame, feature_cols: list) -> list:
    """Compute top-3 SHAP-based risk drivers. Returns [] on failure."""
    try:
        import shap
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(df_row)
        sv         = shap_vals[0]

        abs_series = pd.Series(np.abs(sv), index=feature_cols)
        raw_series = pd.Series(sv,         index=feature_cols)

        drivers = []
        for feat in abs_series.nlargest(3).index:
            raw_val = float(raw_series[feat])
            drivers.append({
                "name"          : feat,
                "abs_shap"      : float(abs_series[feat]),
                "increases_risk": raw_val > 0,
            })
        return drivers
    except Exception:
        return []


def run_prediction(artifact: dict, loan_amnt, annual_inc, dti,
                   int_rate, revol_bal, total_acc, term) -> dict:
    """Full prediction pipeline. Returns results dict."""
    model        = artifact["model"]
    scaler       = artifact["scaler"]
    feature_cols = artifact["feature_cols"]

    df_row = build_single_row(
        loan_amnt, annual_inc, dti, int_rate,
        revol_bal, total_acc, term, feature_cols
    )

    X = df_row.values
    if scaler:
        X = scaler.transform(X)

    pd_score = float(model.predict_proba(X)[0, 1])
    profit   = float(compute_expected_profit(
        np.array([loan_amnt]), np.array([int_rate]),
        np.array([term]),      np.array([pd_score]),
    )[0])
    decision   = str(make_decision(np.array([pd_score]), np.array([profit]))[0])
    risk_level = str(get_risk_level(np.array([pd_score]))[0])
    drivers    = _get_shap_drivers(model, df_row, feature_cols)

    return {
        "pd_score": pd_score, "risk_level": risk_level,
        "profit": profit, "decision": decision, "drivers": drivers,
    }


# ─────────────────────────────────────────────────────────────
# UI RENDERING FUNCTIONS
# ─────────────────────────────────────────────────────────────

def render_header():
    st.markdown("# 🏦 Credit Risk Decision System")
    st.caption("LendingClub · XGBoost · 396K loans · ROC-AUC 0.72")
    st.divider()

def render_executive_dashboard():

    st.markdown("## 📈 Portfolio Intelligence Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="ROC-AUC",
            value="0.722",
        )

    with col2:
        st.metric(
            label="Approval Rate",
            value="9.7%",
        )

    with col3:
        st.metric(
            label="Expected Profit",
            value="$7.05M",
        )

    with col4:
        st.metric(
            label="Approved Default Rate",
            value="4%",
        )

    st.markdown("---")

def render_analytics_dashboard():

    st.markdown("## 📊 Portfolio Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            str(ROOT_DIR / "outputs" / "09_risk_segmentation.png"),
            caption="Risk Segmentation",
            use_container_width=True,
        )

    with col2:
        st.image(
            str(ROOT_DIR / "outputs" / "10_approval_profit_curve.png"),
            caption="Approval vs Profit",
            use_container_width=True,
        )

    st.markdown("## 🤖 Model Benchmarking")

    st.image(
        str(ROOT_DIR / "outputs" / "13_model_comparison.png"),
        caption="Cross-Validated Model Comparison",
        use_container_width=True,
    )

    st.markdown("## 🔬 Explainable AI")

    st.image(
        str(ROOT_DIR / "outputs" / "12_shap_summary.png"),
        caption="SHAP Feature Importance",
        use_container_width=True,
    )

    st.markdown("## 📉 Model Evaluation")

    col3, col4 = st.columns(2)

    with col3:
        st.image(
            str(ROOT_DIR / "outputs" / "06_roc_curve.png"),
            caption="ROC Curve",
            use_container_width=True,
        )

    with col4:
        st.image(
            str(ROOT_DIR / "outputs" / "07_confusion_matrix.png"),
            caption="Confusion Matrix",
            use_container_width=True,
        )

def render_inputs() -> dict:
    """Loan application input form. Returns values dict + submitted flag."""
    st.subheader("📋 Loan Application")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        loan_amnt = st.number_input(
            "Loan Amount ($)", min_value=500, max_value=40_000,
            value=15_000, step=500,
            help="Requested principal"
        )
        annual_inc = st.number_input(
            "Annual Income ($)", min_value=1_000, max_value=500_000,
            value=65_000, step=1_000,
            help="Gross annual income"
        )
        dti = st.number_input(
            "Debt-to-Income Ratio (%)", min_value=0.0, max_value=60.0,
            value=16.9, step=0.1, format="%.1f",
            help="Monthly debt / monthly gross income"
        )
        int_rate = st.number_input(
            "Interest Rate (%)", min_value=5.0, max_value=31.0,
            value=13.3, step=0.1, format="%.2f",
            help="Annual loan interest rate"
        )

    with col2:
        revol_bal = st.number_input(
            "Revolving Balance ($)", min_value=0, max_value=200_000,
            value=11_000, step=500,
            help="Outstanding revolving credit balance"
        )
        total_acc = st.number_input(
            "Total Credit Accounts", min_value=1, max_value=100,
            value=24, step=1,
            help="Total credit lines in borrower history"
        )
        st.write("")
        term = st.radio(
            "Loan Term", options=[36, 60],
            format_func=lambda x: f"{x} months",
            horizontal=True,
        )

    st.write("")
    submitted = st.button(
        "🔍  Analyze Application",
        use_container_width=True,
        type="primary",
    )

    return {
        "loan_amnt": loan_amnt, "annual_inc": annual_inc,
        "dti": dti, "int_rate": int_rate,
        "revol_bal": revol_bal, "total_acc": total_acc,
        "term": term, "submitted": submitted,
    }


def render_decision_badge(decision: str):
    cfg   = DECISION_CONFIG.get(decision, DECISION_CONFIG["REJECT"])
    badge = (
        f'<span class="decision-badge {cfg["badge_cls"]}">'
        f'{cfg["icon"]} {decision}</span>'
    )
    st.markdown(badge, unsafe_allow_html=True)


def render_metrics(pd_score: float, risk_level: str, profit: float):
    """Three-column metrics row."""
    risk_cfg       = RISK_CONFIG.get(risk_level, RISK_CONFIG["High"])
    profit_sign    = "+" if profit >= 0 else "−"
    profit_cls     = "profit-pos" if profit >= 0 else "profit-neg"

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f'<div class="big-metric">{pd_score:.1%}</div>'
            f'<div class="metric-label">Prob. of Default</div>',
            unsafe_allow_html=True,
        )
    with c2:
        pill = (
            f'<span class="risk-pill {risk_cfg["pill_cls"]}">'
            f'{risk_cfg["icon"]} {risk_level}</span>'
        )
        st.markdown(
            f'<div style="margin-top:4px">{pill}</div>'
            f'<div class="metric-label" style="margin-top:6px">Risk Level</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="big-metric {profit_cls}">'
            f'{profit_sign}${abs(profit):,.0f}</div>'
            f'<div class="metric-label">Expected Profit</div>',
            unsafe_allow_html=True,
        )


def render_risk_drivers(drivers: list):
    """Top-3 SHAP risk drivers using st.progress bars."""
    if not drivers:
        st.caption("ℹ️ Install `shap` for feature-level explanations.")
        return

    st.markdown("**Top 3 Risk Drivers**")
    st.caption("SHAP values — contribution to the predicted default probability")
    st.write("")

    max_shap = max(d["abs_shap"] for d in drivers) or 1.0

    for d in drivers:
        display_name    = d["name"].replace("_", " ").title()
        direction_text  = "↑ increases risk" if d["increases_risk"] else "↓ reduces risk"
        direction_cls   = "dir-up" if d["increases_risk"] else "dir-down"
        bar_pct         = int(d["abs_shap"] / max_shap * 100)

        st.markdown(
            f'<div class="driver-label">{display_name} &nbsp;'
            f'<span class="{direction_cls}">{direction_text}</span></div>',
            unsafe_allow_html=True,
        )
        st.progress(bar_pct)


def render_context_note(decision: str, pd_score: float):
    """Callout explaining the decision rule that was triggered."""
    if decision == "APPROVE":
        st.success(
            f"PD of **{pd_score:.1%}** falls within acceptable risk range "
            "with positive expected return. Loan satisfies lending policy."
        )

    elif decision == "REVIEW":
        st.warning(
            f"PD of **{pd_score:.1%}** falls into medium-risk band. "
            "Manual underwriting review is recommended."
        )

    else:
        st.error(
            f"PD of **{pd_score:.1%}** indicates elevated credit risk "
            "or insufficient profitability. Loan rejected under policy."
        )


def render_results(results: dict):
    """Full output section — decision, metrics, drivers, context note."""
    st.divider()
    st.subheader("📊 Decision Output")

    # A. Decision badge
    render_decision_badge(results["decision"])
    st.write("")

    # B. Metrics row
    render_metrics(results["pd_score"], results["risk_level"], results["profit"])
    st.write("")

    # C. Risk drivers
    with st.expander("🔬 Risk Driver Analysis", expanded=True):
        render_risk_drivers(results["drivers"])

    # D. Context note
    st.write("")
    render_context_note(results["decision"], results["pd_score"])


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    render_header()

    render_executive_dashboard()

    render_analytics_dashboard()

    if not os.path.exists(ARTIFACT_PATH):
        st.error(
            "**Model artifact not found.**  \n"
            "Run `python main.py` once to train and save `model_artifact.pkl`.",
            icon="⚠️",
        )
        st.stop()

    artifact = load_artifact(ARTIFACT_PATH)
    inputs   = render_inputs()

    if inputs["submitted"]:
        with st.spinner("Running model …"):
            results = run_prediction(
                artifact,
                loan_amnt  = inputs["loan_amnt"],
                annual_inc = inputs["annual_inc"],
                dti        = inputs["dti"],
                int_rate   = inputs["int_rate"],
                revol_bal  = inputs["revol_bal"],
                total_acc  = inputs["total_acc"],
                term       = inputs["term"],
            )
        render_results(results)


if __name__ == "__main__":
    main()
