# ============================================================
# utils.py — Feature Importance & SHAP Explanation
# ============================================================
# WHY: XGBoost's built-in feature importance helps validate that
# the model is using economically meaningful features (not spurious
# correlations). SHAP provides individual loan-level explanations —
# critical for regulatory interpretability (SR 11-7 / ECOA).

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import OUTPUT_DIR


def _save(fig, name):
    path = f"{OUTPUT_DIR}/{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[Utils] Saved to {path}")


def plot_feature_importance(model, feature_names, top_n: int = 20):
    """
    WHY: Built-in importance (gain) shows which features reduce impurity
    most across all trees — i.e. which features the model actually relies on.
    """
    try:
        importances = model.feature_importances_
    except AttributeError:
        print("[Utils] Model has no feature_importances_. Skipping.")
        return

    imp_df = pd.DataFrame({
        "feature"   : feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(imp_df)))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1],
            color=colors[::-1])
    ax.set_title(f"Top {top_n} Feature Importances (Gain)", fontsize=13)
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    _save(fig, "11_feature_importance")
    return imp_df


def plot_shap_summary(model, X_sample: pd.DataFrame, max_samples: int = 2000):
    """
    WHY: SHAP values explain each prediction: "this loan was rejected
    because int_rate (+0.12) and dti (+0.08) pushed PD up".
    Required for fairness audits and model documentation.
    """
    try:
        import shap
    except ImportError:
        print("[Utils] shap not installed. Skipping SHAP summary.")
        return

    print(f"[Utils] Computing SHAP values on {min(max_samples, len(X_sample))} samples …")
    X_s = X_sample.sample(min(max_samples, len(X_sample)), random_state=42)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_s)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_s, plot_type="bar",
                      max_display=15, show=False)
    plt.title("SHAP Feature Importance — Global (Bar)")
    plt.tight_layout()
    _save(fig, "12_shap_summary")


def print_loan_explanation(model, X_single: pd.DataFrame,
                           pd_score: float, decision: str,
                           expected_profit: float, top_n: int = 5):
    """
    Print a human-readable explanation for a single loan decision.
    """
    try:
        import shap
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_single)
        if hasattr(shap_values, "__len__") and len(shap_values) > 1:
            sv = shap_values[1][0]   # class=1 (default)
        else:
            sv = shap_values[0]

        feat_impact = pd.Series(sv, index=X_single.columns)
        top_drivers = feat_impact.abs().nlargest(top_n).index
        print(f"\n{'─'*50}")
        print(f"  PD Score       : {pd_score:.2%}")
        print(f"  Decision       : {decision}")
        print(f"  Expected Profit: ${expected_profit:,.0f}")
        print(f"  Top Risk Drivers:")
        for f in top_drivers:
            direction = "↑ risk" if feat_impact[f] > 0 else "↓ risk"
            print(f"    {f:35s}: SHAP={feat_impact[f]:+.4f}  {direction}")
        print(f"{'─'*50}")
    except Exception as e:
        print(f"[Utils] SHAP explanation failed: {e}")
