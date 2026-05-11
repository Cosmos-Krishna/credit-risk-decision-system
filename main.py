# ============================================================
# main.py — Full Pipeline Orchestrator
# ============================================================
# Runs the complete credit risk decision system end-to-end.
# Each step is modular — you can swap components independently.

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from src.config import (
    MODEL_TYPE,
    USE_SMOTE,
    OUTPUT_DIR,
    ENABLE_MODEL_COMPARISON,
)

from src.data_loader import load_data, quick_summary
from src.eda import run_eda
from src.preprocess import clean_data
from src.feature_engineering import build_features
from src.model import split_data, train_model
from src.evaluation import evaluate
from src.threshold import find_optimal_threshold
from src.decision_engine import run_decision_engine
from src.portfolio_simulation import run_simulation
from src.utils import plot_feature_importance, plot_shap_summary
from src.model_comparison import compare_models


def main():
    print("\n" + "="*60)
    print("  LENDINGCLUB CREDIT RISK DECISION SYSTEM")
    print("="*60)

    # ── STEP 1: Load & Quick Summary ─────────────────────────
    print("\n[STEP 1] Data Loading & Summary")
    df_raw = load_data()
    quick_summary(df_raw)

    # ── STEP 2: EDA ──────────────────────────────────────────
    print("\n[STEP 2] Exploratory Data Analysis")
    run_eda(df_raw)

    # ── STEP 3: Preprocessing ────────────────────────────────
    print("\n[STEP 3] Data Cleaning")
    df_clean = clean_data(df_raw.copy())

    # ── STEP 4: Feature Engineering ──────────────────────────
    print("\n[STEP 4] Feature Engineering")
    df_feat = build_features(df_clean.copy())

    # ── STEP 5: Train / Test Split ───────────────────────────
    print("\n[STEP 5] Train-Test Split")
    X_train, X_test, y_train, y_test = split_data(df_feat)

    # ============================================================
    # MODEL COMPARISON + CROSS VALIDATION
    # ============================================================

    if ENABLE_MODEL_COMPARISON:

        print("\n[STEP 5A] Model Comparison + Cross Validation")

        comparison_df = compare_models(
            X_train,
            y_train,
        )

    # ── STEP 6: Model Training ───────────────────────────────
    print(f"\n[STEP 6] Training {MODEL_TYPE} (USE_SMOTE={USE_SMOTE})")
    model, scaler = train_model(X_train, y_train,
                                model_type=MODEL_TYPE,
                                use_smote=USE_SMOTE)

    # ── STEP 7: Evaluation at default threshold ───────────────
    print("\n[STEP 7] Evaluation @ threshold=0.5 (baseline)")
    from src.evaluation import get_predictions
    y_proba, _, _ = evaluate(model, X_test, y_test, scaler,
                              threshold=0.5, label="Default Threshold (0.5)")

    # ── STEP 8: Threshold Tuning ──────────────────────────────
    print("\n[STEP 8] Threshold Tuning")
    optimal_threshold, threshold_metrics = find_optimal_threshold(
        y_test, y_proba, metric="f1"
    )

    # ── STEP 9: Evaluation at optimal threshold ────────────────
    print(f"\n[STEP 9] Evaluation @ optimal threshold={optimal_threshold:.2f}")
    y_proba, y_pred, final_metrics = evaluate(
        model, X_test, y_test, scaler,
        threshold=optimal_threshold,
        label=f"Optimal Threshold ({optimal_threshold:.2f})"
    )

    # ── STEP 10: Decision Engine ──────────────────────────────
    print("\n[STEP 10] Decision Engine")
    # Use test set raw features (pre-encoding) for decision labels
    # Reconstruct loan_amnt, int_rate, term from X_test
    raw_test_cols = df_feat[["loan_amnt", "int_rate", "term"]].iloc[
        X_test.index
    ].reset_index(drop=True)

    decisions_df = run_decision_engine(raw_test_cols, y_proba)

    # ── STEP 11: Portfolio Simulation ────────────────────────
    print("\n[STEP 11] Portfolio Simulation")
    y_test_reset = y_test.reset_index(drop=True)
    portfolio_summary = run_simulation(decisions_df, y_test_reset)

    # ── STEP 12: Feature Importance ──────────────────────────
    print("\n[STEP 12] Feature Importance & SHAP")
    feat_imp = plot_feature_importance(model, X_train.columns)
    plot_shap_summary(model, X_test)

    # ── STEP 13: Sample Loan Explanations ─────────────────────
    print("\n[STEP 13] Sample Loan Decisions (first 5 test loans)")
    from src.utils import print_loan_explanation
    for i in range(min(5, len(decisions_df))):
        pd_s = decisions_df.iloc[i]["pd_score"]
        dec  = decisions_df.iloc[i]["decision"]
        ep   = decisions_df.iloc[i]["expected_profit"]
        if scaler:
            x_s = pd.DataFrame(
                scaler.transform(X_test.iloc[[i]]),
                columns=X_test.columns
            )
        else:
            x_s = X_test.iloc[[i]]
        print_loan_explanation(model, x_s, pd_s, dec, ep)

    # ── FINAL SUMMARY ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print(f"  Model          : {MODEL_TYPE}")
    print(f"  ROC-AUC        : {final_metrics['auc']:.4f}")
    print(f"  Recall(default): {final_metrics['recall']:.4f}")
    print(f"  F1(default)    : {final_metrics['f1']:.4f}")
    print(f"  Optimal Thresh : {optimal_threshold:.2f}")
    print(f"  Approval Rate  : {portfolio_summary['approval_rate']:.1%}")
    print(f"  Expected Profit: ${portfolio_summary['expected_profit_total']:,.0f}")
    print(f"  Outputs dir    : {OUTPUT_DIR}")
    print("="*60)

    return model, scaler, decisions_df, final_metrics


if __name__ == "__main__":
    main()
