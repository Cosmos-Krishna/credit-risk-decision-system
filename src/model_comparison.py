# ============================================================
# model_comparison.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier

from src.config import (
    RANDOM_STATE,
    CV_FOLDS,
    OUTPUT_DIR,
)


# ============================================================
# Model Factory
# ============================================================

def get_models(scale_pos_weight=1.0):

    models = {

        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
        ),
    }

    return models


# ============================================================
# Main Comparison Function
# ============================================================

def compare_models(X, y):

    print("\n[ModelComparison] Running model benchmark...")

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    models = get_models(scale_pos_weight)

    skf = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for name, model in models.items():

        print(f"\n[ModelComparison] Evaluating: {name}")

        y_pred_proba = cross_val_predict(
            model,
            X,
            y,
            cv=skf,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "Model": name,
            "ROC_AUC": roc_auc_score(y, y_pred_proba),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1": f1_score(y, y_pred),
        }

        results.append(metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "ROC_AUC",
        ascending=False,
    )

    print("\n=== MODEL COMPARISON ===")
    print(results_df)

    save_results(results_df)

    plot_model_comparison(results_df)

    return results_df


# ============================================================
# Save CSV
# ============================================================

def save_results(results_df):

    path = OUTPUT_DIR / "model_comparison.csv"

    results_df.to_csv(path, index=False)

    print(f"[ModelComparison] Saved CSV to {path}")


# ============================================================
# Plot Results
# ============================================================

def plot_model_comparison(results_df):

    metrics = [
        "ROC_AUC",
        "Precision",
        "Recall",
        "F1",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results_df))

    width = 0.18

    for i, metric in enumerate(metrics):

        ax.bar(
            x + i * width,
            results_df[metric],
            width,
            label=metric,
        )

    ax.set_xticks(x + width * 1.5)

    ax.set_xticklabels(results_df["Model"])

    ax.set_ylabel("Score")

    ax.set_title("Model Performance Comparison")

    ax.legend()

    plt.tight_layout()

    path = OUTPUT_DIR / "13_model_comparison.png"

    plt.savefig(path, dpi=300)

    print(f"[ModelComparison] Saved plot to {path}")