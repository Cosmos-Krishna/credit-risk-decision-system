# ============================================================
# evaluation.py — Model Evaluation
# ============================================================
# WHY: For credit risk, the cost of missing a true defaulter
# (false negative) is much higher than approving a good borrower
# (false positive). We emphasise:
#   • ROC-AUC       to overall discrimination
#   • Recall (def)  to how many defaults we catch
#   • Precision/F1  to tradeoff quality

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score
)
from src.config import OUTPUT_DIR


def _save(fig, name):
    path = f"{OUTPUT_DIR}/{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Saved to {path}")


def get_predictions(model, X_test, scaler=None, threshold: float = 0.5):
    """
    WHY: We use predict_proba() + manual threshold (NOT predict())
    because the default 0.5 threshold is sub-optimal for imbalanced
    data where we want higher recall on the minority class.
    """
    if scaler is not None:
        X_test = scaler.transform(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    return proba, preds


def print_metrics(y_true, y_pred, y_proba, label: str = ""):
    auc  = roc_auc_score(y_true, y_proba)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{'='*50}")
    print(f"EVALUATION REPORT — {label}")
    print(f"{'='*50}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["Paid", "Default"]))
    return {"auc": auc, "precision": prec, "recall": rec, "f1": f1}


def plot_roc_curve(y_true, y_proba, label: str = "Model"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#3498db", linewidth=2,
            label=f"{label} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#3498db")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — Default Prediction")
    ax.legend()
    fig.tight_layout()
    _save(fig, "06_roc_curve")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Paid", "Default"],
                yticklabels=["Paid", "Default"],
                ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    _save(fig, "07_confusion_matrix")


def evaluate(model, X_test, y_test, scaler=None, threshold=0.5, label=""):
    y_proba, y_pred = get_predictions(model, X_test, scaler, threshold)
    metrics = print_metrics(y_test, y_pred, y_proba, label)
    plot_roc_curve(y_test, y_proba, label)
    plot_confusion_matrix(y_test, y_pred)
    return y_proba, y_pred, metrics
