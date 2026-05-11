# ============================================================
# threshold.py — Optimal Threshold Selection
# ============================================================
# WHY: The default 0.5 probability cutoff is arbitrary.
# In credit risk we want to:
#   a) Maximise recall on defaults (catch bad loans)
#   b) Not kill the business by rejecting all loans
# We search across thresholds and pick the one that maximises
# F1 (or optionally recall@precision_floor).

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import OUTPUT_DIR


def _save(fig, name):
    path = f"{OUTPUT_DIR}/{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[Threshold] Saved to {path}")


def find_optimal_threshold(y_true, y_proba,
                            metric: str = "f1",
                            min_precision: float = 0.30):
    """
    Sweep thresholds from 0.05 to 0.90.
    metric='f1'     to maximise F1 on default class
    metric='recall' to maximise recall subject to precision floor

    WHY min_precision: a model that flags everything as default has
    100% recall but 0 business value. The floor prevents this.
    """
    thresholds = np.arange(0.05, 0.90, 0.01)
    f1s, precs, recs = [], [], []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_true, preds, zero_division=0))
        precs.append(precision_score(y_true, preds, zero_division=0))
        recs.append(recall_score(y_true, preds, zero_division=0))

    f1s   = np.array(f1s)
    precs = np.array(precs)
    recs  = np.array(recs)

    if metric == "f1":
        best_idx = np.argmax(f1s)
    else:  # recall subject to precision floor
        valid = precs >= min_precision
        if valid.any():
            best_idx = np.where(valid, recs, -1).argmax()
        else:
            best_idx = np.argmax(f1s)

    best_threshold = thresholds[best_idx]
    print(f"\n[Threshold] Optimal threshold ({metric}): {best_threshold:.2f}")
    print(f"  F1={f1s[best_idx]:.4f}  "
          f"Precision={precs[best_idx]:.4f}  "
          f"Recall={recs[best_idx]:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, f1s,   label="F1",        color="#3498db", linewidth=2)
    ax.plot(thresholds, precs, label="Precision",  color="#2ecc71", linewidth=2)
    ax.plot(thresholds, recs,  label="Recall",     color="#e74c3c", linewidth=2)
    ax.axvline(best_threshold, color="orange", linestyle="--", linewidth=2,
               label=f"Optimal={best_threshold:.2f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Tuning — Precision / Recall / F1 on Default Class")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "08_threshold_tuning")

    return best_threshold, {
        "threshold": best_threshold,
        "f1": f1s[best_idx],
        "precision": precs[best_idx],
        "recall": recs[best_idx],
    }
