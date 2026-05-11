# ============================================================
# model.py — Model Training
# ============================================================
# WHY: We train three models in order of complexity:
#   1. Logistic Regression  to interpretable baseline
#   2. Random Forest        to non-linear, ensemble baseline
#   3. XGBoost              to final production model (speed + AUC)
# class_weight='balanced' handles the 80/20 imbalance without
# over-engineering (no SMOTE by default).

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import TARGET_COL, TEST_SIZE, RANDOM_STATE, MODEL_TYPE, USE_SMOTE


def split_data(df: pd.DataFrame):
    """
    WHY: Stratified split preserves the minority class ratio in both
    train and test sets — critical with 20% default rate.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[Model] Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"[Model] Train default rate: {y_train.mean():.2%}  "
          f"Test default rate: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """
    Optional SMOTE: oversamples the minority (default) class.
    Disabled by default — class_weight='balanced' is sufficient
    and avoids synthetic sample noise.
    """
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[Model] SMOTE applied. New train shape: {X_res.shape}")
    return X_res, y_res


def train_logistic(X_train, y_train):
    """
    WHY: Logistic Regression is the credit-industry baseline.
    Requires scaling (StandardScaler). class_weight='balanced'
    penalises misclassifying the minority class more heavily.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )
    model.fit(X_scaled, y_train)
    print("[Model] Logistic Regression trained.")
    return model, scaler


def train_random_forest(X_train, y_train):
    """
    WHY: RF handles non-linear interactions and doesn't need scaling.
    n_estimators=200 gives stable OOB estimates without being slow.
    """
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print("[Model] Random Forest trained.")
    return model, None   # no scaler needed


def train_xgboost(X_train, y_train):
    """
    WHY: XGBoost is the industry standard for tabular credit data.
    scale_pos_weight handles imbalance; colsample & subsample add
    regularisation to reduce overfitting.
    scale_pos_weight = #negative / #positive (≈ 4:1 here).
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"[Model] XGBoost scale_pos_weight = {spw:.2f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=False)
    print("[Model] XGBoost trained.")
    return model, None


TRAINERS = {
    "logistic"     : train_logistic,
    "random_forest": train_random_forest,
    "xgboost"      : train_xgboost,
}


def train_model(X_train, y_train, model_type: str = MODEL_TYPE,
                use_smote: bool = USE_SMOTE):
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)

    trainer = TRAINERS.get(model_type)
    if trainer is None:
        raise ValueError(f"Unknown model_type: {model_type}. "
                         f"Choose from {list(TRAINERS)}")
    return trainer(X_train, y_train)


if __name__ == "__main__":
    from data_loader import load_data
    from preprocess import clean_data
    from feature_engineering import build_features

    df = build_features(clean_data(load_data()))
    X_tr, X_te, y_tr, y_te = split_data(df)
    model, scaler = train_model(X_tr, y_tr)
    print("Model type:", type(model).__name__)
