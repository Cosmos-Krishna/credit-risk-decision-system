# ============================================================
# config.py — Central Configuration
# ============================================================
# All tunable parameters live here. Change flags/values here
# instead of hunting through individual modules.

from pathlib import Path
import os

# ── Base Directory ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Paths ───────────────────────────────────────────────────
DATA_PATH = BASE_DIR / "data" / "lending_club_loan_two.csv"

OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR = BASE_DIR / "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Target ──────────────────────────────────────────────────
TARGET_COL = "loan_status"
# Mapping: 'Charged Off' to 1 (default), 'Fully Paid' to 0 (non-default)
DEFAULT_LABEL = "Charged Off"
PAID_LABEL    = "Fully Paid"

# ── Columns to DROP (leakage / noise) ───────────────────────
DROP_COLS = ["issue_d", "emp_title", "title", "address", "sub_grade"]

# ── Missing value strategy ───────────────────────────────────
# mort_acc: imputed using group median by total_acc bucket
# numeric:  median
# categorical: mode

# ── Outlier capping (percentile) ────────────────────────────
OUTLIER_COLS = {
    "annual_inc" : (0.01, 0.99),
    "dti"        : (0.01, 0.99),
    "revol_bal"  : (0.01, 0.99),
}

# ── Feature engineering ──────────────────────────────────────
CURRENT_YEAR = 2023   # used for account_age calculation

DTI_BINS   = [0, 10, 20, 30, 40, float("inf")]
DTI_LABELS = ["Very_Low", "Low", "Medium", "High", "Very_High"]

# ── Model ────────────────────────────────────────────────────
MODEL_TYPE  = "xgboost"   # options: "logistic", "random_forest", "xgboost"
USE_SMOTE   = False
USE_TUNING  = False

ENABLE_MODEL_COMPARISON = True
CV_FOLDS = 5

RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ── Decision engine thresholds ───────────────────────────────
PD_APPROVE = 0.20
PD_REVIEW  = 0.45

# ── Portfolio simulation ─────────────────────────────────────
RECOVERY_RATE = 0.10

# Loss Given Default
LGD = 1 - RECOVERY_RATE

# ── Risk Bands ───────────────────────────────────────────────
LOW_RISK_MAX = 0.20
MEDIUM_RISK_MAX = 0.45

# ── Financial Risk Settings ──────────────────────────────────
MIN_EXPECTED_PROFIT = -500