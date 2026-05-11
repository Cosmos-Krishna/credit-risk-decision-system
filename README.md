# LendingClub Credit Risk Decision System
## Final Pipeline Report

---

### Dataset
- **Records**: 396,030 loans | **Features**: 27 raw to 50 engineered
- **Default Rate**: 19.61% (Charged Off) | Class imbalance handled via `scale_pos_weight`

---

### Pipeline Steps

| Step | Module | Key Action |
|------|--------|-----------|
| 1 | data_loader.py | Load CSV, encode target (Charged Off = 1) |
| 2 | eda.py | 5 visualisations: target, interest, grade, correlation, loan_amnt |
| 3 | preprocess.py | Drop leakage cols, impute (mort_acc group-median), cap outliers |
| 4 | feature_engineering.py | 5 domain features + OHE encoding |
| 5 | model.py | XGBoost with scale_pos_weight=4.10 |
| 6 | evaluation.py | ROC-AUC, Precision/Recall/F1, confusion matrix |
| 7 | threshold.py | F1-optimal threshold sweep |
| 8 | decision_engine.py | APPROVE / REVIEW / REJECT with expected profit |
| 9 | portfolio_simulation.py | Business impact charts |
| 10 | utils.py | Feature importance + SHAP |

---

### Model Performance (XGBoost)

| Metric | Default (0.5) | Optimal (0.53) |
|--------|-------------|---------------|
| ROC-AUC | 0.7223 | 0.7223 |
| Recall (default) | 0.6579 | 0.6060 |
| Precision (default) | 0.3246 | 0.3417 |
| F1 (default) | 0.4347 | 0.4370 |

---

### Decision Engine Results (Test Set — 79,206 loans)

| Decision | Count | % |
|----------|-------|---|
| APPROVE | 1,675 | 2.1% |
| REVIEW | 11,408 | 14.4% |
| REJECT | 66,123 | 83.5% |

- **Expected Portfolio Profit (approved)**: $2,121,340
- **Avg profit per approved loan**: $1,266
- **Default rate in approved loans**: 2.0% (vs 19.6% overall)

---

### Top Risk Drivers (SHAP)
1. `int_rate` — highest interest rate = higher default probability
2. `dti` — debt-to-income ratio
3. `annual_inc` — income level
4. `revol_util` — revolving credit utilisation
5. `loan_to_income` — engineered ratio (loan burden vs income)

---

### Business Logic
```
Expected Profit = (1-PD) × (loan × int_rate × term_years) − PD × loan × (1 − recovery_rate)

PD < 0.10  AND profit > 0  to  APPROVE
0.10 ≤ PD ≤ 0.25           to  REVIEW
PD > 0.25  OR  profit < 0  to  REJECT
```

---

### Output Files
| File | Description |
|------|-------------|
| 01_target_distribution.png | Class imbalance chart |
| 02_interest_vs_default.png | Interest rate vs default KDE + grade chart |
| 03_grade_vs_default.png | Default rate and volume by grade |
| 04_correlation_heatmap.png | Feature correlations |
| 05_loan_amount_distribution.png | Loan amount by default status |
| 06_roc_curve.png | ROC curve (AUC = 0.7223) |
| 07_confusion_matrix.png | Confusion matrix at optimal threshold |
| 08_threshold_tuning.png | Precision / Recall / F1 vs threshold sweep |
| 09_risk_segmentation.png | Decision pie + PD distribution + profit histogram |
| 10_approval_profit_curve.png | Approval rate vs cumulative profit curve |
| 11_feature_importance.png | XGBoost top-20 feature importances |
| 12_shap_summary.png | SHAP global feature importance bar chart |
