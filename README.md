# 💳 Lending Risk: Explainable AI-Powered Loan Underwriting Platform

## Overview

Lending Risk is an end-to-end machine learning platform that assists loan underwriting decisions using explainable AI.

The system predicts a borrower's Probability of Default (PD), applies business-driven approval thresholds, generates interpretable explanations using SHAP, and provides actionable recommendations to improve approval chances.

Unlike traditional black-box scoring systems, this platform focuses on:

* Explainability
* Risk-aware decision making
* Threshold optimization
* Human-underwriter review workflows
* Real-time borrower feedback

The platform is built using Streamlit and machine learning models trained on historical LendingClub loan data.

---

# Business Problem

Financial institutions face two major challenges when evaluating loan applications:

1. Approving risky borrowers who later default.
2. Rejecting potentially good borrowers due to conservative rules.

A standard machine learning model using a default threshold of 0.5 often misses a large number of future defaulters.

The objective of this project is to:

* Predict default probability accurately.
* Prioritize recall to identify risky borrowers.
* Provide transparent explanations.
* Support human-underwriter decision making.
* Offer borrowers actionable improvement suggestions.

---

# Dataset

Dataset Source:

* LendingClub Historical Loan Dataset

Target Variable:

| Loan Status           | Target |
| --------------------- | ------ |
| Fully Paid            | 0      |
| Charged Off / Default | 1      |

The dataset contains historical borrower profiles, loan characteristics, repayment behavior, and credit information.

---

# Data Preprocessing

Several preprocessing steps were applied before model training.

## Data Cleaning

* Removed irrelevant columns.
* Standardized percentage fields.
* Converted loan term strings into numeric values.
* Cleaned employment length fields.
* Normalized categorical values.

Example:

```text
"36 months" → 36
"10+ years" → 10
```

---

## Missing Value Handling

Missing values were handled using:

* Median imputation
* Safe defaults
* Domain-aware fallbacks

This ensured robust inference during both training and deployment.

---

## Outlier Treatment

Extreme annual income values were capped using the 99th percentile to reduce skew and prevent model instability.

---

# Feature Engineering

One of the primary strengths of this project is business-driven feature engineering.

The final model uses 22 engineered features.

## Financial Burden Features

### Loan-to-Income Ratio

Measures borrower exposure relative to income.

```text
loan_amount / annual_income
```

---

### EMI-to-Income Ratio

Measures monthly repayment burden.

```text
monthly_emi / monthly_income
```

---

### Revolving Balance-to-Income

Measures existing revolving debt pressure.

```text
revolving_balance / annual_income
```

---

## Credit Behavior Features

### Credit History Years

Derived from earliest credit line.

Measures borrower experience with credit.

---

### Account Opening Velocity

```text
open_accounts / credit_history_years
```

Used as a proxy for aggressive credit seeking behavior.

---

## Interaction Features

### DTI × Grade

Captures compounded risk from:

* Poor credit quality
* High debt burden

---

### Utilization × Loan-to-Income

Captures combined credit stress.

---

## Transformations

### Log Annual Income

Applied due to strong right-skewness.

```python
log_annual_inc = log1p(annual_income)
```

---

# Encoding Strategy

Categorical variables were encoded using business-aware mappings.

Examples:

| Grade | Encoding |
| ----- | -------- |
| A     | 0        |
| B     | 1        |
| C     | 2        |
| D     | 3        |
| E     | 4        |
| F     | 5        |
| G     | 6        |

Ordinal encoding preserves natural credit grade ordering.

---

# Handling Class Imbalance

Loan default prediction is inherently imbalanced.

Instead of using SMOTE, the project uses:

```python
scale_pos_weight
```

Reasons:

* Better performance on tree-based models.
* Avoids synthetic borrower generation.
* Preserves real-world distribution.

---

# Models Evaluated

Five models were compared.

| Model               |
| ------------------- |
| Logistic Regression |
| Random Forest       |
| XGBoost             |
| LightGBM            |
| CatBoost            |

Evaluation was performed using stratified validation splits.

---

# Hyperparameter Optimization

Optuna was used for automated hyperparameter tuning.

Optimization objective:

```text
F-beta Score (β = 2)
```

Why β = 2?

Because recall is significantly more important than precision in lending.

Missing a future defaulter is often more expensive than reviewing additional borrowers.

---

# Cross Validation

The project uses:

```text
Stratified K-Fold Cross Validation
```

Benefits:

* Preserves class distribution.
* Produces more reliable estimates.
* Reduces variance in model evaluation.

---

# Model Selection

After evaluation:

```text
XGBoost
```

was selected as the final production model.

Reasons:

* Highest ROC-AUC
* Strong recall
* Best F-beta performance
* Excellent handling of nonlinear feature interactions

---

# Probability Calibration

Tree-based models often produce overconfident probabilities.

To improve reliability, probability calibration was applied using:

```text
Isotonic Regression
```

This improves trustworthiness of predicted default probabilities.

---

# Threshold Optimization

Instead of using the conventional threshold:

```text
0.50
```

the project performs threshold optimization.

Selected threshold:

```text
0.13
```

The threshold was chosen by maximizing:

```text
F-beta (β = 2)
```

This significantly improves recall and reduces missed defaults.

---

# Decision Engine

Business decisions are generated using probability thresholds.

| Probability of Default | Decision |
| ---------------------- | -------- |
| PD < 0.13              | APPROVE  |
| 0.13 ≤ PD ≤ 0.60       | REVIEW   |
| PD > 0.60              | REJECT   |

This mirrors real-world underwriting workflows where borderline cases are escalated to human review.

---

# Explainable AI (SHAP)

The platform uses SHAP for local model explainability.

For every prediction:

* Positive risk factors are identified.
* Protective factors are identified.
* Feature contributions are ranked.

This allows both borrowers and underwriters to understand model reasoning.

---

# Platform Features

## Borrower Portal

### Dashboard

* Profile summary
* Latest application status
* Recent applications

### Loan Application

* Real-time risk assessment
* Default probability prediction
* Approval decision

### Improve Approval Chances

* Scenario simulation
* AI-driven recommendations
* Risk reduction insights

### Application History

* Historical applications
* Decisions
* Risk trends

---

## Admin Portal

### Applications

* Underwriter review queue
* Approval/rejection workflow
* Override support

### Risk Monitoring

* Model metrics
* Threshold analysis
* Confusion matrix
* Model comparison

### Risk Insights

* Portfolio analytics
* Default behavior analysis
* Feature relationships
* Business intelligence visualizations

---

# Technology Stack

## Frontend

* Streamlit

## Machine Learning

* Scikit-Learn
* XGBoost
* LightGBM
* CatBoost
* Optuna
* SHAP

## Data Processing

* Pandas
* NumPy

## Visualization

* Plotly

## Database

* SQLite

---

# Key Results

| Metric    | Value   |
| --------- | ------- |
| ROC-AUC   | 0.72    |
| Recall    | 0.83    |
| Threshold | 0.13    |
| Model     | XGBoost |

The final system prioritizes default detection while maintaining explainability and business interpretability.

---

# Future Improvements

* Reject inference techniques
* Cost-sensitive learning
* Drift monitoring
* Real-time model retraining
* Advanced portfolio risk simulation
* Champion-challenger model framework

---

# Author

Krishna Sinha

Data Science & Machine Learning Engineer

Built as an end-to-end explainable lending risk platform demonstrating production-oriented machine learning, underwriting analytics, and business-driven AI decision making.
