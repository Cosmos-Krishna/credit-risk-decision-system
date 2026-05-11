# 🏦 LendingClub Credit Risk Decision System

Enterprise-style Machine Learning system for:

* Credit Default Prediction
* Risk-Based Loan Decisioning
* Portfolio Risk Optimization
* Explainable AI (SHAP)
* Business-Aware Lending Decisions

Built using the LendingClub dataset containing ~396K real-world loan applications.

---

# 🚀 Project Overview

Traditional ML projects stop at prediction.

This system goes beyond prediction and simulates how a real BFSI (Banking, Financial Services, and Insurance) institution could:

* estimate probability of default (PD)
* evaluate expected profitability
* optimize loan approvals
* reduce portfolio risk
* explain model decisions using SHAP

The final output is not just a probability score.

The system produces actionable lending decisions:

* ✅ APPROVE
* ⚠️ REVIEW
* ❌ REJECT

based on both:

* predicted risk
* expected business profitability

---

# 📊 Dataset

## LendingClub Loan Dataset

| Property            | Value                      |
| ------------------- | -------------------------- |
| Records             | 396,030 loans              |
| Raw Features        | 27                         |
| Engineered Features | 50+                        |
| Target              | Loan Default (Charged Off) |
| Default Rate        | 19.61%                     |

Target Encoding:

| Loan Status | Encoded Value |
| ----------- | ------------- |
| Fully Paid  | 0             |
| Charged Off | 1             |

---

# 🧠 Core System Architecture

```text
Raw LendingClub Data
            ↓
Data Cleaning & Preprocessing
            ↓
Feature Engineering
            ↓
Model Benchmarking
(Logistic / RF / XGBoost)
            ↓
Cross Validation
            ↓
XGBoost Final Model
            ↓
Probability of Default (PD)
            ↓
Expected Loss + Profit Engine
            ↓
Approve / Review / Reject
            ↓
Portfolio Analytics
            ↓
SHAP Explainability
            ↓
Streamlit Dashboard
```

---

# ⚙️ Key Features

## ✅ Advanced Preprocessing

* Leakage removal
* Missing value imputation
* Group-based median imputation (`mort_acc`)
* Outlier capping
* Feature scaling
* One-hot encoding

---

## ✅ Feature Engineering

Engineered domain-specific financial features:

| Feature            | Description                           |
| ------------------ | ------------------------------------- |
| loan_to_income     | Loan burden relative to income        |
| installment_ratio  | Installment relative to annual income |
| credit_utilization | Revolving balance utilization         |
| account_age        | Credit history age                    |
| dti_bucket         | Risk bucketization for DTI            |

---

## ✅ Model Benchmarking

The system benchmarks multiple ML algorithms using Stratified K-Fold Cross Validation.

### Compared Models

* Logistic Regression
* Random Forest
* XGBoost

### Evaluation Metrics

* ROC-AUC
* Precision
* Recall
* F1-Score

---

# 📈 Model Performance

## Cross-Validated Benchmark Results

| Model              | ROC-AUC | Precision | Recall | F1    |
| ------------------ | ------- | --------- | ------ | ----- |
| XGBoost            | 0.721   | 0.334     | 0.619  | 0.434 |
| RandomForest       | 0.711   | 0.325     | 0.603  | 0.425 |
| LogisticRegression | 0.686   | 0.301     | 0.601  | 0.408 |

### Final Selected Model

✅ XGBoost

Reason:

* highest ROC-AUC
* best recall-performance balance
* superior handling of class imbalance
* strong nonlinear learning capability

---

# 💰 Business Decision Engine

Instead of only predicting default probability, the system converts predictions into lending decisions.

## Core BFSI Concepts Used

* PD (Probability of Default)
* LGD (Loss Given Default)
* Expected Loss
* Expected Profit
* Risk Bands

---

# 📌 Expected Profit Formula

```text
Expected Profit =
(1 - PD) × Interest Revenue
− PD × Expected Loss
```

---

# 📌 Decision Logic

| Condition                   | Decision |
| --------------------------- | -------- |
| PD < 20% AND profit > 0     | APPROVE  |
| 20% ≤ PD < 45%              | REVIEW   |
| PD ≥ 45% OR negative profit | REJECT   |

---

# 📊 Portfolio Simulation Results

## Test Portfolio Size

79,206 loans

---

## Decision Distribution

| Decision | Percentage |
| -------- | ---------- |
| APPROVE  | 9.7%       |
| REVIEW   | 19.8%      |
| REJECT   | 70.5%      |

---

## Portfolio Performance

| Metric                         | Value       |
| ------------------------------ | ----------- |
| Expected Portfolio Profit      | $7.05M      |
| Default Rate in Approved Loans | 4%          |
| Original Dataset Default Rate  | 19.6%       |
| Risk Reduction                 | Significant |

---

# 🔬 Explainable AI (SHAP)

The system uses SHAP (SHapley Additive exPlanations) to explain:

* why a loan was rejected
* which features increased risk
* which features reduced risk

## Top Risk Drivers

* Interest Rate
* Debt-to-Income Ratio
* Revolving Utilization
* Annual Income
* Loan-to-Income Ratio

This makes the model more interpretable and enterprise-friendly.

---

# 🖥️ Streamlit Dashboard

Interactive dashboard includes:

* Loan application form
* Probability of default prediction
* Risk classification
* Expected profit estimation
* Approve / Review / Reject decision
* SHAP-based explanations
* Portfolio analytics
* Model comparison visualizations

---

# 📂 Project Structure

```text
credit-risk-decision-system/
│
├── app/
│   └── ui_app.py
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── model_comparison.py
│   ├── evaluation.py
│   ├── threshold.py
│   ├── decision_engine.py
│   ├── portfolio_simulation.py
│   └── utils.py
│
├── models/
├── outputs/
├── main.py
├── README.md
└── .gitignore
```

---

# 📦 Tech Stack

## Languages

* Python

## Machine Learning

* Scikit-learn
* XGBoost
* SHAP

## Data Processing

* Pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn

## Dashboard

* Streamlit

---

# ▶️ How to Run

## 1. Clone Repository

```bash
git clone https://github.com/Cosmos-Krishna/credit-risk-decision-system.git
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Train Pipeline

```bash
python main.py
```

---

## 4. Launch Dashboard

```bash
streamlit run app/ui_app.py
```

---

# 📸 Dashboard Preview

Add screenshots inside:

```text
assets/
```

Recommended screenshots:

* dashboard
* model comparison
* SHAP explanations
* approval/rejection example
* portfolio analytics

---

# 🎯 Business Impact

This project demonstrates:

* real-world credit risk modeling
* explainable machine learning
* risk-aware portfolio optimization
* business-driven decision systems
* production-style ML architecture
* model benchmarking and validation

---

# 🔮 Future Improvements

Potential enterprise extensions:

* PostgreSQL integration
* Docker deployment
* FastAPI backend
* Real-time scoring API
* Drift monitoring
* Automated retraining pipeline
* Cloud deployment

---

# 👨‍💻 Author

Krishna Sinha

BTech IT Student · Data Science Intern

Focused on:

* Machine Learning
* Explainable AI
* BFSI Analytics
* Applied ML Engineering
