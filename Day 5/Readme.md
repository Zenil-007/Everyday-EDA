# 💳 Credit Risk Prediction & Loan Decision System

## 🚀 Project Overview

This project focuses on building an **end-to-end machine learning system** to predict loan defaults and assist financial institutions in making smarter lending decisions.

We developed a **data-driven credit risk model** using historical loan data and translated predictions into **actionable business insights** such as:

* Risk categorization
* Interest rate recommendation
* Loan approval decisions
* Portfolio-level risk analysis

---

## 🎯 Problem Statement

Financial institutions face significant challenges in:

* Identifying high-risk borrowers
* Minimizing loan defaults
* Pricing loans appropriately

Our objective was to:

> Predict the probability of loan default and convert it into meaningful business decisions.

---

## 📊 Dataset

* Source: LendingClub dataset
* Size: ~2.2 million records
* Features: 145+

### Key Feature Categories:

* Borrower Info (income, employment, home ownership)
* Loan Details (amount, interest rate, term)
* Credit History (DTI, utilization, accounts)
* Loan Performance (loan_status → target)

---

## 🧹 Data Preprocessing

### ✅ Feature Selection

Selected ~15 meaningful features based on:

* Financial strength
* Credit behavior
* Business relevance

### ✅ Target Transformation

* Fully Paid → `0`
* Charged Off → `1`

Removed ambiguous statuses like:

* Current
* Late
* In Grace Period

---

## ⚖️ Class Imbalance

* Dataset ratio ≈ **80:20**
* Handled using:

  * `class_weight='balanced'`
  * Focus on **Recall & ROC-AUC**

---

## 🧼 Missing Value Handling

| Type           | Strategy         |
| -------------- | ---------------- |
| Numerical      | Median           |
| Categorical    | Mode / "Unknown" |
| Count features | 0                |

---

## 🛠️ Feature Engineering

We created meaningful features to enhance model performance:

### 🔹 Income Strength

* `income_to_loan`
* `installment_to_income`

### 🔹 Credit Stress

* `credit_pressure = dti × revol_util`
* `util_per_account`

### 🔹 Credit Behavior

* `accounts_ratio`
* `has_delinquency`

### 🔹 Loan Risk Indicators

* `high_interest`
* `long_term`

### 🔹 Employment Stability

* Converted `emp_length` to numeric

---

## 🔄 Data Encoding

| Feature   | Encoding         |
| --------- | ---------------- |
| term      | Numeric          |
| grade     | Ordinal          |
| sub_grade | Ordinal          |
| Others    | One-Hot Encoding |

---

## 📈 Exploratory Data Analysis (EDA)

### Key Insights:

* Higher **DTI → higher default risk**
* High **revol_util → credit stress**
* Lower **income_to_loan → risky borrowers**
* Lower **grades → higher defaults**

---

## 🤖 Model Building

### Models Used:

1. Logistic Regression (Baseline)
2. Random Forest
3. XGBoost (Final Model)

---

## 📊 Model Evaluation

### Metrics Used:

* ROC-AUC ⭐
* Recall (for defaulters)
* F1-score

### Results:

| Model               | Test ROC-AUC | Behavior   |
| ------------------- | ------------ | ---------- |
| Logistic Regression | ~0.71        | Underfit   |
| Random Forest       | ~0.71        | Overfit    |
| XGBoost             | **~0.71+**   | Balanced ✅ |

---

## 🔍 Cross Validation

* Mean ROC-AUC ≈ **0.68**
* Indicates **stable and generalizable model**

---

## 🏆 Final Model Selection

👉 **XGBoost (Tuned)**

### Reason:

* Best generalization
* Balanced bias-variance tradeoff
* Strong real-world performance

---

## 💡 Business Logic Layer

### 🔹 Risk Bucketing

| Probability | Risk        |
| ----------- | ----------- |
| < 0.3       | Low Risk    |
| 0.3 – 0.6   | Medium Risk |
| > 0.6       | High Risk   |

---

### 💰 Interest Rate Recommendation

| Risk   | Interest Rate |
| ------ | ------------- |
| Low    | 8% – 10%      |
| Medium | 10% – 14%     |
| High   | 14% – 20%     |

---

### 🏦 Loan Decision

| Risk   | Decision |
| ------ | -------- |
| Low    | Approve  |
| Medium | Review   |
| High   | Reject   |

---

## 🖥️ UI & Product Development

### 🔮 Loan Risk Predictor

* User inputs borrower details
* Outputs:

  * Default probability
  * Risk category
  * Interest rate
  * Loan decision

---

### 📂 Portfolio Analyzer (Key Feature)

* Upload loan portfolio CSV
* Outputs:

  * Average risk
  * % high-risk loans
  * Portfolio health
  * Risk alerts

---

## 🧠 Key Learnings

* Feature engineering significantly improves model performance
* ROC-AUC is more reliable than accuracy in imbalanced datasets
* Business translation of ML outputs is critical
* Generalization > Training performance

---

## 💼 Business Impact

* Reduce loan defaults
* Improve risk-based pricing
* Enable data-driven lending decisions
* Provide portfolio-level insights

---

## 🔥 Innovation

> We go beyond prediction by integrating machine learning with real-world financial decision-making.

---

## 🚀 Future Improvements

* SHAP explainability
* Real-time API deployment
* Integration with Power BI dashboard
* Advanced feature engineering

---

## 🏁 Conclusion

This project demonstrates a complete pipeline:

> Data → Insights → Model → Business Decisions → Product

It showcases how machine learning can be applied to solve real-world financial problems effectively.

---
## 👨‍💻 Author

**Zenil Roy (Solace)**
AI & Data Science Enthusiast
