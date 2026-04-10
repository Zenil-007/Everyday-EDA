# High-Level Design (HLD)
## Enterprise Fraud Detection System
**Version:** 1.0 | **Date:** 10 April 2026 

---

## 1. Business Problem Statement

### Context
Vesta Corporation processes millions of e-commerce transactions daily. A small fraction (~3.5%) are fraudulent, yet these cause disproportionate financial and reputational damage. The task is to build a real-time-ready scoring model that assigns each incoming transaction a fraud probability, enabling downstream rule engines to accept, flag, or decline it.

### Cost Asymmetry: Why Accuracy Is the Wrong Metric

| Error Type | Business Event | Estimated Cost |
|---|---|---|
| **False Negative** (fraud missed) | Chargeback + fraud loss + ops investigation | **$500 per transaction** |
| **False Positive** (legit blocked) | Lost transaction revenue + customer churn risk | **$200 per transaction** |

A naive model that flags everything as fraud achieves 100% recall at the cost of blocking every legitimate customer — economically catastrophic. A model that flags nothing misses all fraud. The optimal operating point **minimises Total Cost = FP × $200 + FN × $500**, not misclassification rate.

Because the class ratio is ~28.5:1 (legitimate:fraud), standard accuracy metrics are misleading. A model predicting "all legitimate" would score ~96.5% accuracy while being completely useless.

---

## 2. Data Sources

### 2.1 train_transaction.csv
- **Size:** ~652 MB, ~590,540 rows, 394 columns
- **Key fields:** `TransactionID` (join key), `TransactionDT` (timedelta seconds from reference date), `TransactionAmt`, `ProductCD`, `card1–card6`, `addr1–addr2`, `dist1–dist2`, `P_emaildomain`, `R_emaildomain`, `C1–C14` (count features), `D1–D15` (timedelta features), `M1–M9` (match features), `V1–V339` (Vesta-engineered features), `isFraud` (target)
- **Target distribution:** ~3.5% fraud (~20,663 positive cases)

### 2.2 train_identity.csv
- **Size:** ~25 MB, ~144,233 rows, 41 columns
- **Key fields:** `TransactionID`, `DeviceType`, `DeviceInfo`, `id_01–id_38`
- **Coverage:** Only ~24% of transactions have identity records; absence itself is informative

### 2.3 How Tables Connect
```
train_transaction  ──[LEFT JOIN on TransactionID]──  train_identity
       ↓
   Merged DataFrame (590,540 rows × ~434 columns)
   + binary feature: has_identity (1 if identity row exists)
```
The LEFT JOIN preserves all transaction records. Transactions without identity info are the majority and must not be dropped.

---

## 3. Full Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  RAW DATA LAYER                                             │
│  /data/raw/train_transaction.csv                            │
│  /data/raw/train_identity.csv                               │
└────────────────────────┬────────────────────────────────────┘
                         │  data_loader.py
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  INGESTION & MERGE                                          │
│  • dtype downcast (int64→int32, float64→float32)            │
│  • LEFT JOIN on TransactionID                               │
│  • Add has_identity flag                                    │
│  • Save merged parquet to /data/processed/                  │
└────────────────────────┬────────────────────────────────────┘
                         │  EDA.ipynb
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  EXPLORATORY DATA ANALYSIS                                  │
│  • Class imbalance visualisation                            │
│  • Missing value profile                                    │
│  • Distribution analysis                                    │
│  • Fraud rate by segment                                    │
│  • V-feature correlation clusters                           │
└────────────────────────┬────────────────────────────────────┘
                         │  feature_engineering.py
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  DATA CLEANING & FEATURE ENGINEERING                        │
│  • Drop columns >80% missing                                │
│  • Impute: median (numeric), mode (categorical)             │
│  • Temporal features (hour, day_of_week, is_night)          │
│  • Email features (domain, free email, match)               │
│  • Device parsing (OS from DeviceInfo)                      │
│  • Velocity features (card transaction count)               │
│  • Target encoding (email domain fraud rate)                │
│  • PCA on V-feature clusters                                │
│  • Frequency & label encoding for categoricals              │
└────────────────────────┬────────────────────────────────────┘
                         │  train.py
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  MODEL TRAINING (Time-Based Split)                          │
│  ├── Logistic Regression (baseline)                         │
│  ├── LightGBM (gradient boosting, fast)                     │
│  ├── XGBoost (gradient boosting, robust)                    │
│  └── Random Forest (ensemble baseline)                      │
└────────────────────────┬────────────────────────────────────┘
                         │  evaluate.py
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  EVALUATION & THRESHOLD OPTIMISATION                        │
│  • AUC-ROC, PR-AUC, F1, Precision, Recall                  │
│  • Cost-minimising threshold search                         │
│  • SHAP explainability for best model                       │
│  • Confusion matrix at optimal threshold                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  TEST INFERENCE                                             │
│  • Apply identical pipeline (NO data leakage)              │
│  • Output submission.csv: TransactionID, isFraud (prob)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Metric Selection Rationale

### Why NOT Accuracy
With ~3.5% fraud prevalence, a model predicting all-legitimate scores 96.5% accuracy — useless.

### Why AUC-ROC is Insufficient Alone
ROC curves are optimistic under class imbalance because they weight both classes equally across thresholds. A model can look good on ROC while performing poorly on the minority class.

### Why Precision-Recall AUC (PR-AUC) is Primary
- PR curves focus exclusively on the positive (fraud) class
- PR-AUC degrades appropriately when a model fails to find fraud
- It captures the precision-recall trade-off that directly maps to the FP/FN cost asymmetry
- **A random classifier achieves PR-AUC ≈ prevalence (3.5%); our model must substantially exceed this**

### Operational Metric: Cost-Minimised Threshold
The final deployed threshold is chosen to minimise:
```
Total_Cost = FP × $200 + FN × $500
```
This makes the evaluation metric directly tied to business outcomes, not abstract statistical scores.

---

## 5. Key Assumptions

| Assumption | Risk if Wrong | Mitigation |
|---|---|---|
| TransactionDT is a monotonically increasing timedelta (no gaps or restarts) | Time-based split would be invalid | Verify distribution; check for resets |
| Fraud patterns in the validation period resemble deployment | Model underperforms in production | Monitor feature drift; retrain quarterly |
| $500/$200 cost estimates are reasonable approximations | Threshold set incorrectly | Parameterise cost matrix; expose to stakeholders for review |
| Left join is correct (identity table is optional) | Losing transactions if inner join used | Confirm with data dictionary; never inner join |
| V-features are stable Vesta proprietary signals | Features disappear at deployment | Abstract them behind a feature layer |
| Missing identity records are missing at random (MAR) | Systematic absence encodes fraud signal | Test: fraud rate with vs without identity record |
| No future leakage in V/C/D features | Inflated offline metrics | Treat temporal validation as ground truth |

---

## 6. What Could Break This System

1. **Concept drift:** Fraud rings adapt tactics. A model trained on 2017–2018 patterns may fail on 2024 attacks.
2. **Data pipeline failures:** If identity enrichment fails in production, `has_identity` and all ID features become stale zeros.
3. **New email domains:** Target encoding assigns the global mean to unseen domains — monitor for high-volume new domains.
4. **Card number recycling:** Banks reissue card numbers; `card_transaction_count` becomes stale.
5. **Threshold staleness:** If chargeback costs change, the deployed threshold is no longer optimal without retuning.

---

*Next: See LLD.md for feature-level specifications and model implementation details.*
