# Low-Level Design (LLD)
## Fraud Detection - Feature & Model Specifications
**Version:** 1.0 | **Date:** 10 April 2026 

---

## 1. Feature Inventory

### 1.1 Transaction Table - Core Features

| Feature Name | Type | Source | Missing Strategy | Notes |
|---|---|---|---|---|
| `TransactionID` | int | transaction | None (join key) | Drop after merge |
| `TransactionDT` | int | transaction | None (no nulls expected) | Seconds from reference date |
| `TransactionAmt` | float | transaction | Median impute | Log-transform for modelling |
| `ProductCD` | str (cat) | transaction | Mode impute | W, H, C, S, R |
| `card1` | int | transaction | Median | Card issuer ID |
| `card2` | float | transaction | Median | — |
| `card3` | float | transaction | Median | — |
| `card4` | str (cat) | transaction | Mode | visa/mc/amex/discover |
| `card5` | float | transaction | Median | — |
| `card6` | str (cat) | transaction | Mode | debit/credit |
| `addr1` | float | transaction | Median | Billing zip area code |
| `addr2` | float | transaction | Median | Billing country code |
| `dist1` | float | transaction | Median | Distance (billing to mail) |
| `dist2` | float | transaction | Median | Distance variant |
| `P_emaildomain` | str (cat) | transaction | Mode | Purchaser email domain |
| `R_emaildomain` | str (cat) | transaction | Mode | Recipient email domain |

### 1.2 Transaction Table - C Features (Count-based)

| Feature Name | Type | Source | Missing Strategy | Notes |
|---|---|---|---|---|
| `C1`–`C14` | float | transaction | Median | Anonymised count features (how many addresses, cards, etc. associated with this user) |

### 1.3 Transaction Table - D Features (Timedelta-based)

| Feature Name | Type | Source | Missing Strategy | Notes |
|---|---|---|---|---|
| `D1`–`D15` | float | transaction | Median | Days between transactions; D1 = days since last transaction on this card |

### 1.4 Transaction Table - M Features (Match flags)

| Feature Name | Type | Source | Missing Strategy | Notes |
|---|---|---|---|---|
| `M1`–`M9` | str (T/F/NaN) | transaction | Mode | Binary match flags (e.g. name on card matches billing name). Encode T→1, F→0 |

### 1.5 Transaction Table - V Features (Vesta-engineered)

| Feature Group | Columns | Type | Missing Strategy | Engineering |
|---|---|---|---|---|
| V1–V11 | 11 cols | float | Median | Cluster A — apply PCA |
| V12–V34 | 23 cols | float | Median | Cluster B — apply PCA |
| V35–V52 | 18 cols | float | Median | Cluster C — apply PCA |
| V53–V74 | 22 cols | float | Median | Cluster D — apply PCA |
| V75–V94 | 20 cols | float | Median | Cluster E — apply PCA |
| V95–V137 | 43 cols | float | Median | Cluster F — apply PCA |
| V138–V166 | 29 cols | float | Median | Cluster G — apply PCA |
| V167–V216 | 50 cols | float | Median | Cluster H — apply PCA |
| V217–V278 | 62 cols | float | Median | Cluster I — apply PCA |
| V279–V321 | 43 cols | float | Median | Cluster J — apply PCA |
| V322–V339 | 18 cols | float | Median | Cluster K — apply PCA |

**PCA Strategy for V-features:**
1. Compute Pearson correlation matrix across all V1–V339
2. Apply hierarchical clustering (linkage='complete') on the distance matrix (1 - |corr|)
3. Cut dendrogram at threshold=0.3 to form clusters
4. For each cluster: StandardScaler → PCA → keep components explaining 95% cumulative variance
5. Replace original V columns with PCA components named `V_clusterA_pc1`, `V_clusterA_pc2`, etc.
6. Fit PCA **only on training fold**; transform validation/test using the fitted objects

### 1.6 Identity Table Features

| Feature Name | Type | Source | Missing Strategy | Notes |
|---|---|---|---|---|
| `DeviceType` | str (cat) | identity | Mode | desktop/mobile |
| `DeviceInfo` | str (cat) | identity | Mode | Raw string; parse for OS |
| `id_01`–`id_11` | float | identity | Median | Anonymised identity scores |
| `id_12`–`id_38` | str (cat) | identity | Mode | Categorical identity flags |

---

## 2. Feature Engineering Plan

### 2.1 Temporal Features
```python
REFERENCE_DATE = datetime(2017, 11, 30)  # Inferred from dataset context

df['datetime'] = REFERENCE_DATE + pd.to_timedelta(df['TransactionDT'], unit='s')
df['hour'] = df['datetime'].dt.hour                    # 0–23
df['day_of_week'] = df['datetime'].dt.dayofweek        # 0=Mon, 6=Sun
df['is_night'] = (df['hour'].between(0, 5)).astype(int)  # Fraud spikes at night
```

### 2.2 Transaction Amount Features
```python
df['log_TransactionAmt'] = np.log1p(df['TransactionAmt'])
```
Rationale: TransactionAmt is right-skewed. Log-transform normalises the distribution and improves tree split quality for linear models.

### 2.3 Email Domain Features
```python
FREE_EMAIL_DOMAINS = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
                      'yahoo.fr', 'yahoo.es', 'anonymous.com'}

df['is_free_email'] = df['P_emaildomain'].isin(FREE_EMAIL_DOMAINS).astype(int)

df['sender_receiver_domain_match'] = (
    df['P_emaildomain'] == df['R_emaildomain']
).astype(int)
```

### 2.4 Target Encoding - Email Domain Fraud Rate
```python
# FIT ONLY ON TRAINING FOLD (prevent leakage)
domain_fraud_rate = train_df.groupby('P_emaildomain')['isFraud'].mean()
global_mean = train_df['isFraud'].mean()

df['email_domain_fraud_rate'] = df['P_emaildomain'].map(domain_fraud_rate).fillna(global_mean)
```
**Smoothing:** For domains with fewer than 50 observations, apply Bayesian smoothing:
```
smoothed_rate = (count × domain_mean + k × global_mean) / (count + k)  [k=50]
```

### 2.5 Device OS Parsing
```python
def parse_device_os(device_info: str) -> str:
    """Extract OS family from raw DeviceInfo string."""
    if pd.isna(device_info):
        return 'unknown'
    device_lower = str(device_info).lower()
    if 'windows' in device_lower: return 'windows'
    if 'ios' in device_lower or 'iphone' in device_lower: return 'ios'
    if 'android' in device_lower: return 'android'
    if 'mac' in device_lower: return 'mac'
    if 'linux' in device_lower: return 'linux'
    return 'other'

df['device_os'] = df['DeviceInfo'].apply(parse_device_os)
```

### 2.6 Velocity Features
```python
# Fit only on train; apply to train+test using precomputed map
card_counts = train_df['card1'].value_counts().to_dict()
df['card_transaction_count'] = df['card1'].map(card_counts).fillna(0)
```
High transaction count for a card may indicate either a trusted heavy user OR a compromised card in active use.

### 2.7 Row-Level Null Count
```python
# Compute BEFORE imputation (captures missingness pattern)
df['null_count_per_row'] = df.isnull().sum(axis=1)
```

---

## 3. Encoding Strategy for Categoricals

| Cardinality | Strategy | Columns |
|---|---|---|
| ≤ 10 unique values | Label encoding (LabelEncoder) | ProductCD, card4, card6, DeviceType, M1–M9, device_os |
| > 10 unique values | Frequency encoding (replace with count) | P_emaildomain, R_emaildomain, DeviceInfo, id_30, id_31 |

**Frequency encoding** replaces each category with its count in the training data — captures popularity signal without high-dimensional one-hot expansion.

---

## 4. Data Cleaning Rules

### 4.1 Column Dropping Threshold
Drop any column where `null % > 80%`. Log all dropped columns to `/outputs/dropped_columns.txt`.

Expected drops (from Kaggle EDA literature): Many V-features and identity features exceed this threshold.

### 4.2 Imputation Order
1. Compute `null_count_per_row` **before** any imputation
2. Drop high-null columns
3. Impute: `median` for numeric, `mode` (most frequent) for categorical/object
4. M-features: T→1, F→0, NaN→-1 (preserve "unknown" as a separate signal)

---

## 5. Model Hypotheses

### 5.1 Logistic Regression (Baseline)
- **Hypothesis:** A linear combination of normalised features can separate fraud/non-fraud better than random
- **Purpose:** Establishes a performance floor; fast to train; interpretable coefficients
- **Configuration:** `class_weight='balanced'`, `max_iter=1000`, `C=0.01` (regularised)
- **Expected PR-AUC:** 0.40–0.55

### 5.2 LightGBM
- **Hypothesis:** Gradient boosted trees will outperform linear models due to non-linear interactions in V-features and temporal patterns
- **Configuration:** `num_leaves=63`, `learning_rate=0.05`, `n_estimators=1000`, `is_unbalance=True`, `early_stopping_rounds=50`
- **Expected PR-AUC:** 0.75–0.85
- **Advantages:** Handles missing values natively; very fast; supports `is_unbalance`

### 5.3 XGBoost
- **Hypothesis:** Similar to LGBM but `scale_pos_weight = n_negatives/n_positives ≈ 28.5` handles imbalance at the loss level
- **Configuration:** `max_depth=6`, `learning_rate=0.05`, `n_estimators=1000`, `scale_pos_weight=28.5`, `eval_metric='aucpr'`
- **Expected PR-AUC:** 0.74–0.84

### 5.4 Random Forest
- **Hypothesis:** Uncorrelated decision trees will generalise well without overfitting to training noise
- **Configuration:** `n_estimators=500`, `max_depth=20`, `class_weight='balanced_subsample'`, `n_jobs=-1`
- **Expected PR-AUC:** 0.65–0.75

---

## 6. Evaluation Strategy

### 6.1 Time-Based Train/Validation Split
```
Training set:   Earliest 80% of rows by TransactionDT
Validation set: Latest 20% of rows by TransactionDT
```
**Why time-based (not random)?** Random splitting leaks future patterns into training. In production, the model is always predicting on future data. Time-based split mimics this reality and prevents optimistic bias.

### 6.2 Metrics at Default Threshold (0.5)
- AUC-ROC
- PR-AUC (primary)
- Precision, Recall, F1
- Confusion matrix

### 6.3 Threshold Optimisation
```python
def find_optimal_threshold(y_true, y_prob, cost_fp=200, cost_fn=500):
    """
    Find threshold minimising total misclassification cost.
    
    Parameters
    ----------
    cost_fp : cost of blocking a legitimate customer ($200)
    cost_fn : cost of missing a fraud ($500)
    """
    thresholds = np.linspace(0.01, 0.99, 199)
    costs = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = fp * cost_fp + fn * cost_fn
        costs.append(total_cost)
    return thresholds[np.argmin(costs)]
```

### 6.4 Cost Matrix
```
                 Predicted Fraud    Predicted Legit
Actual Fraud     TP ($0 saved)       FN (-$500 loss)
Actual Legit     FP (-$200 loss)     TN ($0)
```

---

## 7. Feature Store (Final Feature List for Model)

After all engineering, the model input should contain approximately 80–120 features:
- Temporal: `hour`, `day_of_week`, `is_night`
- Amount: `TransactionAmt`, `log_TransactionAmt`
- Card: `card1`–`card6`, `card_transaction_count`
- Address: `addr1`, `addr2`, `dist1`, `dist2`
- Email: `email_domain_fraud_rate`, `is_free_email`, `sender_receiver_domain_match`
- Count features: `C1`–`C14`
- Timedelta features: `D1`–`D15`
- Match features: `M1`–`M9` (encoded)
- Product: `ProductCD`
- V-features: PCA components per cluster (~30–50 total)
- Identity: `DeviceType`, `device_os`, `id_01`–`id_38` (surviving null filter)
- Engineered: `has_identity`, `null_count_per_row`

---

*See HLD.md for business context.*
