# 🚗 Indian Road Accidents - Exploratory Data Analysis (EDA)

## 📌 Overview

This project performs a detailed **Exploratory Data Analysis (EDA)** on an Indian road accidents dataset to uncover patterns related to accident frequency, severity, and contributing factors such as time, weather, state, and cause.

The goal is to move beyond surface-level observations and extract **meaningful, business-relevant insights**.

---

## 🧠 Key Analyses Performed

### 1. Missing Value Analysis

* Identified that the `festival` column contains many missing values.
* Interpreted missing values correctly as **non-festival days**, not data errors.
* Engineered a new feature: `is_festival`.

---

### 2. Festival Impact Analysis

* Compared accidents during festival vs non-festival days.
* Found that **festival days are rare**, so raw counts alone are misleading.
* Shifted focus to **risk_score comparison** instead of counts.

---

### 3. Univariate Analysis

* Explored distributions of numerical and categorical features.
* Observed:

  * `risk_score` is relatively consistent across dataset.
  * Dataset is **imbalanced toward non-festival and non-peak conditions**.

---

### 4. State-wise Analysis

* Maharashtra had the highest number of accidents (volume-based insight).
* However, after normalization:

  * **Average risk across states is very similar (~0.44–0.45)**.
* Conclusion:

  * State is **not a strong predictor of accident severity**.

---

### 5. Advanced Visualization (Beyond Bar Charts)

* Used:

  * Box plots
  * Violin plots
  * Strip plots
* Insight:

  * Despite similar averages, distributions show **minor variation but no dominant state risk pattern**.

---

### 6. Time-based Analysis

* Created `time_period` feature:

  * Morning, Day, Evening (Peak), Night
* Observed:

  * **Night and peak hours tend to have higher risk**
  * Morning is relatively safer

---

### 7. State + Time Analysis

* Compared risk across time periods for each state.
* Found:

  * **Consistent pattern across states**
  * Time of day influences risk more than location

---

### 8. Weather + Time Interaction

* Analyzed combined effect of weather and time.
* Insights:

  * **Night + bad weather = highest risk**
  * Even in clear weather, **peak hours increase risk**

---

### 9. 3D Visualization

* Built 3D surface plot for:

  * Weather + Time → Risk Score
* Used for:

  * Identifying **dangerous combinations**
  * Portfolio-level visualization

---

### 10. Cause vs Accident Severity Analysis

#### Raw Counts:

* Poor road conditions had the highest fatal accidents (absolute count)

#### Normalized Analysis (Key Insight):

* Fatality rates across causes:

| Cause         | Fatal % |
| ------------- | ------- |
| Poor Road     | ~15.67% |
| Drunk Driving | ~15.53% |
| Weather       | ~15.03% |
| Overspeeding  | ~14.98% |
| Distraction   | ~13.46% |

#### Final Insight:

* **All causes have similar fatality rates (~15%)**
* Cause impacts **frequency**, but not strongly **severity**

---

## 💡 Final Conclusions

* Accident severity is **not heavily dependent on state or cause alone**
* **Time of day and environmental conditions** play a more significant role
* Dataset shows signs of:

  * Balanced or synthetic distribution
  * Limited variation in severity patterns

---

## 🚀 Skills Demonstrated

* Data Cleaning & Feature Engineering
* Statistical Thinking (Normalization vs Raw Counts)
* Advanced Visualization Techniques
* Business-Oriented Insight Extraction
* Multi-variable Analysis

---

## 📌 Future Improvements

* Include:

  * Traffic density data
  * Road quality metrics
  * Population normalization
* Apply:

  * Machine Learning models for severity prediction
  * Feature importance analysis

---

## ✍️ Author

**Zenil Roy**
** PG Diploma in AI/ML at IIT Gandhinagar **
---

*“EDA is not about plotting graphs, it's about extracting stories from data.”* 🚀
