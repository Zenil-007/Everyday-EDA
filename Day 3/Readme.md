# 🌍 Global Poverty & Inequality Analysis (EDA Project)

## 📌 Overview

This project performs **Exploratory Data Analysis (EDA)** on a global dataset containing information about:

* Poverty levels
* Income inequality
* GDP & economic indicators
* Income distribution across populations

The goal is to understand:

> **What factors influence poverty across countries and over time?**

---

## 🎯 Objectives

* Analyze global poverty trends
* Study relationship between **GDP, inequality, and poverty**
* Handle missing data using **real-world strategies**
* Prepare dataset for **machine learning (regression modeling)**

---

## 📂 Dataset Features

| Column          | Description                        |
| --------------- | ---------------------------------- |
| country         | Country name                       |
| year            | Year of observation                |
| iso_code        | Country ISO code                   |
| population      | Total population                   |
| gdp             | Total GDP                          |
| gdp_per_capita  | GDP per person                     |
| poverty_rate    | % of population below poverty line |
| gini_index      | Income inequality (0–100 scale)    |
| income_top1     | Income share of top 1%             |
| income_top10    | Income share of top 10%            |
| income_bottom50 | Income share of bottom 50%         |

---

## ⚠️ Data Challenges

* Severe missing values initially:

  * `poverty_rate` → ~75% missing
  * `gini_index` → ~75% missing
* Moderate missing values:

  * GDP-related columns (~5%)
  * Income distribution (~7%)

---

## 🧹 Data Cleaning Strategy

### 1. Target-Aware Filtering

* Retained rows where:

  * `poverty_rate` is available
  * `gini_index` is available

> Result: ~24% of original dataset retained initially

---

### 2. Bias Detection

A custom function was used to compare:

* Mean
* Range

Across original vs cleaned dataset

📊 Result:

* ~33% columns showed distribution change
* Indicates **moderate bias**

---

### 3. Improved Imputation Strategy

Instead of dropping excessive data:

#### ✅ Gini Index

* Filled using **region-based median**
* Preserves economic similarity

#### ✅ GDP & GDP per Capita

* Filled using **country-wise median**
* With global fallback for edge cases

#### ✅ Income Distribution Columns

* Filled using **region-wise median + global fallback**

---

## 🧠 Key Learning

* ❌ Dropping data blindly → introduces bias
* ✅ Smart imputation → preserves structure
* ⚖️ Data science = trade-off between **accuracy vs completeness**

---

## 📊 EDA Performed

* Missing value analysis
* Distribution plots
* Correlation matrix
* GroupBy analysis (region, year)
* Outlier detection
* Feature relationships

---

## 🔍 Key Insights (To Be Updated)

* GDP vs Poverty relationship
* Inequality vs Poverty correlation
* Regional disparities
* Income concentration effects

---

## ⚠️ Limitations

* Initial dataset had heavy missing values
* Moderate bias introduced during cleaning
* Some imputations assume regional similarity

---

## 🚀 Future Work

* Apply **Linear Regression / ML models**
* Feature engineering (log transformations, ratios)
* Advanced imputation (ML-based)
* Time-series trend analysis

---

## 🛠️ Tech Stack

* Python 🐍
* Pandas
* NumPy
* Matplotlib / Seaborn

---

## 💡 Conclusion

This project demonstrates how to:

* Handle **real-world messy data**
* Make **informed cleaning decisions**
* Perform structured EDA

> Not just analyzing data — but thinking like a data scientist.

---

## 👨‍💻 Author

**Zenil Roy**
PG Diploma IIT Gandhinagar
---
