# 🧠 Breast Cancer Classification using Machine Learning

## 📌 Project Overview

This project aims to build a machine learning model to classify breast tumors as **Malignant (M)** or **Benign (B)** using features extracted from digitized images of fine needle aspirate (FNA) of breast masses.

The goal is to assist in early detection of breast cancer by leveraging data-driven insights and predictive modeling.

---

## 📂 Dataset Description

* Dataset: Breast Cancer Wisconsin Dataset
* Total Samples: ~569
* Features: 30 numerical features
* Target Variable:

  * `M` → Malignant (1)
  * `B` → Benign (0)

### 🧩 Feature Categories

Each feature is represented in three forms:

* **Mean** → Average value
* **Standard Error (SE)** → Variation
* **Worst** → Maximum value

### 🔬 Key Features

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Symmetry
* Fractal Dimension

---

## 🧹 Data Preprocessing

Steps performed:

1. Removed unnecessary columns (`id`, `Unnamed: 32`)
2. Converted target variable:

   * `M → 1`, `B → 0`
3. Checked for missing values (none found)
4. Split dataset into training and testing sets (80:20)

---

## 📊 Exploratory Data Analysis (EDA)

Key insights:

* No missing values in dataset
* Some features are highly correlated (e.g., radius, perimeter, area)
* Malignant tumors tend to have:

  * Larger size
  * Higher concavity
  * Irregular shapes
* Slight class imbalance observed

---

## ⚙️ Feature Scaling

Used **StandardScaler** to normalize features:

* Mean ≈ 0
* Standard Deviation ≈ 1

Important:

* Scaler was **fitted only on training data**
* Applied same transformation to test data to avoid data leakage

---

## 🤖 Model Training

### 🔹 Logistic Regression

* Chosen as baseline model
* Requires feature scaling

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
```

---

## 📈 Model Evaluation

### Metrics Used:

* Accuracy
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

### 🎯 Key Focus:

* **Recall for Malignant Class (1)**
  Important to minimize false negatives (missing cancer cases)

---

## 📊 Results

* Accuracy: ~95%
* High recall achieved for malignant tumors
* Model performs well in distinguishing between classes

---

## 🔍 Key Insights

* Features like **concavity, radius, and perimeter** are strong predictors
* “Worst” feature values are more important than mean values
* Logistic Regression performs well due to clear feature separation

---

## ⚠️ Important Considerations

* False Negatives are critical in medical applications
* Threshold tuning can improve recall
* Model should be validated on real-world data before deployment

---

## 🚀 Future Improvements

* Try advanced models:

  * Random Forest
  * XGBoost
* Perform hyperparameter tuning
* Deploy model using Streamlit or Flask
* Handle data drift in production

---

## 🧠 Conclusion

This project demonstrates how machine learning can effectively classify breast tumors with high accuracy. Proper preprocessing, scaling, and evaluation are crucial in building reliable medical prediction systems.

---

## 👨‍💻 Author

**Zenil Roy**
PG Diploma AI/ML IIT Gandhinagar
---
