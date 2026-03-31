🏠 Messy House Price Prediction (EDA + Linear Regression)

This project focuses on analyzing a messy real-world housing dataset and building a Linear Regression model to predict house prices.

The dataset contains inconsistencies, missing values, and noise — making it a perfect case for practicing data cleaning + EDA + regression.

📌 Problem Statement

House prices depend on multiple factors such as area, location, number of rooms, and more.

The objective of this project is to:

Clean a messy dataset 🧹
Perform Exploratory Data Analysis (EDA) 📊
Identify key features affecting house prices
Build a Linear Regression model for prediction
📊 Dataset Description

The dataset includes real estate-related features such as:

Area (size of house)
Location
Number of Bedrooms
Number of Bathrooms
Year Built
Price (Target Variable)

⚠️ The dataset is intentionally messy and includes:

Missing values
Inconsistent data types
Possible outliers
🧹 Data Cleaning

Key cleaning steps performed:

Handling missing values using mean / median / mode
Removing duplicate records
Fixing incorrect data types
Treating outliers
Encoding categorical variables
🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand patterns and relationships in the data.

📈 Key Analysis:
Univariate analysis (distribution of features)
Bivariate analysis (feature vs price)
Correlation analysis
Outlier detection
💡 Key Insights:
Larger area generally leads to higher house prices
Location significantly impacts pricing
Some features showed weak correlation and were removed
Outliers had a noticeable effect on model performance
🤖 Model Building — Linear Regression

A Linear Regression model was trained to predict house prices.

Steps:
Feature selection
Train-test split
Model training
Prediction
📊 Model Evaluation

Evaluation was done using:

R² Score
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
📌 Interpretation:
Model captures general trends well
Some variance remains due to dataset noise
Performance improves after cleaning and feature selection
🧠 What I Learned
How to work with messy real-world datasets
Importance of data cleaning before modeling
How EDA helps in feature selection
Understanding assumptions of Linear Regression
Impact of outliers on model performance
🛠️ Tech Stack
Python 🐍
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
📁 Project Structure
📦 Messy-House-Price
 ┣ 📄 dataset.csv
 ┣ 📄 eda.ipynb
 ┣ 📄 model.ipynb
 ┗ 📄 README.md
🚀 Future Improvements
Apply advanced models (Random Forest, XGBoost)
Perform feature scaling & normalization
Hyperparameter tuning
Try polynomial regression
⚠️ Disclaimer

This project is for learning purposes and may not reflect real-world pricing accuracy.

🔗 Connect With Me
💼 LinkedIn: https://www.linkedin.com/in/zenil-roy-2b8a041a2/
⭐ Support

If you found this project helpful:

⭐ Star this repository
🍴 Fork it
💬 Share feedback
⚡ “Good models come from good data — and good data comes from good EDA.”
