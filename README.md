💼 Expected CTC Prediction Model
📊 Project Overview
🎯 Goal & Objective: The objective of this exercise is to build a model, using historical data, that will determine the salary to be offered to an employee,
minimizing manual judgment in the selection process. 
The approach aims to be robust and eliminate any discrimination in salary among employees with similar profiles.

📂 Dataset Overview :

Data Preprocessing :
✅ Duplicate removal and missing value handling
✅ Smart imputation: Median for numerical, Mode for categorical
✅ Outlier treatment using IQR capping method

Exploratory Data Analysis (EDA) :
Before/after comparison of outlier treatment
Correlation heatmaps identifying key predictors
Feature correlation analysis with Expected CTC
Pairplots for most influential features

Feature Engineering :
Strategic feature selection (removed irrelevant columns)
One-Hot Encoding for categorical variables (Department, Role, Industry, etc.)
Result: 100+ engineered features for optimal model performance

Machine Learning Models :
Random Forest Regressor (Best performer)
Linear Regression (Baseline comparison)
Cross-validation for robust performance evaluation







