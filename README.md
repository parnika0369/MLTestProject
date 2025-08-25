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


<img width="881" height="623" alt="Screenshot 2025-08-25 224303" src="https://github.com/user-attachments/assets/282b8420-5d7b-41bf-adc1-5acfd3957621" />



The correlation heatmap provides a quick overview of how variables relate to each other, highlighting potential multicollinearity and guiding feature selection for machine learning models.
<img width="1919" height="999" alt="Screenshot 2025-08-25 230133" src="https://github.com/user-attachments/assets/5042e337-42c7-47c6-b22f-5800011f331e" />
<img width="1914" height="1005" alt="Screenshot 2025-08-25 225702" src="https://github.com/user-attachments/assets/0926bb3e-9c6a-4bb3-a6c3-af7ce9efce6b" />

<img width="1501" height="1000" alt="Screenshot 2025-08-25 230201" src="https://github.com/user-attachments/assets/d7674a52-4eb8-417c-9a89-e53e063c2c6f" />




For the Linear Regression Plot:
This plot visualizes the relationship between two variables and the line of best fit, showing how well a linear model captures the underlying trend (with the R-value indicating the strength and direction of the correlation).

For the Random Forest Residual Plot:
This plot shows the difference between the model's predictions and the actual values; a random scatter around zero indicates a well-performing model, while any clear patterns suggest the model is missing key trends.


<img width="1096" height="567" alt="Screenshot 2025-08-25 230436" src="https://github.com/user-attachments/assets/8acfb266-35f2-47bb-ad19-2fcc2d1c2f30" />
<img width="1919" height="1011" alt="Screenshot 2025-08-25 225635" src="https://github.com/user-attachments/assets/d1db971d-8524-4108-bae6-663e8534e6eb" />
