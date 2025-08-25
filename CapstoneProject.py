import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
pd.set_option('display.max_columns',200)

#loading the dataset:
df = pd.read_csv('C:\\Users\\Priya\\OneDrive\\Desktop\\vsCode\\ML\\MLCapstoneProject\\expected_ctc.csv')

figures = []#for graphs


#data overview(inspecting first five rows of the dataframe):
print(df.head())
print("Dataset shape:", df.shape)
print(df.info())
print(df.dtypes)
#checking duplicates
dups = df.duplicated()
print("number of duplicated rows = %d" % dups.sum())
#df.drop_duplicates(inplace=True)    #because duplicates are 0



# Numerical columns for outlier analysis
numerical_cols = [
    'Total_Experience', 
    'Total_Experience_in_field_applied',
    'Current_CTC', 
    'Expected_CTC',
    'Number_of_Publications',
    'No_Of_Companies_worked'
]

# Create boxplots BEFORE outlier treatment
fig1=plt.figure(figsize=(20, 12))
plt.suptitle('Boxplots BEFORE Outlier Treatment', fontsize=16, fontweight='bold')

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel('')
plt.tight_layout()
figures.append(fig1)

def remove_outlier(col):
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range

# Outlier treatment for each variable
def treat_outliers(df):
    """
    Treat outliers for all relevant columns in the dataframe
    using the IQR method with capping approach
    """
    df_treated = df.copy()
    
    # Total Experience
    lower_exp, upper_exp = remove_outlier(df_treated['Total_Experience'])
    df_treated['Total_Experience'] = df_treated['Total_Experience'].clip(lower=lower_exp, upper=upper_exp)
    
    # Total Experience in field applied
    lower_exp_field, upper_exp_field = remove_outlier(df_treated['Total_Experience_in_field_applied'])
    df_treated['Total_Experience_in_field_applied'] = df_treated['Total_Experience_in_field_applied'].clip(lower=lower_exp_field, upper=upper_exp_field)
    
    # Number of Publications
    lower_pubs, upper_pubs = remove_outlier(df_treated['Number_of_Publications'])
    df_treated['Number_of_Publications'] = df_treated['Number_of_Publications'].clip(lower=lower_pubs, upper=upper_pubs)
    
    
    # Current CTC
    lower_current_ctc, upper_current_ctc = remove_outlier(df_treated['Current_CTC'])
    df_treated['Current_CTC'] = df_treated['Current_CTC'].clip(lower=lower_current_ctc, upper=upper_current_ctc)
    
    # Number of Companies worked
    lower_companies, upper_companies = remove_outlier(df_treated['No_Of_Companies_worked'])
    df_treated['No_Of_Companies_worked'] = df_treated['No_Of_Companies_worked'].clip(lower=lower_companies, upper=upper_companies)
    
    # Expected CTC
    lower_expected_ctc, upper_expected_ctc = remove_outlier(df_treated['Expected_CTC'])
    df_treated['Expected_CTC'] = df_treated['Expected_CTC'].clip(lower=lower_expected_ctc, upper=upper_expected_ctc)
    
    
    return df_treated

# Apply outlier treatment
df_treated = treat_outliers(df)

# Create boxplots AFTER outlier treatment
fig2=plt.figure(figsize=(20, 12))
plt.suptitle('Boxplots AFTER Outlier Treatment', fontsize=16, fontweight='bold')

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df_treated[col])
    plt.title(f'Boxplot of {col} (After Treatment)')
    plt.ylabel('')
plt.tight_layout()
figures.append(fig2)



#Drop Unnecessary 
print(df.columns)
DropColumns=['IDX',
    'Applicant_ID',
    'Inhand_Offer',
    'Curent_Location',
    'Preferred_location',
    'University_Grad',
    'University_PG',
    'University_PHD',
    'Passing_Year_Of_Graduation',
    'Passing_Year_Of_PG',
    'Passing_Year_Of_PHD',
    'Organization',
    'Designation']

print(df.drop(DropColumns, axis=1,inplace =True))
print(df.dtypes)

#print(df.isna().sum())       #interchangeable
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nBasic statistics:")
print(df.describe(include='all'))
print(df.isnull().sum().sum())

#replacing null values in numerical columns

fill_values = {
    "Total_Experience": df["Total_Experience"].median(),
    "Total_Experience_in_field_applied": df["Total_Experience_in_field_applied"].median(),
    "Current_CTC": df["Current_CTC"].median(),
    "No_Of_Companies_worked": df["No_Of_Companies_worked"].median(),
    "Number_of_Publications": df["Number_of_Publications"].median(),
    "Certifications": df["Certifications"].median(),
    "International_degree_any": df["International_degree_any"].median(),
    "Expected_CTC": df["Expected_CTC"].median()
}
# Fill missing values using fillna()
for column, median_value in fill_values.items():
    df[column] = df[column].fillna(median_value)


#replacing null values in categorial columns
categorical_cols = [
    'Department', 'Role', 'Industry', 'Education',
    'Graduation_Specialization', 'PG_Specialization',
    'PHD_Specialization', 'Last_Appraisal_Rating']
for col in categorical_cols:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
    

#constructing heatmap
correlationCols = [
    'Total_Experience', 
    'Total_Experience_in_field_applied',
    'Current_CTC', 
    'Expected_CTC',
    'Number_of_Publications',
    'No_Of_Companies_worked',
    'Certifications',
    'International_degree_any']

correlation_matrix = df[correlationCols].corr()
fig3=plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Variables', fontsize=16, fontweight='bold')
plt.tight_layout()
figures.append(fig3)
target_correlation = df[correlationCols].corr()['Expected_CTC'].sort_values(ascending=False)

fig4=plt.figure(figsize=(10, 6))
target_correlation.drop('Expected_CTC').plot(kind='bar', color='skyblue')
plt.title('Correlation of Features with Expected CTC', fontsize=14, fontweight='bold')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.axhline(y=0, color='red', linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
figures.append(fig4)

top_correlated_features = target_correlation.drop('Expected_CTC').head(3).index.tolist()
top_correlated_features.append('Expected_CTC')

fig5=sns.pairplot(df[top_correlated_features], diag_kind='kde')
plt.suptitle('Pairplot of Most Correlated Features with Expected CTC', y=1.02)
figures.append(fig5)

# One hot encoding 
print(df.shape)

categorical_cols_for_encoding = ['Department', 'Role', 'Industry', 'Education','Graduation_Specialization', 'PG_Specialization','PHD_Specialization','Last_Appraisal_Rating']

df_encoded = pd.get_dummies(df, columns=categorical_cols_for_encoding, drop_first=True, dtype=int)
print(df_encoded.shape)
print(df_encoded.columns)

X = df_encoded.drop('Expected_CTC', axis=1)
y = df_encoded['Expected_CTC']

print(f"Features shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")
print(f"Total features available: {X.shape[1]}")

# Show some of the feature names
print(X.columns[:10])
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=42) 
print(X_train.shape)
print(X_test.shape)


rf = RandomForestRegressor(random_state=42)
lr = LinearRegression()

rf.fit(X_train, Y_train)
lr.fit(X_train, Y_train)

rf_pred = rf.predict(X_test)
lr_pred = lr.predict(X_test)

print("Random Forest:", mean_absolute_error(Y_test, rf_pred))
print("Linear Regression:", mean_absolute_error(Y_test, lr_pred))

#metrics for rf
rf_mae = mean_absolute_error(Y_test, rf_pred)
rf_mse = mean_squared_error(Y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(Y_test, rf_pred)

#metrics for lr
lr_mae = mean_absolute_error(Y_test,lr_pred)
lr_mse = mean_squared_error(Y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(Y_test,lr_pred)

print(f"MAE: {rf_mae:.4f}, MSE: {rf_mse:.4f}, RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}")
print(f"MAE: {lr_mae:.4f}, MSE: {lr_mse:.4f}, RMSE: {lr_rmse:.4f}, R²: {lr_r2:.4f}")

feature_importances = rf.feature_importances_
feature_names = X.columns

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

fig6 = plt.figure(figsize=(10, 8))
plt.barh(importance_df['feature'][:20], importance_df['importance'][:20])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 5))
figures.append(fig6)

# Random Forest residuals
fig7 = plt.subplot(1, 2, 1)
rf_residuals = Y_test - rf_pred
plt.scatter(rf_pred, rf_residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Random Forest Residual Plot')
figures.append(fig7)

# Linear Regression residuals
fig8 = plt.subplot(1, 2, 2)
lr_residuals = Y_test - lr_pred
plt.scatter(lr_pred, lr_residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Linear Regression Residual Plot')
plt.tight_layout()
figures.append(fig8)



# Actual vs Predicted values plot
fig9 = plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(Y_test, rf_pred, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest: Actual vs Predicted')
figures.append(fig9)

fig10 = plt.subplot(1, 2, 2)
plt.scatter(Y_test, lr_pred, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression: Actual vs Predicted')

plt.tight_layout()
figures.append(fig10)

# Cross-validation for more robust evaluation
from sklearn.model_selection import cross_val_score

# Perform cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_absolute_error')
rf_cv_mae = -rf_cv_scores.mean()

# Perform cross-validation for Linear Regression
lr_cv_scores = cross_val_score(lr, X, y, cv=5, scoring='neg_mean_absolute_error')
lr_cv_mae = -lr_cv_scores.mean()

print(f"\n=== Cross-Validation Results (MAE) ===")
print(f"Random Forest CV MAE: {rf_cv_mae:.4f}")
print(f"Linear Regression CV MAE: {lr_cv_mae:.4f}")

# Compare model performance
comparison_df = pd.DataFrame({
    'Model': ['Random Forest', 'Linear Regression'],
    'Test MAE': [rf_mae, lr_mae],
    'Test R²': [rf_r2, lr_r2],
    'CV MAE': [rf_cv_mae, lr_cv_mae]
})
print(comparison_df)

# Show all figures at once
plt.ioff()  # Turn off interactive mode
plt.show()