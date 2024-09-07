# Healthcare Cost Prediction Using Regression Analysis

# Objective:
# The goal of this project is to develop and evaluate regression models to predict healthcare costs using patient demographics, health metrics, and lifestyle factors. 
# The focus is on identifying key cost drivers, such as smoking status, BMI, and age, to help optimize pricing and resource allocation for healthcare providers and insurers.

# Key Points:
# - This analysis covers end-to-end data preprocessing, model development, and evaluation.
# - Dataset: Kaggle - Medical Cost Personal Dataset
# - Techniques: Linear Regression, Ridge Regression, and Lasso Regression.
# - Insights: Smoking status, BMI, and age are significant predictors of healthcare costs.

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Data Loading and Overview
# The dataset is loaded to contain patient demographics, lifestyle factors, and healthcare charges.
file_path = r"C:\Users\JUDIT\Desktop\Data Sets\insurance.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    raise

# Displaying basic information about the dataset
print(df.head())  # Display the first few rows of the dataset
print(df.describe())  # Summary statistics for numerical features
print(df.info())  # Information about data types and missing values

# 2. Data Preprocessing
# Converting 'sex' to numeric: 1 for male, 0 for female
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)

# Converting 'smoker' to numeric: 1 for yes, 0 for no
df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)

# Dropping the 'region' column as it is not needed for the analysis
df.drop(columns=['region'], inplace=True)

# Separating features and target variable
X = df.drop(columns=['charges'])  # Features
y = df['charges']  # Target variable

# Scaling the features to have zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Variance Inflation Factor (VIF)
# Calculating VIF to check for multicollinearity among the features
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
print(vif_data)

# 4. Train-Test Split
# The dataset is split into training and testing sets to evaluate the model's performance.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Model Development
# Initializing and training the Linear Regression model
linear_model = LinearRegression()
ridge_model = Ridge()
lasso_model = Lasso()

# Training the models
linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

# 6. Model Evaluation
# Predicting on the training and testing sets
y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)

# Calculating performance metrics for Linear Regression
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Linear Regression Performance:')
print(f'Training Mean Squared Error (MSE): {train_mse:.2f}')
print(f'Training R-Squared (R²): {train_r2:.2f}')
print(f'Test Mean Squared Error (MSE): {test_mse:.2f}')
print(f'Test R-Squared (R²): {test_r2:.2f}')
print('-' * 50)

# Evaluating Ridge and Lasso regression models
ridge_train_pred = ridge_model.predict(X_train)
ridge_test_pred = ridge_model.predict(X_test)
lasso_train_pred = lasso_model.predict(X_train)
lasso_test_pred = lasso_model.predict(X_test)

# Ridge Regression Performance
print(f'Ridge Regression Performance:')
print(f'Training Mean Squared Error (MSE): {mean_squared_error(y_train, ridge_train_pred):.2f}')
print(f'Training R-Squared (R²): {r2_score(y_train, ridge_train_pred):.2f}')
print(f'Test Mean Squared Error (MSE): {mean_squared_error(y_test, ridge_test_pred):.2f}')
print(f'Test R-Squared (R²): {r2_score(y_test, ridge_test_pred):.2f}')
print('-' * 50)

# Lasso Regression Performance
print(f'Lasso Regression Performance:')
print(f'Training Mean Squared Error (MSE): {mean_squared_error(y_train, lasso_train_pred):.2f}')
print(f'Training R-Squared (R²): {r2_score(y_train, lasso_train_pred):.2f}')
print(f'Test Mean Squared Error (MSE): {mean_squared_error(y_test, lasso_test_pred):.2f}')
print(f'Test R-Squared (R²): {r2_score(y_test, lasso_test_pred):.2f}')
print('-' * 50)

# 7. Cross-Validation
# 10-fold cross-validation for model evaluation
cv = KFold(n_splits=10, shuffle=True, random_state=42)
linear_cv_scores = cross_val_score(linear_model, X_scaled, y, cv=cv, scoring='r2')
ridge_cv_scores = cross_val_score(ridge_model, X_scaled, y, cv=cv, scoring='r2')
lasso_cv_scores = cross_val_score(lasso_model, X_scaled, y, cv=cv, scoring='r2')

print(f'Linear Regression Average Cross-Validation R-Squared (R²): {linear_cv_scores.mean():.2f}')
print(f'Ridge Regression Average Cross-Validation R-Squared (R²): {ridge_cv_scores.mean():.2f}')
print(f'Lasso Regression Average Cross-Validation R-Squared (R²): {lasso_cv_scores.mean():.2f}')

# 8. Assumption Checking
# Residual Analysis for Homoscedasticity
residuals = y_test - y_test_pred
plt.figure(figsize=(12, 6))
sns.scatterplot(x=y_test_pred, y=residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.savefig(r'C:\Users\JUDIT\Desktop\Data Sets\residuals_vs_fitted.png')
plt.show()

# Q-Q Plot for Normality of Residuals
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals')
plt.savefig(r'C:\Users\JUDIT\Desktop\Data Sets\qqplot_residuals.png')
plt.show()

# Feature Importance Plot for Linear Regression
coefficients = linear_model.coef_
feature_names = X.columns
plt.barh(feature_names, coefficients)
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance for Linear Regression')
plt.savefig(r'C:\Users\JUDIT\Desktop\Data Sets\feature_importance_linear.png')
plt.show()

# Conclusion:
# The analysis of healthcare cost prediction demonstrated good performance for the Linear Regression model with an R² value of 0.78, and similar performance from Ridge and Lasso regressions. The analysis identified key predictors such as smoking status, BMI, and age as significant cost drivers.
# Future work could explore ensemble methods to further improve model accuracy and real-world deployment for practical impact.

# Future Work Recommendations:
# - Explore more complex models, such as ensemble methods (e.g., Random Forest, Gradient Boosting).
# - Incorporate additional features, such as patient comorbidities or treatment types, to refine cost predictions.
# - Deploy the model into real-world healthcare settings to evaluate its impact on resource allocation and cost management.
