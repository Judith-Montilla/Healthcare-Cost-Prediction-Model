# Healthcare Cost Prediction Using Regression Analysis

# Objective:
# Develop a regression model to predict healthcare costs using patient demographics, health metrics, and lifestyle factors.
# The focus is on identifying key cost drivers to help optimize pricing and resource allocation for healthcare providers and insurers.

# Key Points:
# - Feature engineering with interaction terms and polynomial features.
# - Applied log transformation to healthcare costs to manage skewed data.
# - Implemented Ridge and Lasso regression with hyperparameter tuning using GridSearchCV.
# - Assessed model performance using RMSE, R², and Adjusted R².
# - All graphs will be saved as images, including the improved feature importance plot.

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Data Loading and Overview
try:
    # Load the dataset containing patient demographics, lifestyle factors, and healthcare charges
    file_path = r'C:\Users\JUDIT\Desktop\Data Sets\insurance.csv'
    df = pd.read_csv(file_path)
    print("Data loaded successfully")
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")

# Display the first 5 rows and summary statistics
print(df.head())
print(df.describe())
print(df.info())

# 2. Data Preprocessing
# Convert categorical variables to numeric (e.g., 'sex', 'smoker')
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)

# Retain the 'region' column as it is a factor influencing healthcare costs
# Convert 'region' to dummy variables for model inclusion
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# Separate features and target variable
X = df.drop(columns=['charges'])  # Features
y = df['charges']  # Target variable

# Log transformation on target variable to manage skewed data
y_log = np.log(y)

# 3. Feature Engineering: Adding Interaction Terms and Polynomial Features
# Create interaction terms and polynomial features (e.g., Age^2, BMI^2, Age * BMI)
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

# 5. Ridge and Lasso Regression with Hyperparameter Tuning
ridge = Ridge()
lasso = Lasso()

# Define parameter grid for hyperparameter tuning
param_grid = {'alpha': [0.1, 1.0, 10.0]}

# Perform cross-validation with GridSearchCV
ridge_cv = GridSearchCV(ridge, param_grid, cv=10)
lasso_cv = GridSearchCV(lasso, param_grid, cv=10)

# Fit models
ridge_cv.fit(X_train, y_train)
lasso_cv.fit(X_train, y_train)

# Predict on test data
y_pred_ridge = ridge_cv.predict(X_test)
y_pred_lasso = lasso_cv.predict(X_test)

# 6. Model Evaluation
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
ridge_r2 = r2_score(y_test, y_pred_ridge)
lasso_r2 = r2_score(y_test, y_pred_lasso)

# Output the best hyperparameters and RMSE for both models
ridge_best_alpha = ridge_cv.best_params_['alpha']
lasso_best_alpha = lasso_cv.best_params_['alpha']

print(f"Ridge RMSE: {ridge_rmse}, Ridge R²: {ridge_r2}, Best Alpha: {ridge_best_alpha}")
print(f"Lasso RMSE: {lasso_rmse}, Lasso R²: {lasso_r2}, Best Alpha: {lasso_best_alpha}")

# 7. Assumption Checking: Residual Analysis and Homoscedasticity

# Residuals vs Fitted Values Plot (Ridge)
ridge_residuals = y_test - y_pred_ridge

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_ridge, y=ridge_residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values (Ridge)')
plt.savefig('residuals_vs_fitted_ridge.png')  # Save plot as an image
plt.show()

# Q-Q Plot for Normality of Residuals (Ridge)
sm.qqplot(ridge_residuals, line='45')
plt.title('Q-Q Plot of Residuals (Ridge)')
plt.savefig('qq_plot_ridge.png')  # Save Q-Q plot as an image
plt.show()

# 8. Adjusted R² Calculation for Ridge
n = X_test.shape[0]  # number of observations
p = X_test.shape[1]  # number of predictors
ridge_adj_r2 = 1 - (1 - ridge_r2) * (n - 1) / (n - p - 1)

print(f"Adjusted R² (Ridge): {ridge_adj_r2}")

# 9. Improved Feature Importance Plot (Ridge)
coefficients = ridge_cv.best_estimator_.coef_
features = poly.get_feature_names_out()

plt.figure(figsize=(12, 8))  # Increased figure size for better visibility
sns.barplot(x=coefficients, y=features)
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.title('Feature Importance for Ridge Regression')
plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.savefig('feature_importance_ridge_improved.png')  # Save the improved plot
plt.show()

# Conclusion:
# The Ridge regression model performed well with an R² value of 0.8646 and an Adjusted R² of 0.8379. 
# The most significant predictors were interaction terms and polynomial features between age, BMI, and smoking status.
# Future improvements could involve using advanced models like XGBoost to capture non-linear relationships further.
