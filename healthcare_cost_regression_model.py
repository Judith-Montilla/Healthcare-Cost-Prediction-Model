# Healthcare Cost Prediction Using Regression Analysis

# Introduction
# Objective:
# This project aims to develop and evaluate regression models to predict healthcare costs using patient demographics, 
# health metrics, and lifestyle factors. The focus is on identifying key cost drivers, such as smoking status, BMI, and age, 
# to help optimize pricing and resource allocation for healthcare providers and insurers.

# Key Points:
# - This analysis covers end-to-end data preprocessing, model development, and evaluation.
# - Dataset: Kaggle - Medical Cost Personal Dataset.
# - Techniques: Linear Regression, Ridge Regression, and Lasso Regression.
# - Performance Evaluation: Linear regression achieved an R² of 0.78 and a MSE of ~33.98 million on the test set.
#   Both Ridge and Lasso regression also performed similarly, indicating that standard linear regression is effective for this problem.
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
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Error handling: Try-except block to load data
file_path = r"C:\Users\JUDIT\Desktop\Data Sets\insurance.csv"
try:
    df = pd.read_csv(file_path)
    print("Data successfully loaded.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")

# Initial data overview
print(df.head())  # Display the first few rows of the dataset
print(df.describe())  # Summary statistics for numerical features
print(df.info())  # Information about data types and missing values

# 2. Data Preprocessing
# Convert 'sex' to numeric: 1 for male, 0 for female
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)

# Convert 'smoker' to numeric: 1 for yes, 0 for no
df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)

# Drop the 'region' column as it is not needed for the analysis
df.drop(columns=['region'], inplace=True)

# Separate features and target variable
X = df.drop(columns=['charges'])  # Features
y = df['charges']  # Target variable

# 3. Feature Scaling
# Standardize the features to have zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train-Test Split
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Multicollinearity Check using VIF (Variance Inflation Factor)
vif_data = pd.DataFrame()
vif_data['feature'] = df.drop(columns=['charges']).columns
vif_data['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
print(vif_data)

# 6. Model Development and Evaluation for Linear Regression, Ridge, and Lasso
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso()
}

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_train_pred = model.predict(X_train)  # Predictions on training set
    y_test_pred = model.predict(X_test)  # Predictions on test set

    # Calculate performance metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f'{model_name} Performance:')
    print(f'Training Mean Squared Error (MSE): {train_mse:.2f}')
    print(f'Training R-Squared (R²): {train_r2:.2f}')
    print(f'Test Mean Squared Error (MSE): {test_mse:.2f}')
    print(f'Test R-Squared (R²): {test_r2:.2f}')
    print('-' * 50)

# 7. Cross-Validation
# Perform 10-fold cross-validation to evaluate model stability
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')
    print(f'{model_name} Average Cross-Validation R-Squared (R²): {cv_scores.mean():.2f}')
    print('-' * 50)

# 8. Residual Analysis for Assumption Checking
# Residuals vs Predicted Values Plot to check homoscedasticity
residuals = y_test - y_test_pred
plt.figure(figsize=(12, 6))
sns.scatterplot(x=y_test_pred, y=residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.savefig(r"C:\Users\JUDIT\Desktop\Data Sets\residuals_vs_predicted.png")
plt.show()

# Q-Q Plot for Normality of Residuals
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals')
plt.savefig(r"C:\Users\JUDIT\Desktop\Data Sets\qq_plot.png")
plt.show()

# 9. Feature Importance for Linear Regression (since Ridge and Lasso coefficients might shrink)
# Visualizing feature importance for Linear Regression
coef = models['Linear Regression'].coef_
feature_importance = pd.Series(coef, index=df.drop(columns=['charges']).columns)
feature_importance.sort_values().plot(kind='barh')
plt.title('Feature Importance for Linear Regression')
plt.savefig(r"C:\Users\JUDIT\Desktop\Data Sets\feature_importance_linear_regression.png")
plt.show()

# Conclusion:
# The Linear Regression, Ridge Regression, and Lasso Regression models all demonstrated consistent performance, 
# achieving an R² value of 0.78 on the test set. The test Mean Squared Error (MSE) was approximately 33.98 million.
# Smoking status, BMI, and age were identified as key cost drivers, emphasizing their influence on healthcare costs.
# The similarity in performance between Linear, Ridge, and Lasso regression indicates that multicollinearity is not a significant issue, 
# which was further supported by the low Variance Inflation Factors (VIF) for the predictors.
# Future work should explore more advanced models (e.g., ensemble methods), and the model should be tested in real-world settings to assess its impact.
