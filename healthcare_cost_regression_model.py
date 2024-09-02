# Healthcare Cost Prediction Using Regression Analysis

# Objective:
# Develop a regression model to predict healthcare costs using patient demographics, health metrics, and lifestyle factors. 
# The focus is on identifying key cost drivers that help optimize pricing and resource allocation for healthcare providers and insurers.

# Key Points:
# - End-to-end analysis covering data preprocessing, model development, and evaluation.
# - Dataset: Kaggle - Medical Cost Personal Dataset
# - Techniques: Linear regression, feature engineering, and performance evaluation (R²: 0.784, MSE: 33,596,915).
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

# 1. Data Loading and Overview
# Load the dataset containing patient demographics, lifestyle factors, and healthcare charges
file_path = r"C:\Users\YourUser\Desktop\Data Sets\insurance.csv"  # Update the path as needed
df = pd.read_csv(file_path)

# Initial data overview: Understanding the structure and summary statistics of the dataset
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

# 5. Model Development
# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# 6. Model Evaluation
# Predict on the training set and testing set
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate performance metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Training Mean Squared Error (MSE): {train_mse:.2f}')
print(f'Training R-Squared (R²): {train_r2:.2f}')
print(f'Test Mean Squared Error (MSE): {test_mse:.2f}')
print(f'Test R-Squared (R²): {test_r2:.2f}')

# 7. Feature Importance
# Print the coefficients of the features
feature_names = X.columns
coefficients = model.coef_
for feature, coef in zip(feature_names, coefficients):
    print(f'Feature: {feature}, Coefficient: {coef:.2f}')

# 8. Cross-Validation
# Perform 10-fold cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')

print(f'Average Cross-Validation R-Squared (R²): {cv_scores.mean():.2f}')

# 9. Assumption Checking
# Residual Analysis
residuals = y_test - y_test_pred
plt.figure(figsize=(12, 6))
sns.scatterplot(x=y_test_pred, y=residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()

# Q-Q Plot for Normality of Residuals
sm.qqplot(residuals, line ='45')
plt.title('Q-Q Plot of Residuals')
plt.show()

# Conclusion:
# The linear regression model demonstrated good performance with an R² value of 0.78 on the test set, indicating that the model explains a significant proportion of the variance in healthcare costs. The model's ability to identify key predictors such as smoking status, age, and BMI provides valuable insights into healthcare cost drivers. The next steps involve exploring more complex models to potentially improve prediction accuracy and conducting further analysis to understand the impact of each feature in real-world scenarios.

# Future Work Recommendations:
# - Explore additional features and interactions to enhance model performance.
# - Test advanced regression techniques or ensemble methods to improve accuracy.
# - Validate the model on different datasets to ensure robustness and generalizability.
# - Consider deploying the model in a real-world healthcare setting to assess its impact on cost predictions and resource allocation.
