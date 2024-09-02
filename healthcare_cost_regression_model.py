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
X = df.drop(columns=['charges'])  # Features: 'age', 'sex', 'bmi', 'children', 'smoker'
y = df['charges']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Development
# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Model Evaluation
# Calculate Mean Squared Error (MSE) and R-Squared (R²) for test data
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Test Mean Squared Error (MSE): {mse:,.2f}')
print(f'Test R-Squared (R²): {r2:.2f}')

# Feature Importance (Coefficients of the Linear Regression Model)
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Cross-Validation Results
# Perform 10-fold cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

print(f'Average Cross-Validation R-Squared (R²): {cv_scores.mean():.2f}')

# Assumption Checking
# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.show()

# Normality of Residuals
fig = plt.figure(figsize=(10, 6))
sm.qqplot(residuals, line ='45')
plt.title('Q-Q Plot of Residuals')
plt.show()

# 4. Conclusion
# The linear regression model demonstrated a test R² of 0.78, indicating that approximately 78% of the variance in healthcare costs is explained by the model.
# The Mean Squared Error (MSE) was 33,979,257.05, reflecting the average squared difference between predicted and actual charges.
# Key predictors identified include smoking status, age, and BMI, which are crucial for understanding and optimizing healthcare costs.
# Future work could involve exploring regularization techniques (Ridge and Lasso) to handle potential multicollinearity and improve model performance.
