# Healthcare Cost Prediction Using Regression Analysis

# Objective:
# Develop a regression model to predict healthcare costs using patient demographics, health metrics, and lifestyle factors.
# The focus is on identifying key cost drivers to help optimize pricing and resource allocation for healthcare providers and insurers.

# Key Points:
# - End-to-end analysis covering data preprocessing, model development, and evaluation.
# - Dataset: Kaggle - Medical Cost Personal Dataset.
# - Techniques: Linear regression, feature engineering, and performance evaluation (R²: 0.78, MSE: ~33.98 million).
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
try:
    # Load the dataset containing patient demographics, lifestyle factors, and healthcare charges
    file_path = r"C:\Users\JUDIT\Desktop\Data Sets\insurance.csv"
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

# Drop 'region' column as it is not needed for this analysis
df.drop(columns=['region'], inplace=True)

# Separate features and target variable
X = df.drop(columns=['charges'])  # Features
y = df['charges']  # Target variable

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Multicollinearity Check using VIF
# Calculate Variance Inflation Factor (VIF)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
print(vif_data)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Model Development: Linear, Ridge, and Lasso Regression

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_test_pred = model.predict(X_test)
    
    # Calculate performance metrics
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results[name] = {'Test MSE': test_mse, 'Test R²': test_r2}
    
    print(f"{name} Performance:")
    print(f"Test Mean Squared Error (MSE): {test_mse:.2f}")
    print(f"Test R-Squared (R²): {test_r2:.2f}")
    print('-' * 50)

# 6. Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_results = {name: np.mean(cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')) for name, model in models.items()}
print(cv_results)

# 7. Assumption Checking: Residual Analysis and Homoscedasticity

# Residuals vs. Fitted Values Plot
model = LinearRegression()
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
residuals = y_test - y_test_pred

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.savefig('residuals_vs_fitted_values.png')
plt.show()

# Q-Q Plot for Normality of Residuals
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals')
plt.savefig('qq_plot.png')
plt.show()

# 8. Feature Importance Plot for Linear Regression
coefficients = model.coef_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=coefficients, y=features)
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.title('Feature Importance for Linear Regression')
plt.savefig('feature_importance.png')
plt.show()

# Conclusion:
# The linear regression model demonstrated strong performance with an R² value of 0.78 and MSE of 33.98 million on the test set.
# Smoking status, BMI, and age are the most significant factors driving healthcare costs. Ridge and Lasso regression did not provide significant performance improvements.
# The analysis suggests that predictive models can aid healthcare providers in cost optimization and risk-based pricing strategies.
