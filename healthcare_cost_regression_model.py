# Healthcare Cost Prediction Using Regression Analysis

# Objective:
# Develop regression models (Linear, Ridge, and Lasso) to predict healthcare costs using patient demographics, health metrics, and lifestyle factors.
# Focus on identifying key cost drivers, such as smoking status, BMI, and age, to help optimize pricing and resource allocation for healthcare providers and insurers.

# Key Points:
# - End-to-end analysis covering data preprocessing, model development, and evaluation.
# - Dataset: Kaggle - Medical Cost Personal Dataset.
# - Techniques: Linear Regression, Ridge Regression, Lasso Regression, and Assumption Checking.
# - Visualizations: Feature Importance Plot, Residual vs Fitted Values Plot.

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
file_path = r'C:\Users\JUDIT\Desktop\Data Sets\insurance.csv'  # Update the path as needed
try:
    df = pd.read_csv(file_path)
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    raise

# Display dataset structure and summary
print(df.head())
print(df.describe())
print(df.info())

# 2. Data Preprocessing
# Convert 'sex' to numeric: 1 for male, 0 for female
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)

# Convert 'smoker' to numeric: 1 for yes, 0 for no
df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)

# Drop 'region' as it is not needed for the analysis
df.drop(columns=['region'], inplace=True)

# Separate features and target variable
X = df.drop(columns=['charges'])
y = df['charges']

# 3. Feature Scaling
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train-Test Split
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Model Development and Evaluation
# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso()
}

# Train and evaluate each model
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Performance metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f'{name} Performance:')
    print(f'Training MSE: {train_mse:.2f}, Training R²: {train_r2:.2f}')
    print(f'Test MSE: {test_mse:.2f}, Test R²: {test_r2:.2f}')
    print('-' * 50)

# Cross-Validation for Linear Regression
cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(models['Linear Regression'], X_scaled, y, cv=cv, scoring='r2')
print(f'Linear Regression Average Cross-Validation R-Squared (R²): {cv_scores.mean():.2f}')

# 6. Feature Importance Plot for Linear Regression
# Plotting the feature importance for Linear Regression
coefficients = pd.Series(models['Linear Regression'].coef_, index=X.columns)
plt.figure(figsize=(10, 6))
coefficients.sort_values().plot(kind='barh')
plt.title('Feature Importance for Linear Regression')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
# Save the plot as an image
plt.savefig(r'C:\Users\JUDIT\Desktop\Data Sets\feature_importance_plot.png')
plt.show()

# 7. Assumption Checking

# VIF Calculation for Multicollinearity
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
print(vif_data)

# Residuals vs Fitted Values Plot
residuals = y_test - models['Linear Regression'].predict(X_test)
plt.figure(figsize=(12, 6))
sns.scatterplot(x=models['Linear Regression'].predict(X_test), y=residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
# Save the plot as an image
plt.savefig(r'C:\Users\JUDIT\Desktop\Data Sets\residuals_vs_fitted_plot.png')
plt.show()

# Q-Q Plot for Normality of Residuals
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals')
plt.savefig(r'C:\Users\JUDIT\Desktop\Data Sets\qq_plot_residuals.png')
plt.show()

# Conclusion:
# The Linear, Ridge, and Lasso Regression models demonstrated strong performance with an R² value of 0.78 on the test set.
# The test MSE of approximately 33.98 million confirms that the models explain a significant portion of the variance in healthcare costs.
# Significant predictors include smoking status, age, and BMI. Ridge and Lasso regression showed no significant improvement over linear regression.
# Future work includes exploring non-linear models and integrating additional data for better accuracy.
