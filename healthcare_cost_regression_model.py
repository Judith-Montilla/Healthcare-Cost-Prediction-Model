# Healthcare Cost Prediction Using Regression Analysis

# Objective:
# Develop a regression model to predict healthcare costs using patient demographics, health metrics, and lifestyle factors. 
# The focus is on identifying key cost drivers that help optimize pricing and resource allocation for healthcare providers and insurers.

# Key Points:
# - End-to-end analysis covering data preprocessing, model development, and evaluation.
# - Dataset: Kaggle - Medical Cost Personal Dataset
# - Techniques: Linear regression, feature engineering, and performance evaluation (R²: 0.784, MSE: 33,596,915).
# - Insights: Smoking status, BMI, and age are significant predictors of healthcare costs.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv(r'C:\Users\JUDIT\Desktop\Data Sets\insurance.csv')

# Display basic information about the dataset
print("Dataset Information:")
data.info()

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Descriptive statistics for numerical columns
print("\nDescriptive statistics:")
print(data.describe())

# Exploratory Data Analysis (EDA)
def plot_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['charges'], kde=True, color='skyblue')
    plt.title('Distribution of Medical Charges')
    plt.xlabel('Charges')
    plt.ylabel('Frequency')
    plt.savefig('distribution_of_charges.png')  # Save plot for reference
    plt.show()

plot_distribution(data)

# Checking the correlation between numerical features
def plot_correlation_matrix(data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')  # Save plot for reference
    plt.show()

plot_correlation_matrix(data)

# Visualizing the relationship between charges and key factors like smoking, BMI, and age
def plot_charges_by_smoker(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='smoker', y='charges', data=data, palette='Set3')
    plt.title('Charges by Smoking Status')
    plt.xlabel('Smoker')
    plt.ylabel('Charges')
    plt.savefig('charges_by_smoker.png')  # Save plot for reference
    plt.show()

plot_charges_by_smoker(data)

def plot_bmi_vs_charges(data):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='bmi', y='charges', hue='smoker', data=data, palette='coolwarm', s=100)
    plt.title('Charges vs BMI (Colored by Smoking Status)')
    plt.xlabel('BMI')
    plt.ylabel('Charges')
    plt.savefig('bmi_vs_charges.png')  # Save plot for reference
    plt.show()

plot_bmi_vs_charges(data)

def plot_age_vs_charges(data):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='age', y='charges', hue='smoker', data=data, palette='coolwarm', s=100)
    plt.title('Charges vs Age (Colored by Smoking Status)')
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.savefig('age_vs_charges.png')  # Save plot for reference
    plt.show()

plot_age_vs_charges(data)

# Data Preprocessing

# Function to preprocess data
def preprocess_data(data):
    categorical_features = ['sex', 'smoker', 'region']
    numeric_features = ['age', 'bmi', 'children']

    # Creating a column transformer to preprocess both categorical and numerical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])

    X = data.drop('charges', axis=1)
    y = data['charges']
    
    return X, y, preprocessor

X, y, preprocessor = preprocess_data(data)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the pipeline with preprocessing and the linear regression model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Training the model
model_pipeline.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Ensuring the metrics exactly match the case study
print(f"\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-Squared (R²): {r2:.3f}")

# Visualizing Actual vs Predicted Charges with a better regression line
def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='darkorange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title('Actual vs Predicted Charges')
    plt.xlabel('Actual Charges')
    plt.ylabel('Predicted Charges')
    plt.savefig('actual_vs_predicted_charges.png')  # Save plot for reference
    plt.show()

plot_actual_vs_predicted(y_test, y_pred)

# Extracting model coefficients for interpretation
def extract_feature_importance(model_pipeline, numeric_features):
    feature_names = numeric_features + list(model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out())
    coefficients = model_pipeline.named_steps['regressor'].coef_

    coeff_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coeff_df['Impact'] = coeff_df['Coefficient'].apply(lambda x: 'Positive' if x > 0 else 'Negative')

    # Sorting the features by absolute coefficient values for better visualization
    coeff_df = coeff_df.reindex(coeff_df['Coefficient'].abs().sort_values(ascending=False).index)
    
    return coeff_df

coeff_df = extract_feature_importance(model_pipeline, ['age', 'bmi', 'children'])

# Displaying feature importance
print("\nFeature Importance (Coefficients):")
print(coeff_df)

# Bar plot for feature importance
def plot_feature_importance(coeff_df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coeff_df, palette='viridis')
    plt.title('Feature Importance (Coefficient Values)')
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.savefig('feature_importance.png')  # Save plot for reference
    plt.show()

plot_feature_importance(coeff_df)

# Conclusion (Ensure these points are included in the case study):
# 1. The model achieved an R-squared value of approximately 0.784, demonstrating strong predictive power.
# 2. The key drivers of healthcare costs identified by the model are smoking status, BMI, and age, consistent with the written report.
# 3. Future recommendations include exploring Ridge or Lasso regression to further refine the model.
