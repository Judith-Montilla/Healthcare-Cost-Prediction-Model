# Healthcare-Cost-Prediction-Regression-Analysis
This repository features Python code for predicting healthcare costs using linear regression, demonstrating end-to-end data analytics skills. It covers data preprocessing, feature engineering, model development, and performance evaluation, offering practical insights into healthcare cost drivers.

# Objective
Develop a predictive model to estimate healthcare costs based on patient demographics, health metrics, and lifestyle factors. The goal is to identify key cost drivers to help healthcare providers and insurers optimize pricing strategies and improve resource allocation.

# Dataset Description
- Source: Kaggle - Medical Cost Personal Dataset
- Features:
  - Demographics: Age, sex, region
  - Health Metrics: BMI, smoking status
  - Dependents: Number of children
  - Outcome: Medical charges (target variable)

# Methodology Overview
1. Data Cleaning and Preprocessing:
   - Removed duplicates and handled missing data.
   - Encoded categorical variables and scaled numerical features.

2. Exploratory Data Analysis:
   - Analyzed distributions, correlations, and relationships between key features and medical charges.

3. Modeling:
   - Built a linear regression model and evaluated it using R² and MSE.

# Key Findings
- R²: 0.784
- MSE: 33,596,915
- Significant Predictors: Smoking status, BMI, and age are the top drivers of healthcare costs.

