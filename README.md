# Healthcare-Cost-Prediction-Regression-Analysis

This repository contains Python code for predicting healthcare costs using linear regression, demonstrating end-to-end data analytics and modeling skills. It covers data preprocessing, feature engineering, model development, and performance evaluation, offering practical insights into healthcare cost drivers.

## Objective

Develop a regression model to predict healthcare costs based on patient demographics, health metrics, and lifestyle factors. The focus is on identifying key cost drivers to help healthcare providers and insurers optimize pricing and resource allocation.

## Dataset Description

- **Source:** Kaggle - Medical Cost Personal Dataset
- **Features:**
  - **Demographics:** Age, sex
  - **Health Metrics:** BMI, smoking status
  - **Dependents:** Number of children
- **Outcome:** Medical charges (target variable)

## Methodology Overview

- **Data Cleaning and Preprocessing:**
  - Converted categorical variables ('sex' and 'smoker') to numeric format.
  - Dropped non-relevant columns and handled missing data.
  - Scaled numerical features for model training.

- **Exploratory Data Analysis:**
  - Analyzed distributions and correlations between features and medical charges.
  - Identified significant predictors of healthcare costs.

- **Modeling:**
  - Built and evaluated Linear, Ridge, and Lasso regression models.
  - Used Mean Squared Error (MSE) and R² metrics to assess performance.
  - Cross-validation ensured model robustness.

## Key Findings

- **Model Performance:**
  - **Linear Regression:**
    - **Test MSE:** 33.98 million
    - **R² (Test):** 0.78
  - **Ridge Regression:** Similar performance to Linear Regression.
  - **Lasso Regression:** Similar performance to Linear Regression.

- **Significant Predictors:**
  - Smoking status, BMI, and age are the top features influencing healthcare costs.

## Visualizations

- **Feature Importance Plot:** Highlights the most influential features in predicting healthcare costs.
- **Residuals vs. Fitted Values Plot:** Demonstrates the even distribution of residuals, validating the model's reliability.
- **Q-Q Plot of Residuals:** Checks the normality of residuals to ensure assumptions hold.

## Business Impact

The model demonstrates how smoking status, BMI, and age are critical factors in determining healthcare costs. Insights from the model can help healthcare providers better allocate resources, target preventive care strategies, and implement risk-based pricing. By reducing inefficiencies, healthcare organizations can improve financial outcomes and patient care.

## Future Work Recommendations

- Investigate additional features and interactions that could further improve the model's accuracy.
- Explore advanced regression techniques or machine learning models to compare performance.
- Implement and test the model in a real-world healthcare setting to assess its practical impact and accuracy.

## Ethical Considerations

- **Healthcare Regulations:** Compliant with patient data privacy regulations, including HIPAA.
- **Fairness:** Continuous monitoring to ensure no demographic biases are present in the predictions.
