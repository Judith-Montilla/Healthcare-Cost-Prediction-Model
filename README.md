# Healthcare-Cost-Prediction-Regression-Analysis

This repository contains Python code for predicting healthcare costs using regression models, demonstrating end-to-end data analytics skills. It includes data preprocessing, feature engineering, model development, and performance evaluation, offering practical insights into healthcare cost drivers.

## Objective

The objective of this project is to develop regression models to predict healthcare costs based on patient demographics, health metrics, and lifestyle factors. The focus is on identifying key cost drivers to help healthcare providers and insurers optimize pricing and resource allocation.

## Dataset Description

- **Source:** Kaggle - Medical Cost Personal Dataset
- **Features:**
  - **Demographics:** Age, sex
  - **Health Metrics:** BMI, smoking status
  - **Dependents:** Number of children
- **Outcome:** Medical charges (target variable)

## Methodology Overview

- **Data Cleaning and Preprocessing:**
  - Categorical variables, including 'sex' and 'smoker', were converted to numeric format.
  - Non-relevant columns were dropped, and the data was scaled for model training.
  - Numerical features were standardized to improve model performance.

- **Exploratory Data Analysis:**
  - Distributions of features were analyzed, along with their correlations to medical charges.
  - Significant predictors of healthcare costs were identified through this analysis.

- **Modeling:**
  - Linear Regression, Ridge Regression, and Lasso Regression models were developed.
  - Model performance was evaluated using R² and MSE metrics.
  - Cross-validation ensured the robustness and generalizability of the models.

## Key Findings

- **Model Performance:**
  - **Linear Regression:** R² (Training) = 0.74, R² (Test) = 0.78, MSE (Training) = 37,369,582.74, MSE (Test) = 33,979,257.05
  - **Ridge Regression:** R² (Training) = 0.74, R² (Test) = 0.78, MSE (Training) = 37,369,679.22, MSE (Test) = 33,985,434.24
  - **Lasso Regression:** R² (Training) = 0.74, R² (Test) = 0.78, MSE (Training) = 37,369,587.71, MSE (Test) = 33,980,873.97

- **Significant Predictors:** 
  - Smoking status, age, and BMI were identified as key cost drivers.

## Visualizations

The following visualizations were generated during the analysis:
  - **Feature Importance Plot:** Displays the importance of each feature in the regression models.
  - **Residuals vs. Fitted Values Plot:** Assesses the homoscedasticity and performance of the model.
  - **Q-Q Plot:** Evaluates the normality of residuals.

## Business Impact

The model shows that smoking status, age, and BMI are critical factors in determining healthcare costs. By understanding these drivers, healthcare providers can better allocate resources and optimize pricing strategies. Implementing targeted interventions such as smoking cessation programs could significantly reduce overall healthcare expenses.

## Ethical Considerations

In deploying predictive models in healthcare, it is important to comply with healthcare regulations such as **HIPAA** (Health Insurance Portability and Accountability Act) to ensure patient privacy. Models should be monitored continuously to avoid perpetuating biases or inaccuracies, ensuring fairness and ethical use in decision-making.

## Future Work Recommendations

- Further investigation of additional features and interactions to enhance model accuracy.
- Exploration of advanced regression techniques or machine learning models to improve performance.
- Implementation of the model in real-world healthcare settings to assess its practical impact and accuracy.
