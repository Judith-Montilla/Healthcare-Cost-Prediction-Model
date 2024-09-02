# Healthcare-Cost-Prediction-Regression-Analysis

This repository features Python code for predicting healthcare costs using linear regression, demonstrating end-to-end data analytics skills. It covers data preprocessing, feature engineering, model development, and performance evaluation, offering practical insights into healthcare cost drivers.

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
  - Built a linear regression model.
  - Evaluated model performance using R² and MSE metrics.
  - Cross-validation used to ensure robustness.

## Key Findings

- **Model Performance:**
  - **R² (Training):** 0.74
  - **R² (Test):** 0.78
  - **MSE (Training):** 37,369,582.74
  - **MSE (Test):** 33,979,257.05

- **Significant Predictors:** 
  - Smoking status, age, and BMI are significant cost drivers.

## Business Impact

The model demonstrates that smoking status, age, and BMI are critical factors in determining healthcare costs. By understanding these drivers, healthcare providers can better allocate resources and optimize pricing strategies. Future work could include implementing this model in a real-world setting to validate its effectiveness and exploring additional features to enhance predictions.

## Future Work Recommendations

- Investigate additional features and interactions that could further improve the model's accuracy.
- Explore advanced regression techniques or machine learning models to compare performance.
- Implement and test the model in a real-world setting to assess practical impact and accuracy.

## Ethical Considerations

When deploying predictive models in healthcare, it’s crucial to ensure that predictions are used responsibly and do not perpetuate biases or inaccuracies. Regular updates and monitoring of the model’s performance...
