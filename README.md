# Healthcare Cost Prediction Using Regression Analysis

This repository contains Python code for predicting healthcare costs using linear regression, demonstrating end-to-end data analytics and modeling skills. It covers data preprocessing, feature engineering, model development, and performance evaluation, offering practical insights into healthcare cost drivers.

## Objective

Develop a regression model to predict healthcare costs based on patient demographics, health metrics, and lifestyle factors. The focus is on identifying key cost drivers to help healthcare providers and insurers optimize pricing and resource allocation.

## Dataset Description

- **Source:** Kaggle - [Medical Cost Personal Dataset](https://www.kaggle.com/mirichoi0218/insurance)
- **Features:**
  - **Demographics:** Age, sex, and region (dummy encoded for model training).
  - **Health Metrics:** BMI, smoking status.
  - **Dependents:** Number of children.
- **Outcome:** Medical charges (target variable)

## Methodology Overview

### Data Cleaning and Preprocessing:
- Converted categorical variables ('sex', 'smoker', 'region') to numeric format using one-hot encoding for the region.
- Scaled numerical features (age, BMI, children) for model training.
- Checked for multicollinearity using Variance Inflation Factor (VIF), ensuring no high correlations between independent variables.

### Exploratory Data Analysis:
- Analyzed distributions and correlations between features and medical charges.
- Identified significant predictors of healthcare costs, with a focus on lifestyle factors such as smoking and BMI.

### Modeling:
- Built and evaluated three regression models: Linear, Ridge, and Lasso.
- Used Mean Squared Error (MSE) and R² metrics to assess performance.
- Cross-validation ensured model robustness, with 10-fold cross-validation applied for additional validation.

## Model Performance

| Model              | Test MSE          | Test R²  | Cross-Validation R² |
|--------------------|-------------------|----------|---------------------|
| Linear Regression  | 33.60 million     | 0.78     | 0.739               |
| Ridge Regression   | 33.61 million     | 0.78     | 0.739               |
| Lasso Regression   | 33.60 million     | 0.78     | 0.739               |

## Significant Predictors:
- **Smoking Status**: Identified as the top cost driver, with smokers incurring over 20% higher medical expenses due to increased risks of chronic diseases.
- **BMI**: Each unit increase in BMI is associated with a significant rise in healthcare charges, indicating the economic burden of obesity-related conditions.
- **Age**: Older patients incur higher healthcare costs, highlighting the importance of preventive care for aging populations.
- **Region**: Geographic differences exist in healthcare costs, with certain regions (e.g., southeast) showing higher average charges.

## Visualizations

- **Feature Importance Plot**: Highlights the most influential features in predicting healthcare costs.
- **Residuals vs. Fitted Values Plot**: Demonstrates the even distribution of residuals, validating the model's reliability.
- **Q-Q Plot of Residuals**: Checks the normality of residuals to ensure assumptions hold.

## Business Impact

The model demonstrates how smoking status, BMI, and age are critical factors in determining healthcare costs. Insights from the model can help healthcare providers better allocate resources, target preventive care strategies, and implement risk-based pricing. By reducing inefficiencies, healthcare organizations can improve financial outcomes and patient care.

### Examples of Business Impact:
- **Risk-Based Pricing**: Insurers can adjust premiums for high-risk patients (e.g., smokers) based on higher predicted costs.
- **Preventive Care Programs**: Smoking cessation and weight management programs can be developed, leading to healthier populations and reduced overall healthcare expenditures.
- **Resource Allocation**: Predictive models can help hospitals plan resources more effectively, leading to better care delivery.

## Future Work Recommendations

- **Model Improvement**: Explore advanced models such as XGBoost or gradient boosting to further improve accuracy, especially for complex patient populations.
- **Real-Time Predictive Tools**: Implement real-time tools at the point of care to dynamically predict costs, improving resource allocation and patient management.
- **Enhanced Data Integration**: Incorporate additional health metrics such as comorbidities or genetic information for more personalized insights and refined predictions.

## Ethical Considerations

- **Data Privacy**: The dataset used is publicly available and does not include personal identifiable information (PII). All analysis complies with privacy regulations such as HIPAA.
- **Fairness**: Regular checks were performed to ensure no demographic biases were present in the predictions.
