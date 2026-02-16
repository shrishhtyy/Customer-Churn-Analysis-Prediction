# Customer Churn Analysis & Prediction

A machine learning project that predicts which telecom customers are likely to churn, using Random Forest with SMOTE for class imbalance handling.

## Problem

Customer churn is expensive. Acquiring new customers costs 5 to 7x more than retaining existing ones. This project identifies which customers are at risk of leaving and what factors drive churn

## Dataset

- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers
- **Features:** 20 (gender, tenure, contract type, payment method, monthly charges, internet service, etc.)
- **Target:** Churn (Yes / No)
- **Churn rate:** ~26.5%

## Approach

1. **Data cleaning**: Removed customerID, fixed blank TotalCharges values, converted data types
2. **EDA**: Histograms, box plots, correlation heatmap for numerical features; count plots for categorical features
3. **Preprocessing**: Label encoding for all categorical features, saved encoders as pickle
4. **Class balancing**: Applied SMOTE to oversample the minority class (churned customers)
5. **Model comparison**: Trained Decision Tree, Random Forest, and XGBoost with 5-fold cross-validation
6. **Final model**: Random Forest (highest CV accuracy), evaluated on held-out test set
7. **Prediction system**: Built a reusable prediction pipeline using saved model and encoders

## Results

| Model | CV Accuracy |
|-------|-------------|
| Decision Tree | ~82% |
| Random Forest | ~86% |
| XGBoost | ~84% |

Final Random Forest test accuracy: **~79%**

## Tech Stack

Python, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, XGBoost, imbalanced-learn (SMOTE)

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Open the notebook in Google Colab
3. Update the file path to your dataset location
4. Run all cells

