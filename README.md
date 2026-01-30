# Credit Scoring

This repository contains a pet project demonstrating credit risk modeling using machine learning. The goal is to predict the probability of default (PD) for loan applicants and provide interpretable explanations for model decisions.


## Dataset

The project uses the **[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)** dataset. The target is:

- `TARGET`: 1 if the client defaulted on a loan, 0 otherwise.

## Features

- Numerical features (loan amounts, ratios, age, employment years, etc.)
- Categorical features (gender, family status, education, income type, housing, etc.)
- Categorical features are WoE-encoded.
- Stepwise forward selection is used for feature selection.

## Model

- **Logistic Regression** trained on selected features.
- Pipeline:
  - Imputer
  - WoE encoding
  - Standard scaling
  - Feature selection
  - Logistic regression

## Explainability

- **SHAP LinearExplainer** for feature contributions in logit space
- Gain chart for demonstrating model predictions alignment with actual risks
- ROC AUC and IV for selected features

## API

A FastAPI service is provided for:

- Predicting probability of default
- Returning top most important features for application

Example request:
```
POST /predict
{
  "AMT_INCOME_TOTAL": 120000,
  "AMT_CREDIT": 30000,
  "AMT_ANNUITY": 15000,
  "AMT_GOODS_PRICE": 28000,
  "AGE_YEARS": 35,
  "EMPLOYMENT_YEARS": 8,
  "CODE_GENDER": "M",
  "NAME_FAMILY_STATUS": "Married",
  "NAME_EDUCATION_TYPE": "Higher education",
  "NAME_INCOME_TYPE": "Working",
  "OCCUPATION_TYPE": "Managers",
  "NAME_HOUSING_TYPE": "House / apartment",
  "FLAG_OWN_CAR": "Y",
  "FLAG_OWN_REALTY": "Y"
}
```

Response includes:

- Predicted probability of default
- Top most important features

Run the service:

```bash
python -m uvicorn src.api.app:app
```

Swagger UI available at http://127.0.0.1:8000/docs
