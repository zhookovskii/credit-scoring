import joblib
import pandas as pd
from fastapi import FastAPI

from util.util import transform_omit_model
from api.model import Applicant
from data.make_dataset import add_ratio_features


app = FastAPI(title="Credit Scoring API")

pipeline = joblib.load("artifacts/logreg_pipeline.pkl")
explainer = joblib.load("artifacts/shap_explainer.pkl")


def predict_with_explanation(applicant_dict):
    df = pd.DataFrame([applicant_dict])
    df = add_ratio_features(df)

    pd_proba = pipeline.predict_proba(df)[0, 1]

    X = transform_omit_model(pipeline, df)
    shap_values = explainer.shap_values(X)

    feature_names = pipeline[:-1].get_feature_names_out()
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[0]
    })

    shap_df["abs_val"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_val", ascending=False).head(5)

    return pd_proba, shap_df

@app.post("/predict")
def predict(applicant: Applicant):
    pd_proba, shap_df = predict_with_explanation(applicant.dict())

    risk_level = (
        "A" if pd_proba < 0.2 else
        "B" if pd_proba < 0.5 else
        "C"
    )

    return {
        "probability_of_default": round(float(pd_proba), 4),
        "risk_level": risk_level,
        "top_risk_factors": [
            {
                "feature": row["feature"],
                "impact": round(float(row["shap_value"]), 4)
            }
            for _, row in shap_df.iterrows()
        ]
    }

