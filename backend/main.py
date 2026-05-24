from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import joblib
import pandas as pd
import os
import sys
import shap

from macro_scraper import EurostatScraper
from data_pipeline import WinsorizerTransformer

sys.path.append(os.path.abspath('../'))

app = FastAPI(
    title="Consultant Panel API",
)
scraper = EurostatScraper()

MODEL_PATH = "../models/RF_model_v1.joblib"

model_pipeline = None
preprocessor = None
model = None
explainer = None

try:
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        preprocessor = model_pipeline.named_steps['preprocessor']
        model = model_pipeline.named_steps['model']
        explainer = shap.TreeExplainer(model)
        print("Model and SHAP Explainder loaded")
    else:
        print(f"Model hasn't been loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error during model loading. Details: {e}")

class PredictRequest(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    housing: str
    loan: str
    contact: str
    campaign: int
    previous: int
    poutcome: str

    # consultant entries:
    month: str
    day_of_week: str
    year: int

class FeedbackRequest(BaseModel):
    client_id: int
    result: str

@app.get("/clients")
def get_clients_list():
    try:
        conn = sqlite3.connect("clients.db")
        df = pd.read_sql_query("SELECT * FROM clients WHERE contact_status = 'to_call'", conn)
        conn.close()
        return {"clients": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.post("/predict")
def predict_conversion(request: PredictRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Not installed model.")

    macro_data = scraper.get_all_macro(request.year, request.month)
    client_dict = request.dict(exclude={'year'})
    combined_features = {**client_dict, **macro_data}
    X_new = pd.DataFrame([combined_features])

    try:
        probability = model_pipeline.predict_proba(X_new)[0, 1]
        X_transformed = preprocessor.transform(X_new)

        feature_names = preprocessor.get_feature_names_out()
        shap_values = explainer.shap_values(X_transformed)

        if isinstance(shap_values, list):
            row_shap = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            row_shap = shap_values[0, :, 1]
        else:
            row_shap = shap_values[0]

        raw_shap_dict = dict(zip(feature_names, row_shap))

        positive_impact = []
        negative_impact = []

        for feat_name, shap_val in raw_shap_dict.items():
            clean_name = feat_name.split('__')[-1]
            rounded_val = round(float(shap_val), 4)

            if rounded_val > 0.0001:
                positive_impact.append({"feature": clean_name, "impact": rounded_val})
            elif rounded_val < -0.0001:
                negative_impact.append({"feature": clean_name, "impact": rounded_val})

        positive_impact = sorted(positive_impact, key=lambda x: x['impact'], reverse=True)[:3]
        negative_impact = sorted(negative_impact, key=lambda x: x['impact'])[:3]

        return {
            "success": True,
            "probability": float(probability),
            "macro_used": macro_data,
            "shap_explanation": {
                "positive": positive_impact,
                "negative": negative_impact
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

@app.post("/feedback")
def update_client_status(feedback: FeedbackRequest):

    valid_results = ["Success", "Failure"]
    if feedback.result not in valid_results:
        raise HTTPException(status_code=400, detail=f"Result error: {feedback.result}")

    try:
        conn = sqlite3.connect("clients.db")
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE clients SET contact_status = ? WHERE client_id = ?",
            (feedback.result, feedback.client_id)
        )
        conn.commit()

        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Database error: {feedback.result}")
        conn.close()

        return {"success": True, "message": f"Zapisano status '{feedback.result}' dla klienta {feedback.client_id}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")



