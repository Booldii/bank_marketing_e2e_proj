from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import joblib
import pandas as pd
import os
import sys

from macro_scraper import EurostatScraper
from data_pipeline import WinsorizerTransformer

sys.path.append(os.path.abspath('../'))

app = FastAPI(
    title="Consultant Panel API",
)
scraper = EurostatScraper()

MODEL_PATH = "../models/RF_model_v1.joblib"

try:
    model_pipeline = joblib.load(MODEL_PATH)
    print("Model loaded")
except Exception as e:
    print(f"Error during model loading. Details: {e}")
    model_pipeline = None

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

        return {
            "success": True,
            "probability": float(probability),
            "macro_used": macro_data,
            "shap_explanation": "Funkcja w przygotowaniu..."
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")



