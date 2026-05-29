from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
import joblib
import pandas as pd
import os
import sys
import shap

from backend.macro_scraper import EurostatScraper
from backend.data_pipeline import WinsorizerTransformer

DB_URL = "postgresql://bank_admin:supersecret@db:5432/crm_database"

sys.path.append(os.path.abspath('../'))

app = FastAPI(
    title="Consultant Panel API",
)
scraper = EurostatScraper()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "RF_model_v2.joblib")

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

    macro_data: dict

class FeedbackRequest(BaseModel):
    client_id: int
    result: str

@app.get("/clients")
def get_clients_list():
    try:
        conn = psycopg2.connect(DB_URL)
        df = pd.read_sql_query("SELECT * FROM clients WHERE contact_status = 'to_call'", conn)
        conn.close()
        return {"clients": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error on PostgreSQL: {e}")

@app.post("/predict")
def predict_conversion(request: PredictRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Not installed model.")

    macro_data = request.macro_data
    client_dict = request.dict(exclude={'year', 'macro_data'})
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
        hidden_macro = ['euribor3m', 'cons.price.idx', 'cons.conf.idx']

        raw_input_data = request.dict()

        for feat_name, shap_val in raw_shap_dict.items():
            clean_name = feat_name.split('__')[-1]

            if clean_name in hidden_macro:
                continue

            display_value = ""
            base_feature = clean_name.split('_')[0] if '_' in clean_name else clean_name

            if clean_name in raw_input_data:  # Dla numerycznych
                display_value = str(raw_input_data[clean_name])
            else:  # Dla kategorycznych OHE (szukamy w nazwie)
                display_value = clean_name.split('_')[-1] if '_' in clean_name else "Tak"

            rounded_val = round(float(shap_val), 4)

            item = {
                "feature": clean_name.replace('_', ' ').title(),
                "value": display_value,
                "impact": rounded_val
            }

            if rounded_val > 0.001:
                positive_impact.append(item)
            elif rounded_val < -0.001:
                negative_impact.append(item)

        positive_impact = sorted(positive_impact, key=lambda x: x['impact'], reverse=True)[:5]
        negative_impact = sorted(negative_impact, key=lambda x: x['impact'])[:5]

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

    if feedback.result not in ["Success", "Failure"]:
        raise HTTPException(status_code=400, detail="Nieprawidłowy status. Dopuszczalne: 'Success' lub 'Failure'.")

    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE clients SET contact_status = %s WHERE client_id = %s",
            (feedback.result, feedback.client_id)
        )
        conn.commit()
        updated_rows = cursor.rowcount
        conn.close()

        if updated_rows == 0:
            raise HTTPException(status_code=404, detail="Brak klienta o podanym ID w bazie.")

        return {"success": True, "message": f"Zapisano status '{feedback.result}' dla klienta {feedback.client_id}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd zapisu feedbacku w PostgreSQL: {e}")

@app.get("/macro")
def get_macro_indicators(year: int, month: str):
    try:
        macro_data = scraper.get_all_macro(year, month)
        return {"success": True, "macro": macro_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd pobierania danych Eurostat: {e}")



