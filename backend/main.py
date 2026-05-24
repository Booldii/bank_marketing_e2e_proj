from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Consultant Panel API",
)

class ClassFeatures(BaseModel):
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

    # concultrant entries:
    month: str
    day_of_week: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "wszystko zajebiscie"}

@app.get("/clients")
def get_clients_list():
    return {"clients": [{"client_id": 1, "age": 35, "job": "admin."}]}

@app.post("/predict")
def predict_conversion(client: ClassFeatures):

    mock_proba = 0.75
    return {
        "success": True,
        "probability": mock_proba,
        "shap_explanation": {
            "top_positive": ["Wiek (35)", "Brak kredytu hipotecznego"],
            "top_negative": ["Mało udana poprzednia kampania"]
        }
    }

