import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="Telecom Customer Churn Predictor API",
    description="Serverless API for customer churn classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pipeline safely across relative Vercel paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

CANDIDATE_PATHS = [
    os.path.join(BASE_DIR, 'final_pipeline.joblib'),
    os.path.join(PROJECT_ROOT, 'models', 'final_pipeline.joblib'),
    os.path.join(PROJECT_ROOT, 'app', 'final_pipeline.joblib'),
]

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        for path in CANDIDATE_PATHS:
            if os.path.exists(path):
                _pipeline = joblib.load(path)
                break
        if _pipeline is None:
            raise FileNotFoundError("Trained pipeline could not be found.")
    return _pipeline

class CustomerInput(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = Field(2, ge=0, le=72)
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = Field(85.5, ge=0.0)
    TotalCharges: float = Field(171.0, ge=0.0)
    threshold: Optional[float] = Field(0.50, ge=0.10, le=0.90)

def compute_service_count(row: dict) -> int:
    services = [
        row.get('PhoneService') == 'Yes',
        row.get('MultipleLines') == 'Yes',
        row.get('InternetService') in ['DSL', 'Fiber optic'],
        row.get('OnlineSecurity') == 'Yes',
        row.get('OnlineBackup') == 'Yes',
        row.get('DeviceProtection') == 'Yes',
        row.get('TechSupport') == 'Yes',
        row.get('StreamingTV') == 'Yes',
        row.get('StreamingMovies') == 'Yes'
    ]
    return sum(int(s) for s in services)

@app.get("/")
def home():
    public_file = os.path.join(PROJECT_ROOT, 'public', 'index.html')
    if not os.path.exists(public_file):
        public_file = os.path.join(BASE_DIR, '..', 'public', 'index.html')
    if os.path.exists(public_file):
        return FileResponse(public_file)
    return {"status": "online", "message": "Telecom Customer Churn Predictor API"}

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "service": "Telecom Customer Churn Predictor",
        "model": "Logistic Regression (Final Tuned)"
    }

@app.post("/api/predict")
def predict(customer: CustomerInput):
    try:
        pipeline = get_pipeline()
        data = customer.model_dump()
        threshold = data.pop('threshold', 0.50) or 0.50

        data['SubscribedServiceCount'] = compute_service_count(data)
        df_input = pd.DataFrame([data])

        prob = float(pipeline.predict_proba(df_input)[0, 1])
        predicted_class = "Churn" if prob >= threshold else "No Churn"

        if prob >= 0.65:
            risk_label = "High Risk"
            recommendation = "[High Risk] Immediate intervention required. Proactively offer 1-year contract discount or dedicated support."
        elif prob >= 0.35:
            risk_label = "Medium Risk"
            recommendation = "[Medium Risk] Moderate attrition risk. Recommend contract upgrade incentives or bundled tech support."
        else:
            risk_label = "Low Risk"
            recommendation = "[Low Risk] Customer is stable. Good candidate for loyalty rewards or non-intrusive service additions."

        return {
            "predicted_class": predicted_class,
            "probability": round(prob, 4),
            "probability_percent": round(prob * 100, 2),
            "risk_label": risk_label,
            "threshold_used": threshold,
            "recommendation": recommendation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
