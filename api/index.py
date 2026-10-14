import os
import json
import math
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="Telecom Customer Churn Predictor API",
    description="Ultra-lightweight Serverless API for Vercel",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

WEIGHTS_PATHS = [
    os.path.join(BASE_DIR, 'model_weights.json'),
    os.path.join(PROJECT_ROOT, 'models', 'model_weights.json'),
    os.path.join(PROJECT_ROOT, 'api', 'model_weights.json')
]

_model_meta = None

def get_model_meta():
    global _model_meta
    if _model_meta is None:
        for p in WEIGHTS_PATHS:
            if os.path.exists(p):
                with open(p, 'r') as f:
                    _model_meta = json.load(f)
                break
        if _model_meta is None:
            raise FileNotFoundError("Model weights JSON not found.")
    return _model_meta

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
        "runtime": "Vercel Serverless (Ultra-Lightweight)",
        "model": "Logistic Regression (Final Tuned)"
    }

@app.post("/api/predict")
def predict(customer: CustomerInput):
    try:
        meta = get_model_meta()
        data = customer.model_dump()
        threshold = data.pop('threshold', 0.50) or 0.50

        num_cols = meta['num_cols']
        scaler_mean = meta['scaler_mean']
        scaler_scale = meta['scaler_scale']

        # 1. Scale numerical features (x - mean) / scale
        x_num = [
            (float(data[col]) - scaler_mean[i]) / scaler_scale[i]
            for i, col in enumerate(num_cols)
        ]

        # 2. One-hot encode categorical features (drop='first')
        cat_cols = meta['cat_cols']
        categories = meta['categories']
        x_cat = []

        for i, col in enumerate(cat_cols):
            val = str(data[col])
            cats = categories[i]
            for cat in cats[1:]:
                x_cat.append(1.0 if val == cat else 0.0)

        # 3. Linear combination
        x_all = x_num + x_cat
        coefficients = meta['coefficients']
        intercept = meta['intercept']

        linear_sum = sum(a * b for a, b in zip(x_all, coefficients)) + intercept

        # 4. Sigmoid activation
        prob = 1.0 / (1.0 + math.exp(-linear_sum))
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
