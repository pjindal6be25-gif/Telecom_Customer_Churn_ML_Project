import os
import math
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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

# Embedded exact model parameters (2.5 KB) - eliminates all file-path dependency
MODEL_META = {
    "num_cols": ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
    "scaler_mean": [0.16177777777777777, 32.562311111111114, 64.99931555555555, 2301.83952],
    "scaler_scale": [0.3682468308011104, 24.5402396436929, 30.105965373777526, 2275.383801039277],
    "cat_cols": [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ],
    "categories": [
        ["Female", "Male"],
        ["No", "Yes"],
        ["No", "Yes"],
        ["No", "Yes"],
        ["No", "No phone service", "Yes"],
        ["DSL", "Fiber optic", "No"],
        ["No", "No internet service", "Yes"],
        ["No", "No internet service", "Yes"],
        ["No", "No internet service", "Yes"],
        ["No", "No internet service", "Yes"],
        ["No", "No internet service", "Yes"],
        ["No", "No internet service", "Yes"],
        ["Month-to-month", "One year", "Two year"],
        ["No", "Yes"],
        ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]
    ],
    "coefficients": [
        0.07215879174068877, -1.357404055698112, -0.434496641209947, 0.6455077647854222,
        -0.024081440573091813, -0.025124473931883138, -0.23217634762315115, -0.5153368988792981,
        -0.16862708215473482, 0.36278360083384553, 1.1074072422470491, -0.16635916362502223,
        -0.16635916362502223, -0.37318741975078845, -0.16635916362502223, -0.1291605451462777,
        -0.16635916362502223, 0.07426431179724738, -0.16635916362502223, -0.3315783914099988,
        -0.16635916362502223, 0.3724070879929944, -0.16635916362502223, 0.3446492392656797,
        -0.7500681040497957, -1.3648230112394717, 0.2909313589563207, 0.035829136915329955,
        0.379150080601204, 0.07269953384817208
    ],
    "intercept": -1.4270949197916984
}

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

def calculate_prediction(data: dict, threshold: float = 0.50) -> dict:
    num_cols = MODEL_META['num_cols']
    scaler_mean = MODEL_META['scaler_mean']
    scaler_scale = MODEL_META['scaler_scale']

    # 1. Scale numerical features
    x_num = [
        (float(data.get(col, 0.0)) - scaler_mean[i]) / scaler_scale[i]
        for i, col in enumerate(num_cols)
    ]

    # 2. One-hot encode categorical features (drop='first')
    cat_cols = MODEL_META['cat_cols']
    categories = MODEL_META['categories']
    x_cat = []

    for i, col in enumerate(cat_cols):
        val = str(data.get(col, ""))
        cats = categories[i]
        for cat in cats[1:]:
            x_cat.append(1.0 if val == cat else 0.0)

    # 3. Linear combination
    x_all = x_num + x_cat
    coefficients = MODEL_META['coefficients']
    intercept = MODEL_META['intercept']

    linear_sum = sum(a * b for a, b in zip(x_all, coefficients)) + intercept

    # 4. Sigmoid activation
    prob = 1.0 / (1.0 + math.exp(-linear_sum))
    is_churn = prob >= threshold
    predicted_class = "Churn" if is_churn else "No Churn"

    if is_churn:
        if prob >= threshold + 0.15:
            risk_label = "High Risk (Churn)"
            recommendation = f"[High Churn Risk] Probability ({prob*100:.1f}%) significantly exceeds your operating threshold ({threshold*100:.0f}%). Immediate proactive outreach recommended."
        else:
            risk_label = "Moderate Risk (Churn)"
            recommendation = f"[Moderate Churn Risk] Probability ({prob*100:.1f}%) is above your operating threshold ({threshold*100:.0f}%). Offer contract upgrade or service credits."
    else:
        if prob >= threshold - 0.10:
            risk_label = "Borderline (Watchlist)"
            recommendation = f"[Borderline Customer] Probability ({prob*100:.1f}%) is just below your threshold ({threshold*100:.0f}%). Monitor closely for sudden usage drops."
        else:
            risk_label = "Low Risk (Retained)"
            recommendation = f"[Low Churn Risk] Probability ({prob*100:.1f}%) is safely below your threshold ({threshold*100:.0f}%). Customer relationship is healthy."

    return {
        "predicted_class": predicted_class,
        "probability": round(prob, 4),
        "probability_percent": round(prob * 100, 2),
        "risk_label": risk_label,
        "threshold_used": threshold,
        "recommendation": recommendation
    }

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

@app.get("/")
def home():
    public_file = os.path.join(PROJECT_ROOT, 'public', 'index.html')
    if os.path.exists(public_file):
        return FileResponse(public_file)
    return {"status": "online", "message": "Telecom Customer Churn Predictor API"}

@app.api_route("/api/health", methods=["GET", "POST"])
@app.api_route("/health", methods=["GET", "POST"])
def health():
    return {
        "status": "online",
        "service": "Telecom Customer Churn Predictor",
        "runtime": "Vercel Serverless (Embedded Model)",
        "model": "Logistic Regression (Final Tuned)"
    }

@app.api_route("/api/predict", methods=["POST", "OPTIONS"])
@app.api_route("/predict", methods=["POST", "OPTIONS"])
@app.api_route("/api/index.py", methods=["POST", "OPTIONS"])
@app.api_route("/api", methods=["POST", "OPTIONS"])
def predict_endpoint(customer: CustomerInput):
    data = customer.model_dump()
    threshold = data.pop('threshold', 0.50) or 0.50
    return calculate_prediction(data, threshold)
