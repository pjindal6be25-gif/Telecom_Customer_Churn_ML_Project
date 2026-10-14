"""
churn_predictor.py – Streamlit Interactive Web App for Telecom Customer Churn Prediction.

Loads the exact fitted scikit-learn pipeline saved during training.
Guarantees zero inference discrepancy: raw inputs are processed through the identical pipeline.
"""

import os
import json
import streamlit as st
import pandas as pd
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

if os.path.exists(os.path.join(MODELS_DIR, 'final_pipeline.joblib')):
    PIPELINE_PATH = os.path.join(MODELS_DIR, 'final_pipeline.joblib')
    META_PATH     = os.path.join(MODELS_DIR, 'column_meta.json')
else:
    PIPELINE_PATH = os.path.join(BASE_DIR, 'final_pipeline.joblib')
    META_PATH     = os.path.join(BASE_DIR, 'column_meta.json')


@st.cache_resource
def load_model_assets():
    """Loads the unified pipeline and metadata."""
    pipeline = joblib.load(PIPELINE_PATH)
    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    return pipeline, meta


# ── Page Config & Header ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor | MST Mini-Project",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Telecom Customer Churn Predictor")
st.markdown(
    "**Undergraduate MST Mini-Project — Supervised Binary Classification**  \n"
    "This interactive tool predicts customer churn risk using the exact machine learning pipeline fitted on the IBM Telco dataset. "
    "All categorical encodings, imputations, and feature scalings are performed through the saved scikit-learn Pipeline."
)

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Lower threshold (e.g. 0.40) increases recall to catch more churners at the cost of additional false alarms."
    )
    
    st.markdown("### 🏆 Selected Model")
    st.info(
        "**Algorithm:** Logistic Regression  \n"
        "**Selection Metric:** 5-Fold CV F1 = **0.5951**  \n"
        "**Test F1 Score:** **0.6108**  \n"
        "**Test ROC-AUC:** **0.8360**  \n"
        "**Validation:** Stratified 5-Fold CV (Zero Leakage)"
    )
    
    st.markdown("### 💡 Business Recommendation")
    st.caption(
        "At threshold **0.40**, recall improves from 57.5% to 68.5%, capturing 41 additional churners for early retention outreach."
    )

# ── Customer Input Form ───────────────────────────────────────────────────────
st.subheader("📋 Enter Customer Information")

col_demo, col_serv, col_bill = st.columns(3)

with col_demo:
    st.markdown("#### 👤 Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months with company)", min_value=1, max_value=72, value=12)

with col_serv:
    st.markdown("#### 🌐 Telecom Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"] if phone == "Yes" else ["No phone service"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    
    inet_active = internet != "No"
    security = st.selectbox("Online Security", ["Yes", "No"] if inet_active else ["No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No"] if inet_active else ["No internet service"])
    device_prot = st.selectbox("Device Protection", ["Yes", "No"] if inet_active else ["No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"] if inet_active else ["No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"] if inet_active else ["No internet service"])
    streaming_mov = st.selectbox("Streaming Movies", ["Yes", "No"] if inet_active else ["No internet service"])

with col_bill:
    st.markdown("#### 💳 Contract & Billing")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    monthly = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=70.35, step=1.0)
    total = st.number_input("Total Charges ($)", min_value=15.0, max_value=10000.0, value=float(tenure * monthly), step=10.0)

# Calculate SubscribedServiceCount across 9 services
service_flags = [
    phone == "Yes",
    multi_lines == "Yes",
    internet in ["DSL", "Fiber optic"],
    security == "Yes",
    backup == "Yes",
    device_prot == "Yes",
    tech_support == "Yes",
    streaming_tv == "Yes",
    streaming_mov == "Yes"
]
subscribed_service_count = sum(int(f) for f in service_flags)

st.caption(f"ℹ️ Computed **SubscribedServiceCount**: `{subscribed_service_count} / 9` active services.")

predict_btn = st.button("🚀 Evaluate Churn Risk", type="primary", use_container_width=True)

# ── Inference & Prediction Display ────────────────────────────────────────────
if predict_btn:
    try:
        pipeline, meta = load_model_assets()
        
        customer_dict = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multi_lines,
            'InternetService': internet,
            'OnlineSecurity': security,
            'OnlineBackup': backup,
            'DeviceProtection': device_prot,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_mov,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': float(monthly),
            'TotalCharges': float(total),
            'SubscribedServiceCount': subscribed_service_count
        }
        
        df_input = pd.DataFrame([customer_dict])
        prob = float(pipeline.predict_proba(df_input)[0, 1])
        predicted_class = "Churn (Likely to Leave)" if prob >= threshold else "No Churn (Likely to Stay)"
        
        if prob >= 0.70:
            risk_badge = "🔴 High Risk"
            box_style = "error"
        elif prob >= 0.40:
            risk_badge = "🟡 Moderate Risk"
            box_style = "warning"
        else:
            risk_badge = "🟢 Low Risk"
            box_style = "success"
            
        st.divider()
        st.subheader("📊 Prediction Results")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Outcome", "Churn" if prob >= threshold else "Retain")
        m2.metric("Churn Probability", f"{prob:.1%}")
        m3.metric("Risk Category", risk_badge)
        m4.metric("Operating Threshold", f"{threshold:.2f}")
        
        st.progress(prob)
        
        if prob >= threshold:
            st.error(
                f"⚠️ **Customer Classified as CHURN (Probability: {prob:.1%})**  \n"
                f"The model score exceeds the decision threshold of {threshold:.2f}."
            )
        else:
            st.success(
                f"✅ **Customer Classified as NO CHURN (Probability: {prob:.1%})**  \n"
                f"The model score is below the decision threshold of {threshold:.2f}."
            )
            
        # Contextual insights
        st.markdown("### 🔍 Risk Factor Diagnostics")
        risk_factors = []
        protective_factors = []
        
        if contract == "Month-to-month":
            risk_factors.append("Month-to-month contract (highest empirical churn correlation).")
        else:
            protective_factors.append(f"Committed contract ({contract}) significantly reduces attrition.")
            
        if internet == "Fiber optic":
            risk_factors.append("Fiber optic subscription with high monthly fees.")
        elif internet == "No":
            protective_factors.append("No internet service (historically lowest churn category).")
            
        if tech_support == "No" and internet != "No":
            risk_factors.append("Lacks Tech Support add-on.")
        elif tech_support == "Yes":
            protective_factors.append("Subscribed to Tech Support (protective service).")
            
        if payment == "Electronic check":
            risk_factors.append("Electronic check payment method (associated with payment friction).")
        elif "automatic" in payment:
            protective_factors.append("Automated payment method enabled.")
            
        if tenure <= 12:
            risk_factors.append(f"Low tenure ({tenure} months) — within the critical first-year onboarding vulnerability window.")
        elif tenure >= 48:
            protective_factors.append(f"High tenure ({tenure} months) — established customer relationship.")
            
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Identified Risk Signals:**")
            if risk_factors:
                for rf in risk_factors:
                    st.write(f"- ⚠️ {rf}")
            else:
                st.write("- None detected.")
                
        with c_right:
            st.markdown("**Identified Protective Signals:**")
            if protective_factors:
                for pf in protective_factors:
                    st.write(f"- 🛡️ {pf}")
            else:
                st.write("- None detected.")

    except Exception as e:
        st.error(f"Inference error: {e}")
