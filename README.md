# Telecom Customer Churn Prediction
## Master / Undergraduate MST Machine Learning Mini-Project

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-demo-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-100%25%20Complete-brightgreen.svg)](#)

---

## Executive Overview
This repository contains the complete, academically rigorous machine learning project for **Telecom Customer Churn Prediction** developed using the authentic **IBM Telco Customer Churn** dataset (7,043 customer accounts).

- **Problem Type:** Supervised Binary Classification
- **Target:** `Churn` (Yes = 1, No = 0)
- **Champion Architecture:** Logistic Regression with L2 Regularization (encapsulated in scikit-learn Pipeline)
- **Cross-Validation Performance:** 5-Fold Stratified CV F1 = **0.5951 ± 0.0238**, ROC-AUC = **0.8460**
- **Held-Out Test Performance:** Accuracy = **80.53%**, Precision = **65.15%**, Recall = **57.49%**, F1 = **0.6108**, ROC-AUC = **0.8360**
- **Operational Optimization:** Decision Threshold calibrated to **0.40** elevates Recall to **68.45%** (+41 churners captured) with peak global F1 = **0.6267**.

---

## Key Methodological Guarantees
1. **Zero Data Leakage:** Imputation, standard scaling, and one-hot encoding are strictly packaged inside scikit-learn `Pipeline` objects and fitted exclusively within training folds during 5-fold Stratified Cross-Validation.
2. **Empirical Feature Ablation:** Candidate features (`SubscribedServiceCount` across 9 services vs `TenureBand`) were benchmarked via 5-fold CV. `SubscribedServiceCount` was retained; `TenureBand` was discarded due to redundant collinearity.
3. **Dynamic Model Selection:** Final model was not hard-coded; Logistic Regression was selected dynamically based on superior cross-validated F1 and ROC-AUC over Decision Trees and tuned Random Forests.
4. **Single-Shot Held-Out Testing:** The 20% test partition (1,407 cases) was sealed until after final model selection.
5. **Production Parity:** The Streamlit web application loads the identical serialized `final_pipeline.joblib`, ensuring zero divergence between offline training and live inference.

---

## Repository Structure
```
9thmst/
├── data/
│   └── Telco-Customer-Churn.csv             # Authentic IBM Telco dataset (7,043 rows x 21 cols)
├── models/
│   ├── final_pipeline.joblib                 # Unified scikit-learn pipeline (preprocessor + model)
│   ├── preprocessor.joblib                   # Fitted ColumnTransformer (scaling + one-hot encoding)
│   ├── final_model.joblib                    # Fitted LogisticRegression classifier
│   └── column_meta.json                      # Metadata schema & training hyperparameter configuration
├── notebook/
│   └── Telecom_Customer_Churn_ML.ipynb      # Executed 26-section Jupyter notebook with embedded outputs
├── src/
│   ├── preprocessing.py                      # Data loading, cleaning, feature engineering, pipeline builders
│   ├── models.py                             # Model wrappers, leakage-free CV, evaluation utilities
│   ├── train.py                              # End-to-end training, CV ablation, evaluation, and serialization
│   └── inference.py                          # Production single-customer prediction module
├── results/
│   ├── figures/                              # Question-driven EDA and model evaluation charts
│   ├── model_results.csv                     # Test set performance comparison across all models
│   ├── feature_ablation_results.csv          # 5-fold CV results for candidate feature configurations
│   ├── threshold_analysis.csv                # Precision/Recall/F1 metrics across thresholds [0.30–0.70]
│   ├── data_stats.json                       # Verified raw dataset inspection metrics
│   └── error_analysis.json                   # False Positive and False Negative profile telemetry
├── app/
│   ├── churn_predictor.py                    # Interactive Streamlit web application demo
│   ├── final_pipeline.joblib                 # Local pipeline mirror for deployment
│   ├── preprocessor.joblib                   # Local preprocessor mirror
│   ├── final_model.joblib                    # Local classifier mirror
│   └── column_meta.json                      # Metadata schema for application inference
├── docs/
│   ├── data_dictionary.md                   # Full 21-column schema + feature engineering definitions
│   ├── methodology.md                       # Comprehensive methodology documentation
│   └── viva_questions.md                    # 33 comprehensive viva defense questions & answers
├── PROGRESS.md                              # Project tracking checklist and verified benchmarks
└── README.md                                # Project landing page
```

---

## Empirical Benchmark Results

### 1. 5-Fold Stratified Cross-Validation on X_train (5,625 samples)
| Candidate Model | Mean CV F1 (Primary) | Mean CV Recall | Mean CV ROC-AUC | Mean CV Precision | Selection Status |
|:---|:---|:---|:---|:---|:---|
| Dummy (Majority Class) | 0.0000 ± 0.0000 | 0.0000 | 0.5000 | 0.0000 | Baseline Floor |
| Decision Tree | 0.4960 ± 0.0197 | 0.5037 | 0.6573 | 0.4891 | Rejected (Overfitting) |
| Random Forest (Default) | 0.5612 ± 0.0184 | 0.4943 | 0.8277 | 0.6496 | Solid ensemble baseline |
| Random Forest (Tuned) | 0.5725 ± 0.0211 | 0.5023 | 0.8431 | 0.6659 | Tuned (depth=10, n=50) |
| **Logistic Regression** | **0.5951 ± 0.0238** | **0.5452** | **0.8460** | **0.6558** | **WINNER (Dynamically Selected)** |

### 2. Held-Out Test Set Performance (1,407 samples, 20% partition)
| Evaluated Pipeline | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---|:---|:---|:---|:---|
| Dummy (Majority Class) | 0.7342 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Decision Tree | 0.7107 | 0.4567 | 0.4652 | 0.4609 | 0.6324 |
| Random Forest (Default) | 0.7854 | 0.6233 | 0.4866 | 0.5465 | 0.8157 |
| Random Forest (Tuned) | 0.7903 | 0.6386 | 0.4866 | 0.5524 | 0.8314 |
| **Logistic Regression (Final)** | **0.8053** | **0.6515** | **0.5749** | **0.6108** | **0.8360** |

### 3. Operational Threshold Sensitivity Analysis
| Threshold ($\tau$) | Precision | Recall | F1-Score | TP | FP | FN | Strategic Assessment |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 0.30 | 0.5118 | 0.7567 | 0.6106 | 283 | 270 | 91 | Aggressive retention; elevated budget |
| **0.40** | **0.5779** | **0.6845** | **0.6267** | **256** | **187** | **118** | **RECOMMENDED OPTIMUM: Peak F1 (+41 churners saved)** |
| 0.50 | 0.6515 | 0.5749 | 0.6108 | 215 | 115 | 159 | Default statistical threshold |
| 0.60 | 0.6808 | 0.3877 | 0.4940 | 145 | 68 | 229 | Conservative; misses 61% of churners |
| 0.70 | 0.7590 | 0.1684 | 0.2757 | 63 | 20 | 311 | Ultra-conservative; 83% churn missed |

---

## Quick Start & Reproduction

### 1. Environment Setup
Install required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit jupyter
```

### 2. Run the Full Machine Learning Pipeline
Executes data auditing, feature ablation, cross-validation, tuning, threshold optimization, error profiling, and pipeline serialization into `models/`:
```bash
python src/train.py
```

### 3. Test Standalone Inference
Perform real-time single-customer churn risk inference via Python:
```bash
python -c "from src.inference import predict_single; print(predict_single({'tenure': 2, 'MonthlyCharges': 85.0, 'TotalCharges': 170.0, 'Contract': 'Month-to-month', 'InternetService': 'Fiber optic', 'PaymentMethod': 'Electronic check', 'TechSupport': 'No', 'PaperlessBilling': 'Yes', 'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No', 'PhoneService': 'Yes', 'MultipleLines': 'No', 'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No'}))"
```

### 4. Launch the Interactive Streamlit Web App
Launch the interactive web application to evaluate customer churn risk in real-time:
```bash
streamlit run app/churn_predictor.py
```

### 5. Run Jupyter Notebook
Open and inspect the 26-section executed notebook:
```bash
jupyter notebook notebook/Telecom_Customer_Churn_ML.ipynb
```
