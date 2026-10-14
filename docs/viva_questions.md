# Viva Questions & Comprehensive Answers
## Telecom Customer Churn Prediction – MST Mini-Project

---

### Section 1: Problem Understanding & Objective

**Q1. What is the business problem statement?**  
In the telecommunications sector, acquiring a new subscriber costs 5 to 7 times more than retaining an existing account. The business problem is identifying, in advance, subscribers who exhibit high probabilities of terminating their services (churning) so that proactive, targeted retention offers can be deployed before they formally cancel.

**Q2. What is the technical objective of the project?**  
To formulate, train, evaluate, and interpret a supervised binary classification machine learning pipeline that maps customer demographics, service subscriptions, and billing patterns to:
1. A binary predicted class: Churn (1) vs. No Churn (0)
2. A calibrated continuous probability: $P(Y=1|X) \in [0, 1]$

**Q3. What is the target variable and how is it distributed?**  
The target variable is `Churn` ('Yes' / 'No'), encoded as 1 ('Yes') and 0 ('No'). In the cleaned dataset of 7,032 records:
- **No Churn (0):** 5,163 customers (73.42%)
- **Churn (1):** 1,869 customers (26.58%)
This represents a moderate class imbalance (~2.77:1).

**Q4. Why is this a classification problem and not regression or time-series?**  
The outcome is a categorical status (churned vs. retained). Regression models continuous real-valued targets. Furthermore, the dataset is a static cross-sectional historical snapshot rather than longitudinal time-series telemetry. We do not predict continuous-time hazard survival or claim temporal causality.

**Q5. What is the difference between reactive and proactive retention?**  
- **Reactive retention:** Intervening only after a customer contacts customer service to terminate their contract. By this point, customer dissatisfaction is cemented, resulting in poor save rates (<15%).
- **Proactive retention:** Utilizing ML risk scores to identify at-risk customers months prior to contract expiration, enabling low-cost interventions (discounts, service upgrades) before the customer decides to exit.

---

### Section 2: Dataset & Inspection

**Q6. What dataset was used and where does it originate?**  
The original IBM Telco Customer Churn dataset (7,043 rows, 21 columns). It contains customer demographic details, telecom service subscriptions, contract types, and billing history.

**Q7. What data quality anomalies were discovered during initial inspection?**  
1. `TotalCharges` was stored as an `object` (string) column rather than numeric.
2. Exactly 11 rows contained whitespace strings (`" "`).
3. Cross-referencing revealed that all 11 records had `tenure = 0` (brand-new accounts enrolled in month 0 who had never completed a monthly billing cycle).
4. No other duplicate rows or duplicate `customerID` entries existed.

**Q8. Why were the 11 rows dropped instead of imputed?**  
Because `tenure = 0`, these customers had truly incurred zero monthly billing cycles. Imputing lifetime charges would fabricate non-existent billing history. Dropping 11 rows out of 7,043 represents less than 0.16% of the dataset (< 0.2%), introducing zero statistical distortion and retaining 7,032 complete records.

**Q9. Why was `customerID` excluded from the feature matrix?**  
`customerID` is an arbitrary unique string identifier with cardinality equal to sample size ($N$). It carries zero generalizable predictive signal and would invite spurious memorization and overfitting.

---

### Section 3: Preprocessing & Leakage Elimination

**Q10. What is data leakage in cross-validation and why is it dangerous?**  
Data leakage occurs when information from outside the training partition (such as validation or test set statistics) influences the model during training. In standard ML coursework, students often apply `StandardScaler` or `OneHotEncoder` across the entire training set (or full dataset) before calling `cross_val_score`. This leaks validation fold means, variances, and category distributions into the training process, artificially inflating CV performance.

**Q11. How does your architecture guarantee zero data leakage?**  
All feature transformations are encapsulated inside a scikit-learn `Pipeline([('preprocessor', ColumnTransformer(...)), ('classifier', estimator)])`. During 5-fold cross-validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), the pipeline fits transformers strictly on the 4 fold-training partitions (80%) and independently transforms the 5th validation partition (20%). The held-out 20% test set was sealed until after final model selection.

**Q12. Why was `OneHotEncoder(drop='first')` utilized?**  
Dropping the first category prevents the "dummy variable trap" (perfect multicollinearity). For a categorical variable with $K$ states, $K-1$ binary dummies completely capture the information space. In linear models like Logistic Regression, collinear columns produce singular covariance matrices and erratic coefficient estimates.

**Q13. Why is standard scaling required for Logistic Regression?**  
Logistic Regression utilizes gradient-based optimization (L-BFGS). Features on disparate scales (e.g., `tenure` in [0, 72] vs `MonthlyCharges` in [18, 118] vs binary indicators in [0, 1]) cause elongated elliptical loss contours, causing gradient oscillations and slow convergence. Standardization ($\mu = 0, \sigma = 1$) ensures isotropic gradient descent.

---

### Section 4: Feature Engineering & CV Ablation

**Q14. How was `SubscribedServiceCount` engineered and why was it corrected?**  
It aggregates active customer subscriptions across 9 telecom services: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, and `StreamingMovies`.  
*Critical Correction:* Raw `InternetService` values are `'DSL'`, `'Fiber optic'`, and `'No'` (never `'Yes'`). Previous scripts that checked `x == 'Yes'` completely missed internet subscriptions. Our logic explicitly checks `InternetService.isin(['DSL', 'Fiber optic'])`.

**Q15. What is feature ablation and what did your CV ablation study reveal?**  
Feature ablation evaluates model performance across candidate feature subsets via 5-Fold Stratified CV on `X_train`:
- *Baseline (Raw Features):* LR CV F1 = 0.5964 ± 0.0237 | RF CV F1 = 0.5573 ± 0.0201
- *With SubscribedServiceCount:* LR CV F1 = 0.5951 ± 0.0238 | RF CV F1 = 0.5612 ± 0.0184
- *With TenureBand Only:* LR CV F1 = 0.5906 ± 0.0135 | RF CV F1 = 0.5585 ± 0.0225
- *With Both:* LR CV F1 = 0.5887 ± 0.0119 | RF CV F1 = 0.5559 ± 0.0175

**Q16. Why was `TenureBand` discarded?**  
Discretizing continuous `tenure` into 6 arbitrary categorical brackets added 5 dummy features that were collinear with continuous `tenure`. The cross-validation evidence showed that `TenureBand` degraded Logistic Regression CV F1 (from 0.5951 down to 0.5906). Adhering to the principle of parsimony, it was discarded. `SubscribedServiceCount` was retained as it improved Random Forest performance and provides direct ecosystem stickiness information.

---

### Section 5: Machine Learning Models & Mechanics

**Q17. What four model families were evaluated and why?**  
1. **Dummy Classifier:** Majority class baseline floor.
2. **Logistic Regression:** Linear parametric benchmark with well-calibrated probabilities and direct coefficient interpretability.
3. **Decision Tree Classifier:** Non-parametric single-tree baseline to capture non-linear interactions and illustrate overfitting dynamics.
4. **Random Forest Classifier:** Bagged ensemble reducing single-tree variance through bootstrap sampling and feature subspace randomization.

**Q18. Explain the mathematical step-by-step mechanics of Logistic Regression.**  
1. *Linear Score:* Compute weighted sum $z = \mathbf{w}^T \mathbf{x} + b$.
2. *Sigmoid Activation:* Map $z$ to probability $\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$.
3. *Loss Minimization:* Optimize parameters by minimizing L2-regularized binary cross-entropy (log-loss):
   $$\mathcal{L}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \hat{p}_i + (1 - y_i) \ln(1 - \hat{p}_i) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$
4. *Decision Threshold:* Assign class $\hat{y} = 1$ if $\hat{p} \ge \tau$, else 0.

**Q19. Why does Decision Tree underperform on this dataset?**  
The Decision Tree achieved a CV F1 of 0.4960 and test F1 of 0.4609. Single unconstrained trees split greedily on Gini impurity, creating deep orthogonal partitions that overfit training noise. On tabular telecom data with smooth linear log-odds, single trees exhibit high variance and poor test generalization.

**Q20. What hyperparameters were tuned in Random Forest and what were the optimal values?**  
GridSearchCV with 5-fold Stratified CV was conducted over:
- `classifier__n_estimators`: [50, 100]
- `classifier__max_depth`: [5, 10, None]
- `classifier__min_samples_split`: [2, 5]  
*Optimal Parameters:* `max_depth = 10`, `min_samples_split = 5`, `n_estimators = 50`. Tuning elevated RF CV F1 from 0.5612 to 0.5725.

---

### Section 6: Model Selection & Test Evaluation

**Q21. Why was Logistic Regression selected over Random Forest?**  
Selection was driven dynamically by empirical cross-validation evidence on `X_train`:
- **Primary Metric (Mean CV F1):** Logistic Regression (0.5951 ± 0.0238) > Tuned RF (0.5725 ± 0.0211) > Default RF (0.5612) > Decision Tree (0.4960).
- **Supporting Metric 1 (Mean CV Recall):** Logistic Regression (54.52%) > Tuned RF (50.23%).
- **Supporting Metric 2 (Mean CV ROC-AUC):** Logistic Regression (0.8460) > Tuned RF (0.8431).
In tabular data where risk scales monotonically with tenure decay and fee increases, regularized linear models often generalize better than axis-aligned tree ensembles.

**Q22. What were the final test metrics of the winning Logistic Regression pipeline?**  
On the 1,407 held-out test samples:
- **Accuracy:** 0.8053 (80.53%)
- **Precision:** 0.6515 (65.15%)
- **Recall:** 0.5749 (57.49%)
- **F1-Score:** 0.6108
- **ROC-AUC:** 0.8360

**Q23. What are the exact values of the test Confusion Matrix (at threshold 0.50)?**  
- **True Negatives (TN):** 918 (retained customers correctly classified)
- **False Positives (FP):** 115 (false alarms: predicted to churn, but stayed)
- **False Negatives (FN):** 159 (missed churners: predicted to stay, but churned)
- **True Positives (TP):** 215 (churners successfully captured)

**Q24. Explain the Accuracy Paradox in this project.**  
The Dummy Classifier achieved 73.42% accuracy by universally predicting "No Churn". However, it had Recall = 0.0000 and F1 = 0.0000, missing 100% of churnable revenue. Logistic Regression achieved 80.53% accuracy (+7.11% gain) while capturing 215 churners with 65.15% precision, demonstrating why accuracy alone is fundamentally deceptive.

---

### Section 7: Operational Threshold Analysis

**Q25. What is the business rationale for evaluating alternative classification thresholds?**  
Default classification uses $\tau = 0.50$, assuming equal error costs. In telecom churn, a False Negative costs ~$1,000 in lost customer lifetime value, while a False Positive costs only ~$30 for a retention credit. False Negatives are 20 to 30 times more expensive than False Positives. Adjusting $\tau$ aligns the classifier with financial reality.

**Q26. What happens across the evaluated thresholds [0.30, 0.40, 0.50, 0.60, 0.70]?**  
- $\tau = 0.30$: Precision = 0.5118, Recall = 0.7567, F1 = 0.6106 (TP 283, FP 270, FN 91)
- $\tau = 0.40$: Precision = 0.5779, Recall = 0.6845, **F1 = 0.6267** (TP 256, FP 187, FN 118)
- $\tau = 0.50$: Precision = 0.6515, Recall = 0.5749, F1 = 0.6108 (TP 215, FP 115, FN 159)
- $\tau = 0.60$: Precision = 0.6808, Recall = 0.3877, F1 = 0.4940 (TP 145, FP 68, FN 229)
- $\tau = 0.70$: Precision = 0.7590, Recall = 0.1684, F1 = 0.2757 (TP 63, FP 20, FN 311)

**Q27. Why is threshold 0.40 recommended as the operational optimum?**  
At threshold 0.40:
1. Peak global F1-Score (0.6267) is attained.
2. Recall increases from 57.49% to 68.45% (+11.0% absolute increase).
3. The model captures **41 additional churners** (reducing missed churners from 159 to 118).
4. The 72 extra false positives cost ~$2,100 in promotional offers, while saving 41 subscribers protects ~$41,000 in customer lifetime value (a ~19:1 return on investment).

---

### Section 8: Model Interpretation & Error Analysis

**Q28. What are the strongest predictive features identified by the model?**  
- **Top Protective Drivers (Negative Coefficients):**
  - `Contract_Two year` ($\beta = -1.38$) and `Contract_One year` ($\beta = -0.68$): Contractual switching barriers.
  - `tenure` ($\beta = -0.79$): Customer longevity and brand attachment.
  - `TechSupport_Yes` ($\beta = -0.28$) and `OnlineSecurity_Yes` ($\beta = -0.32$): Operational friction reduction.
- **Top Risk Drivers (Positive Coefficients):**
  - `InternetService_Fiber optic` ($\beta = +0.86$): Premium fee dissatisfaction and installation friction.
  - `PaymentMethod_Electronic check` ($\beta = +0.31$): Manual monthly payment friction.
  - `PaperlessBilling_Yes` ($\beta = +0.18$): Tech-savvy subscribers with lower switching inertia.

**Q29. What customer profiles does the model struggle to classify?**  
- **False Positives (115 cases – The "Resilient Fragile" Customer):** Low tenure (mean 15.5 months), high monthly charges ($77.83), mostly Month-to-Month contracts (82.6%). Their profile looks like churners, yet they stay due to unobserved factors like local broadband monopolies, landlord restrictions, or employer reimbursement.
- **False Negatives (159 cases – The "Silent Attrition" Customer):** Moderate-to-long tenure (mean 25.8 months), lower monthly charges ($68.42). Their static profile appears healthy, but they churned unexpectedly due to sudden unmeasured shocks: competitor promotions, service outages, or relocations.

**Q30. Does feature importance imply causation?**  
No. Statistical coefficients measure observational correlation within this dataset. For instance, having a two-year contract is associated with lower churn, but forcing dissatisfied customers into two-year contracts will not magically eliminate the root causes of their dissatisfaction. Causal inference requires randomized controlled trials (A/B testing) or quasi-experimental instrumental variable methods.

---

### Section 9: Business Playbook & Operational Strategy

**Q31. What are your 4 core business recommendations?**  
1. **First-Year Onboarding Stabilization:** Churn density peaks in months 1–6. Establish automated Day 14, 45, and 90 onboarding touchpoints with VIP customer support.
2. **Month-to-Month Contract Conversion:** Month-to-month subscribers account for 88% of all churn. Deploy proactive campaigns offering a $10/mo discount in exchange for signing an annual contract.
3. **Fiber Optic Service Bundling:** Fiber optic has 41.9% churn, but Tech Support reduces churn to 15.2%. Automatically bundle free complimentary Tech Support and Device Protection into all fiber installations.
4. **Digital Auto-Pay Migration:** Electronic check users experience 45.3% churn compared to ~15% for automatic billing. Offer a one-time $15 statement credit to migrate users to ACH or credit card auto-pay.

---

### Section 10: Limitations & Future Enhancements

**Q32. What are the primary limitations of this study?**  
1. *Cross-Sectional Data:* The dataset provides a single historical snapshot lacking sequential event logs, call-detail records (CDR), and customer service complaint transcripts.
2. *Unmeasured Exogenous Shocks:* External factors such as regional competitor infrastructure upgrades or economic downturns cannot be observed.
3. *Survivor Bias:* Subscribers with 60+ months of tenure inherently represent individuals with high baseline brand loyalty.

**Q33. How would you advance this project in a corporate telecommunications environment?**  
1. Implement **Survival Analysis** (Cox Proportional Hazards and Weibull AFT) to model time-to-churn hazard curves: $S(t) = P(T > t)$.
2. Incorporate **Cost-Sensitive Loss Functions** directly into model training to penalize false negatives proportionally during gradient descent.
3. Ingest real-time weekly streaming telemetry (broadband latency, packet loss, billing query tickets) into recurrent neural networks (LSTM) or Temporal Fusion Transformers.
4. Deploy model monitoring dashboards to track feature drift (Population Stability Index - PSI) and performance degradation over time.
