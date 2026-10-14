# Telecom Customer Churn Prediction – Data Dictionary

## 1. Source & Attribution
- **Dataset:** IBM Telco Customer Churn
- **Origin:** IBM Cognos Analytics / Telco Customer Churn Repository
- **Context:** Residential telecommunications customer accounts in California

---

## 2. Raw Attribute Specifications (21 Columns)

### A. Identifier Column (1)
| Column Name | Storage Type | Description | Modeling Treatment |
|:---|:---|:---|:---|
| `customerID` | String / Object | Unique alphanumeric account identifier (e.g. `7590-VHVEG`). | **Excluded before modeling.** High cardinality ($N=7,043$) with zero generalizable predictive signal; risks spurious memorization. |

### B. Customer Demographics (4)
| Column Name | Type | Permitted Values | Definition |
|:---|:---|:---|:---|
| `gender` | Categorical | `Male`, `Female` | Customer biological gender. |
| `SeniorCitizen` | Binary Integer | `0`, `1` | Indicates whether customer is 65 years or older (1 = Senior, 0 = Non-Senior). |
| `Partner` | Categorical | `Yes`, `No` | Indicates whether customer has a spouse or live-in partner. |
| `Dependents` | Categorical | `Yes`, `No` | Indicates whether customer lives with financial dependents (children, elderly relatives). |

### C. Account Relationship (1)
| Column Name | Type | Range | Definition |
|:---|:---|:---|:---|
| `tenure` | Integer | `0` to `72` months | Total number of months the customer has maintained an active subscription with the operator. |

### D. Telecom Services (9)
| Column Name | Type | Permitted Values | Definition |
|:---|:---|:---|:---|
| `PhoneService` | Categorical | `Yes`, `No` | Whether the customer subscribes to fixed voice service. |
| `MultipleLines` | Categorical | `Yes`, `No`, `No phone service` | Whether customer subscribes to multiple telephone lines. |
| `InternetService` | Categorical | `DSL`, `Fiber optic`, `No` | Underlying fixed broadband technology provider. |
| `OnlineSecurity` | Categorical | `Yes`, `No`, `No internet service` | Add-on service for online threat prevention and firewall protection. |
| `OnlineBackup` | Categorical | `Yes`, `No`, `No internet service` | Cloud storage add-on for automated file backups. |
| `DeviceProtection` | Categorical | `Yes`, `No`, `No internet service` | Warranty / insurance protection for subscriber hardware devices. |
| `TechSupport` | Categorical | `Yes`, `No`, `No internet service` | Dedicated priority technical troubleshooting assistance. |
| `StreamingTV` | Categorical | `Yes`, `No`, `No internet service` | Subscription to third-party or operator IPTV streaming content. |
| `StreamingMovies` | Categorical | `Yes`, `No`, `No internet service` | Subscription to premium on-demand movie streaming. |

### E. Contract & Billing (5)
| Column Name | Type | Permitted Values / Range | Definition |
|:---|:---|:---|:---|
| `Contract` | Categorical | `Month-to-month`, `One year`, `Two year` | Current contractual commitment duration. Strongest inverse predictor of churn. |
| `PaperlessBilling` | Categorical | `Yes`, `No` | Whether subscriber receives electronic statements instead of physical paper mail. |
| `PaymentMethod` | Categorical | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | Payment mechanism used for monthly invoice settlement. |
| `MonthlyCharges` | Float | `$18.25` to `$118.75` | Current recurring monthly charge in USD. |
| `TotalCharges` | Float (coerced) | `$18.80` to `$8684.80` | Cumulative lifetime charges billed to customer in USD. Coerced from string; 11 zero-tenure rows with blank strings were dropped. |

### F. Target Variable (1)
| Column Name | Type | Permitted Values | Definition |
|:---|:---|:---|:---|
| `Churn` | Categorical | `Yes`, `No` | Ground-truth binary target indicating whether subscriber discontinued service within the preceding month. Mapped: `Yes` -> 1, `No` -> 0. |

---

## 3. Feature Engineering & Ablation Summary

| Feature Name | Type | Construction Formula | Evaluation Status & Justification |
|:---|:---|:---|:---|
| `SubscribedServiceCount` | Integer (0–9) | Sum of active flags across: PhoneService ('Yes'), MultipleLines ('Yes'), InternetService ('DSL'/'Fiber optic'), OnlineSecurity ('Yes'), OnlineBackup ('Yes'), DeviceProtection ('Yes'), TechSupport ('Yes'), StreamingTV ('Yes'), StreamingMovies ('Yes'). | **RETAINED:** Empirically validated via 5-fold CV. Increases Random Forest CV F1 from 0.5573 to 0.5612. Captures total ecosystem stickiness without inflating model complexity. |
| `TenureBand` | Categorical (6 bins) | Grouping `tenure` into `0-1 Yr`, `1-2 Yrs`, `2-3 Yrs`, `3-4 Yrs`, `4-5 Yrs`, `5-6 Yrs`. | **DISCARDED:** 5-fold CV ablation proved it adds redundant parameters collinear with continuous `tenure`, lowering Logistic Regression CV F1 from 0.5951 to 0.5906. Excluded adhering to the principle of parsimony. |
