# End-to-End-Machine-Learning-System-for-Customer-Churn-Prediction-and-Business-Risk-Analysis
AML Customer Churn Prediction and Business Risk Analysis

# Telco Customer Churn: Predictive Analytics & Value-Based Retention

This repository houses an end-to-end, reproducible machine learning pipeline designed to predict customer churn, evaluate model reliability, extract feature-level interpretability, and optimize retention campaign budgets using cost-sensitive decision theory. 

Using the **Telco Customer Churn dataset (7,043 subscribers)**, this project establishes a repeatable workflow: **Raw Customer Data → Preprocessing Pipeline → Churn Probability Scoring → Risk Tier Segmentation → Targeted Financial Action**.

---

## 🚀 Key Project Results

*   **Champion Classifier:** Tuned Gradient Boosting Classifier.
*   **Ranking Separation:** **0.845 ROC-AUC** on the untouched holdout test set (\\(n=1,409\\)).
*   **Operational Success:** At the default 0.50 cutoff, the model secures **67.3% Precision** (minimizing wasted incentives) while capturing **51.1% of all active churners**.
*   **Financial Impact:** Transitioning from a flat 0.50 threshold to an optimized cost-sensitive threshold of **\\(T = 0.12\\)** catches **83.7% of all churners** (reducing missed churners from 183 down to 61) and saves **\$46,650 in net revenue** (147.6% Campaign ROI) on the holdout partition.

---

## 🛠️ Data Preprocessing & Leakage Prevention

The target variable is moderately imbalanced with **26.54% churners** vs. **73.46% non-churners**. To prevent optimistic evaluation bias, the dataset was split into an **80/20 stratified partition** before any data transformations took place.

```
          [ Raw Input Data ]
                  │
        ┌─────────┴─────────┐
    [ Train Set ]       [ Test Set ] (Untouched Holdout)
          │                 │
    [ Pipeline ]            │
    ├── Fit Preprocessing ──┼──────> [ Transform Test Data ] (No Fit!)
    └── Fit Model           │                 │
                            └─────────────────┼─────────┐
                                              ▼         ▼
                                        [ Evaluation Metrics ]
```

### 1. Preprocessing Pipeline (`ColumnTransformer`)
*   **Zero-Tenure Imputation:** Profiling revealed exactly 11 missing values in `TotalCharges`. Because all 11 records corresponded to customers with a `tenure` of 0 months, they were **imputed with 0** using domain reasoning rather than a naive column median.
*   **Continuous Features:** Standardized using `StandardScaler` to accommodate margin-sensitive algorithms like Support Vector Machines.
*   **Categorical Features:** Encoded using **One-Hot Encoding** to prevent nominal features from being falsely treated as ordinal.
*   **Feature Engineering:** Three predictive features were constructed to capture customer lifecycle behaviors:
    1.  `AvgMonthlySpend`: Continuous measure of spending velocity (`TotalCharges` / `tenure`).
    2.  `TenureGroup`: Categorical lifecycle bands grouping customer maturity.
    3.  `ProtectionServiceCount`: Numeric sum of active security and support add-ons.

### 2. Leakage Prevention
All transformations are wrapped inside a unified `scikit-learn Pipeline`. Preprocessing metrics are fit strictly on the training set and applied as passive transforms on validation folds and test data, eliminating any possibility of future-data leakage.

---

## 📊 Model Evaluation & Hyperparameter Tuning

To find the optimal classifier, five distinct supervised classification algorithms were trained, validated, and tuned.

### 1. Baseline Performance (5-Fold Stratified CV)
Models were initially evaluated on the training set using 5-fold cross-validation. Standard deviations (\\(\sigma\\)) were analyzed to guarantee the model's stability across different data folds:

| Model Candidate | Mean CV ROC-AUC | CV Standard Deviation (\\(\sigma\\)) | Mean CV Accuracy |
| :--- | :---: | :---: | :---: |
| **Gradient Boosting** | **0.848** | **0.011** | **80.6%** |
| Logistic Regression | 0.846 | 0.009 | 0.804 |
| Random Forest | 0.842 | 0.010 | 79.9% |
| Support Vector Machine (SVM) | 0.833 | 0.006 | 79.8% |
| Decision Tree | 0.822 | 0.014 | 78.2% |

### 2. Randomized Search Tuning (3-Fold CV Scheme)
For hyperparameter tuning via `RandomizedSearchCV`, a **3-fold cross-validation scheme** was deliberately deployed. 
*   *Computational Trade-Off:* Tuning evaluates hundreds of hyperparameter permutations. Swapping from a 5-fold to a 3-fold scheme during search optimization preserved validation signals while drastically reducing CPU overhead. Baseline model comparisons remained anchored on 5-fold cross-validation to maintain experimental integrity.

### 3. Tuned Performance (Training Set CV)
Tuning refined rather than inverted baseline ranks, securing the **Gradient Boosting Classifier** as the final champion:

| Model Candidate | Tuned CV ROC-AUC | Tuned CV Std (\\(\sigma\\)) | ROC-AUC Delta |
| :--- | :---: | :---: | :---: |
| **Gradient Boosting** | **0.8487** | **0.0091** | **+0.0007** |
| Logistic Regression | 0.8460 | 0.0085 | +0.0010 |
| Random Forest | 0.8445 | 0.0058 | +0.0021 |
| SVM | 0.8355 | 0.0038 | +0.0025 |
| Decision Tree | 0.8296 | 0.0089 | +0.0076 |

---

## 🔍 Model Interpretability & Business Insights

To establish transparent model decisions, feature contributions were evaluated using a dual-interpretability framework:

### 1. Non-Linear Feature Importances (Gradient Boosting)
Ranks variables by their overall predictive contribution to the tree splits:
*   **Contract Type (Month-to-Month): 0.409** (By far the dominant feature; **2.97×** more influential than tenure).
*   **Tenure (Months): 0.138** (Represents established customer loyalty).
*   **Internet Service (Fiber Optic): 0.094** (Friction signal).
*   **Online Security (No): 0.075** (Lack of structural switching barriers).

### 2. Linear Odds Ratios (Logistic Regression)
Provides direct, multiplicative impact on churn odds (\\(OR = e^\beta\\)), holding all other variables constant:
*   **Fiber Optic Internet (OR = 3.49×):** Fiber optic subscribers have **3.49 times the odds** of churning compared to reference DSL users, highlighting a major pricing or connection vulnerability.
*   **Month-to-Month Contract (OR = 1.86×):** Nearly doubles the odds of churn compared to annual plans.
*   **Two-Year Contract (OR = 0.43×):** Reduces churn odds by **57%** (strong loyalty anchor).
*   **DSL Internet Service (OR = 0.33×):** Highly stable; cuts churn odds by **67%**.

---

## 📈 Cost-Sensitive Decision Optimization (Appendix)

In real-world customer retention, **prediction errors carry asymmetric costs**. Failing to identify a churner is far more expensive than unnecessarily sending an incentive to a loyal customer.

### 1. Operational Cost Matrix Parameters
*   **Customer Lifetime Value (CLV):** **\$500** (revenue lost if a customer leaves)
*   **Promotional Incentive Cost:** **\$50** (incentive spent on flagged accounts)
*   **Retention Success Rate (\\(r\\)):** **50%** (probability of successfully saving an at-risk customer)
*   **Error Penalties:**
    *   *False Positive (Type I):* Cost = **\$50** (unnecessary discount, customer stays)
    *   *False Negative (Type II):* Cost = **\$500** (lost CLV)
    *   *True Positive (Correctly Flagged):* Expected Cost = \$50 incentive + 50% chance of losing remaining CLV = **\$300**

### 2. Threshold Strategy Matrix

Evaluating different threshold cutoffs on the untouched holdout test set (\\(n=1,409\\)) yields the following financial outcomes:

| Strategy | Threshold (\\(T\\)) | TP | FP | FN | TN | Budget Spent | Expected Business Loss | Net Revenue Saved |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Do Nothing** | — | 0 | 0 | 374 | 1,035 | \$0 | \$187,000 | Baseline (\$0) |
| **Contact All** | 0.00 | 374 | 1,035 | 0 | 0 | \$70,450 | \$163,950 | \$23,050 |
| **Balanced** | 0.50 | 191 | 93 | 183 | 942 | \$14,200 | \$153,450 | \$33,550 |
| **Optimal Aggressive** | **0.12** | **313** | **319** | **61** | **716** | **\$31,600** | **\$140,350** | **\$46,650** |
| **Oracle (Perfect)** | — | 374 | 0 | 0 | 1,035 | \$18,700 | \$112,200 | \$74,800 |

### 3. Actionable Risk Segmentation
Customers are placed into prioritized operational queues based on their calibrated probability scores:
*   **High-Risk (\\(\ge 0.70\\)):** Narrow peak representing maximum revenue risk. Action: Outbound retention calls, personal account managers, and deep contract conversion credits.
*   **Medium-Risk (\\(0.40 - 0.69\\)):** Middle tier. Action: Lower-cost digital engagement and target-to-annual email campaigns.
*   **Low-Risk (\\(< 0.40\\)):** Wide, stable base. Action: Monitor and route through standard automated customer experience loops.

---

## 📂 Repository Structure

```
├── data/
│   └── telco_customer_churn.csv        # Primary customer dataset
├── notebooks/
│   └── AMLCustomerChurnPA2.ipynb       # Modeling notebook (EDA, training, validation)
├── exports/
│   ├── customer_churn_model.joblib     # Pre-fitted scikit-learn Pipeline binary
│   └── prediction_scoring.csv          # Calibrated customer probability and risk scores
├── reports/
│   ├── customer_churn_project_report.docx   # Written business report
│   └── customer_churn_analytics_deck.pptx   # Executive presentation slides
├── src/
│   ├── preprocess.py                   # Custom ColumnTransformer pipeline steps
│   └── simulate_cost_optimization.py   # Cost-sensitive loss-function equations
└── README.md                           # This file
```

---

## 💻 Setup & Reproducibility

To recreate the python environment and execute the pipeline:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/telco-churn-value-retention.git
    cd telco-churn-value-retention
    ```

2.  **Install Dependencies:**
    This project is locked to Python 3.12 with environment seeds recorded for standard reproducibility:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute the Scoring Pipeline:**
    To score new customer data using your saved `.joblib` pipeline:
    ```python
    import joblib
    import pandas as pd

    # Load the pre-fitted scikit-learn pipeline
    pipeline = joblib.load('exports/customer_churn_model.joblib')

    # Score new subscribers (must match primary input feature schema)
    new_subscribers = pd.read_csv('data/new_subscribers.csv')
    probabilities = pipeline.predict_proba(new_subscribers)[:, 1]
    ```
```

***

