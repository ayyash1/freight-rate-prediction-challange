# 2-3 Minute Loom Video Presentation Script
**Assessment Title:** Freight Rate Prediction Machine Learning Challenge  
**Target Duration:** 2 minutes 30 seconds - 3 minutes  
**Presenter:** Ahamed Ayyash  
**Date:** October 6, 2026  

---

## Screen Setup Checklist Before Recording
1. **Tab/Window 1 (0:00 - 0:40):** Report Figures (`report/figures/rate_vs_distance.png` and `report/figures/day_of_week_seasonality.png`)
2. **Tab/Window 2 (0:40 - 1:15):** VS Code / IDE showing `src/data_cleaner.py` and `src/feature_engineering.py`
3. **Tab/Window 3 (1:15 - 2:05):** `src/models.py`, `src/validation.py` & CV Comparison Table in Report
4. **Tab/Window 4 (2:05 - 2:40):** `scorer_results/candidate_december.png` and terminal output from running `python predict.py`

---

## Section-by-Section Timing & Spoken Script

### [0:00 - 0:40] Key Findings from Exploratory Data Analysis (EDA)
> **What to show on screen:** `report/figures/rate_vs_distance.png` and the EDA section of the report.

**Spoken Transcript:**
> "Hi everyone, thank you for reviewing my submission for the Freight Rate Prediction Challenge. Today, I'll walk you through my end-to-end machine learning solution.
>
> In exploring the 48,000 labeled development loads, I identified four critical market behaviors:
> 1. **Distance Economies of Scale:** While haul distance strongly dictates total freight cost ($r = 0.91$), the **Rate Per Mile (RPM)** declines non-linearly. Short hauls under 300 miles command an average rate of **$2.58/mile** due to fixed loading and terminal overhead, whereas long hauls over 1,000 miles drop to **$1.96/mile**.
> 2. **Equipment Premiums:** Reefer loads command a notable premium averaging **$2.38/mile** to offset refrigeration fuel and compliance costs, and Flatbeds average **$2.29/mile**, while Dry Vans sit at the standard baseline of **$2.12/mile**.
> 3. **Intra-Week Cyclicality:** Freight rates follow a strict 7-day demand cycle—cresting mid-week on Tuesdays and Thursdays during commercial dispatch surges, and dropping over weekends.
> 4. **Market Index Interaction:** The upstream `quote_signal` and macroeconomic `market_index` provide powerful baseline cost anchors when multiplied by haul distance."

---

### [0:40 - 1:15] Data Quality Issues Identified & Remediation
> **What to show on screen:** `src/data_cleaner.py` (lines showing `weight.abs()` and median imputation).

**Spoken Transcript:**
> "During data auditing, I identified and resolved four critical data-quality anomalies:
> 1. **Inverted Negative Cargo Weights:** 292 training records and 145 validation records contained negative weights (e.g., -36,559 lbs). Analyzing the distribution confirmed these were sign-inverted commercial shipments. I applied absolute value sign correction (`np.abs(weight)`), restoring realistic commercial truckload weights between 10,000 and 47,500 lbs.
> 2. **Missing Weights:** 300 training and 165 validation rows had null weights, which I imputed using equipment-specific category medians alongside a missingness indicator flag.
> 3. **Missing Market Indices:** 374 training and 249 validation loads lacked `market_index`. Since this index reflects day-level macroeconomic capacity, I imputed missing values from matching daily cross-sectional medians.
> 4. **Spatial Integrity:** Verified 1-to-1 deterministic coordinate mappings for all cities and built a unified master geocoding dictionary across all splits."

---

### [1:15 - 1:50] Training & Validation Strategy (Preventing Lookahead Leakage)
> **What to show on screen:** `src/validation.py` (ExpandingTimeSeriesCV code) and the Cross-Validation table in the report.

**Spoken Transcript:**
> "To validate our models, I intentionally rejected standard random K-Fold cross-validation. In non-stationary time-series freight data, random shuffling causes lookahead leakage by allowing models to peek at future market states to predict the past.
>
> Instead, I implemented an **Expanding-Window Time Series Cross-Validation** strategy across 5 monthly folds (training on months 1 to $k-1$ and validating strictly on month $k$, from June through October 2025). This guaranteed that validation scores faithfully mirror our final deployment on unobserved November and December shipments."

---

### [1:50 - 2:25] Model Architecture & Target Formulation Reasoning
> **What to show on screen:** `src/models.py` (FreightEnsembleModel and target transformation).

**Spoken Transcript:**
> "For the model architecture, my key breakthrough was re-framing the problem:
> Instead of directly predicting total load rate—which suffers from severe heteroscedasticity across short and long hauls—I formulated the target as **Rate Per Mile (RPM)**:
> $$\text{predicted\_rate} = \hat{y}_{\text{RPM}} \times \text{distance}$$
>
> This single formulation reduced out-of-fold Mean Absolute Error (MAE) from $141.63 down to **$120.11**—a 15% reduction in error.
>
> For our final model, I trained a **Weighted Multi-Model Gradient Boosted Ensemble**:
> - **40% LightGBM** (optimized with L1 loss for outlier resistance)
> - **40% CatBoost** (with symmetric trees for robust regularization)
> - **20% XGBoost** (with Huber loss)
>
> This ensemble achieved an out-of-fold **MAE of $123.82**, **R² of 0.8225**, and a **MAPE of 5.10%** across temporal cross-validation folds."

---

### [2:25 - 2:55] Code Walkthrough & score.py Verification
> **What to show on screen:** Terminal showing `python predict.py` output and `scorer_results/candidate_december.png`.

**Spoken Transcript:**
> "Finally, let's look at the implementation and verification:
> - The codebase is organized modularly under `src/` with dedicated modules for data cleaning, feature engineering, modeling, validation, and end-to-end pipeline execution.
> - Running `python predict.py` automatically fits the production ensemble on all 48,000 labeled loads, generates all 12,000 required validation predictions in `validation_predictions.csv`, completes `data/december_chart_inputs.csv`, and executes the provided `score.py`.
> - As you can see, `score.py` verified all 12,000 predictions with 100% compliance and generated the official **December 2025 Prediction Chart** (`candidate_december.png`).
> - The December forecast for the Lexington to Fort Wayne dry van lane exhibits realistic rate stability (~$805 average, ~$2.24/mi), captures weekly demand oscillations, and reflects the pre-holiday shipping surge and post-holiday normalization.
>
> Thank you for your time, and I look forward to discussing the solution further!"

---

## Key Talking Points Summary Table

| Category | Key Takeaway | Spoken Emphasis |
| :--- | :--- | :--- |
| **EDA** | Distance economy of scale; equipment premiums (Reefer: $2.38, Flatbed: $2.29, Dry Van: $2.12); weekly seasonality. | Highlights deep understanding of freight domain mechanics. |
| **Data Quality** | Fixed inverted negative weights via `abs()`; median imputation for missing cargo weights and market indices. | Demonstrates diligence and clean engineering practices. |
| **Validation** | 5-Fold Expanding-Window Time Series CV (zero temporal lookahead leakage). | Emphasizes rigorous ML methodology over naive k-fold. |
| **Model Choice** | Rate Per Mile (RPM) formulation + Weighted Ensemble (LightGBM + CatBoost + XGBoost). | Proves strong performance gains (MAE $123.82, MAPE 5.10%). |
| **Verification** | `score.py` passed 100% with 12,000 predictions & December 2025 chart generated. | Complete, verified, production-ready deliverable. |
