# TN-MVD-outcomes

**Predicting Post-Operative Pain Outcomes in Microvascular Decompression for Trigeminal Neuralgia**

A machine learning classification analysis on 322 patients who underwent Microvascular Decompression (MVD) at the University of Pittsburgh Medical Center (UPMC). The goal is to predict treatment failure at last follow-up using pre-operative and intra-operative clinical features.

---

## Clinical Background

**Trigeminal Neuralgia (TN)** is a chronic facial pain condition characterised by sudden, severe, electric shock-like attacks along the trigeminal nerve. **Microvascular Decompression (MVD)** is the gold-standard surgical treatment, achieving ~80% long-term success. This project aims to identify which patients are at risk of treatment failure before surgery.

Outcomes are measured using the **Barrow Neurological Institute (BNI) Pain Score**:
- **Success (class 0):** BNI I–IIIb — pain-free or adequately controlled
- **Failure (class 1):** BNI IV–V — inadequate pain control

---

## Repository Contents

| File | Description |
|---|---|
| `TN_MVD_Analysis.ipynb` | Full analysis notebook: EDA → preprocessing → model training → evaluation → SHAP |
| `TN_MVD_ML_Report.md` | Comprehensive written report with figures, results, and clinical interpretation |
| `*.png` | 15 output figures referenced in the report |
| `.gitignore` | Excludes raw data files (CSV/XLSX) |

---

## Methods Overview

### Data
- **322 patients**, all MVD surgery, UPMC
- **Target:** Binary — treatment failure at last follow-up (19.6% failure rate)
- **39 features:** demographics, comorbidities, pain characteristics, neurovascular anatomy, surgical details, medication history

### Pipeline
1. **EDA** — Missing value analysis, class distribution, feature distributions, outlier detection
2. **Preprocessing** — Median/mode imputation, ordinal encoding, one-hot encoding, StandardScaler; all within a scikit-learn Pipeline to prevent data leakage
3. **Feature Engineering** — Duration quartile bins, age groups, duration in years
4. **Modeling** — 9 classifiers compared via 5-fold stratified cross-validation:
   - Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
   - XGBoost, LightGBM, SVM (RBF), KNN, Dummy baseline
5. **Class Imbalance** — `class_weight='balanced'`, SMOTE, `scale_pos_weight`
6. **Hyperparameter Tuning** — RandomizedSearchCV (50 iterations) for top 3 models
7. **Evaluation** — AUC-ROC, sensitivity/specificity, threshold analysis, calibration curves
8. **Interpretability** — SHAP values (XGBoost + Random Forest), permutation importance, logistic regression coefficients

---

## Key Results

| Model | Test AUC-ROC |
|---|---|
| **XGBoost (tuned)** | **0.660** |
| Random Forest (tuned) | 0.614 |
| KNN | 0.604 |
| LightGBM (tuned) | 0.572 |
| Dummy baseline | 0.500 |

At a classification threshold of **0.15** (below the default 0.50), the best model achieves:
- **Sensitivity: 46%** — catches ~6 of 13 failures in the test set
- **Specificity: 77%** — correctly reassures 40 of 52 successes

### Top Predictors (SHAP)
1. `duration_months` — time from TN diagnosis to surgery (longer → worse outcome)
2. `age_surgery` — younger patients have better outcomes
3. `vessel_intraop_Arterial` — arterial compression predicts success
4. `nvc_intraop` / `nvc_radiologic` — higher NVC severity confirms surgical target
5. `baclofen` use — may indicate centrally-mediated pain subtype

---

## Figures

| Figure | Description |
|---|---|
| `missing_heatmap.png` | Missing value heatmap |
| `class_balance.png` | Target class distribution |
| `categorical_failure_rates.png` | Failure rate by categorical feature |
| `numeric_distributions.png` | Numeric feature distributions by class |
| `correlation_matrix.png` | Pearson correlation matrix |
| `cv_auc_comparison.png` | Cross-validation AUC comparison across all models |
| `roc_curves.png` | ROC curves — all models on held-out test set |
| `confusion_matrices.png` | Confusion matrices — top 4 models |
| `calibration_curves.png` | Calibration curves — top 4 models |
| `threshold_analysis.png` | Sensitivity/specificity vs. classification threshold |
| `shap_summary_xgb.png` | SHAP beeswarm plot — XGBoost |
| `shap_importance_bar.png` | SHAP importance bar — XGBoost |
| `shap_importance_rf.png` | SHAP importance bar — Random Forest |
| `permutation_importance.png` | Permutation importance — XGBoost, RF, LightGBM |
| `lr_coefficients.png` | Logistic Regression standardised coefficients |

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
imbalanced-learn
shap
jupyter
```

Install with: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn shap jupyter`

---

## Usage

1. Place `UpdatedTNMVD.csv` in the same directory as the notebook
2. Open `TN_MVD_Analysis.ipynb` and run all cells
3. Figures are saved as PNG files in the same directory
4. See `TN_MVD_ML_Report.md` for full results and clinical interpretation

---

## Limitations

- Single-centre retrospective data (UPMC only)
- 63 failure cases — small minority class leads to high variance in performance estimates
- Variable follow-up duration across patients
- No external validation cohort

External validation on an independent multi-centre cohort is required before clinical deployment.

---

*UPMC Zenonos Lab · Department of Neurosurgery · University of Pittsburgh*
