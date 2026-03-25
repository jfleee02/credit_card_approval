[README_credit_approvals.md](https://github.com/user-attachments/files/26228783/README_credit_approvals.md)
# Credit Card Approval Prediction

A full-pipeline machine learning study benchmarking six classifier families to predict credit card approval outcomes.

---

## Overview

This project builds and evaluates a complete credit card approval prediction pipeline. Starting from raw applicant data, we perform thorough EDA, principled feature engineering, and systematic model comparison — all evaluated under a recall-weighted F-beta objective reflecting the real business cost of incorrectly rejecting qualified applicants. The project concludes with a demographic bias audit across ethnicity groups and employment sectors.

---

## Key Results

- **Best model:** Tuned Random Forest — achieved near-perfect recall on the test set while maintaining strong precision and the highest Fβ (β=2) score across all models tested.
- **Best interpretable model:** LASSO Logistic Regression (Test Accuracy: 0.861, Test Fβ: 0.892, Optimal Threshold: 0.38).
- **Recommended deployment:** Dual-model strategy — Random Forest for prediction performance, LASSO for regulatory transparency and explainability.

---

## Methods

| Stage | Approach |
|---|---|
| EDA | Boxplots, correlation heatmaps, approval rate breakdowns by group |
| Feature engineering | Decision-tree credit score binning (entropy), log-income transform, interaction testing via LR tests |
| Interaction testing | Likelihood ratio + AIC/BIC comparison for 5 candidate terms; 2 retained |
| Model evaluation | Stratified 5-fold CV with F-beta (β=2) out-of-fold threshold selection |
| Explainability | SHAP values on tuned Random Forest |
| Bias audit | Controlled logistic regression across ethnicity and industry groups |

---

## Models Compared

Logistic Regression (Baseline, LASSO, Ridge, Elastic Net) · Naive Bayes · Decision Tree · Random Forest · XGBoost · SVM (Linear + RBF)

---

## Top Predictors

Prior default history was the dominant predictor (coef = +1.41, p < 0.001), followed by log-income (+0.62), bank customer status (+0.55), and employment tenure. The Debt × Moderate Credit Score interaction term was retained as a significant negative modifier.

---

## Bias Findings

Latino applicants showed an approval rate of ~14% vs. 42–63% for other groups. After controlling for all financial predictors, the disparity remained statistically significant (OR ≈ 0.25, p < 0.05), though the subsample is small (n = 57). Industry-level disparities largely disappeared after controlling for applicant financial profiles.

---

## Tech Stack

Python · scikit-learn · XGBoost · statsmodels · pandas · numpy · seaborn · SHAP · matplotlib

---

## Repository Structure

```
├── Group_2_Credit_Card_Approval_FINAL.ipynb   # Full analysis notebook
└── index.html                                  # GitHub Pages project site
```

---
