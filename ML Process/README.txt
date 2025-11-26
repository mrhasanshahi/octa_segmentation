# ML Pipeline Overview

This package contains four cleaned Python modules representing a full machine‑learning workflow:

## 1. Data Splitting
Implements grouped patient‑wise stratified splitting to avoid data leakage. Ensures no patient appears in more than one subset.

## 2. Feature Selection
Runs multiple selection techniques  Outputs best features and visual summaries.

## 3. Model Tuning & Selection
Uses Optuna with 8‑fold stratified CV to tune and evaluate models including XGBoost, RF, KNN, LightGBM, SVM, ANN, Logistic Regression, and more.

## 4. Ensemble Model
Searches weight combinations for multiple models to find the best‑performing ensemble using soft and hard voting.

