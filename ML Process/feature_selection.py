2.FEATURE SELECTION MODULE
------------------------

This script loads the training, testing, and full dataset, then performs several
feature-selection techniques to identify the most informative predictors for
classification.

Included feature-selection methods:
    • Univariate feature selection (ANOVA F-test)
    • Recursive Feature Elimination (RFE) with Logistic Regression
    • Random Forest feature importance ranking

The script then outputs:
    • Top 5 features from each method
    • Common features across methods
    • Union of all selected features
    • Example Random Forest accuracy using the selected features

All file paths are user-editable so you can plug in your own dataset.


# ============================================================
# USER-EDITABLE INPUTS
# ============================================================

TEST_PATH = "path/to/test.csv"
FULL_DATA_PATH = "path/to/full_training_data.csv"
SEPARATED_TEST_PATH = "path/to/holdout_test.csv"

TARGET_COLUMN = "CONDITION"   # dependent variable
GROUP_COLUMN = "ID"           # patient ID column
FEATURE_START = 3             # starting column index for features
FEATURE_END = 25              # ending column index for features

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# LOAD DATA
# ============================================================

test_data = pd.read_csv(TEST_PATH)
full_input_data = pd.read_csv(FULL_DATA_PATH)
separated_test = pd.read_csv(SEPARATED_TEST_PATH)

# Feature columns (editable range)
feature_columns = full_input_data.columns[FEATURE_START:FEATURE_END]

# Extract X and y
X = full_input_data[feature_columns]
y = full_input_data[TARGET_COLUMN]


# ============================================================
# FEATURE SELECTION METHODS
# ============================================================

# ------------------------------
# 1. Univariate Feature Selection
# ------------------------------
selector_kbest = SelectKBest(score_func=f_classif, k=5)
selector_kbest.fit(X, y)

kbest_features = [feature_columns[i] for i in selector_kbest.get_support(indices=True)]
print("Univariate Feature Selection - Top 5 Features:", kbest_features)

# Plot scores
kbest_df = pd.DataFrame({
    "Feature": feature_columns,
    "Score": selector_kbest.scores_
}).sort_values(by="Score", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(kbest_df["Feature"], kbest_df["Score"])
plt.xlabel("Score")
plt.ylabel("Features")
plt.title("Univariate Feature Selection (ANOVA F-test)")
plt.gca().invert_yaxis()
plt.show()


# ------------------------------
# 2. Recursive Feature Elimination (RFE)
# ------------------------------
rfe_model = LogisticRegression(max_iter=2000)
rfe = RFE(estimator=rfe_model, n_features_to_select=5)
rfe.fit(X, y)

rfe_features = [feature_columns[i] for i in range(len(feature_columns)) if rfe.support_[i]]
print("RFE - Top 5 Features:", rfe_features)

# Plot rankings
rfe_df = pd.DataFrame({
    "Feature": feature_columns,
    "Ranking": rfe.ranking_
}).sort_values(by="Ranking")

plt.figure(figsize=(10, 6))
plt.barh(rfe_df["Feature"], rfe_df["Ranking"])
plt.xlabel("Ranking")
plt.ylabel("Features")
plt.title("Recursive Feature Elimination (RFE)")
plt.gca().invert_yaxis()
plt.show()


# ------------------------------
# 3. Random Forest Feature Importance
# ------------------------------
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)

rf_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

rf_features = rf_df["Feature"][:5].tolist()
print("Random Forest - Top 5 Features:", rf_features)

plt.figure(figsize=(10, 6))
plt.barh(rf_df["Feature"], rf_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.show()


# ============================================================
# FEATURE SET COMBINATIONS
# ============================================================

common_features = list(set(kbest_features) & set(rfe_features) & set(rf_features))
print("Common Features Across All Methods:", common_features)

union_features = list(set(kbest_features) | set(rfe_features) | set(rf_features))
print("Union of All Selected Features:", union_features)


# ============================================================
# EXAMPLE MODEL USING SELECTED FEATURES
# ============================================================

X_union = X[union_features]
X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(
    X_union, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_fs, y_train_fs)
preds = model.predict(X_test_fs)

print("Accuracy Using Union of Selected Features:", accuracy_score(y_test_fs, preds))