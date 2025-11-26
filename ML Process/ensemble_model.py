4. ensemble model :




"""
===============================================================
 ENSEMBLE WEIGHT SEARCH SCRIPT
===============================================================

 This script automatically finds the best ensemble of models
 by testing all weight combinations (that sum to 1.0) using a
 specified step size.

 YOU CAN FREELY REPLACE THE FOLLOWING:
 --------------------------------------------------------------
 1. MODEL PATHS:
      - Add/remove models
      - Use any .joblib compatible model
      - Order of models determines weight order

 2. FEATURE COLUMNS:
      - Replace with any set of feature names from your dataset
      - Set feature_columns = [...]

 3. DATASET FILE PATHS:
      - Train set (optional)
      - Test set (optional)
      - Full training CSV (optional)
      - Hold-out final evaluation set (X_holdout, y_holdout REQUIRED)

 4. WEIGHT SEARCH:
      - weight_step controls granularity (e.g., 0.1, 0.05, 0.2)
      - Weight combinations automatically generated for all models

 5. OUTPUT:
      - top_k defines how many best ensembles to print/save

 This script outputs a CSV ("ensemble_top_results.csv") containing:
      - voting type (soft/hard)
      - weight combination
      - accuracy
      - sensitivity
      - specificity (binary only)
      - precision
      - AUC (binary soft voting only)

===============================================================
"""

import joblib
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, roc_auc_score,
    confusion_matrix
)

# ---------- USER CONFIGURATION (edit these) ----------
# Replace with your model file paths (add/remove models freely)
model_paths = [
    r"C:/path/to/gb_model.joblib",
    r"C:/path/to/knn_model.joblib",
    r"C:/path/to/rf_model.joblib",
    r"C:/path/to/xgb_model.joblib",
    r"C:/path/to/svm_model.joblib",
]

# Replace with your CSV files
train_csv = "splitted-train.csv"
test_csv = "splitted-test.csv"
full_csv = "train-full.csv"
holdout_csv = "test-holdout.csv"     # REQUIRED for evaluation

# Replace with your feature list
feature_columns = [
    "DVC-Form Factor", "SVC-Convexity", "SVC-Parafovea VAD", "SVC-Form Factor",
    "DVC-Retina VSD", "DVC-Convexity", "SVC-Solidity", "DVC-Solidity", "DVC-Roundness"
]

# Target column
target_column = "CONDITION"

# Ensemble search resolution and output count
weight_step = 0.1      # smaller = more combinations
top_k = 10             # show and save top K ensembles
# -----------------------------------------------------

# Load data
full_input_data = pd.read_csv(full_csv)
holdout_data = pd.read_csv(holdout_csv)

X_holdout = holdout_data[feature_columns].values
y_holdout = holdout_data[target_column].values

# Load models
models = [joblib.load(p) for p in model_paths]
n_models = len(models)

# Determine class labels
ref_classes = None
for m in models:
    if hasattr(m, "classes_"):
        ref_classes = list(m.classes_)
        break
if ref_classes is None:
    ref_classes = sorted(np.unique(y_holdout).tolist())
ref_classes = np.array(ref_classes)
n_classes = len(ref_classes)

# Prediction probability alignment
def get_aligned_proba(model, X, ref_classes):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        model_classes = np.array(model.classes_)
        if np.array_equal(model_classes, ref_classes):
            return probs

        aligned = np.zeros((probs.shape[0], len(ref_classes)))
        for i, cls in enumerate(model_classes):
            idx = np.where(ref_classes == cls)[0]
            if idx.size:
                aligned[:, idx[0]] = probs[:, i]
        row_sums = aligned.sum(axis=1, keepdims=True)
        mask = row_sums.squeeze() > 0
        aligned[mask] /= row_sums[mask]
        return aligned
    else:
        preds = model.predict(X)
        aligned = np.zeros((len(preds), len(ref_classes)))
        for i, p in enumerate(preds):
            idx = np.where(ref_classes == p)[0]
            if idx.size:
                aligned[i, idx[0]] = 1.0
        return aligned

# Precompute predictions
preds_list = [m.predict(X_holdout) for m in models]
proba_list = [get_aligned_proba(m, X_holdout, ref_classes) for m in models]

# Metrics
def evaluate(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)

    try:
        cm = confusion_matrix(y_true, y_pred, labels=ref_classes)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            spec = None
    except:
        spec = None

    sens = recall_score(y_true, y_pred,
                        average="binary" if len(ref_classes) == 2 else "macro",
                        pos_label=ref_classes[1] if len(ref_classes) == 2 else None)

    prec = precision_score(y_true, y_pred,
                           average="binary" if len(ref_classes) == 2 else "macro",
                           pos_label=ref_classes[1] if len(ref_classes) == 2 else None)

    auc = None
    if y_score is not None and len(ref_classes) == 2:
        try:
            auc = roc_auc_score(y_true, y_score)
        except:
            auc = None

    return {
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec) if spec is not None else None,
        "precision": float(prec),
        "auc": float(auc) if auc is not None else None
    }

# Weight generation
def generate_weight_combinations(n, step):
    steps = int(round(1.0 / step)) + 1
    raw = [i * step for i in range(steps)]
    combos = []

    def rec(sofar, k, remaining):
        if k == 1:
            val = round(remaining, 10)
            if 0 <= val <= 1:
                combos.append(sofar + [val])
            return
        for v in raw:
            if v <= remaining + 1e-9:
                rec(sofar + [v], k - 1, remaining - v)

    rec([], n, 1.0)
    return combos

weight_combinations = generate_weight_combinations(n_models, weight_step)

results = []

# Evaluate all weight combinations
for weights in weight_combinations:
    w = np.array(weights)

    # Soft voting
    weighted_prob = sum(w[i] * proba_list[i] for i in range(n_models))
    soft_pred_idx = np.argmax(weighted_prob, axis=1)
    soft_preds = ref_classes[soft_pred_idx]

    y_auc = weighted_prob[:, 1] if len(ref_classes) == 2 else None
    soft_metrics = evaluate(y_holdout, soft_preds, y_auc)

    results.append({
        "voting": "soft",
        "weights": weights,
        **soft_metrics
    })

    # Hard voting
    one_hot = np.zeros((len(y_holdout), len(ref_classes)))
    for i in range(n_models):
        preds = preds_list[i]
        for j, p in enumerate(preds):
            idx = np.where(ref_classes == p)[0]
            if idx.size:
                one_hot[j, idx[0]] += w[i]
    hard_pred_idx = np.argmax(one_hot, axis=1)
    hard_preds = ref_classes[hard_pred_idx]

    hard_metrics = evaluate(y_holdout, hard_preds, None)
    results.append({
        "voting": "hard",
        "weights": weights,
        **hard_metrics
    })

# Sort & output
df = pd.DataFrame(results)
df["_auc"] = df["auc"].fillna(-1)
df = df.sort_values(by=["accuracy", "_auc"], ascending=False).drop(columns=["_auc"])

top_results = df.head(top_k)
top_results.to_csv("ensemble_top_results.csv", index=False)

print("Saved: ensemble_top_results.csv\n")
print(top_results.to_string(index=False))