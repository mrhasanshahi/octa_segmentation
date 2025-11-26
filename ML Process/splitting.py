1. Data Splitting :
To prevent data leakage and ensure a reliable evaluation of the model, we applied a group-wise splitting strategy using patient IDs as grouping labels. Because the dataset contains samples from both eyes of some patients, splitting at the row level could place data from the same patient into different subsets, causing the model to indirectly “see” similar samples across training and testing. To avoid this, all samples belonging to each patient were kept within the same subset. In the first step, we performed an initial split of the full dataset into 80% for training/validation and 20% as an independent hold-out test set. Then, within the 80% training portion, we performed an additional split—again using grouped sampling—to divide it into 80% training and 20% test/validation. This approach ensures that no patient contributes data to more than one split and that all evaluation metrics remain free from leakage. We recommend following this same grouped splitting procedure for any future experiments or replications.


# IMPORTS

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# --------------------------
# User inputs
# --------------------------
INPUT_CSV_PATH = "path/to/your/input.csv"   # <- replace with your CSV file path
OUTPUT_TEST_PATH = "path/to/save/test.csv"

TARGET_COLUMN = "CONDITION"   # target variable
GROUP_COLUMN = "ID"           # patient ID column
# --------------------------

# Load dataset
data = pd.read_csv(INPUT_CSV_PATH)

# Separate features, target, and grouping column
X = data.drop(columns=[TARGET_COLUMN])
y = data[TARGET_COLUMN]
groups = data[GROUP_COLUMN]

# StratifiedGroupKFold for an 80/20 split
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=50)

# Perform the split (take the first fold only → 80/20)
for train_idx, test_idx in sgkf.split(X, y, groups):
    train_data = data.iloc[train_idx].copy()
    test_data = data.iloc[test_idx].copy()
    break

# Reset indices for clean outputs
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# Save results
test_data.to_csv(OUTPUT_TEST_PATH, index=False)

# Summary
print(f"Train set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

# Optional: class distribution check
print("\nTrain class distribution:")
print(train_data[TARGET_COLUMN].value_counts(normalize=True))

print("\nTest class distribution:")
print(test_data[TARGET_COLUMN].value_counts(normalize=True))