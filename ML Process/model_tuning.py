import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier as KNN
train_data = pd.read_csv('splitted-train.csv')
test_data = pd.read_csv('splitted-test.csv')
full_input_data = pd.read_csv('train.csv')
seperated_test = pd.read_csv('test.csv')
feature_columns = full_input_data.columns[3:25]

custom_features = ['DVC-Form Factor', 'SVC-Convexity', 'SVC-Parafovea VAD', 'SVC-Form Factor', 'DVC-Retina VSD', 'DVC-Convexity', 'SVC-Solidity', 'DVC-Solidity', 'DVC-Roundness']
feature_columns = custom_features


X_train = train_data[feature_columns]
y_train = train_data['CONDITION']
X_test = test_data[feature_columns]
y_test = test_data['CONDITION']


X_seperated = seperated_test[feature_columns]
y_seperated = seperated_test['CONDITION']


X = full_input_data[feature_columns]
y = full_input_data['CONDITION']


feature_columns
import optuna
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np

# Define the objective function for Optuna optimization
def objective(trial):
    # Hyperparameter search space for XGBoost
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'scale_pos_weight': trial.suggest_uniform('scale_pos_weight', 0.5, 3.0)
    }
    
    # Prepare cross-validation
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    accuracy_list = []
    
    for train_idx, test_idx in skf.split(X, y):
        # Split the data into train and test for the current fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train the XGBoost model with the trial's hyperparameters
        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results for each fold
        accuracy_list.append(accuracy)
    
    # Return the average performance across all folds
    mean_accuracy = np.mean(accuracy_list)
    return mean_accuracy

# Load your data (assuming it's already loaded in the variable `df`)
# X = df.drop('target', axis=1)  # Features
# y = df['target']  # Target variable

# Run the optimization with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best parameters found by Optuna
best_params = study.best_params
print(f"Best hyperparameters found: {best_params}")

# Train and evaluate the model with the best parameters using 8-fold CV
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
accuracy_list = []
sensitivity_list = []
specificity_list = []
precision_list = []
auc_list = []

for train_idx, test_idx in skf.split(X, y):
    # Split the data into train and test for the current fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train the XGBoost model with the best hyperparameters found by Optuna
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    
    # Confusion matrix for specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    auc = roc_auc_score(y_test, y_prob)
    
    # Store results for each fold
    accuracy_list.append(accuracy)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    precision_list.append(precision)
    auc_list.append(auc)

# Calculate and print average and standard deviation of the metrics
average_accuracy = np.mean(accuracy_list)
sd_accuracy = np.std(accuracy_list)

average_sensitivity = np.mean(sensitivity_list)
sd_sensitivity = np.std(sensitivity_list)

average_specificity = np.mean(specificity_list)
sd_specificity = np.std(specificity_list)

average_precision = np.mean(precision_list)
sd_precision = np.std(precision_list)

average_auc = np.mean(auc_list)
sd_auc = np.std(auc_list)

print(f"Average Accuracy: {average_accuracy:.4f} ± {sd_accuracy:.4f}")
print(f"Average Sensitivity (Recall): {average_sensitivity:.4f} ± {sd_sensitivity:.4f}")
print(f"Average Specificity: {average_specificity:.4f} ± {sd_specificity:.4f}")
print(f"Average Precision: {average_precision:.4f} ± {sd_precision:.4f}")
print(f"Average AUC: {average_auc:.4f} ± {sd_auc:.4f}")
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd

# Define the objective function for Optuna optimization
def objective(trial):
    # Hyperparameter search space for Random Forest
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
    }
    
    # Prepare cross-validation
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    accuracy_list = []
    
    for train_idx, test_idx in skf.split(X, y):
        # Split the data into train and test for the current fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train the Random Forest model with the trial's hyperparameters
        model = RandomForestClassifier(**param, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results for each fold
        accuracy_list.append(accuracy)
    
    # Return the average performance across all folds
    mean_accuracy = np.mean(accuracy_list)
    return mean_accuracy

# Load your data (assuming it's already loaded in the variable `df`)
# X = df.drop('target', axis=1)  # Features
# y = df['target']  # Target variable

# Run the optimization with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best parameters found by Optuna
best_params = study.best_params
print(f"Best hyperparameters found: {best_params}")

# Train and evaluate the model with the best parameters using 8-fold CV
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
accuracy_list = []
sensitivity_list = []
specificity_list = []
precision_list = []
auc_list = []

for train_idx, test_idx in skf.split(X, y):
    # Split the data into train and test for the current fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train the Random Forest model with the best hyperparameters found by Optuna
    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    
    # Confusion matrix for specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    auc = roc_auc_score(y_test, y_prob)
    
    # Store results for each fold
    accuracy_list.append(accuracy)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    precision_list.append(precision)
    auc_list.append(auc)

# Calculate and print average and standard deviation of the metrics
average_accuracy = np.mean(accuracy_list)
sd_accuracy = np.std(accuracy_list)

average_sensitivity = np.mean(sensitivity_list)
sd_sensitivity = np.std(sensitivity_list)

average_specificity = np.mean(specificity_list)
sd_specificity = np.std(specificity_list)

average_precision = np.mean(precision_list)
sd_precision = np.std(precision_list)

average_auc = np.mean(auc_list)
sd_auc = np.std(auc_list)

print(f"Average Accuracy: {average_accuracy:.4f} ± {sd_accuracy:.4f}")
print(f"Average Sensitivity (Recall): {average_sensitivity:.4f} ± {sd_sensitivity:.4f}")
print(f"Average Specificity: {average_specificity:.4f} ± {sd_specificity:.4f}")
print(f"Average Precision: {average_precision:.4f} ± {sd_precision:.4f}")
print(f"Average AUC: {average_auc:.4f} ± {sd_auc:.4f}")
import optuna
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd

# Define the objective function for Optuna optimization
def objective(trial):
    # Hyperparameter search space for KNN
    param = {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),  # Number of neighbors
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),  # Weight function
        'p': trial.suggest_int('p', 1, 2),  # Power parameter for Minkowski distance (1 for Manhattan, 2 for Euclidean)
    }
    
    # Prepare cross-validation
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    accuracy_list = []
    
    for train_idx, test_idx in skf.split(X, y):
        # Split the data into train and test for the current fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train the KNN model with the trial's hyperparameters
        model = KNeighborsClassifier(**param)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results for each fold
        accuracy_list.append(accuracy)
    
    # Return the average performance across all folds
    mean_accuracy = np.mean(accuracy_list)
    return mean_accuracy

# Load your data (assuming it's already loaded in the variable `df`)
# X = df.drop('target', axis=1)  # Features
# y = df['target']  # Target variable

# Run the optimization with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best parameters found by Optuna
best_params = study.best_params
print(f"Best hyperparameters found: {best_params}")

# Train and evaluate the model with the best parameters using 8-fold CV
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
accuracy_list = []
sensitivity_list = []
specificity_list = []
precision_list = []
auc_list = []

for train_idx, test_idx in skf.split(X, y):
    # Split the data into train and test for the current fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train the KNN model with the best hyperparameters found by Optuna
    model = KNeighborsClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    
    # Confusion matrix for specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    auc = roc_auc_score(y_test, y_prob)
    
    # Store results for each fold
    accuracy_list.append(accuracy)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    precision_list.append(precision)
    auc_list.append(auc)

# Calculate and print average and standard deviation of the metrics
average_accuracy = np.mean(accuracy_list)
sd_accuracy = np.std(accuracy_list)

average_sensitivity = np.mean(sensitivity_list)
sd_sensitivity = np.std(sensitivity_list)

average_specificity = np.mean(specificity_list)
sd_specificity = np.std(specificity_list)

average_precision = np.mean(precision_list)
sd_precision = np.std(precision_list)

average_auc = np.mean(auc_list)
sd_auc = np.std(auc_list)

print(f"Average Accuracy: {average_accuracy:.4f} ± {sd_accuracy:.4f}")
print(f"Average Sensitivity (Recall): {average_sensitivity:.4f} ± {sd_sensitivity:.4f}")
print(f"Average Specificity: {average_specificity:.4f} ± {sd_specificity:.4f}")
print(f"Average Precision: {average_precision:.4f} ± {sd_precision:.4f}")
print(f"Average AUC: {average_auc:.4f} ± {sd_auc:.4f}")
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd

# Define the objective function for Optuna optimization
def objective(trial):
    # Hyperparameter search space for LightGBM
    param = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),  # Number of leaves in one tree
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),  # Learning rate
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),  # Number of trees
        'max_depth': trial.suggest_int('max_depth', 3, 12),  # Maximum depth of a tree
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Fraction of samples to train on
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Fraction of features to train on
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),  # Minimum number of data points in a leaf
    }
    
    # Prepare cross-validation
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    accuracy_list = []
    
    for train_idx, test_idx in skf.split(X, y):
        # Split the data into train and test for the current fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train the LightGBM model with the trial's hyperparameters
        model = LGBMClassifier(**param)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results for each fold
        accuracy_list.append(accuracy)
    
    # Return the average performance across all folds
    mean_accuracy = np.mean(accuracy_list)
    return mean_accuracy

# Load your data (assuming it's already loaded in the variable `df`)
# X = df.drop('target', axis=1)  # Features
# y = df['target']  # Target variable

# Run the optimization with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best parameters found by Optuna
best_params = study.best_params
print(f"Best hyperparameters found: {best_params}")

# Train and evaluate the model with the best parameters using 8-fold CV
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
accuracy_list = []
sensitivity_list = []
specificity_list = []
precision_list = []
auc_list = []

for train_idx, test_idx in skf.split(X, y):
    # Split the data into train and test for the current fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train the LightGBM model with the best hyperparameters found by Optuna
    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    
    # Confusion matrix for specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    auc = roc_auc_score(y_test, y_prob)
    
    # Store results for each fold
    accuracy_list.append(accuracy)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    precision_list.append(precision)
    auc_list.append(auc)

# Calculate and print average and standard deviation of the metrics
average_accuracy = np.mean(accuracy_list)
sd_accuracy = np.std(accuracy_list)

average_sensitivity = np.mean(sensitivity_list)
sd_sensitivity = np.std(sensitivity_list)

average_specificity = np.mean(specificity_list)
sd_specificity = np.std(specificity_list)

average_precision = np.mean(precision_list)
sd_precision = np.std(precision_list)

average_auc = np.mean(auc_list)
sd_auc = np.std(auc_list)

print(f"Average Accuracy: {average_accuracy:.4f} ± {sd_accuracy:.4f}")
print(f"Average Sensitivity (Recall): {average_sensitivity:.4f} ± {sd_sensitivity:.4f}")
print(f"Average Specificity: {average_specificity:.4f} ± {sd_specificity:.4f}")
print(f"Average Precision: {average_precision:.4f} ± {sd_precision:.4f}")
print(f"Average AUC: {average_auc:.4f} ± {sd_auc:.4f}")
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd

# Define the objective function for Optuna optimization
def objective(trial):
    # Hyperparameter search space for Gradient Boosting
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),  # Number of boosting stages
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),  # Step size shrinking
        'max_depth': trial.suggest_int('max_depth', 3, 12),  # Maximum depth of the individual trees
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),  # Minimum number of samples required to split an internal node
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # Fraction of samples used for fitting each base estimator
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),  # Minimum number of samples required to be at a leaf node
    }
    
    # Prepare cross-validation
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    accuracy_list = []
    
    for train_idx, test_idx in skf.split(X, y):
        # Split the data into train and test for the current fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train the Gradient Boosting model with the trial's hyperparameters
        model = GradientBoostingClassifier(**param)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results for each fold
        accuracy_list.append(accuracy)
    
    # Return the average performance across all folds
    mean_accuracy = np.mean(accuracy_list)
    return mean_accuracy

# Load your data (assuming it's already loaded in the variable `df`)
# X = df.drop('target', axis=1)  # Features
# y = df['target']  # Target variable

# Run the optimization with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best parameters found by Optuna
best_params = study.best_params
print(f"Best hyperparameters found: {best_params}")

# Train and evaluate the model with the best parameters using 8-fold CV
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
accuracy_list = []
sensitivity_list = []
specificity_list = []
precision_list = []
auc_list = []

for train_idx, test_idx in skf.split(X, y):
    # Split the data into train and test for the current fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train the Gradient Boosting model with the best hyperparameters found by Optuna
    model = GradientBoostingClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    
    # Confusion matrix for specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    auc = roc_auc_score(y_test, y_prob)
    
    # Store results for each fold
    accuracy_list.append(accuracy)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    precision_list.append(precision)
    auc_list.append(auc)

# Calculate and print average and standard deviation of the metrics
average_accuracy = np.mean(accuracy_list)
sd_accuracy = np.std(accuracy_list)

average_sensitivity = np.mean(sensitivity_list)
sd_sensitivity = np.std(sensitivity_list)

average_specificity = np.mean(specificity_list)
sd_specificity = np.std(specificity_list)

average_precision = np.mean(precision_list)
sd_precision = np.std(precision_list)

average_auc = np.mean(auc_list)
sd_auc = np.std(auc_list)

print(f"Average Accuracy: {average_accuracy:.4f} ± {sd_accuracy:.4f}")
print(f"Average Sensitivity (Recall): {average_sensitivity:.4f} ± {sd_sensitivity:.4f}")
print(f"Average Specificity: {average_specificity:.4f} ± {sd_specificity:.4f}")
print(f"Average Precision: {average_precision:.4f} ± {sd_precision:.4f}")
print(f"Average AUC: {average_auc:.4f} ± {sd_auc:.4f}")
import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd

# Define the objective function for Optuna optimization
def objective(trial):
    # Hyperparameter search space for Decision Tree
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 12),  # Maximum depth of the tree
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),  # Minimum number of samples required to split an internal node
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),  # Minimum number of samples required to be at a leaf node
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),  # Features considered for splits
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),  # Split quality function
    }
    
    # Prepare cross-validation
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    accuracy_list = []
    
    for train_idx, test_idx in skf.split(X, y):
        # Split the data into train and test for the current fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train the Decision Tree model with the trial's hyperparameters
        model = DecisionTreeClassifier(**param)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_list.append(accuracy)
    
    # Return the average performance across all folds
    mean_accuracy = np.mean(accuracy_list)
    return mean_accuracy

# Load your data (assuming it's already loaded in the variable `df`)
# X = df.drop('target', axis=1)  # Features
# y = df['target']  # Target variable

# Run the optimization with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=150)

# Best parameters found by Optuna
best_params = study.best_params
print(f"Best hyperparameters found: {best_params}")

# Train and evaluate the model with the best parameters using 8-fold CV
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
accuracy_list = []
sensitivity_list = []
specificity_list = []
precision_list = []
auc_list = []

for train_idx, test_idx in skf.split(X, y):
    # Split the data into train and test for the current fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train the Decision Tree model with the best hyperparameters found by Optuna
    model = DecisionTreeClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    
    # Confusion matrix for specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    # Store results for each fold
    accuracy_list.append(accuracy)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    precision_list.append(precision)
    if auc is not None:
        auc_list.append(auc)

# Calculate and print average and standard deviation of the metrics
average_accuracy = np.mean(accuracy_list)
sd_accuracy = np.std(accuracy_list)

average_sensitivity = np.mean(sensitivity_list)
sd_sensitivity = np.std(sensitivity_list)

average_specificity = np.mean(specificity_list)
sd_specificity = np.std(specificity_list)

average_precision = np.mean(precision_list)
sd_precision = np.std(precision_list)

average_auc = np.mean(auc_list) if auc_list else None
sd_auc = np.std(auc_list) if auc_list else None

# Display results
print(f"Average Accuracy: {average_accuracy:.4f} ± {sd_accuracy:.4f}")
print(f"Average Sensitivity (Recall): {average_sensitivity:.4f} ± {sd_sensitivity:.4f}")
print(f"Average Specificity: {average_specificity:.4f} ± {sd_specificity:.4f}")
print(f"Average Precision: {average_precision:.4f} ± {sd_precision:.4f}")
if auc_list:
    print(f"Average AUC: {average_auc:.4f} ± {sd_auc:.4f}")
import optuna
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd

# Define the objective function for Optuna optimization
def objective(trial):
    # Hyperparameter search space for SVM
    param = {
        'C': trial.suggest_loguniform('C', 1e-5, 1e5),  # Regularization parameter
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),  # Type of kernel
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),  # Kernel coefficient
        'degree': trial.suggest_int('degree', 2, 5),  # Degree of the polynomial kernel (if kernel='poly')
    }
    
    # Prepare cross-validation
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    precision_list = []
    auc_list = []
    
    for train_idx, test_idx in skf.split(X, y):
        # Split the data into train and test for the current fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train the SVM model with the trial's hyperparameters
        model = SVC(**param, probability=True, random_state=42)  # Use probability=True to calculate AUC
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        sensitivity = recall_score(y_test, y_pred)
        
        # Confusion matrix for specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        
        # Store results for each fold
        accuracy_list.append(accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        precision_list.append(precision)
        if auc is not None:
            auc_list.append(auc)
    
    # Return the average performance across all folds
    mean_accuracy = np.mean(accuracy_list)
    return mean_accuracy

# Load your data (assuming it's already loaded in the variable `df`)
# X = df.drop('target', axis=1)  # Features
# y = df['target']  # Target variable

# Run the optimization with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25)

# Best parameters found by Optuna
best_params = study.best_params
print(f"Best hyperparameters found: {best_params}")

# Train and evaluate the model with the best parameters using 8-fold CV
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
accuracy_list = []
sensitivity_list = []
specificity_list = []
precision_list = []
auc_list = []
accuracy_per_fold = []

for train_idx, test_idx in skf.split(X, y):
    # Split the data into train and test for the current fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train the SVM model with the best hyperparameters found by Optuna
    model = SVC(**best_params, probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    
    # Confusion matrix for specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    # Store results for each fold
    accuracy_list.append(accuracy)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    precision_list.append(precision)
    if auc is not None:
        auc_list.append(auc)
    accuracy_per_fold.append(accuracy)

# Calculate and print average and standard deviation of the metrics
average_accuracy = np.mean(accuracy_list)
sd_accuracy = np.std(accuracy_list)

average_sensitivity = np.mean(sensitivity_list)
average_specificity = np.mean(specificity_list)
average_precision = np.mean(precision_list)
average_auc = np.mean(auc_list) if auc_list else None

sd_accuracy_fold = np.std(accuracy_per_fold)

print(f"Average Accuracy: {average_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {sd_accuracy:.4f}")
print(f"Average Sensitivity (Recall): {average_sensitivity:.4f}")
print(f"Average Specificity: {average_specificity:.4f}")
print(f"Average Precision: {average_precision:.4f}")
if auc_list:
    print(f"Average AUC: {average_auc:.4f}")
print(f"Accuracy per fold: {accuracy_per_fold}")
print(f"Standard Deviation of Accuracy per fold: {sd_accuracy_fold:.4f}")
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np
import joblib

# Define the objective function for Optuna optimization
def objective(trial):
    # Hyperparameter search space for ANN
    param = {
        'num_layers': trial.suggest_int('num_layers', 1, 3),  # Number of hidden layers
        'neurons_per_layer': trial.suggest_int('neurons_per_layer', 32, 128),  # Neurons per hidden layer
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid']),  # Activation function
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),  # Learning rate
        'batch_size': trial.suggest_int('batch_size', 32, 256),  # Batch size
        'epochs': trial.suggest_int('epochs', 10, 50),  # Number of epochs
    }

    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    accuracy_list = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = Sequential()
        for _ in range(param['num_layers']):
            model.add(Dense(param['neurons_per_layer'], activation=param['activation'], input_dim=X_train.shape[1] if _ == 0 else None))
        model.add(Dense(1, activation='sigmoid'))  # Output layer
        
        optimizer = Adam(learning_rate=param['learning_rate'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, batch_size=param['batch_size'], epochs=param['epochs'], verbose=0)
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_list.append(accuracy)

    return np.mean(accuracy_list)

# Run the optimization with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Best parameters found by Optuna
best_params = study.best_params
print(f"Best hyperparameters found: {best_params}")

# Stratified K-Fold CV with best hyperparameters
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
accuracy_list = []
sensitivity_list = []
specificity_list = []
precision_list = []
auc_list = []
final_model = None

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model = Sequential()
    for _ in range(best_params['num_layers']):
        model.add(Dense(best_params['neurons_per_layer'], activation=best_params['activation'], input_dim=X_train.shape[1] if _ == 0 else None))
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=best_params['learning_rate'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, batch_size=best_params['batch_size'], epochs=best_params['epochs'], verbose=0)
    if final_model is None:
        final_model = model
    
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_prob = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    accuracy_list.append(accuracy)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    precision_list.append(precision)
    if auc is not None:
        auc_list.append(auc)

# Calculate average and standard deviation for each metric
average_accuracy = np.mean(accuracy_list)
sd_accuracy = np.std(accuracy_list)

average_sensitivity = np.mean(sensitivity_list)
sd_sensitivity = np.std(sensitivity_list)

average_specificity = np.mean(specificity_list)
sd_specificity = np.std(specificity_list)

average_precision = np.mean(precision_list)
sd_precision = np.std(precision_list)

average_auc = np.mean(auc_list) if auc_list else None
sd_auc = np.std(auc_list) if auc_list else None

# Output results
print(f"Average Accuracy: {average_accuracy:.4f} ± {sd_accuracy:.4f}")
print(f"Average Sensitivity (Recall): {average_sensitivity:.4f} ± {sd_sensitivity:.4f}")
print(f"Average Specificity: {average_specificity:.4f} ± {sd_specificity:.4f}")
print(f"Average Precision: {average_precision:.4f} ± {sd_precision:.4f}")
if auc_list:
    print(f"Average AUC: {average_auc:.4f} ± {sd_auc:.4f}")
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np

# Define the objective function for Optuna optimization
def objective(trial):
    # Hyperparameter search space for Logistic Regression
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga', 'lbfgs'])  # Solver choices
    penalty = 'none' if solver in ['saga', 'lbfgs'] else 'l2'  # Ensure penalty compatibility

    param = {
        'C': trial.suggest_loguniform('C', 1e-5, 1e5),  # Regularization strength
        'penalty': penalty,
        'solver': solver,
        'max_iter': trial.suggest_int('max_iter', 100, 500),  # Max iterations
        'tol': trial.suggest_loguniform('tol', 1e-5, 1e-2),  # Tolerance
    }

    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    accuracy_list = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train the Logistic Regression model with the trial's hyperparameters
        model = LogisticRegression(**param, random_state=42)
        model.fit(X_train, y_train)

        # Predictions and scoring
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_list.append(accuracy)

    return np.mean(accuracy_list)

# Run the optimization with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best hyperparameters found by Optuna
best_params = study.best_params
print(f"Best hyperparameters found: {best_params}")

# Stratified K-Fold CV with the best hyperparameters
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
accuracy_list = []
sensitivity_list = []
specificity_list = []
precision_list = []
auc_list = []

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train Logistic Regression with the best parameters
    model = LogisticRegression(**best_params, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and scoring
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for AUC calculation

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    auc = roc_auc_score(y_test, y_prob)

    # Store metrics for each fold
    accuracy_list.append(accuracy)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    precision_list.append(precision)
    auc_list.append(auc)

# Calculate averages and standard deviations
average_accuracy = np.mean(accuracy_list)
sd_accuracy = np.std(accuracy_list)

average_sensitivity = np.mean(sensitivity_list)
sd_sensitivity = np.std(sensitivity_list)

average_specificity = np.mean(specificity_list)
sd_specificity = np.std(specificity_list)

average_precision = np.mean(precision_list)
sd_precision = np.std(precision_list)

average_auc = np.mean(auc_list)
sd_auc = np.std(auc_list)

# Output results
print(f"Average Accuracy: {average_accuracy:.4f} ± {sd_accuracy:.4f}")
print(f"Average Sensitivity (Recall): {average_sensitivity:.4f} ± {sd_sensitivity:.4f}")
print(f"Average Specificity: {average_specificity:.4f} ± {sd_specificity:.4f}")
print(f"Average Precision: {average_precision:.4f} ± {sd_precision:.4f}")
print(f"Average AUC: {average_auc:.4f} ± {sd_auc:.4f}")