# ================================
# Import Libraries
# ================================

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# ================================
# Load Dataset
# ================================

data = pd.read_csv("ML_clustered_signals_FULL (1).csv")

# ================================
# 1. FEATURES
# ================================

feature_cols = data.drop(columns=['timestamp', 'time_seconds', 'cluster']).columns
X = data[feature_cols]

# ================================
# 2. BINARY LABEL
# ================================

# Convert clusters into binary classes
# 0 stays 0
# 1 and 2 become 1
y = data["cluster"].apply(lambda x: 0 if x == 0 else 1)

# ================================
# 3. TRAIN / TEST SPLIT
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================================
# 4. RANDOM FOREST MODEL
# ================================

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
joblib.dump(rf, "stress_rf_model.pkl")

# ================================
# 5. PREDICTIONS
# ================================

y_pred = rf.predict(X_test)

# ================================
# 6. CONFUSION MATRIX
# ================================

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("\n=== Confusion Matrix ===")
print(f"TP = {TP}")
print(f"FP = {FP}")
print(f"FN = {FN}")
print(f"TN = {TN}")

# ================================
# 7. CLASSIFICATION REPORT
# ================================

print("\n=== Classification Report ===")

target_names = ["Class 0", "Class 1"]

print(classification_report(y_test, y_pred, target_names=target_names))
