# =========================================
# 1. IMPORTS
# =========================================
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# =========================================
# 2. LOAD DATASET
# =========================================
data = pd.read_csv("ML_clustered_signals_FULL (1).csv")

# Keep only rows with a valid recording_id
data = data.dropna(subset=["recording_id"]).copy()

# =========================================
# 3. DEFINE COLUMNS
# =========================================
NON_FEATURE_COLS = ["timestamp", "time_seconds", "cluster", "recording_id", "label"]
feature_cols = [col for col in data.columns if col not in NON_FEATURE_COLS]

# Binary labels: 0 = calm, 1 = stress
data["label"] = data["cluster"].apply(lambda x: 0 if x == 0 else 1)

X_all = data[feature_cols].copy()
y_all = data["label"].copy()
groups = data["recording_id"].copy()

# =========================================
# 4. MODEL SETTINGS
# =========================================
default_threshold = 0.50
adjusted_threshold = 0.65

def build_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=200,
        min_samples_split=400,
        max_features=0.3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

# =========================================
# 5. EVALUATION FUNCTION
# =========================================
def evaluate_predictions(y_true, y_pred, name="Model"):
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    print("\nConfusion Matrix")
    print(f"TP = {TP}")
    print(f"FP = {FP}")
    print(f"FN = {FN}")
    print(f"TN = {TN}")

    print("\nClassification Report")
    print(classification_report(y_true, y_pred, digits=4))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TP": TP
    }

# =========================================
# 6. LEAVE-ONE-RECORDING-OUT EVALUATION
# =========================================
logo = LeaveOneGroupOut()

fold_results = []
all_y_true = []
all_y_prob = []
all_test_groups = []

print("\n" + "=" * 60)
print("LEAVE-ONE-RECORDING-OUT EVALUATION")
print("=" * 60)

for fold_num, (train_idx, test_idx) in enumerate(logo.split(X_all, y_all, groups), start=1):
    train_data = data.iloc[train_idx].copy()
    test_data = data.iloc[test_idx].copy()

    test_recording_id = test_data["recording_id"].iloc[0]

    X_train = train_data[feature_cols]
    y_train = train_data["label"]

    X_test = test_data[feature_cols]
    y_test = test_data["label"]

    rf = build_model()
    rf.fit(X_train, y_train)

    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)

    y_prob_test = rf.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob_test >= default_threshold).astype(int)
    y_pred_adjusted = (y_prob_test >= adjusted_threshold).astype(int)

    print(f"\n--- Fold {fold_num}: Test Recording {test_recording_id} ---")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")

    metrics_default = evaluate_predictions(
        y_test, y_pred_default,
        name=f"Recording {test_recording_id} | Threshold = {default_threshold}"
    )

    metrics_adjusted = evaluate_predictions(
        y_test, y_pred_adjusted,
        name=f"Recording {test_recording_id} | Threshold = {adjusted_threshold}"
    )

    fold_results.append({
        "fold": fold_num,
        "test_recording_id": test_recording_id,
        "train_accuracy": train_acc,
        "test_accuracy_default_threshold": metrics_default["accuracy"],
        "test_accuracy_adjusted_threshold": metrics_adjusted["accuracy"],
        "TP": metrics_adjusted["TP"],
        "FP": metrics_adjusted["FP"],
        "FN": metrics_adjusted["FN"],
        "TN": metrics_adjusted["TN"]
    })

    all_y_true.extend(y_test.tolist())
    all_y_prob.extend(y_prob_test.tolist())
    all_test_groups.extend([test_recording_id] * len(y_test))

# =========================================
# 7. OVERALL POOLED RESULTS
# =========================================
all_y_true = np.array(all_y_true)
all_y_prob = np.array(all_y_prob)

all_y_pred_default = (all_y_prob >= default_threshold).astype(int)
all_y_pred_adjusted = (all_y_prob >= adjusted_threshold).astype(int)

print("\n" + "=" * 60)
print("OVERALL POOLED RESULTS ACROSS ALL HELD-OUT RECORDINGS")
print("=" * 60)

evaluate_predictions(
    all_y_true,
    all_y_pred_default,
    name=f"Pooled | Threshold = {default_threshold}"
)

evaluate_predictions(
    all_y_true,
    all_y_pred_adjusted,
    name=f"Pooled | Threshold = {adjusted_threshold}"
)

# =========================================
# 8. FOLD SUMMARY TABLE
# =========================================
fold_results_df = pd.DataFrame(fold_results)

print("\n=== Fold Summary ===")
print(fold_results_df)

print("\n=== Mean Fold Accuracy ===")
print("Default/Stored threshold accuracy mean:",
      fold_results_df["test_accuracy_adjusted_threshold"].mean())

# =========================================
# 9. TRAIN FINAL MODEL ON ALL DATA
# =========================================
final_model = build_model()
final_model.fit(X_all, y_all)

print("\n" + "=" * 60)
print("FINAL MODEL TRAINED ON ALL RECORDINGS")
print("=" * 60)
print("Final Training Accuracy:", final_model.score(X_all, y_all))

# =========================================
# 10. FEATURE IMPORTANCE FROM FINAL MODEL
# =========================================
importances = pd.Series(final_model.feature_importances_, index=feature_cols)
print("\n=== Top 15 Features (Final Model) ===")
print(importances.sort_values(ascending=False).head(15))

# =========================================
# 11. SAVE MODEL PACKAGE
# =========================================
model_package = {
    "model": final_model,
    "feature_cols": feature_cols,
    "non_feature_cols": NON_FEATURE_COLS,
    "decision_threshold": adjusted_threshold,
    "normalization": {
        "type": "pipeline_only",
        "notes": "Feature normalization is applied in the preprocessing pipeline before training and inference."
    },
    "validation": {
        "type": "leave_one_recording_out",
        "num_recordings": int(data["recording_id"].nunique())
    }
}

joblib.dump(model_package, "stress_rf_model_package.pkl")
print("\nSaved model package to stress_rf_model_package.pkl")

# =========================================
# 12. SAVE FOLD RESULTS
# =========================================
fold_results_df.to_csv("loro_fold_results.csv", index=False)
print("Saved fold-by-fold results to loro_fold_results.csv")

# =========================================
# 13. OPTIONAL: RECORDING LABEL BALANCE CHECK
# =========================================
print("\n=== Label Balance Per Recording ===")
print(data.groupby("recording_id")["label"].value_counts(normalize=True))
