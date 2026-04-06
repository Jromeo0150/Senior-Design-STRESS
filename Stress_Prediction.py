# =====================================
# IMPORT LIBRARIES
# =====================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# =====================================
# LOAD TRAINED MODEL PACKAGE
# =====================================

model_package = joblib.load("stress_rf_model_package.pkl")

model = model_package["model"]
expected_features = model_package["feature_cols"]
decision_threshold = model_package.get("decision_threshold", 0.5)

print("\nLoaded model package.")
print(f"Decision threshold: {decision_threshold}")

# =====================================
# LOAD NORMALIZED MODEL INPUT FILE
# =====================================

data = pd.read_csv("ML_model_input.csv")

# =====================================
# CHECK REQUIRED FEATURES
# =====================================

missing = set(expected_features) - set(data.columns)
extra = set(data.columns) - set(expected_features) - {"timestamp", "time_seconds"}

if missing:
    raise ValueError(f"Missing features in input data: {sorted(missing)}")

if extra:
    print(f"Warning: extra columns found and ignored: {sorted(extra)}")

X = data[expected_features].copy()

# =====================================
# RUN PROBABILITY PREDICTIONS
# =====================================

probabilities = model.predict_proba(X)[:, 1]

# Apply saved threshold instead of default model.predict()
predictions = (probabilities >= decision_threshold).astype(int)

# =====================================
# OPTIONAL: SMOOTH PROBABILITIES
# =====================================

# If time_seconds exists, estimate window from time.
if "time_seconds" in data.columns and len(data) > 1:
    dt = np.mean(np.diff(data["time_seconds"]))
    smooth_sec = 5.0
    smooth_window = max(1, int(smooth_sec / dt))
else:
    smooth_window = 200

stress_smooth = (
    pd.Series(probabilities)
    .rolling(smooth_window, center=True, min_periods=1)
    .mean()
    .values
)

# Optional smoothed classification for summary
smoothed_predictions = (stress_smooth >= decision_threshold).astype(int)

# =====================================
# RECORDING SUMMARY
# =====================================

avg_stress_probability = np.mean(probabilities)
avg_smoothed_probability = np.mean(stress_smooth)

percent_stress_raw = np.mean(predictions) * 100
percent_stress_smoothed = np.mean(smoothed_predictions) * 100

print("\n====== RECORDING SUMMARY ======")
print(f"Average Stress Probability (raw)      : {avg_stress_probability:.3f}")
print(f"Average Stress Probability (smoothed) : {avg_smoothed_probability:.3f}")
print(f"Percent Classified as Stress (raw)    : {percent_stress_raw:.2f}%")
print(f"Percent Classified as Stress (smooth) : {percent_stress_smoothed:.2f}%")

# =====================================
# TIME AXIS
# =====================================

time = data["time_seconds"] if "time_seconds" in data.columns else np.arange(len(probabilities))

# =====================================
# PLOT RESULTS
# =====================================

plt.figure(figsize=(12, 5))

plt.plot(time, stress_smooth, label="Stress Probability (smoothed)")
plt.plot(time, probabilities, alpha=0.25, linewidth=0.8, label="Raw Probability")
plt.axhline(decision_threshold, color='orange', linestyle='--', label=f"Decision Threshold ({decision_threshold:.2f})")
plt.axhline(avg_smoothed_probability, color='red', linestyle='--', label="Average Smoothed Stress")

plt.title("Stress Prediction Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Stress Probability")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
