# =====================================
# IMPORT LIBRARIES
# =====================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# =====================================
# LOAD TRAINED MODEL
# =====================================

model = joblib.load("stress_rf_model.pkl")

# =====================================
# LOAD NEW SIGNAL FILE
# =====================================

data = pd.read_csv("ML_ready_signals.csv")

# =====================================
# ALIGN FEATURES TO WHAT MODEL EXPECTS
# =====================================

expected_features = model.feature_names_in_  # works for sklearn RF/SVM
missing = set(expected_features) - set(data.columns)
if missing:
    raise ValueError(f"Missing features in input data: {missing}")

X = data[expected_features]  # ensures correct column order

# =====================================
# RUN PREDICTIONS
# =====================================

predictions   = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]  # probability of stress class

# =====================================
# RECORDING SUMMARY
# =====================================

avg_stress_probability = np.mean(probabilities)
percent_stress         = np.mean(predictions) * 100

print("\n====== RECORDING SUMMARY ======")
print(f"Average Stress Probability      : {avg_stress_probability:.3f}")
print(f"Percent Classified as Stress    : {percent_stress:.2f}%")

# =====================================
# TIME AXIS
# =====================================

time = data["time_seconds"] if "time_seconds" in data.columns else np.arange(len(probabilities))

# =====================================
# SMOOTH PROBABILITY
# =====================================

smooth_window = 200

stress_smooth = (
    pd.Series(probabilities)
    .rolling(smooth_window, center=True, min_periods=1)
    .mean()
)

# =====================================
# PLOT RESULTS
# =====================================

plt.figure(figsize=(12, 5))

plt.plot(time, stress_smooth, label="Stress Probability (smoothed)")  # ← fixed: was probabilities
plt.plot(time, probabilities, alpha=0.25, linewidth=0.8, label="Raw Probability")  # optional background
plt.axhline(avg_stress_probability, color='red', linestyle="--", label="Average Stress")

plt.title("Stress Prediction Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Stress Probability")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
