# STRESS — Signal Tracking and Recognition of Emotional Stress State
**Senior Design Project | Team 305**

---

## Overview
STRESS is a multimodal biosignal analysis system designed to detect and quantify human stress levels using physiological data from wearable sensors. The system integrates EEG (brain activity) and fNIRS (blood oxygenation) signals, applies signal processing techniques, and uses machine learning to generate continuous stress probability outputs.

Rather than relying only on binary classification, this system models stress as a continuous and dynamic state, allowing for more realistic and interpretable results.

---

## Project Goals
- Detect stress using wearable sensor data
- Convert raw EEG and fNIRS signals into meaningful features
- Train a machine learning model to classify calm vs stress states
- Output probability-based stress levels
- Provide an interactive dashboard for visualization

---

## Key Features
- Multimodal signal fusion (EEG + fNIRS)
- End-to-end pipeline (raw data → prediction → visualization)
- Probability-based stress detection
- Signal smoothing for stability
- Streamlit dashboard for user interaction
- Leave-One-Recording-Out validation for robust evaluation

---

## System Pipeline

Raw Muse Data (EEG + fNIRS)  
↓  
Signal Extraction  
↓  
Filtering and Feature Engineering  
↓  
Feature Normalization  
↓  
Machine Learning Model (Random Forest)  
↓  
Stress Probability Output  
↓  
Visualization Dashboard  

---

## Tech Stack

Languages & Libraries:
- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

Concepts:
- Digital Signal Processing
- EEG bandpower feature extraction
- Multimodal data fusion
- Supervised machine learning
- Cross-validation
- Time-series smoothing

---

## Project Structure

├── Full_Muse_to_Signals.py  
├── ML_Model.py  
├── Stress_Prediction.py  
├── Dashboard.py  
├── stress_rf_model_package.pkl  
├── ML_ready_signals.csv  
├── ML_model_input.csv  
├── loro_fold_results.csv  

---

## How It Works

### 1. Signal Processing
- Extracts EEG and fNIRS data from raw Muse recordings
- Applies bandpass filtering (0.5–40 Hz)
- Separates EEG into frequency bands:
  - Alpha (8–15 Hz)
  - Beta (15–30 Hz)
  - Theta (4–8 Hz)
- Computes:
  - Band power
  - Alpha/Beta ratio
  - Theta/Beta ratio
- Aligns EEG and fNIRS using timestamps

---

### 2. Feature Engineering
- Applies baseline normalization
- Uses z-score normalization per recording
- Generates stable and comparable features across sessions

---

### 3. Model Training
- Model: Random Forest Classifier
- Labels:
  - 0 = Calm
  - 1 = Stress
- Validation:
  - Leave-One-Recording-Out
- Uses class balancing and an adjusted threshold (0.65)

---

### 4. Prediction
- Outputs stress probability over time
- Applies smoothing to reduce noise
- Generates summary metrics:
  - Average stress
  - Peak stress
  - Percent time in stress

---

### 5. Dashboard
- Upload raw Muse recording
- Upload trained model package
- Visualize stress probability over time
- Highlight stress intervals
- View session metrics
- Download results

---

## Output Interpretation

Instead of:
0 = Calm  
1 = Stress  

The system outputs:

Stress Probability (0.0 → 1.0)

Stress Levels:
- Low: 0.00 – 0.39
- Moderate: 0.40 – 0.79
- High: 0.80 – 1.00

---

## How to Run

1. Install dependencies:
pip install pandas numpy matplotlib scikit-learn streamlit scipy joblib

2. Run signal pipeline:
python Full_Muse_to_Signals.py

3. Train model:
python ML_Model.py

4. Run predictions:
python Stress_Prediction.py

5. Launch dashboard:
streamlit run Dashboard.py

---

## Results
- Strong performance with clean signals
- Noise affects predictions
- Smoothing improves stability
- Multimodal data improves robustness

---

## Limitations
- Limited dataset size
- Not clinically validated
- Sensitive to noise and artifacts
- Real-time streaming limitations

---

## Future Work
- Collect more data
- Implement deep learning models (LSTM, CNN)
- Improve real-time processing
- Better noise filtering
- Personalized stress modeling

---

## Team
Team 305  
Signal Tracking and Recognition of Emotional Stress State (STRESS)

Roles:
- Software and Program Engineer
- Signal Processing Engineer
- Machine Learning Engineer

---

## License
Academic and research use only.

---

## Summary
This project demonstrates a full pipeline that converts raw biosignals into meaningful stress probability outputs. By treating stress as a continuous variable instead of a binary state, the system provides more realistic and interpretable insights.
