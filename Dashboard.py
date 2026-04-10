import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy import signal
import tempfile
import os

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Stress Prediction Dashboard",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

GLOBAL_BAND   = (0.5, 40)
THETA_BAND    = (4, 8)
ALPHA_BAND    = (8, 15)
BETA_BAND     = (15, 30)

SMOOTH_SEC    = 5
FILTER_ORDER  = 4
BASELINE_SEC  = 30.0
EPS           = 1e-6

MERGE_TOLERANCE_SEC = 0.20

EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']

FNIRS_COLS = [
    '730nm HbR left inner',
    '730nm HbR right inner',
    '850nm HbO left inner',
    '850nm HbO right inner'
]

NON_FEATURE_COLS = ['timestamp', 'time_seconds']

EEG_FEATURE_COLS = [
    f'{ch}_{feat}'
    for ch in EEG_CHANNELS
    for feat in (
        'alpha_power',
        'beta_power',
        'theta_power',
        'alpha_beta_ratio',
        'theta_beta_ratio'
    )
]

MODEL_FEATURE_COLS = EEG_FEATURE_COLS + FNIRS_COLS


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def bandpass_filter(data: np.ndarray, fs: float,
                    lowcut: float, highcut: float,
                    order: int = FILTER_ORDER) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def rolling_power(x: np.ndarray, window_samples: int) -> np.ndarray:
    return (
        pd.Series(x ** 2)
        .rolling(window_samples, center=True, min_periods=1)
        .mean()
        .values
    )


def zscore_per_recording(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    means = df[feature_cols].mean()
    stds = df[feature_cols].std().replace(0, EPS).fillna(EPS)
    df[feature_cols] = (df[feature_cols] - means) / (stds + EPS)
    return df


# ═══════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# These preserve your existing feature-generation logic
# ═══════════════════════════════════════════════════════════════

def extract_eeg(df: pd.DataFrame) -> pd.DataFrame:
    eeg_rows = df[df['osc_address'] == '/muse/eeg'].copy()

    if eeg_rows.empty:
        raise ValueError("No EEG rows found in the file.")

    channels = (
        eeg_rows['osc_data']
        .astype(str)
        .str.strip('"')
        .str.split(',', expand=True)
    )

    if channels.shape[1] < 4:
        raise ValueError(
            f"Not enough EEG columns found. Expected at least 4, got {channels.shape[1]}."
        )

    channels = channels.iloc[:, :4].apply(pd.to_numeric, errors='coerce')
    channels.columns = EEG_CHANNELS

    out = pd.DataFrame({
        'timestamp': eeg_rows['timestamp'].values,
        'TP9': channels['TP9'].values,
        'AF7': channels['AF7'].values,
        'AF8': channels['AF8'].values,
        'TP10': channels['TP10'].values,
    })

    out = out.dropna().reset_index(drop=True)
    out['time_seconds'] = (out['timestamp'] - out['timestamp'].iloc[0]) / 1000.0

    return out[['timestamp', 'time_seconds'] + EEG_CHANNELS]


def extract_fnirs(df: pd.DataFrame) -> pd.DataFrame:
    fnirs_rows = df[df['osc_address'] == '/muse/optics'].copy()

    if fnirs_rows.empty:
        raise ValueError("No fNIRS rows found in the file.")

    channels = (
        fnirs_rows['osc_data']
        .astype(str)
        .str.strip('"')
        .str.split(',', expand=True)
    )

    if channels.shape[1] >= 4:
        channels = channels.iloc[:, :4]
    elif channels.shape[1] >= 4:
        channels = channels.iloc[:, :4]
    else:
        raise ValueError(
            f"Not enough fNIRS columns found. Expected at least 4, got {channels.shape[1]}."
        )

    channels = channels.apply(pd.to_numeric, errors='coerce')
    channels.columns = FNIRS_COLS

    out = pd.DataFrame({
        'timestamp': fnirs_rows['timestamp'].values
    })

    for col in FNIRS_COLS:
        out[col] = channels[col].values

    out = out.dropna().reset_index(drop=True)
    out['time_seconds'] = (out['timestamp'] - out['timestamp'].iloc[0]) / 1000.0

    return out[['timestamp', 'time_seconds'] + FNIRS_COLS]


def filter_eeg(eeg: pd.DataFrame) -> pd.DataFrame:
    eeg = eeg.copy()
    time_seconds = eeg['time_seconds'].values

    if len(time_seconds) < 2:
        raise ValueError("Not enough EEG samples to estimate sampling rate.")

    fs = 1.0 / np.mean(np.diff(time_seconds))
    window_samples = max(1, int(SMOOTH_SEC * fs))

    features = {
        'timestamp': eeg['timestamp'].values,
        'time_seconds': time_seconds,
    }

    baseline_mask = time_seconds < BASELINE_SEC
    if baseline_mask.sum() < 5:
        baseline_mask = np.ones_like(time_seconds, dtype=bool)

    for ch in EEG_CHANNELS:
        raw = eeg[ch].values.astype(float)
        raw = raw - np.nanmean(raw)

        global_sig = bandpass_filter(raw, fs, *GLOBAL_BAND)
        alpha_sig  = bandpass_filter(global_sig, fs, *ALPHA_BAND)
        beta_sig   = bandpass_filter(global_sig, fs, *BETA_BAND)
        theta_sig  = bandpass_filter(global_sig, fs, *THETA_BAND)

        alpha_pwr = rolling_power(alpha_sig, window_samples)
        beta_pwr  = rolling_power(beta_sig, window_samples)
        theta_pwr = rolling_power(theta_sig, window_samples)

        alpha_baseline = np.mean(alpha_pwr[baseline_mask]) + EPS
        beta_baseline  = np.mean(beta_pwr[baseline_mask]) + EPS
        theta_baseline = np.mean(theta_pwr[baseline_mask]) + EPS

        alpha_pwr_norm = alpha_pwr / alpha_baseline
        beta_pwr_norm  = beta_pwr / beta_baseline
        theta_pwr_norm = theta_pwr / theta_baseline

        features[f'{ch}_alpha_power'] = alpha_pwr_norm
        features[f'{ch}_beta_power'] = beta_pwr_norm
        features[f'{ch}_theta_power'] = theta_pwr_norm
        features[f'{ch}_alpha_beta_ratio'] = alpha_pwr_norm / (beta_pwr_norm + EPS)
        features[f'{ch}_theta_beta_ratio'] = theta_pwr_norm / (beta_pwr_norm + EPS)

    return pd.DataFrame(features)


def merge_modalities(eeg_features: pd.DataFrame,
                     fnirs: pd.DataFrame,
                     tolerance_sec: float = MERGE_TOLERANCE_SEC) -> pd.DataFrame:
    eeg_features = eeg_features.sort_values('timestamp').reset_index(drop=True)
    fnirs = fnirs.sort_values('timestamp').reset_index(drop=True)

    tolerance_ms = int(tolerance_sec * 1000)

    merged = pd.merge_asof(
        eeg_features,
        fnirs[['timestamp'] + FNIRS_COLS],
        on='timestamp',
        direction='nearest',
        tolerance=tolerance_ms
    )

    merged = merged.dropna(subset=MODEL_FEATURE_COLS).reset_index(drop=True)

    return merged[['timestamp', 'time_seconds'] + MODEL_FEATURE_COLS]


def prepare_model_input(ml_ready: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in MODEL_FEATURE_COLS if c not in ml_ready.columns]
    if missing:
        raise ValueError(f"Missing expected model features: {missing}")

    model_df = ml_ready[['timestamp', 'time_seconds'] + MODEL_FEATURE_COLS].copy()
    model_df = zscore_per_recording(model_df, MODEL_FEATURE_COLS)
    return model_df


def run_pipeline_from_dataframe(raw: pd.DataFrame):
    eeg = extract_eeg(raw)
    fnirs = extract_fnirs(raw)
    eeg_features = filter_eeg(eeg)
    ml_ready = merge_modalities(eeg_features, fnirs)
    model_input = prepare_model_input(ml_ready)
    return ml_ready, model_input


# ═══════════════════════════════════════════════════════════════
# PREDICTION + DASHBOARD PREP
# ═══════════════════════════════════════════════════════════════

def get_stress_intervals(time_values, labels):
    intervals = []
    in_stress = False
    start_time = None

    for i in range(len(labels)):
        if labels[i] == 1 and not in_stress:
            start_time = time_values[i]
            in_stress = True
        elif labels[i] == 0 and in_stress:
            end_time = time_values[i - 1]
            intervals.append((start_time, end_time))
            in_stress = False

    if in_stress:
        intervals.append((start_time, time_values[-1]))

    return intervals


def run_prediction(model_input: pd.DataFrame, model_package):
    model = model_package["model"]
    expected_features = model_package["feature_cols"]
    decision_threshold = model_package.get("decision_threshold", 0.5)

    missing = set(expected_features) - set(model_input.columns)
    extra = set(model_input.columns) - set(expected_features) - {"timestamp", "time_seconds"}

    if missing:
        raise ValueError(f"Missing features in input data: {sorted(missing)}")

    X = model_input[expected_features].copy()

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= decision_threshold).astype(int)

    if "time_seconds" in model_input.columns and len(model_input) > 1:
        dt = np.mean(np.diff(model_input["time_seconds"]))
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

    smoothed_predictions = (stress_smooth >= decision_threshold).astype(int)

    time = (
        model_input["time_seconds"].values
        if "time_seconds" in model_input.columns
        else np.arange(len(probabilities))
    )

    results = pd.DataFrame({
        "timestamp": model_input["timestamp"].values if "timestamp" in model_input.columns else np.arange(len(probabilities)),
        "time_seconds": time,
        "stress_probability_raw": probabilities,
        "stress_probability_smoothed": stress_smooth,
        "predicted_label_raw": predictions,
        "predicted_label_smoothed": smoothed_predictions
    })

    summary = {
        "decision_threshold": decision_threshold,
        "avg_stress_probability_raw": float(np.mean(probabilities)),
        "avg_stress_probability_smoothed": float(np.mean(stress_smooth)),
        "percent_stress_raw": float(np.mean(predictions) * 100),
        "percent_stress_smoothed": float(np.mean(smoothed_predictions) * 100),
        "peak_stress_raw": float(np.max(probabilities)),
        "peak_stress_smoothed": float(np.max(stress_smooth)),
        "session_length_sec": float(time[-1] - time[0]) if len(time) > 1 else 0.0,
        "num_samples": int(len(results)),
        "extra_columns_ignored": sorted(extra) if extra else []
    }

    return results, summary


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def make_main_plot(results: pd.DataFrame, decision_threshold: float):
    fig, ax = plt.subplots(figsize=(16, 6))

    time = results["time_seconds"].values
    raw = results["stress_probability_raw"].values
    smooth = results["stress_probability_smoothed"].values
    labels = results["predicted_label_smoothed"].values

    intervals = get_stress_intervals(time, labels)

    for start, end in intervals:
        ax.axvspan(start, end, alpha=0.18)

    ax.plot(time, smooth, linewidth=2.5, label="Stress Probability (smoothed)")
    ax.plot(time, raw, alpha=0.25, linewidth=1.0, label="Raw Probability")

    avg_smoothed_probability = np.mean(smooth)
    ax.axhline(avg_smoothed_probability, linestyle='--', linewidth=2, label=f"Average Smoothed Stress ({avg_smoothed_probability:.2f})")

    peak_idx = np.argmax(smooth)
    peak_time = time[peak_idx]
    peak_value = smooth[peak_idx]

    ax.scatter([peak_time], [peak_value], s=100, label="Peak Stress")
    ax.annotate(
        f"Peak @ {peak_time:.1f}s",
        (peak_time, peak_value),
        xytext=(10, 10),
        textcoords="offset points"
    )

    ax.set_title("Stress Prediction Over Time", fontsize=16)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Stress Probability")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig


def make_distribution_plot(summary):
    calm = max(0.0, 100.0 - summary["percent_stress_smoothed"])
    stress = summary["percent_stress_smoothed"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [calm, stress],
        labels=["Calm", "Stress"],
        autopct="%1.1f%%"
    )
    ax.set_title("Session Distribution")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# APP UI
# ═══════════════════════════════════════════════════════════════

st.title("Stress Prediction Session Dashboard")
st.markdown("Upload a raw Muse recording file and your trained model package to generate the full session dashboard.")

left, right = st.columns(2)

with left:
    recording_file = st.file_uploader(
        "Upload raw Muse recording CSV",
        type=["csv"]
    )

with right:
    model_file = st.file_uploader(
        "Upload trained model package (.pkl)",
        type=["pkl"]
    )

run_button = st.button("Run Full Dashboard", use_container_width=True)

if run_button:
    if recording_file is None:
        st.error("Please upload the raw Muse recording CSV.")
        st.stop()

    if model_file is None:
        st.error("Please upload the trained model package (.pkl).")
        st.stop()

    try:
        with st.spinner("Reading recording..."):
            raw_df = pd.read_csv(recording_file)

        with st.spinner("Loading model package..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_model:
                tmp_model.write(model_file.read())
                tmp_model_path = tmp_model.name

            model_package = joblib.load(tmp_model_path)

        with st.spinner("Running signal pipeline..."):
            ml_ready, model_input = run_pipeline_from_dataframe(raw_df)

        with st.spinner("Running stress prediction..."):
            results, summary = run_prediction(model_input, model_package)

        if os.path.exists(tmp_model_path):
            os.remove(tmp_model_path)

        st.success("Dashboard generated successfully.")

        # Top metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Average Stress", f"{summary['avg_stress_probability_smoothed']:.2f}")
        col2.metric("Peak Stress", f"{summary['peak_stress_smoothed']:.2f}")
        col3.metric("Time in Stress", f"{summary['percent_stress_smoothed']:.1f}%")
        col4.metric("Session Length", f"{summary['session_length_sec']:.1f}s")

        # Main plot
        st.subheader("Stress Score Over Time")
        main_fig = make_main_plot(results, summary["decision_threshold"])
        st.pyplot(main_fig, use_container_width=True)

        # Interpretation + distribution
        colA, colB = st.columns([2, 1])

        with colA:
            st.subheader("Session Interpretation")

            avg = summary["avg_stress_probability_smoothed"]
            pct = summary["percent_stress_smoothed"]

            if avg < 0.35:
                interpretation = "This recording appeared mostly calm, with limited periods of elevated stress."
            elif avg < 0.65:
                interpretation = "This recording showed moderate stress activity, with some portions crossing the stress threshold."
            else:
                interpretation = "This recording showed sustained elevated stress activity across a large portion of the session."

            st.write(f"**Average smoothed stress level:** {avg:.3f}")
            st.write(f"**Percent of session classified as stress:** {pct:.2f}%")
            st.write(interpretation)
            st.caption("Highlighted regions on the graph indicate intervals classified as stress using the smoothed prediction.")

        with colB:
            dist_fig = make_distribution_plot(summary)
            st.pyplot(dist_fig, use_container_width=True)

        # Optional details
        with st.expander("Show Model Input Data"):
            st.dataframe(model_input, use_container_width=True)

        with st.expander("Show Prediction Results"):
            st.dataframe(results, use_container_width=True)

        # Downloads
        st.subheader("Download Results")
        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                label="Download Model Input CSV",
                data=model_input.to_csv(index=False).encode("utf-8"),
                file_name="ML_model_input.csv",
                mime="text/csv"
            )

        with d2:
            st.download_button(
                label="Download Prediction Results CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="stress_prediction_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {str(e)}")
