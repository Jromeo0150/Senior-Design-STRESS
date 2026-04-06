import pandas as pd
import numpy as np
from scipy import signal

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

# Maximum time difference allowed when aligning EEG and fNIRS
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
    """Zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def rolling_power(x: np.ndarray, window_samples: int) -> np.ndarray:
    """Rolling mean power of a signal."""
    return (
        pd.Series(x ** 2)
        .rolling(window_samples, center=True, min_periods=1)
        .mean()
        .values
    )


def zscore_per_recording(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Apply the same per-recording z-score normalization used in model training.
    For a single new file, this is effectively per-file normalization.
    """
    df = df.copy()
    means = df[feature_cols].mean()
    stds = df[feature_cols].std().replace(0, EPS).fillna(EPS)
    df[feature_cols] = (df[feature_cols] - means) / (stds + EPS)
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 1 — EXTRACT RAW EEG
# ═══════════════════════════════════════════════════════════════

def extract_eeg(df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw EEG samples from the Muse OSC recording."""
    eeg_rows = df[df['osc_address'] == '/muse/eeg'].copy()
    print(f"  EEG rows found : {len(eeg_rows)}")

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


# ═══════════════════════════════════════════════════════════════
# STEP 2 — EXTRACT RAW fNIRS
# ═══════════════════════════════════════════════════════════════

def extract_fnirs(df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw fNIRS (optics) samples from the Muse OSC recording."""
    fnirs_rows = df[df['osc_address'] == '/muse/optics'].copy()
    print(f"  fNIRS rows found: {len(fnirs_rows)}")

    if fnirs_rows.empty:
        raise ValueError("No fNIRS rows found in the file.")

    channels = (
        fnirs_rows['osc_data']
        .astype(str)
        .str.strip('"')
        .str.split(',', expand=True)
    )

    # Your earlier comment suggested the useful inner channels are columns 4:8.
    # This version uses 4:8 when available. If your actual export format differs,
    # this is one of the first things to verify.
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


# ═══════════════════════════════════════════════════════════════
# STEP 3 — FILTER EEG → FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def filter_eeg(eeg: pd.DataFrame) -> pd.DataFrame:
    """Compute EEG bandpower and ratio features."""
    eeg = eeg.copy()
    time_seconds = eeg['time_seconds'].values

    if len(time_seconds) < 2:
        raise ValueError("Not enough EEG samples to estimate sampling rate.")

    fs = 1.0 / np.mean(np.diff(time_seconds))
    print(f"  EEG sampling rate: {fs:.2f} Hz")

    window_samples = max(1, int(SMOOTH_SEC * fs))

    features = {
        'timestamp': eeg['timestamp'].values,
        'time_seconds': time_seconds,
    }

    baseline_mask = time_seconds < BASELINE_SEC
    if baseline_mask.sum() < 5:
        print("  Warning: very short baseline window detected. Using full recording as baseline.")
        baseline_mask = np.ones_like(time_seconds, dtype=bool)

    for ch in EEG_CHANNELS:
        raw = eeg[ch].values.astype(float)

        # Remove DC offset before filtering
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


# ═══════════════════════════════════════════════════════════════
# STEP 4 — ALIGN EEG + fNIRS
# ═══════════════════════════════════════════════════════════════

def merge_modalities(eeg_features: pd.DataFrame,
                     fnirs: pd.DataFrame,
                     tolerance_sec: float = MERGE_TOLERANCE_SEC) -> pd.DataFrame:
    """
    Align EEG and fNIRS using nearest-timestamp merge instead of exact outer join.
    This is usually much more appropriate for multimodal biosignal streams.
    """
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

    before = len(merged)
    merged = merged.dropna(subset=MODEL_FEATURE_COLS).reset_index(drop=True)
    after = len(merged)

    print(f"  Rows before alignment cleanup: {before:,} -> ML-ready rows: {after:,}")

    return merged[['timestamp', 'time_seconds'] + MODEL_FEATURE_COLS]


# ═══════════════════════════════════════════════════════════════
# STEP 5 — PREPARE MODEL INPUT
# ═══════════════════════════════════════════════════════════════

def prepare_model_input(ml_ready: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same per-recording/per-file z-score normalization
    used during model training.
    """
    missing = [c for c in MODEL_FEATURE_COLS if c not in ml_ready.columns]
    if missing:
        raise ValueError(f"Missing expected model features: {missing}")

    model_df = ml_ready[['timestamp', 'time_seconds'] + MODEL_FEATURE_COLS].copy()
    model_df = zscore_per_recording(model_df, MODEL_FEATURE_COLS)
    return model_df


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(input_file: str,
                 output_ml_ready_file: str = 'ML_ready_signals.csv',
                 output_model_input_file: str = 'ML_model_input.csv') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full Muse EEG + fNIRS pipeline.

    Returns
    -------
    ml_ready:
        Feature-engineered signals before final z-score normalization

    model_input:
        Final normalized features ready for the trained model
    """
    print(f"\n{'='*60}")
    print(" Muse EEG + fNIRS Pipeline")
    print(f"{'='*60}")

    print("\n[1/5] Reading raw recording ...")
    raw = pd.read_csv(input_file)
    print(f"  Total rows in file: {len(raw):,}")

    print("\n[2/5] Extracting EEG & fNIRS ...")
    eeg = extract_eeg(raw)
    fnirs = extract_fnirs(raw)

    print("\n[3/5] Filtering EEG and building EEG features ...")
    eeg_features = filter_eeg(eeg)

    print("\n[4/5] Aligning EEG + fNIRS ...")
    ml_ready = merge_modalities(eeg_features, fnirs)

    print("\n[5/5] Applying model normalization ...")
    model_input = prepare_model_input(ml_ready)

    ml_ready.to_csv(output_ml_ready_file, index=False)
    model_input.to_csv(output_model_input_file, index=False)

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"ML-ready output shape   : {ml_ready.shape}")
    print(f"Model-input shape       : {model_input.shape}")
    print(f"Saved ML-ready signals  : {output_ml_ready_file}")
    print(f"Saved model input       : {output_model_input_file}")
    print(f"{'='*60}\n")

    return ml_ready, model_input


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    INPUT_FILE = 'muselab_recording(10).csv'
    OUTPUT_ML_READY_FILE = 'ML_ready_signals.csv'
    OUTPUT_MODEL_INPUT_FILE = 'ML_model_input.csv'

    ml_ready, model_input = run_pipeline(
        INPUT_FILE,
        OUTPUT_ML_READY_FILE,
        OUTPUT_MODEL_INPUT_FILE
    )
