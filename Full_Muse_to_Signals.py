import pandas as pd
import numpy as np
from scipy import signal


# ═══════════════════════════════════════════════════════════════
#  PARAMETERS
# ═══════════════════════════════════════════════════════════════

GLOBAL_BAND = (0.5, 40)
ALPHA_BAND  = (8, 15)
BETA_BAND   = (15, 30)
THETA_BAND = (4, 8)   # add this near the top with your other band constants
SMOOTH_SEC  = 5        # smoothing window in seconds
FILTER_ORDER = 4


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def bandpass_filter(data: np.ndarray, fs: float,
                    lowcut: float, highcut: float,
                    order: int = FILTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq  = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, data)


# ═══════════════════════════════════════════════════════════════
#  STEP 1 — EXTRACT RAW EEG
# ═══════════════════════════════════════════════════════════════

def extract_eeg(df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw EEG samples from the Muse OSC recording."""
    eeg_rows = df[df['osc_address'] == 'samples/eeg'].copy()
    print(f"  EEG rows found : {len(eeg_rows)}")

    channels = (eeg_rows['osc_data']
                .str.strip('"')
                .str.split(',', expand=True))
    channels.columns = ['TP9', 'AF7', 'AF8', 'TP10']
    channels = channels.astype(float)

    out = pd.DataFrame({
        'timestamp'   : eeg_rows['timestamp'].values,
        'time_seconds': None,                           # filled below
        'TP9'         : channels['TP9'].values,
        'AF7'         : channels['AF7'].values,
        'AF8'         : channels['AF8'].values,
        'TP10'        : channels['TP10'].values,
    })
    out['time_seconds'] = (out['timestamp'] - out['timestamp'].iloc[0]) / 1000.0
    return out.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
#  STEP 2 — EXTRACT RAW fNIRS
# ═══════════════════════════════════════════════════════════════

FNIRS_COLS = [
    '730nm HbR left outer',  '730nm HbR right outer',
    '850nm HbO left outer',  '850nm HbO right outer',
    '730nm HbR left inner',  '730nm HbR right inner',
    '850nm HbO left inner',  '850nm HbO right inner',
]

def extract_fnirs(df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw fNIRS (optics) samples from the Muse OSC recording."""
    fnirs_rows = df[df['osc_address'] == 'samples/optics'].copy()
    print(f"  fNIRS rows found: {len(fnirs_rows)}")

    channels = (fnirs_rows['osc_data']
                .str.strip('"')
                .str.split(',', expand=True))
    channels.columns = FNIRS_COLS
    channels = channels.astype(float)

    out = pd.DataFrame({'timestamp': fnirs_rows['timestamp'].values})
    for col in FNIRS_COLS:
        out[col] = channels[col].values
    out['time_seconds'] = (out['timestamp'] - out['timestamp'].iloc[0]) / 1000.0
    return out[['timestamp', 'time_seconds'] + FNIRS_COLS].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
#  STEP 3 — FILTER EEG → ALPHA / BETA FEATURES
# ═══════════════════════════════════════════════════════════════

EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']

def filter_eeg(eeg: pd.DataFrame) -> pd.DataFrame:
    time_seconds = eeg['time_seconds'].values
    fs = 1.0 / np.mean(np.diff(time_seconds))
    print(f"  EEG sampling rate: {fs:.2f} Hz")

    window_samples = max(1, int(SMOOTH_SEC * fs))
    features = {
        'timestamp'   : eeg['timestamp'].values,
        'time_seconds': time_seconds,
    }

    for ch in EEG_CHANNELS:
        raw     = eeg[ch].values - eeg[ch].mean()
        global_ = bandpass_filter(raw, fs, *GLOBAL_BAND)
        alpha   = bandpass_filter(global_, fs, *ALPHA_BAND)
        beta    = bandpass_filter(global_, fs, *BETA_BAND)
        theta   = bandpass_filter(global_, fs, *THETA_BAND)

        alpha_pwr = (pd.Series(alpha ** 2)
                    .rolling(window_samples, center=True, min_periods=1)
                    .mean().values)
        beta_pwr  = (pd.Series(beta ** 2)
                    .rolling(window_samples, center=True, min_periods=1)
                    .mean().values)
        theta_pwr = (pd.Series(theta ** 2)
                    .rolling(window_samples, center=True, min_periods=1)
                    .mean().values)

        baseline_mask  = time_seconds < 30.0
        alpha_baseline = np.mean(alpha_pwr[baseline_mask]) + 1e-10
        beta_baseline  = np.mean(beta_pwr[baseline_mask])  + 1e-10
        theta_baseline = np.mean(theta_pwr[baseline_mask]) + 1e-10

        alpha_pwr_norm = alpha_pwr / alpha_baseline
        beta_pwr_norm  = beta_pwr  / beta_baseline
        theta_pwr_norm = theta_pwr / theta_baseline

        features[f'{ch}_alpha_power']      = alpha_pwr_norm
        features[f'{ch}_beta_power']       = beta_pwr_norm
        features[f'{ch}_theta_power']      = theta_pwr_norm
        features[f'{ch}_alpha_beta_ratio'] = alpha_pwr_norm / beta_pwr_norm
        features[f'{ch}_theta_beta_ratio'] = theta_pwr_norm / beta_pwr_norm

    return pd.DataFrame(features)  # ← outside the loop


# ═══════════════════════════════════════════════════════════════
#  STEP 4 — MERGE & DROP INCOMPLETE ROWS
# ═══════════════════════════════════════════════════════════════

EEG_FEATURE_COLS = [
    f'{ch}_{feat}'
    for ch   in EEG_CHANNELS
    for feat in ('alpha_power', 'beta_power', 'theta_power',
                 'alpha_beta_ratio', 'theta_beta_ratio')
]

def merge_and_clean(eeg_features: pd.DataFrame,
                    fnirs: pd.DataFrame) -> pd.DataFrame:
    """
    Outer-join EEG features and fNIRS on timestamp, then drop any row
    that is missing values in either modality.
    """
    combined = pd.merge(
        eeg_features, fnirs,
        on='timestamp', how='outer',
        suffixes=('_eeg', '_fnirs')
    )

    # Reconcile duplicate time_seconds columns introduced by the merge
    combined['time_seconds'] = (combined['time_seconds_eeg']
                                .fillna(combined['time_seconds_fnirs']))
    combined.drop(columns=['time_seconds_eeg', 'time_seconds_fnirs'],
                  inplace=True)
    combined.sort_values('timestamp', inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # Keep only rows where every EEG feature AND every fNIRS channel exist
    before = len(combined)
    ml_ready = combined.dropna(subset=EEG_FEATURE_COLS + FNIRS_COLS).reset_index(drop=True)
    print(f"  Rows before drop: {before:,}  →  ML-ready rows: {len(ml_ready):,}")

    # Tidy column order
    return ml_ready[['timestamp', 'time_seconds'] + EEG_FEATURE_COLS + FNIRS_COLS]


# ═══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(input_file: str,
                 output_file: str = 'ML_ready_signals.csv') -> pd.DataFrame:
    """
    Full Muse EEG + fNIRS pipeline.

    Parameters
    ----------
    input_file  : path to the raw Muse SDK CSV recording
    output_file : destination path for the cleaned, ML-ready CSV

    Returns
    -------
    ml_ready    : the final DataFrame (also saved to output_file)
    """
    print(f"\n{'='*60}")
    print(f" Muse EEG + fNIRS Pipeline")
    print(f"{'='*60}")

    print("\n[1/4] Reading raw recording …")
    raw = pd.read_csv(input_file)
    print(f"  Total rows in file: {len(raw):,}")

    print("\n[2/4] Extracting EEG & fNIRS …")
    eeg   = extract_eeg(raw)
    fnirs = extract_fnirs(raw)

    print("\n[3/4] Filtering EEG (alpha / beta) …")
    eeg_features = filter_eeg(eeg)

    print("\n[4/4] Merging modalities & removing incomplete rows …")
    ml_ready = merge_and_clean(eeg_features, fnirs)

    ml_ready.to_csv(output_file, index=False)
    print(f"\n{'='*60}")
    print(f" Done! Output shape : {ml_ready.shape}")
    print(f" Saved to           : {output_file}")
    print(f"{'='*60}\n")

    return ml_ready


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    INPUT_FILE  = 'muselab_recording(4).csv'
    OUTPUT_FILE = 'ML_ready_signals.csv'

    ml_ready = run_pipeline(INPUT_FILE, OUTPUT_FILE)
