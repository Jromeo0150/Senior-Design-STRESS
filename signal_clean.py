import pandas as pd
import numpy as np


def extract_eeg_data(df):
    """Extract and clean EEG data from Muse recording dataframe."""
    eeg_data = df[df['osc_address'] == 'Ryan/eeg'].copy()
    print(f"Found {len(eeg_data)} EEG data points")

    eeg_channels = eeg_data['osc_data'].str.strip('"').str.split(',', expand=True)
    eeg_channels.columns = ['TP9', 'AF7', 'AF8', 'TP10']
    for col in eeg_channels.columns:
        eeg_channels[col] = eeg_channels[col].astype(float)

    eeg_clean = pd.DataFrame({
        'timestamp': eeg_data['timestamp'].values,
        'TP9': eeg_channels['TP9'].values,
        'AF7': eeg_channels['AF7'].values,
        'AF8': eeg_channels['AF8'].values,
        'TP10': eeg_channels['TP10'].values
    })

    eeg_clean['time_seconds'] = (eeg_clean['timestamp'] - eeg_clean['timestamp'].iloc[0]) / 1000.0
    eeg_clean = eeg_clean[['timestamp', 'time_seconds', 'TP9', 'AF7', 'AF8', 'TP10']]
    return eeg_clean


def extract_fnirs_data(df):
    """Extract and clean fNIRS (optics) data from Muse recording dataframe."""
    fnirs_data = df[df['osc_address'] == 'Ryan/optics'].copy()
    print(f"Found {len(fnirs_data)} fNIRS data points")

    # Each row has 8 comma-separated values:
    # Outer channels (indices 0-3): 730nm HbR Left, 730nm HbR Right, 850nm HbO Left, 850nm HbO Right
    # Inner channels (indices 4-7): 730nm HbR Left, 730nm HbR Right, 850nm HbO Left, 850nm HbO Right
    fnirs_cols = [
        '730nm HbR left outer',
        '730nm HbR right outer',
        '850nm HbO left outer',
        '850nm HbO right outer',
        '730nm HbR left inner',
        '730nm HbR right inner',
        '850nm HbO left inner',
        '850nm HbO right inner'
    ]

    fnirs_channels = fnirs_data['osc_data'].str.strip('"').str.split(',', expand=True)
    fnirs_channels.columns = fnirs_cols
    for col in fnirs_channels.columns:
        fnirs_channels[col] = fnirs_channels[col].astype(float)

    fnirs_clean = pd.DataFrame({'timestamp': fnirs_data['timestamp'].values})
    for col in fnirs_cols:
        fnirs_clean[col] = fnirs_channels[col].values

    fnirs_clean['time_seconds'] = (fnirs_clean['timestamp'] - fnirs_clean['timestamp'].iloc[0]) / 1000.0
    fnirs_clean = fnirs_clean[['timestamp', 'time_seconds'] + fnirs_cols]
    return fnirs_clean


def extract_combined_data(df):
    """
    Create a combined dataframe with all EEG + fNIRS columns merged on timestamp.
    Rows will have NaN where a modality didn't record at that exact timestamp.
    """
    eeg_clean = extract_eeg_data(df).copy()
    fnirs_clean = extract_fnirs_data(df).copy()

    # Tag each source before merging
    eeg_clean['source'] = 'EEG'
    fnirs_clean['source'] = 'fNIRS'

    combined = pd.merge(eeg_clean, fnirs_clean, on='timestamp', how='outer', suffixes=('_eeg', '_fnirs'))

    # Reconcile duplicate time_seconds columns
    combined['time_seconds'] = combined['time_seconds_eeg'].fillna(combined['time_seconds_fnirs'])
    combined = combined.drop(columns=['time_seconds_eeg', 'time_seconds_fnirs', 'source_eeg', 'source_fnirs'])
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    # Reorder columns cleanly
    eeg_cols = ['TP9', 'AF7', 'AF8', 'TP10']
    fnirs_cols = [
        '730nm HbR left outer', '730nm HbR right outer',
        '850nm HbO left outer', '850nm HbO right outer',
        '730nm HbR left inner', '730nm HbR right inner',
        '850nm HbO left inner', '850nm HbO right inner'
    ]
    combined = combined[['timestamp', 'time_seconds'] + eeg_cols + fnirs_cols]
    return combined


def extract_all_data(input_file,
                     eeg_output='muse_eeg_clean.csv',
                     fnirs_output='muse_fnirs_clean.csv',
                     combined_output='muse_combined_clean.csv'):
    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file)

    # --- EEG ---
    eeg_clean = extract_eeg_data(df)
    eeg_clean.to_csv(eeg_output, index=False)
    print(f"Saved EEG data to: {eeg_output}  |  Shape: {eeg_clean.shape}")
    print(f"EEG sampling rate: ~{len(eeg_clean) / eeg_clean['time_seconds'].iloc[-1]:.2f} Hz\n")

    # --- fNIRS ---
    fnirs_clean = extract_fnirs_data(df)
    fnirs_clean.to_csv(fnirs_output, index=False)
    print(f"Saved fNIRS data to: {fnirs_output}  |  Shape: {fnirs_clean.shape}")
    print(f"fNIRS sampling rate: ~{len(fnirs_clean) / fnirs_clean['time_seconds'].iloc[-1]:.2f} Hz\n")

    # --- Combined (outer-joined on timestamp) ---
    combined = extract_combined_data(df)
    combined.to_csv(combined_output, index=False)
    print(f"Saved combined data to: {combined_output}  |  Shape: {combined.shape}")

    print("\n--- fNIRS Preview ---")
    print(fnirs_clean.head())
    print("\n--- fNIRS Statistics ---")
    print(fnirs_clean.describe())

    return eeg_clean, fnirs_clean, combined


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # For Google Colab, upload your file first:
    #   from google.colab import files
    #   uploaded = files.upload()

    input_file = 'muselab_recording(1).csv'   # ← change to your filename

    eeg_df, fnirs_df, combined_df = extract_all_data(
        input_file,
        eeg_output='muse_eeg_clean.csv',
        fnirs_output='muse_fnirs_clean.csv',
        combined_output='muse_combined_clean.csv'
    )

    # To download results in Colab:
    # from google.colab import files
    # files.download('muse_eeg_clean.csv')
    # files.download('muse_fnirs_clean.csv')
    # files.download('muse_combined_clean.csv')
