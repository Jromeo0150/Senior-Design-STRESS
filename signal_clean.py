import pandas as pd
import numpy as np

def extract_eeg_data(input_file, output_file='muse_eeg_clean.csv'):

    # Read the CSV file
    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file)

    # Filter only EEG data rows
    eeg_data = df[df['osc_address'] == 'test/eeg'].copy()
    print(f"Found {len(eeg_data)} EEG data points")

    # Parse the osc_data column (which contains comma-separated EEG channel values)
    # The 4 channels correspond to: TP9, AF7, AF8, TP10 (standard Muse electrode positions)
    eeg_channels = eeg_data['osc_data'].str.strip('"').str.split(',', expand=True)
    eeg_channels.columns = ['TP9', 'AF7', 'AF8', 'TP10']

    # Convert to float
    for col in eeg_channels.columns:
        eeg_channels[col] = eeg_channels[col].astype(float)

    # Create clean dataframe with timestamp and EEG channels
    eeg_clean = pd.DataFrame({
        'timestamp': eeg_data['timestamp'].values,
        'TP9': eeg_channels['TP9'].values,
        'AF7': eeg_channels['AF7'].values,
        'AF8': eeg_channels['AF8'].values,
        'TP10': eeg_channels['TP10'].values
    })

    # Optional: Convert timestamp to relative time in seconds from start
    eeg_clean['time_seconds'] = (eeg_clean['timestamp'] - eeg_clean['timestamp'].iloc[0]) / 1000.0

    # Reorder columns to have time_seconds first
    eeg_clean = eeg_clean[['timestamp', 'time_seconds', 'TP9', 'AF7', 'AF8', 'TP10']]

    # Save to CSV
    eeg_clean.to_csv(output_file, index=False)
    print(f"Saved cleaned EEG data to: {output_file}")
    print(f"\nData shape: {eeg_clean.shape}")
    print(f"Sampling rate: ~{len(eeg_clean) / eeg_clean['time_seconds'].iloc[-1]:.2f} Hz")
    print(f"\nFirst few rows:")
    print(eeg_clean.head())
    print(f"\nBasic statistics:")
    print(eeg_clean.describe())

    return eeg_clean


# Example usage for Google Colab:
if __name__ == "__main__":
    # Upload your file in Colab using:
    # from google.colab import files
    # uploaded = files.upload()

    # Then extract EEG data
    input_file = 'muselab_recording(1).csv'  # Change this to your uploaded filename
    eeg_df = extract_eeg_data(input_file, output_file='muse_eeg_clean.csv')

    # Optional: Download the cleaned file
    # from google.colab import files
    # files.download('muse_eeg_clean.csv')
