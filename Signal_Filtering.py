"""
Simple Muse EEG Brainwave Extraction using only scipy
Minimal dependencies: pandas, numpy, scipy, matplotlib
"""

import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Define brainwave frequency bands (Hz)
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

def extract_brainwave(eeg_signal, sampling_rate, lowcut, highcut, order=4):
    """
    Extract a specific frequency band using scipy's butterworth filter.

    Parameters:
    -----------
    eeg_signal : array
        Raw EEG signal
    sampling_rate : float
        Sampling rate in Hz
    lowcut : float
        Low frequency cutoff in Hz
    highcut : float
        High frequency cutoff in Hz
    order : int
        Filter order (default: 4)

    Returns:
    --------
    filtered : array
        Bandpass filtered signal
    """
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')

    # Apply filter (zero-phase)
    filtered = signal.filtfilt(b, a, eeg_signal)

    return filtered


# ========== EXAMPLE USAGE FOR GOOGLE COLAB ==========

# Read the Muse CSV file
df = pd.read_csv('muselab_recording(1).csv')

# Extract only EEG rows
eeg_data = df[df['osc_address'] == 'test/eeg'].copy()

# Parse the 4 EEG channels
eeg_channels = eeg_data['osc_data'].str.strip('"').str.split(',', expand=True)
eeg_channels.columns = ['TP9', 'AF7', 'AF8', 'TP10']
eeg_channels = eeg_channels.astype(float)

# Get time in seconds
timestamps = eeg_data['timestamp'].values
time_seconds = (timestamps - timestamps[0]) / 1000.0

# Calculate sampling rate
sampling_rate = len(time_seconds) / time_seconds[-1]
print(f"Sampling rate: {sampling_rate:.2f} Hz")

# Extract brainwaves from TP9 channel
tp9_signal = eeg_channels['TP9'].values

# Extract each brainwave band
tp9_delta = extract_brainwave(tp9_signal, sampling_rate, 0.5, 4)
tp9_theta = extract_brainwave(tp9_signal, sampling_rate, 4, 8)
tp9_alpha = extract_brainwave(tp9_signal, sampling_rate, 8, 13)
tp9_beta = extract_brainwave(tp9_signal, sampling_rate, 13, 30)
tp9_gamma = extract_brainwave(tp9_signal, sampling_rate, 30, 50)

print("Extracted all brainwave bands!")

# Plot results (first 10 seconds)
duration = 10
mask = time_seconds <= duration
t = time_seconds[mask]

fig, axes = plt.subplots(6, 1, figsize=(12, 10))

# Original
axes[0].plot(t, tp9_signal[mask], 'k', linewidth=0.5)
axes[0].set_ylabel('Raw EEG')
axes[0].grid(True, alpha=0.3)

# Delta
axes[1].plot(t, tp9_delta[mask], 'blue', linewidth=0.8)
axes[1].set_ylabel('Delta\n0.5-4 Hz')
axes[1].grid(True, alpha=0.3)

# Theta
axes[2].plot(t, tp9_theta[mask], 'green', linewidth=0.8)
axes[2].set_ylabel('Theta\n4-8 Hz')
axes[2].grid(True, alpha=0.3)

# Alpha
axes[3].plot(t, tp9_alpha[mask], 'orange', linewidth=0.8)
axes[3].set_ylabel('Alpha\n8-13 Hz')
axes[3].grid(True, alpha=0.3)

# Beta
axes[4].plot(t, tp9_beta[mask], 'red', linewidth=0.8)
axes[4].set_ylabel('Beta\n13-30 Hz')
axes[4].grid(True, alpha=0.3)

# Gamma
axes[5].plot(t, tp9_gamma[mask], 'purple', linewidth=0.8)
axes[5].set_ylabel('Gamma\n30-50 Hz')
axes[5].grid(True, alpha=0.3)

axes[5].set_xlabel('Time (seconds)')
plt.tight_layout()
plt.show()

# If you want to process all 4 channels at once:
all_brainwaves = {}
for channel in ['TP9', 'AF7', 'AF8', 'TP10']:
    channel_signal = eeg_channels[channel].values
    all_brainwaves[channel] = {
        'delta': extract_brainwave(channel_signal, sampling_rate, 0.5, 4),
        'theta': extract_brainwave(channel_signal, sampling_rate, 4, 8),
        'alpha': extract_brainwave(channel_signal, sampling_rate, 8, 13),
        'beta': extract_brainwave(channel_signal, sampling_rate, 13, 30),
        'gamma': extract_brainwave(channel_signal, sampling_rate, 30, 50)
    }

print("\nAccess any brainwave with: all_brainwaves['TP9']['alpha']")

# Save all brainwaves to CSV file
output_df = pd.DataFrame({
    'time_seconds': time_seconds,
    # TP9 channel
    'TP9_delta': all_brainwaves['TP9']['delta'],
    'TP9_theta': all_brainwaves['TP9']['theta'],
    'TP9_alpha': all_brainwaves['TP9']['alpha'],
    'TP9_beta': all_brainwaves['TP9']['beta'],
    'TP9_gamma': all_brainwaves['TP9']['gamma'],
    # AF7 channel
    'AF7_delta': all_brainwaves['AF7']['delta'],
    'AF7_theta': all_brainwaves['AF7']['theta'],
    'AF7_alpha': all_brainwaves['AF7']['alpha'],
    'AF7_beta': all_brainwaves['AF7']['beta'],
    'AF7_gamma': all_brainwaves['AF7']['gamma'],
    # AF8 channel
    'AF8_delta': all_brainwaves['AF8']['delta'],
    'AF8_theta': all_brainwaves['AF8']['theta'],
    'AF8_alpha': all_brainwaves['AF8']['alpha'],
    'AF8_beta': all_brainwaves['AF8']['beta'],
    'AF8_gamma': all_brainwaves['AF8']['gamma'],
    # TP10 channel
    'TP10_delta': all_brainwaves['TP10']['delta'],
    'TP10_theta': all_brainwaves['TP10']['theta'],
    'TP10_alpha': all_brainwaves['TP10']['alpha'],
    'TP10_beta': all_brainwaves['TP10']['beta'],
    'TP10_gamma': all_brainwaves['TP10']['gamma']
})

# Save to CSV
output_filename = 'muse_brainwaves_filtered.csv'
output_df.to_csv(output_filename, index=False)
print(f"\nSaved filtered brainwaves to: {output_filename}")
print(f"Output shape: {output_df.shape}")
print(f"Columns: {list(output_df.columns)}")

# Optional: Download in Google Colab
# from google.colab import files
# files.download(output_filename)
