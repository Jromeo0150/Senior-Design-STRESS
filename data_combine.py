import pandas as pd

# Load the files properly
eeg_filter = pd.read_csv('eeg_alpha_beta_full_resolution.csv')
fnirs_filter = pd.read_csv('muse_fnirs_clean.csv')

# eeg_filter currently only has 'time_seconds' but not 'timestamp'.
# We need 'timestamp' to merge with fnirs_filter.
# We can get the 'timestamp' from the original eeg_df which was created
# in an earlier cell and should be aligned by 'time_seconds'.

# Ensure 'timestamp' is present in eeg_filter by merging with the original eeg_df
# Select only 'timestamp' and 'time_seconds' from eeg_df for the merge
eeg_timestamps = eeg_df[['timestamp', 'time_seconds']].drop_duplicates(subset=['time_seconds'])
eeg_filter = pd.merge(eeg_filter, eeg_timestamps, on='time_seconds', how='left')

# Reorder columns to put timestamp at the beginning for consistency
eeg_filter = eeg_filter[['timestamp', 'time_seconds'] + [col for col in eeg_filter.columns if col not in ['timestamp', 'time_seconds']]]

# Tag each source before merging
eeg_filter['source'] = 'EEG'
fnirs_filter['source'] = 'fNIRS'

# Merge on timestamp
combined = pd.merge(
    eeg_filter,
    fnirs_filter,
    on='timestamp',
    how='outer',
    suffixes=('_eeg', '_fnirs')
)

# Reconcile duplicate time_seconds columns
combined['time_seconds'] = combined['time_seconds_eeg'].fillna(
    combined['time_seconds_fnirs']
)

combined = combined.drop(
    columns=[
        'time_seconds_eeg',
        'time_seconds_fnirs',
        'source_eeg',
        'source_fnirs'
    ]
)

combined = combined.sort_values('timestamp').reset_index(drop=True)

# Reorder columns cleanly
eeg_cols = [
    'TP9_alpha_power', 'TP9_beta_power', 'TP9_alpha_beta_ratio',
    'AF7_alpha_power', 'AF7_beta_power', 'AF7_alpha_beta_ratio',
    'AF8_alpha_power', 'AF8_beta_power', 'AF8_alpha_beta_ratio',
    'TP10_alpha_power', 'TP10_beta_power', 'TP10_alpha_beta_ratio'
]

fnirs_cols = [
    '730nm HbR left outer', '730nm HbR right outer',
    '850nm HbO left outer', '850nm HbO right outer',
    '730nm HbR left inner', '730nm HbR right inner',
    '850nm HbO left inner', '850nm HbO right inner'
]

# Ensure all eeg_cols and fnirs_cols are present in combined before reordering
final_eeg_cols = [col for col in eeg_cols if col in combined.columns]
final_fnirs_cols = [col for col in fnirs_cols if col in combined.columns]

combined = combined[['timestamp', 'time_seconds'] + final_eeg_cols + final_fnirs_cols]

# Save correctly
combined.to_csv('muse_combined_filtered.csv', index=False)

# Load the merged dataset
df = pd.read_csv('muse_combined_filtered.csv')

# Define EEG columns
eeg_cols = [
    'TP9_alpha_power', 'TP9_beta_power', 'TP9_alpha_beta_ratio',
    'AF7_alpha_power', 'AF7_beta_power', 'AF7_alpha_beta_ratio',
    'AF8_alpha_power', 'AF8_beta_power', 'AF8_alpha_beta_ratio',
    'TP10_alpha_power', 'TP10_beta_power', 'TP10_alpha_beta_ratio'
]

# Define fNIRS columns
fnirs_cols = [
    '730nm HbR left outer', '730nm HbR right outer',
    '850nm HbO left outer', '850nm HbO right outer',
    '730nm HbR left inner', '730nm HbR right inner',
    '850nm HbO left inner', '850nm HbO right inner'
]

# Keep only rows where ALL EEG and ALL fNIRS values exist
ml_ready = df.dropna(subset=eeg_cols + fnirs_cols)

# Optional: reset index
ml_ready = ml_ready.reset_index(drop=True)

# Save to new CSV
ml_ready.to_csv('ML_ready_signals.csv', index=False)

print("Original rows:", len(df))
print("ML-ready rows:", len(ml_ready))
print("File saved as ML_ready_signals.csv")
