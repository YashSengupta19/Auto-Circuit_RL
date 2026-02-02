import os
import pandas as pd
import numpy as np

# Folder containing your data files
data_folder = "../5_data"

# Parameters for noise detection
RELATIVE_JUMP_THRESHOLD = 0.3    # gain change >30% considered a spike
SPIKE_FRACTION_THRESHOLD = 0.06  # >10% of data has spikes → noisy

noisy_files = []

for filename in sorted(os.listdir(data_folder)):
    if filename.startswith("target_"):
        filepath = os.path.join(data_folder, filename)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Skipping {filename}: cannot read file ({e})")
            continue

        if 'Gain' not in df.columns:
            print(f"Skipping {filename}: 'Gain' column not found")
            continue

        gain = df['Gain'].values
        if len(gain) < 5:
            continue  # not enough data

        # Compute relative gain changes between points
        rel_change = np.abs(np.diff(gain) / (np.abs(gain[:-1]) + 1e-12))

        # Find how many points are "spikes"
        spike_fraction = np.mean(rel_change > RELATIVE_JUMP_THRESHOLD)

        # If many spikes across the curve, mark as noisy
        if spike_fraction > SPIKE_FRACTION_THRESHOLD:
            noisy_files.append(filename)

# Print results
if noisy_files:
    print("Noisy files detected:")
    for f in noisy_files:
        print("  ", f)
else:
    print("✅ No noisy files detected.")
