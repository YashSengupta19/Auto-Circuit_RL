import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Input and output directories
data_folder = "../20_data"
output_folder = "../20_data_images"

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate over all CSV files in the data folder
for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(data_folder, file)
        
        # Read the CSV
        df = pd.read_csv(file_path)
        
        # Extract columns
        f = df["Frequency"].values
        gain = df["Gain"].values
        phase = df["Phase"].values  # assumed already in radians
        
        # Convert gain to dB
        gain_db = 20 * np.log10(np.abs(gain))
        
        # Convert phase to degrees
        phase_deg = np.degrees(phase)
        
        # Create figure with two subplots (Gain & Phase)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        
        # Plot Gain (dB)
        ax1.semilogx(f, gain_db, label="Gain (dB)")
        ax1.set_ylabel("Gain (dB)")
        ax1.set_title(file)
        ax1.grid(True, which="both", ls="--")
        ax1.legend()
        
        # Plot Phase (degrees)
        ax2.semilogx(f, phase_deg, color='orange', label="Phase (°)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (°)")
        ax2.grid(True, which="both", ls="--")
        ax2.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_file = os.path.splitext(file)[0] + ".png"
        plt.savefig(os.path.join(output_folder, output_file), dpi=300, bbox_inches="tight")
        plt.close()

print("✅ All Gain + Phase plots saved to:", output_folder)
