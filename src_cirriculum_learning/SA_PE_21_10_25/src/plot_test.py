import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load the two responses ===
target_graph = pd.read_csv(r"C:\Users\jishu\OneDrive\Desktop\SA_PE\test_data\target_9.csv")
obtained_graph = pd.read_csv(r"C:\Users\jishu\OneDrive\Desktop\SA_PE\src_cirriculum_learning\ltspice_test_env\response.csv")

# === Ensure frequency alignment (optional interpolation if needed) ===
# Interpolate the obtained graph to match target frequencies
obtained_interp_gain = np.interp(target_graph["Frequency"], obtained_graph["Frequency"], obtained_graph["Gain"])
obtained_interp_phase = np.interp(target_graph["Frequency"], obtained_graph["Phase"], obtained_graph["Phase"])

# === Convert to dB ===
target_gain_db = 20 * np.log10(target_graph["Gain"])
obtained_gain_db = 20 * np.log10(obtained_interp_gain)

# === Plot Gain Comparison (Bode Magnitude Plot) ===
plt.figure(figsize=(8, 5))
plt.semilogx(target_graph["Frequency"], target_gain_db, label="Target Gain (dB)", linewidth=2)
plt.semilogx(target_graph["Frequency"], obtained_gain_db, label="Obtained Gain (dB)", linestyle="--", linewidth=2)
plt.title("Bode Magnitude Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain (dB)")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# # === Plot Phase Comparison (Bode Phase Plot) ===
# plt.figure(figsize=(8, 5))
# plt.semilogx(target_graph["Frequency"], target_graph["Phase"], label="Target Phase", linewidth=2)
# plt.semilogx(target_graph["Frequency"], obtained_interp_phase, label="Obtained Phase", linestyle="--", linewidth=2)
# plt.title("Bode Phase Comparison")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Phase (radians)")
# plt.grid(True, which="both", ls="--", lw=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # === Optional: Plot Error ===
# gain_error = target_gain_db - obtained_gain_db
# phase_error = target_graph["Phase"] - obtained_interp_phase

# plt.figure(figsize=(8, 4))
# plt.semilogx(target_graph["Frequency"], gain_error, label="Gain Error (dB)")
# plt.title("Gain Error (Target - Obtained)")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Error (dB)")
# plt.grid(True, which="both", ls="--", lw=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 4))
# plt.semilogx(target_graph["Frequency"], phase_error, label="Phase Error (radians)")
# plt.title("Phase Error (Target - Obtained)")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Error (radians)")
# plt.grid(True, which="both", ls="--", lw=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()
