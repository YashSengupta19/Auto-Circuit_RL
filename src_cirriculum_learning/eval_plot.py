import numpy as np

# Load the evaluations file
data = np.load(r".\eval_logs\evaluations.npz")

# Print all the keys stored in the file
print("Keys in the file:", data.files)

# Inspect each key’s contents
for key in data.files:
    print(f"\n=== {key} ===")
    print(data[key])
    print(f"Shape: {data[key].shape}")
