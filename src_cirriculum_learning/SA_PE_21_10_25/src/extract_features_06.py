import numpy as np

# ============================================================
# 1️⃣ Extract Feature Vectors
# ============================================================
def extract_feature_vector(file_path, atol=1e-8, rtol=1e-5):
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    freq = data[:, 0]
    mag = data[:, 1]
    phase_deg = data[:, 2]
    
    def safe_normalize(x):
        x_min, x_max = np.min(x), np.max(x)
        if np.isclose(x_max, x_min, atol=atol, rtol=rtol) or np.isnan(x_max) or np.isnan(x_min):
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    mag_norm = safe_normalize(mag)
    phase_norm = safe_normalize(phase_deg)
    return mag_norm, phase_norm


# ============================================================
# 2️⃣ Basic Metrics
# ============================================================
def rmse(v1, v2):
    return np.sqrt(np.mean((np.array(v1) - np.array(v2)) ** 2))

def rmse_similarity(v1, v2):
    """Converts RMSE to bounded similarity [0,1]."""
    return 1 / (1 + rmse(v1, v2))


def normalized_cross_correlation(v1, v2):
    """Robust NCC that handles constant vectors."""
    v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
    std1, std2 = np.std(v1), np.std(v2)
    mean1, mean2 = np.mean(v1), np.mean(v2)
    
    if std1 == 0 and std2 == 0:
        return 1.0 if np.isclose(mean1, mean2, atol=1e-6) else 0.0
    if std1 == 0 or std2 == 0:
        return 0.0

    v1 = (v1 - mean1) / std1
    v2 = (v2 - mean2) / std2
    corr = np.correlate(v1, v2, mode='valid')[0]
    return np.clip(corr / len(v1), -1, 1)


def derivative_similarity(v1, v2):
    """Compares shapes based on slope patterns."""
    d1 = np.gradient(v1)
    d2 = np.gradient(v2)
    
    norm1 = np.linalg.norm(d1)
    norm2 = np.linalg.norm(d2)
    if norm1 == 0 and norm2 == 0:
        return 1.0
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cosine_sim = np.dot(d1, d2) / (norm1 * norm2)
    return np.clip((cosine_sim + 1) / 2, 0, 1)  # map [-1,1] → [0,1]


# ============================================================
# 3️⃣ Combined Similarity Metric
# ============================================================
def combined_similarity(v1, v2, w_ncc=0.5, w_deriv=0.3, w_rmse=0.2):
    ncc = normalized_cross_correlation(v1, v2)
    deriv_sim = derivative_similarity(v1, v2)
    rmse_sim = rmse_similarity(v1, v2)
    
    score = (w_ncc * ncc) + (w_deriv * deriv_sim) + (w_rmse * rmse_sim)
    return float(np.clip(score, 0, 1))


# ============================================================
# 4️⃣ Test Script
# ============================================================
def main():
    vec1_mag, vec1_phase = extract_feature_vector(r"C:\Users\jishu\OneDrive\Desktop\SA_PE\5_data\target_14.csv")
    vec2_mag, vec2_phase = extract_feature_vector(r"C:\Users\jishu\OneDrive\Desktop\SA_PE\5_data\target_12.csv")

    # Convert magnitude to dB scale
    vec1_mag_db = 20 * np.log10(np.clip(vec1_mag, 1e-8, None))
    vec2_mag_db = 20 * np.log10(np.clip(vec2_mag, 1e-8, None))

    # Similarity for Magnitude and Phase
    mag_sim = combined_similarity(vec1_mag_db, vec2_mag_db)
    phase_sim = combined_similarity(vec1_phase, vec2_phase)

    print(f"🔹 Magnitude Similarity: {mag_sim:.4f}")
    print(f"🔹 Phase Similarity:     {phase_sim:.4f}")
    print(f"🔸 Combined Score:       {(0.7*mag_sim + 0.3*phase_sim):.4f}")


if __name__ == "__main__":
    main()
