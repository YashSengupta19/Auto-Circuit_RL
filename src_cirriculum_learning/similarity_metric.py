# Retry computing similarity metrics between the two provided feature vectors.
import numpy as np
import extract_features_03 as fe

import numpy as np

def compare_metrics(v1metric, v2metric, tol=1e-6):
    """
    Compare two circuits based on uncancelled poles, zeros, residues, and d/e.
    
    v1metric = [poles1, zeros1, residues1, d1, e1]
    v2metric = [poles2, zeros2, residues2, d2, e2]
    
    Returns a similarity score between 0 and 1 (1 = identical)
    """
    poles1, zeros1, residues1, d1, e1 = v1metric
    poles2, zeros2, residues2, d2, e2 = v2metric

    # --- Step 1: Penalize different number of uncancelled poles ---
    n_poles_diff = abs(len(poles1) - len(poles2))
    max_poles = max(len(poles1), len(poles2), 1)
    pole_count_similarity = 1.0 - (n_poles_diff / max_poles)
    pole_count_similarity = max(0.0, pole_count_similarity)  # clamp 0-1

    # --- Step 2: Compare poles (if same length, otherwise only count similarity) ---
    min_len = min(len(poles1), len(poles2))
    if min_len > 0:
        poles1_arr = np.array(poles1[:min_len])
        poles2_arr = np.array(poles2[:min_len])
        poles_diff = np.linalg.norm(poles1_arr - poles2_arr)
        poles_max = np.sqrt(min_len * (2.0**2))
        poles_similarity = max(0.0, 1.0 - poles_diff / poles_max)
    else:
        poles_similarity = 0.0

    # --- Step 3: Compare zeros similarly ---
    min_len_z = min(len(zeros1), len(zeros2))
    if min_len_z > 0:
        zeros1_arr = np.array(zeros1[:min_len_z])
        zeros2_arr = np.array(zeros2[:min_len_z])
        zeros_diff = np.linalg.norm(zeros1_arr - zeros2_arr)
        zeros_max = np.sqrt(min_len_z * (2.0**2))
        zeros_similarity = max(0.0, 1.0 - zeros_diff / zeros_max)
    else:
        zeros_similarity = 0.0

    # --- Step 4: Compare residues similarly ---
    min_len_r = min(len(residues1), len(residues2))
    if min_len_r > 0:
        residues1_arr = np.array(residues1[:min_len_r])
        residues2_arr = np.array(residues2[:min_len_r])
        residues_diff = np.linalg.norm(residues1_arr - residues2_arr)
        residues_max = np.sqrt(min_len_r * (2.0**2))
        residues_similarity = max(0.0, 1.0 - residues_diff / residues_max)
    else:
        residues_similarity = 0.0

    # --- Step 5: Compare d/e terms ---
    d_diff = abs(d1 - d2)
    e_diff = abs(e1 - e2)
    de_max = 2.0  # normalized
    de_similarity = max(0.0, 1.0 - (d_diff + e_diff) / de_max)

    # --- Step 6: Combine similarities ---
    # Give more weight to residues, then poles/zeros, then d/e
    weights = np.array([0.2, 0.2, 0.4, 0.2])  # poles, zeros, residues, d/e
    combined_similarity = np.average(
        [poles_similarity, zeros_similarity, residues_similarity, de_similarity], 
        weights=weights
    )

    # --- Step 7: Penalize for different number of poles ---
    combined_similarity *= pole_count_similarity

    return combined_similarity


v1, v1_metrics, _ = fe.extractFeatures(3, r"C:\Users\jishu\OneDrive\Desktop\SA_PE\3_data\target_37.csv")
v1 = np.array(v1)
v2, v2_metrics, _ = fe.extractFeatures(3, r"C:\Users\jishu\OneDrive\Desktop\SA_PE\3_data\target_39.csv")
v2 = np.array(v2)

sim_score = compare_metrics(v1_metrics, v2_metrics)
print(sim_score)
# print(v1)
# print(v2)

# assert v1.shape == v2.shape
# n = v1.size

# d = np.linalg.norm(v1 - v2)
# d_max = np.sqrt(n * (2.0**2))
# euclid_similarity = max(0.0, 1.0 - d / d_max)

# cos_sim = (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-15) + 1) / 2.0

# blocks = [14,14,14,1]
# idx = np.cumsum([0] + blocks)
# block_sims = []
# for i in range(len(blocks)):
#     a = v1[idx[i]:idx[i+1]]
#     b = v2[idx[i]:idx[i+1]]
#     db = np.linalg.norm(a - b)
#     db_max = np.sqrt(blocks[i] * (2.0**2))
#     sb = max(0.0, 1.0 - db / db_max)
#     block_sims.append(sb)

# block_weighted_similarity = float(np.mean(block_sims))

# mse = np.mean((v1 - v2)**2)
# mse_similarity = max(0.0, 1.0 - mse / 4.0)

# # Print results
# print("Euclidean distance d =", d)
# print("Euclidean similarity (0-1) =", euclid_similarity)
# print("Cosine similarity (mapped 0-1) =", cos_sim)
# print("Block sims (poles, zeros, residues, d/e):", block_sims)
# print("Block-weighted similarity =", block_weighted_similarity)
# print("MSE =", mse)
# print("MSE-based similarity (0-1) =", mse_similarity)
