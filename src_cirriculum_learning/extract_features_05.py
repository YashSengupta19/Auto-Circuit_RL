
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
from skrf.vectorFitting import VectorFitting
import warnings

# =========================
# --- FEATURE EXTRACTION ---
# =========================

def load_network_from_csv(file_path):
    """
    Load frequency, magnitude, and phase from CSV
    and create a 2-port Network with S21 = H.
    """
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    freq = data[:, 0]
    mag = data[:, 1]
    phase_deg = data[:, 2]

    phase_rad = np.deg2rad(phase_deg)
    H = mag * np.exp(1j * phase_rad)

    s_params = np.zeros((len(freq), 2, 2), dtype=complex)
    s_params[:, 1, 0] = H
    network = rf.Network(frequency=rf.Frequency.from_f(freq, unit='hz'), s=s_params, name='H_fit')

    return network, freq, H

# -------------------------
def fit_vector(network, freq, n_real, n_cmplx):
    """
    Fit the network using Vector Fitting with given number of poles.
    """
    vf = VectorFitting(network)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vf.vector_fit(
            n_poles_real=n_real,
            n_poles_cmplx=n_cmplx,
            init_pole_spacing='log',
            parameter_type='s',
            fit_constant=True,
            fit_proportional=True
        )
        for warning in w:
            if "Vector Fitting: The pole relocation process stopped" in str(warning.message):
                raise RuntimeError("Fitting did not converge")

    H_fit = vf.get_model_response(1, 0, freqs=freq)
    error = np.mean(np.abs(network.s[:, 1, 0] - H_fit)**2)
    return vf, H_fit, error

# -------------------------
def vf_to_fixed_vector(poles, zeros, residues, d, h, N_poles=6, N_zeros=6):
    """
    Convert vector fitting outputs into a fixed-length, normalized vector.
    """
    def pad_truncate(arr, N):
        arr = arr[:N]
        return np.pad(arr, (0, max(0, N-len(arr))), 'constant')
    
    print(f"Poles = {poles}")
    print(f"Zeros = {zeros}")
    print(f"Residues = {residues}")
    
    # Split real/imag
    p_real = pad_truncate(np.real(poles), N_poles)
    p_imag = pad_truncate(np.imag(poles), N_poles)
    z_real = pad_truncate(np.real(zeros), N_zeros)
    z_imag = pad_truncate(np.imag(zeros), N_zeros)
    r_real = pad_truncate(np.real(residues), N_poles)
    r_imag = pad_truncate(np.imag(residues), N_poles)
    
    # Log normalization with sign preservation
    def log_scale(x):
        return np.sign(x) * np.log1p(np.abs(x))
    
    p_real, p_imag = log_scale(p_real), log_scale(p_imag)
    z_real, z_imag = log_scale(z_real), log_scale(z_imag)
    r_real, r_imag = log_scale(r_real), log_scale(r_imag)
    
    d_scaled, h_scaled = log_scale(d), log_scale(h)
    
    feature_vector = np.concatenate([
        p_real, p_imag,
        z_real, z_imag,
        r_real, r_imag,
        [d_scaled, h_scaled]
    ])
    return feature_vector

# -------------------------
# def compute_zeros(poles, residues, d, h):
#     """
#     Compute zeros from poles, residues, d, h.
#     """
#     den = np.poly(poles)
#     num = np.polymul([h, 0], den)
#     num = np.polyadd(num, np.pad(d*den, (len(num)-len(den), 0)))

#     for rk, pk in zip(residues, poles):
#         others = np.poly([p for p in poles if p != pk])
#         num = np.polyadd(num, rk * others)

#     return np.roots(num)

def compute_zeros(poles, residues, d, h):
    den = np.poly(poles)
    num = np.polymul([h, 0], den) if h != 0 else np.zeros_like(den)
    num = np.polyadd(num, d * den)

    for rk, pk in zip(residues, poles):
        others = np.poly([p for p in poles if p != pk])
        num = np.polyadd(num, rk * others)

    return np.roots(num)


# =========================
# --- SIMILARITY FUNCTION ---
# =========================

def normalized_circuit_similarity(vec1, vec2):
    """
    Compute a single similarity score (0-1) between two circuit feature vectors.
    Higher score = more similar.
    """
    eu_dist = np.linalg.norm(vec1 - vec2)
    eu_sim = 1 / (1 + eu_dist)
    
    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cos_sim_norm = (cos_sim + 1) / 2
    
    return (eu_sim + cos_sim_norm) / 2

# =========================
# --- MAIN SCRIPT ---
# =========================

def extract_feature_vector(file_path, N_real=3, N_cmplx=3):
    """
    Extract normalized fixed-length feature vector from CSV file.
    """
    network, freq, H = load_network_from_csv(file_path)
    
    # Fit vector
    best_error = np.inf
    best_vf = None
    for n_real in range(0, N_real+1):
        for n_cmplx in range(0, N_cmplx+1):
            if n_real + n_cmplx > N_real + N_cmplx:
                continue
            try:
                vf, H_fit, err = fit_vector(network, freq, n_real, n_cmplx)
                if err < best_error:
                    best_error = err
                    best_vf = vf
            except RuntimeError:
                continue
            
    print("Printing Best vf parameters")
    print(f"best_vf.poles = {best_vf.poles}")
    print(f"best_vf.reisdues = {best_vf.residues}")
    print(f"best_vf.constant_coeff = {best_vf.constant_coeff}")
    print(f"best_vf.proportional_coeff = {best_vf.proportional_coeff}")

    
    # Extract VF parameters
    poles = best_vf.poles
    residues = best_vf.residues[2]
    d = best_vf.constant_coeff[2]
    h = best_vf.proportional_coeff[2]
    zeros = compute_zeros(poles, residues, d, h)
    
    # Convert to fixed-length feature vector
    vec = vf_to_fixed_vector(poles, zeros, residues, d, h)
    return vec, False

# -------------------------
def compare_circuits(file1, file2):
    """
    Extract feature vectors from two CSV files and compute similarity.
    """
    vec1, _ = extract_feature_vector(file1)
    vec2, _ = extract_feature_vector(file2)
    score = normalized_circuit_similarity(vec1, vec2)
    return score

# =========================
# --- EXAMPLE USAGE ---
# =========================

if __name__ == "__main__":
    file_target = r"C:\Users\jishu\OneDrive\Desktop\SA_PE\3_data\target_6.csv"
    file_model  = r"C:\Users\jishu\OneDrive\Desktop\SA_PE\3_data\target_6.csv"
    
    similarity_score = compare_circuits(file_target, file_model)
    print(f"Similarity between circuits: {similarity_score:.4f}")
