# import numpy as np
# import skrf as rf
# from skrf.vectorFitting import VectorFitting

# import numpy as np

# import numpy as np

# def get_uncancelled_modes(poles, zeros, residues, tol=1e-6):
#     """
#     Identify and remove nearly cancelling pole-zero pairs.
    
#     Parameters:
#         poles: array-like, complex poles
#         zeros: array-like, complex zeros
#         residues: array-like, complex residues corresponding to poles
#         tol: float, cancellation threshold (default 1e-6)
        
#     Returns:
#         uncancelled_poles: list of poles not cancelled by zeros
#         uncancelled_zeros: list of zeros not cancelled by poles
#         uncancelled_residues: list of residues corresponding to uncancelled poles
#     """
#     poles = np.array(poles)
#     zeros = np.array(zeros)
#     residues = np.array(residues)

#     # Flags to keep track of uncancelled indices
#     keep_poles = np.ones(len(poles), dtype=bool)
#     keep_zeros = np.ones(len(zeros), dtype=bool)

#     # Check each pole against each zero
#     for i, p in enumerate(poles):
#         for j, z in enumerate(zeros):
#             if abs(p - z) < tol:
#                 keep_poles[i] = False
#                 keep_zeros[j] = False
#                 break  # a pole can only cancel one zero

#     uncancelled_poles = poles[keep_poles].tolist()
#     uncancelled_zeros = zeros[keep_zeros].tolist()

#     # Only keep residues corresponding to uncancelled poles
#     uncancelled_residues = residues[keep_poles].tolist()

#     return uncancelled_poles, uncancelled_zeros, uncancelled_residues


# def create_feature_vector(poles_number,poles, zeros, residues, d, e, eps=1e-12):
    
#     max_poles = 2 * poles_number
#     max_residues = max_poles
#     max_zeros = max_poles
    
#     uncancelled_poles, uncancelled_zeros, uncancelled_residues = get_uncancelled_modes(poles, zeros, residues)
#     """
#     Convert VF outputs into a fixed 86-dimensional real feature vector (ML-friendly).

#     This version preserves magnitude and phase information:
#     - Complex numbers are converted to log-magnitude + phase.
#     - Padding is done with zeros.
#     - Scalars d and e are normalized to [-1,1] using tanh.

#     Parameters
#     ----------
#     poles, zeros, residues : complex arrays
#         Complex poles, zeros, and residues from vector fitting.
#     d, e : scalar
#         Constant and proportional terms from vector fitting.
#     max_poles, max_residues, max_zeros : int
#         Maximum number of poles, residues, zeros to include.
#     eps : float
#         Small value to avoid log(0).

#     Returns
#     -------
#     feature_vector : np.ndarray, shape (86,)
#         Real-valued, normalized feature vector.
#         Structure:
#             - 28 entries: poles (14 log-mag + 14 phase)
#             - 28 entries: residues (14 log-mag + 14 phase)
#             - 28 entries: zeros (14 log-mag + 14 phase)
#             - 2 entries: d and e
#     """

#     def log_mag_phase(arr, max_len):
#         # Pad or truncate
#         if len(arr) < max_len:
#             arr = np.pad(arr, (0, max_len - len(arr)), 'constant')
#         else:
#             arr = arr[:max_len]
#         mag = np.log(np.abs(arr) + eps)      # log-magnitude
#         phase = np.angle(arr)                # phase in radians
#         return np.hstack([mag, phase])

#     poles_feat = log_mag_phase(poles, max_poles)
#     residues_feat = log_mag_phase(residues, max_residues)
#     zeros_feat = log_mag_phase(zeros, max_zeros)

#     # Normalize each group (z-score) to [-1,1] via tanh
#     def normalize_zscore(arr):
#         mean = np.mean(arr)
#         std = np.std(arr) + eps
#         return np.tanh((arr - mean)/std)

#     poles_norm = normalize_zscore(poles_feat)
#     residues_norm = normalize_zscore(residues_feat)
#     zeros_norm = normalize_zscore(zeros_feat)

#     # Scalars d and e normalized with tanh
#     d_norm = np.tanh(d)
#     e_norm = np.tanh(e)

#     # Combine
#     feature_vector = np.hstack([poles_norm, residues_norm, zeros_norm, [d_norm, e_norm]])

#     assert feature_vector.shape[0] == (max_poles * 6 + 2), f"Feature vector size mismatch: {feature_vector.shape[0]}"
    
#     comparison_metrics = [uncancelled_poles, uncancelled_zeros, uncancelled_residues, d, e]
    
#     # print("Comparison Metrics")
#     # print(comparison_metrics)

#     return feature_vector, comparison_metrics


# def extractFeatures(poles_number, file_path):
#     """
#     Extract normalized 86-dimensional feature vector from CSV using VF.
#     Returns: feature_vector, svd_failed
#     """
#     svd_failed = False
#     try:
#         # === Load data ===
#         data = np.loadtxt(file_path, delimiter=",", skiprows=1)
#         f = data[:, 0]           # frequency [Hz]
#         mag = data[:, 1]         # linear gain
#         phase = data[:, 2]       # radians

#         # Complex response
#         H = mag * np.exp(1j * phase)

#         # Wrap into Network
#         freq = rf.Frequency.from_f(f, unit='hz')
#         s_params = H[:, np.newaxis, np.newaxis]
#         nw = rf.Network(frequency=freq, s=s_params)

#         # Vector fitting (always poles_number complex poles)
#         vf = VectorFitting(nw)
#         vf.vector_fit(n_poles_real=0, n_poles_cmplx=poles_number,
#                       fit_constant=True, fit_proportional=True)

#         # Extract model parameters
#         poles = vf.poles
#         residues = vf.residues[0]
#         d = vf.constant_coeff[0]
#         e = vf.proportional_coeff[0]

#         # Compute zeros
#         den = np.poly(poles)
#         num = d * den
#         num = np.polyadd(num, np.polymul([e, 0], den))
#         for rk, pk in zip(residues, poles):
#             others = np.poly([p for p in poles if p != pk])
#             num = np.polyadd(num, rk * others)
#         zeros = np.roots(num)

#         # Build feature vector
#         feature_vector, comparison_metric = create_feature_vector(poles_number, poles, zeros, residues, d, e)

#     except Exception as ex:
#         print(f"Vector fitting failed: {ex}")
#         feature_vector = np.zeros(86)
#         svd_failed = True

#     return feature_vector, comparison_metric, svd_failed


# if __name__ == "__main__":
#     file_path = r"C:\Users\jishu\OneDrive\Desktop\SA_PE\3_data\target_24.csv"
#     poles_number = 3
#     feature_vector, comparison_metric, svd_failed = extractFeatures(poles_number, file_path)
#     print("SVD Failed:", svd_failed)
#     print("Feature vector shape:", feature_vector.shape)
#     print("Feature vector:", feature_vector)
import numpy as np
import skrf as rf
from skrf.vectorFitting import VectorFitting
import warnings

def get_uncancelled_modes(poles, zeros, residues, tol=1e-6):
    """Remove nearly cancelling pole-zero pairs."""
    poles = np.array(poles)
    zeros = np.array(zeros)
    residues = np.array(residues)

    keep_poles = np.ones(len(poles), dtype=bool)
    keep_zeros = np.ones(len(zeros), dtype=bool)

    for i, p in enumerate(poles):
        for j, z in enumerate(zeros):
            if abs(p - z) < tol:
                keep_poles[i] = False
                keep_zeros[j] = False
                break

    uncancelled_poles = poles[keep_poles].tolist()
    uncancelled_zeros = zeros[keep_zeros].tolist()
    uncancelled_residues = residues[keep_poles].tolist()
    return uncancelled_poles, uncancelled_zeros, uncancelled_residues

def create_feature_vector(poles_number, poles, zeros, residues, d, e, eps=1e-12):
    """Convert VF outputs into a fixed 86-dimensional real feature vector."""
    max_poles = 2 * poles_number
    max_residues = max_poles
    max_zeros = max_poles

    def log_mag_phase(arr, max_len):
        arr = np.array(arr, dtype=np.complex128)
        if len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)), 'constant')
        else:
            arr = arr[:max_len]
        mag = np.log(np.abs(arr) + eps)
        phase = np.angle(arr)
        return np.hstack([mag, phase])

    def normalize_zscore(arr):
        mean = np.mean(arr)
        std = np.std(arr) + eps
        return np.tanh((arr - mean) / std)

    poles_feat = normalize_zscore(log_mag_phase(poles, max_poles))
    residues_feat = normalize_zscore(log_mag_phase(residues, max_residues))
    zeros_feat = normalize_zscore(log_mag_phase(zeros, max_zeros))

    d_norm = np.tanh(d)
    e_norm = np.tanh(e)

    feature_vector = np.hstack([poles_feat, residues_feat, zeros_feat, [d_norm, e_norm]])

    # Sanity check: should be 86
    assert feature_vector.shape[0] == (max_poles * 6 + 2), f"Feature vector size mismatch: {feature_vector.shape[0]}"

    comparison_metrics = get_uncancelled_modes(poles, zeros, residues) + [d, e]
    return feature_vector, comparison_metrics


def extractFeatures(file_path, max_poles_try=7):
    """Extract normalized 86-dimensional feature vector from CSV using VF."""
    svd_failed = False
    best_fit = None
    best_err = np.inf
    best_poles_number = 0
    feature_vector = np.zeros(86)
    comparison_metric = [[], [], [], 0, 0]

    try:
        # === Load data ===
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        f = data[:, 0]
        mag_db = data[:, 1]
        phase_deg = data[:, 2]

        # Convert magnitude from dB to linear
        mag = 10 ** (mag_db / 20)
        phase = np.deg2rad(phase_deg)
        H = mag * np.exp(1j * phase)

        # Create skrf Network
        freq = rf.Frequency.from_f(f, unit='hz')
        s_params = H[:, np.newaxis, np.newaxis]
        nw = rf.Network(frequency=freq, s=s_params)

        # Try vector fitting with increasing pole numbers
        for n_cmplx in range(1, max_poles_try + 1):
            try:
                vf = VectorFitting(nw)
                vf.vector_fit(n_poles_real=0, n_poles_cmplx=n_cmplx,
                              fit_constant=True, fit_proportional=True)
                
                err = vf.get_rms_error(parameter_type='s')
                
                if err < best_err:
                    best_err = err
                    best_fit = vf
                    best_poles_number = n_cmplx

            except Exception as inner_ex:
                continue  # Skip if fitting fails for this number of poles

        if best_fit is None:
            raise ValueError("No stable vector fit found.")

        # Extract poles, residues, and coefficients
        poles = best_fit.poles
        residues = best_fit.residues[0]
        d = best_fit.constant_coeff[0]
        e = best_fit.proportional_coeff[0]

        # Construct numerator and zeros
        den = np.poly(poles)
        num = d * den
        num = np.polyadd(num, np.polymul([e, 0], den))
        for rk, pk in zip(residues, poles):
            others = np.poly([p for p in poles if p is not pk])
            num = np.polyadd(num, rk * others)
        zeros = np.roots(num)

        # Create feature vector
        feature_vector, comparison_metric = create_feature_vector(
            best_poles_number, poles, zeros, residues, d, e
        )

    except Exception as ex:
        print(f"Vector fitting failed: {ex}")
        svd_failed = True

    return feature_vector, comparison_metric, svd_failed


if __name__ == "__main__":
    file_path = r"C:\Users\jishu\OneDrive\Desktop\SA_PE\3_data\target_23.csv"
    feature_vector, comparison_metric, svd_failed = extractFeatures(file_path)
    print("SVD Failed:", svd_failed)
    print("Feature vector shape:", feature_vector.shape)
    print("Feature vector:", feature_vector)
