import numpy as np
import scipy.stats as stats
import antropy as ant
from MFDFA import MFDFA
import pywt

def compute_moments(data):
    """
    Computes RMS, Skewness, and Kurtosis.
    """
    rms = np.sqrt(np.mean(data**2))
    skew = stats.skew(data, nan_policy='omit')
    kurt = stats.kurtosis(data, nan_policy='omit')
    return rms, skew, kurt

def compute_structure_functions(data, lags, max_order=6):
    """
    Computes structure functions S_q(tau) = < |B(t+tau) - B(t)|^q >
    for q=1 to max_order.
    Returns a dictionary of S_q for each lag, and scale-dependent flatness.
    """
    sf = {q: [] for q in range(1, max_order + 1)}
    flatness = []
    
    for lag in lags:
        diffs = data[lag:] - data[:-lag]
        abs_diffs = np.abs(diffs)
        for q in range(1, max_order + 1):
            sf[q].append(np.mean(abs_diffs**q))
            
        # Flatness = S_4 / (S_2)^2
        f_tau = np.mean(diffs**4) / (np.mean(diffs**2)**2) if np.mean(diffs**2) != 0 else np.nan
        flatness.append(f_tau)
            
    # Convert lists to arrays
    for q in sf:
        sf[q] = np.array(sf[q])
        
    return sf, np.array(flatness)

def compute_intermittency_index(sf, lags):
    """
    Computes the intermittency index based on the deviation from Kolmogorov scaling.
    Uses the scaling exponents zeta_q of the structure functions.
    S_q(tau) ~ tau^(zeta_q)
    """
    if len(lags) < 2:
        return np.nan
        
    log_tau = np.log10(lags)
    zeta = []
    
    for q in sorted(sf.keys()):
        log_sq = np.log10(sf[q])
        coeffs = np.polyfit(log_tau, log_sq, 1)
        zeta.append(coeffs[0])
        
    zeta = np.array(zeta)
    q_vals = np.arange(1, len(sf) + 1)
    
    # Kolmogorov (K41) predicts zeta_q = q/3 for velocity, 
    # For magnetic field, deviations from linear relation indicate intermittency
    # We can measure intermittency (mu) by fitting zeta_q = q/3 - mu/18 * q * (q-3)  (lognormal model)
    # A simpler metric is just the difference between zeta_4 and 4/3 or zeta_6 and 2
    if len(zeta) >= 6:
        # Intermittency parameter from p-model or log-normal fit
        # Simple empirical measure: zeta_6 / 6 vs zeta_3 / 3
        # If no intermittency, zeta_q is linear.
        mu = 2.0 - zeta[5] # zeta_6 should be 2 without intermittency
        return mu
    return np.nan

def compute_dfa(data):
    """
    Computes standard Detrended Fluctuation Analysis (DFA) scaling exponent.
    """
    N = len(data)
    lag = np.unique(np.logspace(1.5, np.log10(N/10), 20).astype(int))
    
    try:
        lag, dfa = MFDFA(data, lag=lag, q=np.array([2]), order=1)
        log_lag = np.log10(lag)
        log_dfa = np.log10(dfa[:,0])
        popt, pcov = np.polyfit(log_lag, log_dfa, 1, cov=True)
        alpha = popt[0]
        alpha_err = np.sqrt(np.diag(pcov))[0]
        return alpha, alpha_err
    except Exception:
        return np.nan, np.nan

def compute_hurst_mfdfa(data):
    """
    Computes the Generalized Hurst Exponent using MFDFA.
    """
    N = len(data)
    lag = np.unique(np.logspace(1.5, np.log10(N/10), 20).astype(int))
    
    q = 2 # standard Hurst corresponds to q=2
    try:
        lag, dfa = MFDFA(data, lag=lag, q=np.array([q]), order=1)
        popt, pcov = np.polyfit(np.log10(lag), np.log10(dfa[:,0]), 1, cov=True)
        H_hat = popt[0]
        H_err = np.sqrt(np.diag(pcov))[0]
        return H_hat, H_err
    except Exception:
        return np.nan, np.nan

def compute_multifractal_width(data):
    """
    Computes the width of the multifractal spectrum delta alpha.
    """
    N = len(data)
    lag = np.unique(np.logspace(1.5, np.log10(N/10), 10).astype(int))
    q = np.linspace(-5, 5, 21)
    q = q[q != 0] # remove q=0 to avoid division by zero
    
    try:
        lag, dfa = MFDFA(data, lag=lag, q=q, order=1)
        
        # Calculate generalized Hurst exponents h(q)
        h_q = np.zeros(len(q))
        for i in range(len(q)):
            h_q[i] = np.polyfit(np.log10(lag), np.log10(dfa[:,i]), 1)[0]
            
        # Calculate tau(q) = q*h(q) - 1
        tau = q * h_q - 1
        
        # Calculate singularity strength alpha = d(tau)/dq
        alpha = np.gradient(tau, q)
        
        return np.max(alpha) - np.min(alpha)
    except Exception:
        return np.nan

def compute_higuchi_fd(data, kmax=10):
    """
    Computes Higuchi Fractal Dimension (HFD).
    """
    N = len(data)
    if N < kmax * 2:
        return np.nan
        
    L = np.zeros(kmax)
    for k in range(1, kmax + 1):
        Lk = 0
        for m in range(k):
            indices = np.arange(m, N, k)
            if len(indices) > 1:
                Lmk = np.sum(np.abs(np.diff(data[indices])))
                # normalization
                Lmk = Lmk * (N - 1) / (len(indices) * k)
                Lk += Lmk
        L[k-1] = Lk / k
        
    log_k = np.log10(np.arange(1, kmax + 1))
    log_L = np.log10(L)
    
    try:
        coeffs = np.polyfit(log_k, log_L, 1)
        hfd = -coeffs[0] # Slope is -HFD
        return hfd
    except Exception:
        return np.nan

def compute_entropies(data, fs):
    """
    Computes Shannon Entropy, Permutation Entropy, and Wavelet Entropy.
    """
    # Shannon Entropy
    hist, bin_edges = np.histogram(data, bins='auto', density=True)
    p_shannon = hist * np.diff(bin_edges)
    p_shannon = p_shannon[p_shannon > 0]
    shannon_entropy = -np.sum(p_shannon * np.log2(p_shannon))
    
    # Permutation entropy (order 3, delay 1 is typical)
    pe = ant.perm_entropy(data, order=3, delay=1, normalize=True)
    
    # Wavelet entropy
    wavelet = 'cmor1.5-1.0'
    scales = np.arange(1, 64)
    cwtmatr, _ = pywt.cwt(data, scales, wavelet, 1.0/fs)
    
    # Mean energy at each scale
    energy = np.mean(np.abs(cwtmatr)**2, axis=1)
    total_energy = np.sum(energy)
    
    if total_energy > 0:
        p = energy / total_energy
        p = p[p > 0]
        we = -np.sum(p * np.log2(p))
        # Normalize wavelet entropy
        we = we / np.log2(len(scales))
    else:
        we = np.nan
        
    return shannon_entropy, pe, we

if __name__ == '__main__':
    # Test
    np.random.seed(42)
    data = np.random.randn(1000).cumsum() # Random walk (Hurst ~ 0.5 for increments, 1.5 for data)
    inc = np.diff(data)
    
    rms, skew, kurt = compute_moments(inc)
    print(f"RMS: {rms:.2f}, Skew: {skew:.2f}, Kurtosis: {kurt:.2f}")
    
    lags = np.arange(1, 20)
    sf = compute_structure_functions(inc, lags)
    interm = compute_intermittency_index(sf, lags)
    print(f"Intermittency index: {interm:.2f}")
    
    hurst = compute_hurst_mfdfa(inc)
    print(f"Hurst exponent: {hurst:.2f}")
    
    mf_width = compute_multifractal_width(inc)
    print(f"Multifractal width: {mf_width:.2f}")
    
    pe, we = compute_entropies(inc, 64.0)
    print(f"Permutation Entropy: {pe:.2f}")
    print(f"Wavelet Entropy: {we:.2f}")
