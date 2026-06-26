import numpy as np
import scipy.signal as signal
import pywt
from scipy.optimize import curve_fit

def compute_psd_welch(time_series, fs, nperseg=None):
    """
    Computes the Power Spectral Density using Welch's method.
    """
    if nperseg is None:
        nperseg = min(len(time_series), int(fs * 60 * 10)) # 10 minute windows default
        
    freqs, psd = signal.welch(time_series, fs=fs, nperseg=nperseg, window='hann')
    # Skip DC component (freq=0)
    if freqs[0] == 0:
        freqs = freqs[1:]
        psd = psd[1:]
        
    return freqs, psd

def fit_spectral_slope(freqs, psd, f_min=None, f_max=None):
    """
    Fits a power-law (straight line in log-log space) to the PSD within a frequency band.
    """
    if f_min is None:
        f_min = freqs.min()
    if f_max is None:
        f_max = freqs.max()
        
    mask = (freqs >= f_min) & (freqs <= f_max)
    f_band = freqs[mask]
    p_band = psd[mask]
    
    if len(f_band) < 3:
        return np.nan, np.nan, np.nan
        
    log_f = np.log10(f_band)
    log_p = np.log10(p_band)
    
    # Linear fit log_p = slope * log_f + intercept
    # Use scipy.optimize.curve_fit to get covariance matrix for uncertainty
    def linear_model(x, a, b):
        return a * x + b
        
    try:
        popt, pcov = curve_fit(linear_model, log_f, log_p)
        slope = popt[0]
        intercept = popt[1]
        slope_err = np.sqrt(np.diag(pcov))[0]
    except Exception:
        coeffs = np.polyfit(log_f, log_p, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        slope_err = np.nan
        
    return slope, intercept, slope_err

def compute_spectral_break(freqs, psd, f_min=None, f_max=None):
    """
    Detects the spectral break frequency by fitting a two-segment piecewise linear model in log-log space.
    """
    if f_min is None: f_min = freqs.min()
    if f_max is None: f_max = freqs.max()
        
    mask = (freqs >= f_min) & (freqs <= f_max)
    f_band = freqs[mask]
    p_band = psd[mask]
    
    if len(f_band) < 10:
        return np.nan
        
    log_f = np.log10(f_band)
    log_p = np.log10(p_band)
    
    def piecewise_linear(x, x0, y0, k1, k2):
        return np.piecewise(x, [x < x0], 
                            [lambda x: k1*x + y0 - k1*x0, 
                             lambda x: k2*x + y0 - k2*x0])
                             
    # Initial guesses: break at median freq
    x0_guess = np.median(log_f)
    y0_guess = np.median(log_p)
    k1_guess = -1.0
    k2_guess = -2.0
    
    try:
        popt, _ = curve_fit(piecewise_linear, log_f, log_p, p0=[x0_guess, y0_guess, k1_guess, k2_guess], maxfev=2000)
        break_freq = 10**(popt[0])
        # Ensure break freq is within bounds
        if f_min < break_freq < f_max:
            return break_freq
        return np.nan
    except Exception:
        return np.nan

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """
    Applies a Butterworth bandpass filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Handle cases where highcut is too close to nyquist
    if high >= 1.0:
        b, a = signal.butter(order, low, btype='high')
    else:
        b, a = signal.butter(order, [low, high], btype='band')
        
    y = signal.filtfilt(b, a, data)
    return y

def compute_wavelet_energy(data, fs, freq_band):
    """
    Computes total wavelet energy in a specific frequency band using CWT.
    """
    # Use Complex Morlet wavelet
    wavelet = 'cmor1.5-1.0'
    
    # Determine scales for the target frequency band
    # center frequency of wavelet / scale = target frequency
    # scale = center frequency / (target frequency / fs)
    fc = pywt.central_frequency(wavelet)
    
    # Create an array of target frequencies within the band
    f_min, f_max = freq_band
    target_freqs = np.logspace(np.log10(f_min), np.log10(f_max), 20)
    
    scales = fc * fs / target_freqs
    
    cwtmatr, freqs_out = pywt.cwt(data, scales, wavelet, 1.0/fs)
    
    # Wavelet power spectrum is absolute value squared
    power = np.abs(cwtmatr)**2
    
    # Total energy in this band over time (summing over scales)
    # Using integration approximation
    energy = np.sum(power, axis=0) 
    
    return energy

if __name__ == '__main__':
    # Simple test
    fs = 64.0
    t = np.arange(0, 10, 1/fs)
    data = np.sin(2*np.pi*1.0*t) + 0.5 * np.random.randn(len(t))
    
    f, p = compute_psd_welch(data, fs)
    slope, inter, err = fit_spectral_slope(f, p, 0.1, 10.0)
    
    print(f"Spectral Slope (0.1-10Hz): {slope:.2f} ± {err:.2f}")
    
    f_break = compute_spectral_break(f, p, 0.1, 10.0)
    print(f"Spectral Break Frequency: {f_break:.2f} Hz")
    
    bp_data = bandpass_filter(data, fs, 0.5, 2.0)
    print(f"Bandpass filtered RMS: {np.sqrt(np.mean(bp_data**2)):.2f}")
    
    energy = compute_wavelet_energy(data, fs, (0.5, 2.0))
    print(f"Mean Wavelet Energy in (0.5-2Hz): {np.mean(energy):.2f}")
