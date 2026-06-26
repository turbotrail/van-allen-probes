import numpy as np
import pandas as pd
from src.turbulence.spectral_analysis import bandpass_filter, compute_psd_welch, compute_wavelet_energy

# Frequency bands (Hz)
BANDS = {
    'Pc5': (0.002, 0.007),
    'Pc4': (0.007, 0.022),
    'Pc3': (0.022, 0.100),
    'Pc2': (0.100, 0.200),
    'Pc1': (0.200, 5.000)
}

def analyze_wave_activity(time_series, fs):
    """
    Analyzes the wave activity across different frequency bands.
    Returns a dictionary of metrics for each band.
    """
    results = {}
    
    for band_name, (lowcut, highcut) in BANDS.items():
        # Bandpass filter the data
        filtered = bandpass_filter(time_series, fs, lowcut, highcut)
        
        # Power and RMS
        power = np.var(filtered)
        rms = np.sqrt(np.mean(filtered**2))
        
        # PSD to find dominant frequency
        # For lower frequencies, need longer windows. Use length of data.
        f, p = compute_psd_welch(filtered, fs, nperseg=len(filtered))
        
        if len(p) > 0:
            dom_idx = np.argmax(p)
            dom_freq = f[dom_idx]
        else:
            dom_freq = np.nan
            
        # Wavelet energy
        we = compute_wavelet_energy(time_series, fs, (lowcut, highcut))
        mean_we = np.mean(we)
        
        results[band_name] = {
            'power': power,
            'rms': rms,
            'dominant_frequency': dom_freq,
            'mean_wavelet_energy': mean_we
        }
        
    return results

def detect_wave_events(time_series, time_array, fs, band_name, threshold_factor=3.0, min_cycles=3.0):
    """
    Detects continuous intervals of enhanced wave activity.
    min_cycles: minimum number of wave periods (at lowcut freq) to qualify as an event.
    """
    lowcut, highcut = BANDS[band_name]
    filtered = bandpass_filter(time_series, fs, lowcut, highcut)
    
    # Frequency dependent duration: e.g. Pc5 (2 mHz) -> period=500s. 3 cycles = 1500s.
    # Pc1 (200 mHz) -> period=5s. 3 cycles = 15s. (Enforce absolute minimum of 60s for Pc1)
    min_duration = max(60, min_cycles / lowcut)
    
    
    # Compute envelope or running variance for power
    window = int(fs * min_duration) # sliding window
    if window == 0:
        window = 1
        
    power_ts = pd.Series(filtered**2).rolling(window, center=True).mean().values
    
    mean_p = np.nanmean(power_ts)
    std_p = np.nanstd(power_ts)
    threshold = mean_p + threshold_factor * std_p
    
    # Identify intervals where power > threshold
    is_active = (power_ts > threshold).astype(int)
    
    # Find contiguous segments
    diffs = np.diff(is_active)
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    
    if is_active[0] == 1:
        starts = np.insert(starts, 0, 0)
    if is_active[-1] == 1:
        ends = np.append(ends, len(is_active) - 1)
        
    events = []
    for s, e in zip(starts, ends):
        duration = (e - s) / fs
        if duration >= min_duration:
            segment = filtered[s:e]
            
            # Integrated wave power (sum of squared amplitude * dt)
            integrated_power = np.sum(segment**2) / fs
            
            # Dominant frequency & Bandwidth
            f, p = compute_psd_welch(segment, fs, nperseg=len(segment))
            if len(p) > 0:
                dom_idx = np.argmax(p)
                dom_freq = f[dom_idx]
                
                # Spectral width (bandwidth): half-power width
                max_p = p[dom_idx]
                half_power = max_p / 2.0
                indices_above_half = np.where(p >= half_power)[0]
                if len(indices_above_half) > 1:
                    bandwidth = f[indices_above_half[-1]] - f[indices_above_half[0]]
                else:
                    bandwidth = 0.0
            else:
                dom_freq = np.nan
                bandwidth = np.nan
                
            events.append({
                'band': band_name,
                'start_time': time_array.iloc[s],
                'end_time': time_array.iloc[e],
                'duration': duration,
                'max_power': np.nanmax(power_ts[s:e]),
                'integrated_power': integrated_power,
                'dominant_frequency': dom_freq,
                'bandwidth': bandwidth
            })
            
    return pd.DataFrame(events)

if __name__ == '__main__':
    # Test
    fs = 64.0
    t = np.arange(0, 3600, 1/fs)
    data = np.sin(2*np.pi*0.05*t) + 0.5 * np.random.randn(len(t))
    
    results = analyze_wave_activity(data, fs)
    print("Wave Activity Analysis:")
    for band, metrics in results.items():
        print(f"{band}: RMS={metrics['rms']:.3f}, Fdom={metrics['dominant_frequency']:.3f} Hz")
        
    # Inject a 300s event at 0.5 Hz
    start_idx = int(1000 * fs)
    end_idx = start_idx + int(300 * fs)
    data[start_idx:end_idx] += 2.0 * np.sin(2*np.pi*0.5*t[start_idx:end_idx])
    
    times = pd.date_range('2015-01-01', periods=len(t), freq=f'{1000/fs}ms')
    events = detect_wave_events(data, times, fs, 'Pc1', min_duration=60)
    print("\nDetected Events in Pc1:")
    print(events)
