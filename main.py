import os
import glob
import gc
import numpy as np
import pandas as pd
import h5py
from src.data.data_loader import load_mag_data_day, compute_fluctuations, perform_qc
from src.physics.storm_phases import load_omni_data, segment_storm_phases
from src.turbulence.spectral_analysis import compute_psd_welch, fit_spectral_slope, compute_spectral_break
from src.turbulence.turbulence_stats import (
    compute_moments, compute_structure_functions, compute_intermittency_index,
    compute_dfa, compute_hurst_mfdfa, compute_higuchi_fd, compute_entropies
)
from src.physics.wave_events import analyze_wave_activity, detect_wave_events, BANDS

def process_day(mag_file, ephem_file, fs=64.0, window_minutes=10):
    """
    Processes a single day of data: loads, computes stats in intervals, and detects events.
    """
    print(f"--- Processing {os.path.basename(mag_file)} ---")
    df = load_mag_data_day(mag_file, ephem_file)
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()
        
    df = compute_fluctuations(df, window=f'{window_minutes}min')
    df, qc_summary = perform_qc(df)
    
    # Drop rows flagged by QC (qc_flag > 0)
    df_valid = df[df['qc_flag'] == 0].copy()
    if len(df_valid) == 0:
        return pd.DataFrame(), pd.DataFrame(), qc_summary
        
    # 1. Detect wave events on the full valid day
    all_events = []
    # We will use dBmag for wave detection
    times_pd = df_valid['time']
    data_np = df_valid['dBmag'].values
    
    for band_name in BANDS.keys():
        print(f"Detecting {band_name} wave events...")
        events = detect_wave_events(data_np, times_pd, fs, band_name)
        all_events.append(events)
        
    df_events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    
    # Enrich events with spatial metrics
    if not df_events.empty:
        # Interpolate L, MLT, MLAT at the start_time of the event
        df_events = pd.merge_asof(df_events.sort_values('start_time'), 
                                  df_valid[['time', 'L', 'MLT', 'MLAT']].sort_values('time'),
                                  left_on='start_time', right_on='time', direction='nearest')
        df_events.drop(columns=['time'], inplace=True)
        
    # 2. Compute turbulence statistics in windows
    interval_samples = int(window_minutes * 60 * fs)
    num_intervals = len(data_np) // interval_samples
    
    stats_list = []
    
    for i in range(num_intervals):
        start_idx = i * interval_samples
        end_idx = (i + 1) * interval_samples
        
        segment = data_np[start_idx:end_idx]
        segment_time = times_pd.iloc[start_idx]
        
        # Get mean spatial coordinates for this interval
        mean_L = df_valid['L'].iloc[start_idx:end_idx].mean()
        mean_MLT = df_valid['MLT'].iloc[start_idx:end_idx].mean()
        mean_MLAT = df_valid['MLAT'].iloc[start_idx:end_idx].mean()
        mean_R = df_valid['R'].iloc[start_idx:end_idx].mean()
        
        # Skip if missing data
        if np.isnan(segment).sum() > len(segment) * 0.1:
            continue
            
        # Fill remaining NaNs with 0 (since it's a fluctuation series)
        segment = np.nan_to_num(segment)
        
        # Compute stats
        rms, skew, kurt = compute_moments(segment)
        
        # Structure functions & intermittency (lags 1 to 100)
        lags = np.arange(1, 101, 5)
        sf, flatness = compute_structure_functions(segment, lags)
        interm = compute_intermittency_index(sf, lags)
        
        # Scale-dependent flatness at lag=10 (approx 0.15s)
        flatness_10 = flatness[1] if len(flatness) > 1 else np.nan
        
        # Hurst & Entropies (downsample to speed up MFDFA/Entropies if needed)
        seg_ds = segment[::10]
        try:
            alpha_dfa, alpha_err = compute_dfa(seg_ds)
            hurst, hurst_err = compute_hurst_mfdfa(seg_ds)
            shannon, pe, we = compute_entropies(seg_ds, fs/10)
            higuchi = compute_higuchi_fd(seg_ds)
        except Exception:
            alpha_dfa, alpha_err, hurst, hurst_err = np.nan, np.nan, np.nan, np.nan
            shannon, pe, we, higuchi = np.nan, np.nan, np.nan, np.nan
            
        # Spectral slope (Pc3-Pc5 range: 0.002 to 0.1 Hz)
        f, p = compute_psd_welch(segment, fs)
        slope, intercept, slope_err = fit_spectral_slope(f, p, 0.002, 0.1)
        break_freq = compute_spectral_break(f, p, 0.002, 0.5)
        
        stats_list.append({
            'time': segment_time,
            'L': mean_L,
            'MLT': mean_MLT,
            'MLAT': mean_MLAT,
            'R': mean_R,
            'RMS': rms,
            'Skewness': skew,
            'Kurtosis': kurt,
            'Intermittency': interm,
            'Scale_Flatness': flatness_10,
            'DFA_alpha': alpha_dfa,
            'DFA_alpha_err': alpha_err,
            'Hurst': hurst,
            'Hurst_err': hurst_err,
            'Higuchi_FD': higuchi,
            'Shannon_Entropy': shannon,
            'Permutation_Entropy': pe,
            'Wavelet_Entropy': we,
            'Spectral_Slope': slope,
            'Spectral_Slope_err': slope_err,
            'Spectral_Break': break_freq
        })
        
    df_stats = pd.DataFrame(stats_list)
    
    # Free memory
    del df
    del df_valid
    del data_np
    gc.collect()
    
    return df_stats, df_events, qc_summary

def run_pipeline(year='2015', month=None, test_mode=False):
    print("=== Loading OMNI Data and Segmenting Storm Phases ===")
    omni = load_omni_data()
    storms = segment_storm_phases(omni)
    storms.to_csv('results/storm_catalog.csv', index=False)
    
    month_str = month if month else '*'
    mag_files = sorted(glob.glob(f'uvw/{year}/rbsp-a_magnetometer_uvw_emfisis-l2_{year}{month_str}*.cdf'))
    ephem_files = sorted(glob.glob(f'ect-mag-ephem/{year}/rbsp-a_mag-ephem_def-5min-ts04d_{year}{month_str}*_v01.cdf'))
    
    # Map ephem dates
    ephem_dict = {}
    for ef in ephem_files:
        date_str = os.path.basename(ef).split('_')[3]
        ephem_dict[date_str] = ef
        
    if test_mode:
        mag_files = mag_files[:1] # Process only 1 day in test mode
        
    all_stats = []
    all_events = []
    all_qc = []
    
    for mag_file in mag_files:
        date_str = os.path.basename(mag_file).split('_')[4][:8]
        ephem_file = ephem_dict.get(date_str)
        
        try:
            df_stats, df_events, qc_summary = process_day(mag_file, ephem_file)
            
            qc_summary['date'] = date_str
            all_qc.append(qc_summary)
            
            if not df_stats.empty:
                all_stats.append(df_stats)
            if not df_events.empty:
                all_events.append(df_events)
        except Exception as e:
            print(f"Error processing {mag_file}: {e}")
            
    # Save QC Report
    pd.DataFrame(all_qc).to_csv('results/quality_control_report.csv', index=False)
    
    if all_stats:
        final_stats = pd.concat(all_stats, ignore_index=True)
        final_stats.to_csv('results/turbulence_statistics.csv', index=False)
        
        # Merge with storm phases to label each interval
        def assign_storm_phase(time):
            # Default is Quiet
            for _, storm in storms.iterrows():
                if storm['Onset_Time'] <= time < storm['Min_Time']:
                    return 'Main Phase'
                elif storm['Min_Time'] <= time <= storm['End_Time']:
                    return 'Recovery Phase'
                elif time < storm['Onset_Time'] and time >= (storm['Onset_Time'] - pd.Timedelta(hours=2)):
                    return 'Onset'
            return 'Quiet'
            
        final_stats['Storm_Phase'] = final_stats['time'].apply(assign_storm_phase)
        final_stats.to_csv('results/turbulence_statistics.csv', index=False)
        
        # Save to HDF5
        with pd.HDFStore('results/processed_data.h5') as store:
            store.put('turbulence_stats', final_stats)
            
    if all_events:
        final_events = pd.concat(all_events, ignore_index=True)
        # Assign storm phase to events
        final_events['Storm_Phase'] = final_events['start_time'].apply(assign_storm_phase)
        final_events.to_csv('results/wavelet_results.csv', index=False)
        with pd.HDFStore('results/processed_data.h5') as store:
            store.put('wave_events', final_events)

if __name__ == '__main__':
    run_pipeline(year='2015', month='03', test_mode=False)
