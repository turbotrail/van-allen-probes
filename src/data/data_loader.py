import cdflib
import pandas as pd
import numpy as np
import glob
import os
import gc

def load_mag_data_day(mag_file, ephem_file=None):
    """
    Loads one day of EMFISIS 64Hz magnetometer data and merges with ephemeris.
    """
    print(f"Loading magnetometer data: {mag_file}")
    cdf_mag = cdflib.CDF(mag_file)
    
    epoch = cdf_mag.varget('Epoch')
    mag = cdf_mag.varget('Mag')
    magnitude = cdf_mag.varget('Magnitude')
    
    # Filter fill values (-1e30, -90000, etc.)
    mag = np.where((mag < -90000) | (mag > 90000), np.nan, mag)
    magnitude = np.where((magnitude < -90000) | (magnitude > 90000), np.nan, magnitude)
    
    # Convert Epoch to datetime
    epoch_dt = cdflib.cdfepoch.to_datetime(epoch)
    
    df_mag = pd.DataFrame({
        'time': epoch_dt,
        'Bu': mag[:, 0],
        'Bv': mag[:, 1],
        'Bw': mag[:, 2],
        'Bmag': magnitude
    })
    
    # Drop NaNs to save memory
    df_mag = df_mag.dropna().reset_index(drop=True)
    
    del cdf_mag
    del epoch
    del mag
    del magnitude
    del epoch_dt
    gc.collect()
    
    if ephem_file and os.path.exists(ephem_file):
        print(f"Loading ephemeris data: {ephem_file}")
        cdf_ephem = cdflib.CDF(ephem_file)
        
        ephem_epoch = cdf_ephem.varget('Epoch')
        ephem_dt = cdflib.cdfepoch.to_datetime(ephem_epoch)
        
        L = cdf_ephem.varget('Lsimple')
        MLT = cdf_ephem.varget('CDMAG_MLT')
        MLAT = cdf_ephem.varget('CDMAG_MLAT')
        R = cdf_ephem.varget('CDMAG_R')
        
        # Replace fill values with NaN
        L = np.where(L < -1e30, np.nan, L)
        MLT = np.where(MLT < -1e30, np.nan, MLT)
        MLAT = np.where(MLAT < -1e30, np.nan, MLAT)
        R = np.where(R < -1e30, np.nan, R)
        
        df_ephem = pd.DataFrame({
            'time': ephem_dt,
            'L': L,
            'MLT': MLT,
            'MLAT': MLAT,
            'R': R
        })
        
        df_ephem = df_ephem.dropna().sort_values('time')
        
        del cdf_ephem
        gc.collect()
        
        # Merge using merge_asof (nearest past value) to interpolate ephemeris onto 64Hz
        df_mag = df_mag.sort_values('time')
        df_merged = pd.merge_asof(df_mag, df_ephem, on='time', direction='nearest')
        
        return df_merged
    else:
        return df_mag

def compute_fluctuations(df, window='10min'):
    """
    Computes the background magnetic field (rolling mean) and the fluctuating field (delta B).
    """
    print(f"Computing fluctuations with rolling window: {window}")
    
    # Parse window to samples (assume 64 Hz)
    if isinstance(window, str) and window.endswith('min'):
        minutes = int(window.replace('min', ''))
        window_samples = int(minutes * 60 * 64.0)
    else:
        window_samples = 38400 # fallback to 10 min
        
    df.set_index('time', inplace=True)
    
    # Use integer window for O(1) rolling mean which is >100x faster than time-based
    rolling_mean = df[['Bu', 'Bv', 'Bw', 'Bmag']].rolling(window=window_samples, center=True, min_periods=1).mean()
    
    df['Bu0'] = rolling_mean['Bu']
    df['Bv0'] = rolling_mean['Bv']
    df['Bw0'] = rolling_mean['Bw']
    df['Bmag0'] = rolling_mean['Bmag']
    
    df['dBu'] = df['Bu'] - df['Bu0']
    df['dBv'] = df['Bv'] - df['Bv0']
    df['dBw'] = df['Bw'] - df['Bw0']
    df['dBmag'] = df['Bmag'] - df['Bmag0']
    
    df.reset_index(inplace=True)
    return df

def perform_qc(df):
    """
    Performs Quality Control on the dataframe.
    Flags spikes, unphysical jumps, and invalid ephemeris.
    Returns the dataframe with a 'qc_flag' column, and a summary dictionary.
    """
    print("Performing Quality Control...")
    df['qc_flag'] = 0 # 0 means good
    
    # 1. Missing data in essential columns
    missing_mask = df[['Bmag', 'L', 'MLT', 'MLAT']].isna().any(axis=1)
    df.loc[missing_mask, 'qc_flag'] = 1
    
    # 2. Saturation (Bmag > 60000 nT is unlikely for Van Allen Probes except very close to Earth, 
    # but EMFISIS saturates at 65536 nT)
    sat_mask = df['Bmag'] > 60000
    df.loc[sat_mask, 'qc_flag'] = 2
    
    # 3. Unrealistic jumps (dB/dt > 100 nT/s)
    # Since fs = 64Hz, dt = 0.015625s. 100 nT/s = 1.56 nT/sample
    dt = df['time'].diff().dt.total_seconds()
    db = df['Bmag'].diff()
    jump_mask = (np.abs(db / dt) > 1000) & (df['qc_flag'] == 0) # Use 1000 nT/s to be safe against true waves
    df.loc[jump_mask, 'qc_flag'] = 3
    
    # 4. Spikes in fluctuations (|dBmag| > 500 nT)
    if 'dBmag' in df.columns:
        spike_mask = (np.abs(df['dBmag']) > 500) & (df['qc_flag'] == 0)
        df.loc[spike_mask, 'qc_flag'] = 4
        
    # Summary
    qc_summary = {
        'total_points': len(df),
        'missing_data': missing_mask.sum(),
        'saturation': sat_mask.sum(),
        'jumps': jump_mask.sum(),
        'spikes': spike_mask.sum() if 'dBmag' in df.columns else 0,
        'valid_points': (df['qc_flag'] == 0).sum()
    }
    
    return df, qc_summary

if __name__ == '__main__':
    # Test on a single file
    mag_files = sorted(glob.glob('uvw/2015/rbsp-a_magnetometer_uvw_emfisis-l2_2015*.cdf'))
    ephem_files = sorted(glob.glob('ect-mag-ephem/2015/rbsp-a_mag-ephem_def-5min-ts04d_2015*_v01.cdf'))
    
    if mag_files and ephem_files:
        df = load_mag_data_day(mag_files[0], ephem_files[0])
        print(f"Loaded {len(df)} 64Hz points.")
        print(df.head())
        
        # Take a 1-hour subset to test fluctuation computation
        df_sub = df.head(64 * 60 * 60).copy()
        df_sub = compute_fluctuations(df_sub)
        print(df_sub.head())
    else:
        print("Could not find data files.")
