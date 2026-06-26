import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from src.data.data_loader import load_mag_data_day, compute_fluctuations
import pywt
from scipy import stats

# Set publication quality plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def plot_background_verification():
    mag_files = sorted(glob.glob('uvw/2015/rbsp-a_magnetometer_uvw_emfisis-l2_2015*.cdf'))
    if not mag_files: return
        
    df = load_mag_data_day(mag_files[0], None)
    if df.empty: return
    df = df.head(64 * 3600 * 2).copy()
    df = compute_fluctuations(df, window='10min')
    
    db = df['dBmag'].dropna()
    mean_db = np.mean(db)
    median_db = np.median(db)
    std_db = np.std(db)
    skew_db = stats.skew(db)
    kurt_db = stats.kurtosis(db)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    axes[0].plot(df['time'], df['Bmag'], color='gray', alpha=0.5, label='Raw B (64 Hz)')
    axes[0].plot(df['time'], df['Bmag0'], color='red', linewidth=2, label='Background B0 (10-min mean)')
    axes[0].set_ylabel('Magnetic Field (nT)')
    axes[0].set_title('Background Field Removal Verification')
    axes[0].legend()
    
    axes[1].plot(df['time'], df['dBmag'], color='blue', linewidth=0.5, label='Fluctuations (δB)')
    axes[1].set_ylabel('δB (nT)')
    axes[1].legend()
    
    sns.histplot(db, bins=100, ax=axes[2], color='blue', kde=True)
    axes[2].set_title(f'δB Distribution | Mean: {mean_db:.2f}, Median: {median_db:.2f}, Std: {std_db:.2f}\nSkew: {skew_db:.2f}, Kurtosis: {kurt_db:.2f}')
    axes[2].set_xlabel('δB (nT)')
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/figures/background_verification.png')
    plt.close()

def plot_event_spectrogram(events_csv='results/wavelet_results.csv'):
    if not os.path.exists(events_csv): return
    df_events = pd.read_csv(events_csv, parse_dates=['start_time', 'end_time'])
    if df_events.empty: return
    
    # Pick top event
    event = df_events.loc[df_events['max_power'].idxmax()]
    start_time, end_time = event['start_time'], event['end_time']
    day_str = start_time.strftime('%Y%m%d')
    mag_files = sorted(glob.glob(f'uvw/2015/rbsp-a_magnetometer_uvw_emfisis-l2_{day_str}*.cdf'))
    if not mag_files: return
        
    df = load_mag_data_day(mag_files[0], None)
    df = compute_fluctuations(df, window='10min')
    
    mask = (df['time'] >= start_time - pd.Timedelta(minutes=5)) & (df['time'] <= end_time + pd.Timedelta(minutes=5))
    df_event = df[mask].copy()
    if df_event.empty: return
        
    # Downsample for visualization to prevent hanging (CWT on 64Hz is too slow for long events)
    # Target ~10 Hz (take every 6th point)
    df_event_ds = df_event.iloc[::6].copy()
    fs_ds = 64.0 / 6.0
    
    freqs = np.logspace(np.log10(0.002), np.log10(5.0), 100)
    scales = 1.0 / (freqs * (1.0 / fs_ds))
    print("Computing Wavelet Transform for Spectrogram...")
    cwtmatr, _ = pywt.cwt(df_event_ds['dBmag'].values, scales, 'cmor1.5-1.0', 1.0/fs_ds)
    power = np.abs(cwtmatr)**2
    
    import matplotlib.dates as mdates
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(df_event_ds['time'], df_event_ds['dBmag'], 'k-', linewidth=0.5)
    axes[0].axvspan(start_time, end_time, color='red', alpha=0.2, label='Detected Event Window')
    axes[0].set_ylabel('δB (nT)')
    axes[0].set_title(f"Wave Event Verification ({event['band']})")
    axes[0].legend()
    
    t_min, t_max = mdates.date2num(df_event_ds['time'].min()), mdates.date2num(df_event_ds['time'].max())
    im = axes[1].imshow(power, extent=[t_min, t_max, freqs.min(), freqs.max()], 
                        aspect='auto', origin='lower', cmap='viridis',
                        vmin=np.percentile(power, 10), vmax=np.percentile(power, 99))
    axes[1].axvspan(mdates.date2num(start_time), mdates.date2num(end_time), color='red', alpha=0.2, fill=False, hatch='//', edgecolor='red')
    axes[1].xaxis_date()
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (UTC)')
    axes[1].set_yscale('log')
    plt.colorbar(im, ax=axes[1], label='Wavelet Power')
    
    plt.tight_layout()
    plt.savefig('results/figures/event_spectrogram.png')
    plt.close()

def plot_radial_dependence(stats_csv='results/turbulence_statistics.csv'):
    if not os.path.exists(stats_csv): return
    df = pd.read_csv(stats_csv)
    df = df.dropna(subset=['L', 'RMS', 'Spectral_Slope', 'Hurst'])
    
    df['L_bin'] = pd.cut(df['L'], bins=np.arange(1.0, 6.5, 0.25))
    df_agg = df.groupby('L_bin')[['RMS', 'Spectral_Slope', 'Hurst']].agg(['mean', 'std', 'count']).reset_index()
    l_centers = df_agg['L_bin'].apply(lambda x: x.mid).astype(float)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    metrics = ['RMS', 'Spectral_Slope', 'Hurst']
    titles = ['RMS Amplitude (nT)', 'Spectral Slope', 'Hurst Exponent']
    
    for ax, metric, title in zip(axes, metrics, titles):
        mean = df_agg[metric]['mean']
        std = df_agg[metric]['std']
        count = df_agg[metric]['count']
        ci = 1.96 * std / np.sqrt(count) # 95% CI
        
        ax.plot(l_centers, mean, 'ko-')
        ax.fill_between(l_centers, mean - ci, mean + ci, color='gray', alpha=0.3, label='95% CI')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.axvspan(1.0, 2.0, color='red', alpha=0.1, label='Inner Belt')
        ax.axvspan(2.0, 3.0, color='green', alpha=0.1, label='Slot Region')
        ax.axvspan(3.0, 6.0, color='blue', alpha=0.1, label='Outer Belt')
        ax.axvline(4.0, color='black', linestyle='--', label='Plasmapause (~L=4)')
        
    axes[0].legend()
    axes[-1].set_xlabel('L-shell')
    plt.tight_layout()
    plt.savefig('results/figures/radial_dependence.png')
    plt.close()

def plot_storm_phase_comparison(stats_csv='results/turbulence_statistics.csv'):
    if not os.path.exists(stats_csv): return
    df = pd.read_csv(stats_csv).dropna(subset=['Storm_Phase', 'RMS', 'Hurst'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    order = ['Quiet', 'Onset', 'Main Phase', 'Recovery Phase']
    
    sns.violinplot(data=df, x='Storm_Phase', y='RMS', order=order, ax=axes[0], inner='quartile', palette='muted')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('RMS Amplitude (nT)')
    axes[0].set_title('Turbulence Amplitude by Storm Phase')
    
    sns.violinplot(data=df, x='Storm_Phase', y='Hurst', order=order, ax=axes[1], inner='quartile', palette='muted')
    axes[1].set_ylabel('Hurst Exponent')
    axes[1].set_title('Fractal Scaling by Storm Phase')
    
    plt.tight_layout()
    plt.savefig('results/figures/storm_phase_comparison.png')
    plt.close()

def plot_heatmaps(stats_csv='results/turbulence_statistics.csv'):
    if not os.path.exists(stats_csv): return
    df = pd.read_csv(stats_csv).dropna(subset=['L', 'MLT', 'Hurst'])
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
    sc = ax.scatter(df['MLT'] * np.pi / 12, df['L'], c=df['Hurst'], cmap='viridis', s=10, alpha=0.8)
    ax.set_theta_zero_location('S')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.arange(0, 2*np.pi, np.pi/4))
    ax.set_xticklabels(['00', '03', '06', '09', '12', '15', '18', '21'])
    ax.set_ylim(1, 6)
    plt.colorbar(sc, label='Hurst Exponent', pad=0.1)
    plt.title('Equatorial Spatial Distribution of Hurst Exponent (L vs MLT)')
    plt.savefig('results/figures/spatial_heatmap.png')
    plt.close()

def plot_superposed_epoch(stats_csv='results/turbulence_statistics.csv', storm_csv='results/storm_catalog.csv'):
    if not os.path.exists(stats_csv) or not os.path.exists(storm_csv): return
    df = pd.read_csv(stats_csv, parse_dates=['time']).dropna(subset=['RMS', 'Hurst'])
    storms = pd.read_csv(storm_csv, parse_dates=['Min_Time'])
    
    superposed_data = []
    
    for _, storm in storms.iterrows():
        t0 = storm['Min_Time']
        # Extract data from -48h to +72h around t0
        mask = (df['time'] >= t0 - pd.Timedelta(hours=48)) & (df['time'] <= t0 + pd.Timedelta(hours=72))
        storm_data = df[mask].copy()
        
        if len(storm_data) > 0:
            # Calculate delta_t in hours
            storm_data['delta_t'] = (storm_data['time'] - t0).dt.total_seconds() / 3600.0
            # Bin into 1-hour bins
            storm_data['t_bin'] = np.round(storm_data['delta_t'])
            
            # Aggregate per bin for this storm
            bin_data = storm_data.groupby('t_bin').agg({'RMS': 'mean', 'Hurst': 'mean', 'Intermittency': 'mean', 'Spectral_Slope': 'mean'}).reset_index()
            superposed_data.append(bin_data)
            
    if not superposed_data: return
    df_sup = pd.concat(superposed_data, ignore_index=True)
    
    # Ensemble average across all storms for each time bin
    ensemble = df_sup.groupby('t_bin').agg(['mean', 'std']).reset_index()
    t_bins = ensemble['t_bin']
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    metrics = ['RMS', 'Hurst', 'Intermittency', 'Spectral_Slope']
    titles = ['RMS Amplitude (nT)', 'Hurst Exponent', 'Intermittency Index', 'Spectral Slope']
    
    for ax, metric, title in zip(axes, metrics, titles):
        mean = ensemble[metric]['mean']
        std = ensemble[metric]['std']
        
        ax.plot(t_bins, mean, 'b-', linewidth=2)
        ax.fill_between(t_bins, mean - std, mean + std, color='blue', alpha=0.2)
        ax.axvline(0, color='red', linestyle='--', label='Minimum Sym-H (t=0)')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        if metric == 'RMS':
            ax.set_yscale('log')
            
    axes[0].legend()
    axes[-1].set_xlabel('Time from Minimum Sym-H (hours)')
    axes[-1].set_xlim(-48, 72)
    plt.tight_layout()
    plt.savefig('results/figures/superposed_epoch.png')
    plt.close()

if __name__ == '__main__':
    os.makedirs('results/figures', exist_ok=True)
    plot_background_verification()
    plot_event_spectrogram()
    plot_radial_dependence()
    plot_storm_phase_comparison()
    plot_heatmaps()
    plot_superposed_epoch()
