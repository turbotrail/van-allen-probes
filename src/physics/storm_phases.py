import pandas as pd
import numpy as np

def load_omni_data(filepath='omni/omni_min2015.asc'):
    """
    Loads 1-minute OMNI data to extract Sym-H and IMF Bz.
    """
    cols_to_use = [0, 1, 2, 3, 18, 41]
    col_names = ['Year', 'Day', 'Hour', 'Minute', 'Bz', 'SymH']
    
    df = pd.read_csv(
        filepath, 
        sep=r'\s+', 
        header=None, 
        usecols=cols_to_use, 
        names=col_names
    )
    
    df['Bz'] = df['Bz'].replace(9999.99, np.nan)
    df['SymH'] = df['SymH'].replace(99999, np.nan)
    
    df['datetime'] = pd.to_datetime(
        df['Year'].astype(str) + '-' + df['Day'].astype(str), format='%Y-%j'
    ) + pd.to_timedelta(df['Hour'], unit='h') + pd.to_timedelta(df['Minute'], unit='m')
    
    df.set_index('datetime', inplace=True)
    df.drop(columns=['Year', 'Day', 'Hour', 'Minute'], inplace=True)
    
    return df

def segment_storm_phases(omni_df):
    """
    Segments the time into Quiet, Onset, Main Phase, and Recovery.
    Quiet: Sym-H > -20 nT for at least 6 continuous hours.
    Storm Onset: Continuous decrease in Sym-H with southward IMF (Bz < 0).
    Main Phase: Sym-H <= -50 nT (moderate), <= -100 nT (intense).
    Recovery: From Sym-H minimum until it recovers above -20 nT.
    Returns a dataframe of storm events.
    """
    symh = omni_df['SymH'].interpolate(method='time', limit=60)
    bz = omni_df['Bz'].interpolate(method='time', limit=60)
    
    events = []
    
    in_storm = False
    storm_start = None
    storm_min_val = 0
    storm_min_time = None
    
    for time, val in symh.items():
        if pd.isna(val): continue
        
        if not in_storm:
            if val <= -50:
                in_storm = True
                storm_min_val = val
                storm_min_time = time
                
                trace_back_window = symh.loc[time - pd.Timedelta(hours=24):time]
                if not trace_back_window.empty:
                    onset_time = trace_back_window.idxmax()
                else:
                    onset_time = time - pd.Timedelta(hours=6)
                storm_start = onset_time
        else:
            if val < storm_min_val:
                storm_min_val = val
                storm_min_time = time
                
            if val > -20 and time > storm_min_time + pd.Timedelta(hours=2):
                storm_end = time
                
                intensity = 'Moderate'
                if storm_min_val <= -100:
                    intensity = 'Intense'
                    
                events.append({
                    'Onset_Time': storm_start,
                    'Min_Time': storm_min_time,
                    'End_Time': storm_end,
                    'Min_SymH': storm_min_val,
                    'Intensity': intensity
                })
                
                in_storm = False
                storm_start = None
                
    return pd.DataFrame(events)

if __name__ == '__main__':
    print("Loading OMNI data...")
    omni = load_omni_data()
    print("Detecting storms...")
    storms = segment_storm_phases(omni)
    print(f"Detected {len(storms)} storm events.")
    print(storms)
