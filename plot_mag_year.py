import cdflib
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import gc

# Find all CDF files for 2015
file_pattern = 'uvw/2015/rbsp-a_magnetometer_uvw_emfisis-l2_2015*.cdf'
files = sorted(glob.glob(file_pattern))
print(f"Found {len(files)} files.")

# To avoid memory issues, we'll downsample.
# The data is sampled at 64 Hz, so 5529600 points per day.
# Taking every 3840th point gives 1 point per minute (1440 points per day).
step = 3840
fill_val = -1e30

all_epochs = []
all_mag = []
all_magnitude = []

for idx, file_path in enumerate(files):
    if idx % 10 == 0:
        print(f"Processing file {idx+1}/{len(files)}: {os.path.basename(file_path)}")
    try:
        cdf_file = cdflib.CDF(file_path)
        
        # Read variables and downsample immediately
        epoch = cdf_file.varget('Epoch')[::step]
        mag = cdf_file.varget('Mag')[::step]
        magnitude = cdf_file.varget('Magnitude')[::step]
        
        # Filter fill values and unrealistic large values (e.g. -1e31, -100000, -99999)
        mag = np.where((mag < -90000) | (mag > 90000), np.nan, mag)
        magnitude = np.where((magnitude < -90000) | (magnitude > 90000), np.nan, magnitude)
        
        # Convert epoch to datetime objects
        epoch_dt = cdflib.cdfepoch.to_datetime(epoch)
        
        all_epochs.extend(epoch_dt)
        all_mag.append(mag)
        all_magnitude.extend(magnitude)
        
        # Free memory
        del cdf_file
        del epoch
        del mag
        del magnitude
        del epoch_dt
        gc.collect()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print("Concatenating data...")
all_mag = np.vstack(all_mag)
all_magnitude = np.array(all_magnitude)
all_epochs = np.array(all_epochs)

labels = ['U', 'V', 'W']

print("Plotting...")
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

for i in range(3):
    axes[i].plot(all_epochs, all_mag[:, i], label=f'Mag {labels[i]}', linewidth=0.5)
    axes[i].set_ylabel(f'{labels[i]} (nT)')
    axes[i].legend(loc='upper right')
    axes[i].grid(True)

axes[3].plot(all_epochs, all_magnitude, label='Magnitude', color='black', linewidth=0.5)
axes[3].set_ylabel('|B| (nT)')
axes[3].set_xlabel('Time (UTC)')
axes[3].legend(loc='upper right')
axes[3].grid(True)

plt.suptitle('Van Allen Probes A (RBSP-A) EMFISIS Magnetometer Data - 2015 (1-min resolution)')
plt.tight_layout()
plt.savefig('mag_plot_2015_year.png', dpi=300)
print("Plot saved to mag_plot_2015_year.png")
