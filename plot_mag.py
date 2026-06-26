import cdflib
import matplotlib.pyplot as plt
import numpy as np

# Load the CDF file
file_path = 'uvw/2015/rbsp-a_magnetometer_uvw_emfisis-l2_20150101_v1.6.2.cdf'
cdf_file = cdflib.CDF(file_path)

# Extract variables
epoch = cdf_file.varget('Epoch')
mag = cdf_file.varget('Mag')
magnitude = cdf_file.varget('Magnitude')

# Convert epoch to datetime objects
epoch_dt = cdflib.cdfepoch.to_datetime(epoch)

# mag is typically a Nx3 array for U, V, W components (or similar coordinates)
# We can find out the labels if Mag_LABL_1 exists
try:
    labels = cdf_file.varget('Mag_LABL_1')
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    labels = [lbl.decode('utf-8') if isinstance(lbl, bytes) else lbl for lbl in labels]
except Exception:
    labels = ['U', 'V', 'W']

# Check for fill values and replace them with NaN
# Standard CDF fill values are often -1.0E31 or similar
fill_val = -1e30
mag = np.where(mag < fill_val, np.nan, mag)
magnitude = np.where(magnitude < fill_val, np.nan, magnitude)

# Create the plot
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

# Plot components
for i in range(3):
    axes[i].plot(epoch_dt, mag[:, i], label=f'Mag {labels[i]}')
    axes[i].set_ylabel(f'{labels[i]} (nT)')
    axes[i].legend(loc='upper right')
    axes[i].grid(True)

# Plot magnitude
axes[3].plot(epoch_dt, magnitude, label='Magnitude', color='black')
axes[3].set_ylabel('|B| (nT)')
axes[3].set_xlabel('Time (UTC)')
axes[3].legend(loc='upper right')
axes[3].grid(True)

plt.suptitle('Van Allen Probes A (RBSP-A) EMFISIS Magnetometer Data\n' + file_path.split('/')[-1])
plt.tight_layout()
plt.savefig('mag_plot_20150101.png', dpi=300)
print("Plot saved to mag_plot_20150101.png")
