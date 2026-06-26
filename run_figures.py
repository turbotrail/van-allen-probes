import os
from src.visualization.publication_figures import (
    plot_background_verification,
    plot_event_spectrogram,
    plot_radial_dependence,
    plot_storm_phase_comparison,
    plot_heatmaps,
    plot_superposed_epoch
)

if __name__ == '__main__':
    os.makedirs('results/figures', exist_ok=True)
    plot_background_verification()
    plot_event_spectrogram()
    plot_radial_dependence()
    plot_storm_phase_comparison()
    plot_heatmaps()
    plot_superposed_epoch()
