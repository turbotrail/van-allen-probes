# Van Allen Probes Magnetic Turbulence Analysis

This repository contains a comprehensive, publication-quality data analysis pipeline designed to process high-resolution (64 Hz) EMFISIS magnetometer data from the Van Allen Probes (RBSP-A). The pipeline computes scale-dependent turbulence properties, identifies distinct ULF wave events (Pc1-Pc5), maps these metrics to the spacecraft's physical ephemeris ($L$-shell, $MLT$), and performs rigorous non-parametric statistical validation across different geomagnetic storm phases.

## Installation

This project is configured to run using `uv` for seamless dependency management.

1. Ensure `uv` is installed on your system.
2. The project relies on several key scientific libraries which will be automatically resolved by `uv`:
   - `numpy`, `pandas`, `scipy`
   - `matplotlib`, `seaborn`
   - `h5py`, `cdflib`
   - `pywavelets`
   - `MFDFA`, `antropy`
   - `statsmodels`, `scikit-posthocs`

## Core Pipeline Architecture

The processing pipeline is modularized into dedicated scientific scripts:

*   **`data_loader.py`**: Handles loading standard `.cdf` EMFISIS and ECT-MAG-EPHEM files. Includes rigorous `perform_qc()` protocols to drop `nan` gaps, instrument saturation, and physically invalid signals ($> 1000 \text{ nT/s}$). Computes the fluctuating magnetic field ($\delta B$) by detrending a centered rolling background window via an optimized $O(1)$ algorithm.
*   **`storm_phases.py`**: Parses 1-minute OMNI solar wind dataset (`omni_min2015.asc`) to delineate `Quiet`, `Onset`, `Main Phase`, and `Recovery Phase` timeframes based on $Sym-H$ minimums and $B_z$ reversals.
*   **`wave_events.py`**: Dynamically isolates continuous intervals of enhanced wave power within specific bands (Pc1-Pc5). Calculates integrated wave power, dominant frequencies, and half-power spectral bandwidths for detected events.
*   **`spectral_analysis.py`**: Calculates Power Spectral Densities (PSD) via Welch's Method. Computes continuous wavelet transforms (CWT) and fits robust spectral slopes and spectral break frequencies with uncertainty propagation.
*   **`turbulence_stats.py`**: Computes multi-scale scaling properties (Generalized Hurst Exponent via MFDFA, standard DFA, Higuchi Fractal Dimension). Calculates $S_q$ Structure Functions, Intermittency indices (via scale-dependent flatness), and complexity metrics via Shannon, Permutation, and Wavelet Entropies.
*   **`statistical_analysis.py`**: Executes rigorous non-parametric statistical checks across storm phases using the Kruskal-Wallis H-test and Dunn's post-hoc comparisons. Measures significance using Mann-Whitney U and effect size via Cliff's Delta. Analyzes spatial gradients via Spearman rank correlations.
*   **`publication_figures.py`**: Contains advanced plotting routines for publication-ready visualizations (Heatmaps, Superposed Epoch Analysis, Spectrograms, Violin plots, and Radial Dependence overlays).

## How to Run

1. **Process the Data:** 
   Execute the main orchestrator to parse the CDF files, compute all turbulence statistics, and detect wave events. 
   *(Note: Processing high-frequency data for prolonged timeframes can be computationally expensive; let it run uninterrupted.)*
   ```bash
   uv run main.py
   ```
2. **Perform Statistical Validation:**
   Once `main.py` generates the core `.csv` results, validate the significance of the findings.
   ```bash
   uv run run_statistics.py
   ```
3. **Generate Publication Figures:**
   Render the final visualizations based on the calculated turbulence characteristics.
   ```bash
   uv run run_figures.py
   ```

## Output Artifacts

The pipeline generates several deliverables routed into the `results/` directory:

### Data Tables
*   `processed_data.h5` - Raw high-performance storage of core processing arrays.
*   `turbulence_statistics.csv` - The aggregated 10-minute non-overlapping turbulence metrics mapped to spatial parameters.
*   `wavelet_results.csv` - The continuous wave-event catalog.
*   `quality_control_report.csv` - A summary of data points dropped due to instrument noise.
*   `storm_catalog.csv` - Timestamps delineating distinct geomagnetic storm events.
*   `statistical_summary.csv` & `statistical_validation.txt` - Formal output of the non-parametric tests.

### Figures (`results/figures/`)
*   `background_verification.png`: Proof-of-concept visual showing $\delta B$ statistics tracking.
*   `event_spectrogram.png`: Verifies wave event detection with Continuous Wavelet Transforms over specific Pc bands.
*   `radial_dependence.png`: Binned spatial averages ($\Delta L=0.25$) showcasing $95\%$ Confidence Intervals.
*   `spatial_heatmap.png`: Equatorial L vs MLT polar mapping of parameters like the Hurst exponent.
*   `storm_phase_comparison.png`: Violin plots indicating phase-dependent turbulence amplifications.
*   `superposed_epoch.png`: Sea-level alignment at minimum Sym-H ($t=0$) demonstrating macroscopic ensemble trends across multiple storms.
