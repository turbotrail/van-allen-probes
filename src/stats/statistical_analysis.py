import pandas as pd
import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp
import os

def cliffs_delta(lst1, lst2):
    """Returns the Cliff's delta effect size for non-parametric comparisons."""
    n1, n2 = len(lst1), len(lst2)
    m1, m2 = np.meshgrid(lst1, lst2)
    res = np.sign(m1 - m2)
    return np.sum(res) / (n1 * n2)

def run_statistical_tests(stats_csv='results/turbulence_statistics.csv', output_csv='results/statistical_summary.csv', output_txt='results/statistical_validation.txt'):
    if not os.path.exists(stats_csv):
        print(f"File not found: {stats_csv}")
        return
        
    df = pd.read_csv(stats_csv)
    # Drop rows with NaNs in required columns
    cols_to_check = ['RMS', 'Hurst', 'L', 'Storm_Phase']
    available_cols = [c for c in cols_to_check if c in df.columns]
    df = df.dropna(subset=available_cols)
    
    phases = ['Quiet', 'Onset', 'Main Phase', 'Recovery Phase']
    metrics = ['RMS', 'Hurst', 'Spectral_Slope', 'Intermittency', 'Shannon_Entropy', 'Scale_Flatness', 'Higuchi_FD']
    
    summary_data = []
    
    with open(output_txt, 'w') as f:
        f.write("=== Scientific Validation: Non-Parametric Statistical Tests ===\n\n")
        f.write("1. Turbulence Modulation by Geomagnetic Storm Phase\n")
        f.write("-" * 55 + "\n")
        
        for metric in metrics:
            if metric not in df.columns:
                continue
                
            f.write(f"\n--- {metric} ---\n")
            
            # Extract groups
            groups = [df[df['Storm_Phase'] == phase][metric].dropna().values for phase in phases if len(df[df['Storm_Phase'] == phase]) > 0]
            valid_phases = [phase for phase in phases if len(df[df['Storm_Phase'] == phase]) > 0]
            
            if len(groups) > 1:
                # Kruskal-Wallis H-test
                h_stat, p_val = stats.kruskal(*groups)
                f.write(f"Kruskal-Wallis: H = {h_stat:.2f}, p = {p_val:.2e}\n")
                
                summary_data.append({
                    'Metric': metric,
                    'Test': 'Kruskal-Wallis',
                    'Statistic': h_stat,
                    'P-Value': p_val,
                    'Significance': p_val < 0.05
                })
                
                if p_val < 0.05:
                    f.write(f"-> Significant differences exist between storm phases for {metric}.\n")
                    # Dunn's Post-Hoc
                    dunn = sp.posthoc_dunn(df.dropna(subset=[metric]), val_col=metric, group_col='Storm_Phase', p_adjust='holm')
                    f.write("Dunn's Post-Hoc Test Results (p-values):\n")
                    f.write(str(dunn))
                    f.write("\n")
                    
                    # Compute Mann-Whitney U and Cliff's Delta between Quiet and Main Phase
                    q_data = df[df['Storm_Phase'] == 'Quiet'][metric].dropna().values
                    m_data = df[df['Storm_Phase'] == 'Main Phase'][metric].dropna().values
                    if len(q_data) > 0 and len(m_data) > 0:
                        u_stat, u_p = stats.mannwhitneyu(q_data, m_data)
                        c_delta = cliffs_delta(m_data, q_data) # Main vs Quiet
                        f.write(f"Mann-Whitney U (Main vs Quiet): U = {u_stat:.2f}, p = {u_p:.2e}\n")
                        f.write(f"Cliff's Delta (Main vs Quiet): {c_delta:.3f}\n")
                        
                        summary_data.append({
                            'Metric': metric,
                            'Test': 'Cliff Delta (Main vs Quiet)',
                            'Statistic': c_delta,
                            'P-Value': u_p,
                            'Significance': u_p < 0.05
                        })
                else:
                    f.write(f"-> No significant differences between storm phases for {metric}.\n")
                    
        # 2. Radial Dependence (Spearman Rank Correlation)
        f.write("\n2. Radial Dependence (L-shell) of Turbulence\n")
        f.write("-" * 55 + "\n")
        
        for metric in metrics:
            if metric not in df.columns or 'L' not in df.columns:
                continue
                
            mask = ~np.isnan(df[metric]) & ~np.isnan(df['L'])
            if mask.sum() > 2:
                rho, p_val = stats.spearmanr(df.loc[mask, 'L'], df.loc[mask, metric])
                
                f.write(f"{metric} vs L-shell:\n")
                f.write(f"  Spearman rho = {rho:.3f}, p = {p_val:.2e}\n")
                
                summary_data.append({
                    'Metric': metric,
                    'Test': 'Spearman Correlation vs L',
                    'Statistic': rho,
                    'P-Value': p_val,
                    'Significance': p_val < 0.05
                })
                
    pd.DataFrame(summary_data).to_csv(output_csv, index=False)
    print(f"Saved statistical reports to {output_txt} and {output_csv}")

if __name__ == '__main__':
    run_statistical_tests()
