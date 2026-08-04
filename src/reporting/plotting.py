import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

def create_comparison_plots(df):
    """
    Create comprehensive comparison plots showing actual values
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # Get unique k values for x-axis (smaller values on the right)
    k_values = sorted(df['k'].unique())
    M_values = sorted(df['M'].unique())
    
    # Define better colors - darker blue and orange
    colors = {'5': '#4682B4', '50': '#FF8C00'}  # Steel blue for M=5, Dark orange for M=25
    
    # Plot 1: App Accuracy
    ax1 = axes[0, 0]
    for M in [5, 50]:
        subset = df[df['M'] == M]
        color = colors[str(M)]
        
        # Independent model
        means_single = subset.groupby('k')['acc_app_single'].mean()
        stds_single = subset.groupby('k')['acc_app_single'].std()
        ax1.errorbar(k_values, means_single, yerr=stds_single, marker='o', 
                    label=f'Independent $M_{{max}}={M}$', color=color, linestyle='--', 
                    capsize=0, capthick=2, alpha=0.7, markersize=6)
        
        # Joint model
        means_multi = subset.groupby('k')['acc_app_multi'].mean()
        stds_multi = subset.groupby('k')['acc_app_multi'].std()
        ax1.errorbar(k_values, means_multi, yerr=stds_multi, marker='s', 
                    label=f'Joint $M_{{max}}={M}$', color=color, linestyle='-', 
                    capsize=0, capthick=2, markersize=6)
    
    ax1.set_xlabel('k (Number of Features)', fontsize = 16)
    ax1.set_ylabel('Accuracy', fontsize = 16)
    ax1.set_title('App Classification', fontsize = 18)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.legend(loc='best', fontsize = 15)
    ax1.set_facecolor('#f0f0f0')  # Gray background
    ax1.invert_xaxis()
    ax1.set_xticks(k_values)
    ax1.grid(True, alpha=1.0, color='white', linewidth=1.5)  # White grid lines
    ax1.text(0.01, 1.03, 'a)', transform=ax1.transAxes, fontsize=16)
    
    # Plot 2: DDoS Accuracy
    ax2 = axes[0, 1]
    for M in [5, 50]:
        subset = df[df['M'] == M]
        color = colors[str(M)]
        
        # Independent model
        means_single = subset.groupby('k')['acc_ddos_single'].mean()
        stds_single = subset.groupby('k')['acc_ddos_single'].std()
        ax2.errorbar(k_values, means_single, yerr=stds_single, marker='o', 
                    color=color, linestyle='--', 
                    capsize=0, capthick=2, alpha=0.7, markersize=6)
        
        # Joint model
        means_multi = subset.groupby('k')['acc_ddos_multi'].mean()
        stds_multi = subset.groupby('k')['acc_ddos_multi'].std()
        ax2.errorbar(k_values, means_multi, yerr=stds_multi, marker='s', 
                    color=color, linestyle='-', 
                    capsize=0, capthick=2, markersize=6)
    
    ax2.set_xlabel('k (Number of Features)', fontsize = 16)
    ax2.set_ylabel('Accuracy', fontsize = 16)
    ax2.set_title('DDoS Detection', fontsize = 18)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_facecolor('#f0f0f0')
    ax2.invert_xaxis()
    ax2.set_xticks(k_values)
    ax2.grid(True, alpha=1.0, color='white', linewidth=1.5)
    ax2.text(0.01, 1.03, 'b)', transform=ax2.transAxes, fontsize=16)
    
    # Plot 3: Blocks (Resource Usage)
    ax3 = axes[0, 2]
    for M in [5, 50]:
        subset = df[df['M'] == M]
        color = colors[str(M)]
        
        # Independent model
        means_single = subset.groupby('k')['blocks_single_total'].mean()
        stds_single = subset.groupby('k')['blocks_single_total'].std()
        ax3.errorbar(k_values, means_single, yerr=stds_single, marker='o', 
                    color=color, linestyle='--', 
                    capsize=0, capthick=2, alpha=0.7, markersize=6)
        
        # Joint model
        means_multi = subset.groupby('k')['blocks_multi_total'].mean()
        stds_multi = subset.groupby('k')['blocks_multi_total'].std()
        ax3.errorbar(k_values, means_multi, yerr=stds_multi, marker='s', 
                    color=color, linestyle='-', 
                    capsize=0, capthick=2, markersize=6)
    
    ax3.set_xlabel('k (Number of Features)', fontsize = 16)
    ax3.set_ylabel('Number of TCAM Blocks', fontsize = 16)
    ax3.set_title('TCAM Blocks', fontsize = 18)
    ax3.tick_params(axis='both', which='major', labelsize=15)
    ax3.set_facecolor('#f0f0f0')
    ax3.invert_xaxis()
    ax3.set_xticks(k_values)
    ax3.grid(True, alpha=1.0, color='white', linewidth=1.5)
    ax3.text(0.01, 1.03, 'c)', transform=ax3.transAxes, fontsize=16)

    df['blocks_improvement'] = ((df['blocks_single_total'] - df['blocks_multi_total']) / df['blocks_single_total']) * 100

    
    # Plot 4: App Accuracy Difference
    ax4 = axes[1, 0]
    for M in M_values:
        subset = df[df['M'] == M]
        
        means_diff = subset.groupby('k')['acc_app_diff'].mean()
        stds_diff = subset.groupby('k')['acc_app_diff'].std()
        ax4.errorbar(k_values, means_diff, yerr=stds_diff, marker='o', 
                    label=f'$M_{{max}}={M}$', linestyle='-', 
                    capsize=0, capthick=2, markersize=6)
    
    ax4.set_xlabel('k (Number of Features)', fontsize = 16)
    ax4.set_ylabel('Accuracy Difference', fontsize = 16)
    ax4.set_title('App Classification', fontsize = 18)
    ax4.legend(loc='best', fontsize = 15)
    ax4.set_facecolor('#f0f0f0')
    ax4.grid(True, alpha=1.0, color='white', linewidth=1.5)
    ax4.tick_params(axis='both', which='major', labelsize=15)
    ax4.set_xticks(k_values)
    ax4.invert_xaxis()
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.text(0.01, 1.03, 'd)', transform=ax4.transAxes, fontsize=16)
    
    # Plot 5: DDoS Accuracy Difference
    ax5 = axes[1, 1]
    for M in M_values:
        subset = df[df['M'] == M]
        
        means_diff = subset.groupby('k')['acc_ddos_diff'].mean()
        stds_diff = subset.groupby('k')['acc_ddos_diff'].std()
        ax5.errorbar(k_values, means_diff, yerr=stds_diff, marker='o', 
                    linestyle='-', 
                    capsize=0, capthick=2, markersize=6)
    
    ax5.set_xlabel('k (Number of Features)', fontsize = 16)
    ax5.set_ylabel('Accuracy Difference', fontsize = 16)
    ax5.set_title('DDoS Detection', fontsize = 18)
    ax5.set_facecolor('#f0f0f0')
    ax5.grid(True, alpha=1.0, color='white', linewidth=1.5)
    ax5.tick_params(axis='both', which='major', labelsize=15)
    ax5.set_xticks(k_values)
    ax5.invert_xaxis()
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.text(0.01, 1.03, 'e)', transform=ax5.transAxes, fontsize=16)
    
    # Plot 6: Blocks % Improvement
    ax6 = axes[1, 2]
    for M in M_values:
        subset = df[df['M'] == M]
        
        means_improvement = subset.groupby('k')['blocks_improvement'].mean()
        stds_improvement = subset.groupby('k')['blocks_improvement'].std()
        ax6.errorbar(k_values, means_improvement, yerr=stds_improvement, marker='o', 
                    linestyle='-', 
                    capsize=0, capthick=2, markersize=6)
    
    ax6.set_xlabel('k (Number of Features)', fontsize = 16)
    ax6.set_ylabel('Savings in Blocks, %', fontsize = 16)
    ax6.set_title('TCAM Blocks', fontsize = 18)
    ax6.set_facecolor('#f0f0f0')
    ax6.grid(True, alpha=1.0, color='white', linewidth=1.5)
    ax6.tick_params(axis='both', which='major', labelsize=15)
    ax6.set_xticks(k_values)
    ax6.invert_xaxis()
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax6.text(0.01, 1.03, 'f)', transform=ax6.transAxes, fontsize=16)
    
    # Adjust spacing between plots
    plt.tight_layout(h_pad=1, w_pad=1.0)
    plt.savefig('feature_selection_comparison_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_comparison_results_by_k(results_df, name, save_plots=True):
    """
    Create comprehensive plots comparing single-task vs multi-task approaches by number of features (k).
    """
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Calculate summary statistics grouped by k and gamma
    summary_stats = results_df.groupby(['k', 'gamma']).agg({
        'acc_avg_diff': ['mean', 'std'],
        'f1_avg_diff': ['mean', 'std'],
        'stages_total_diff': ['mean', 'std'],
        'blocks_total_diff': ['mean', 'std'],
    }).round(4)
    
    print("\nSummary Statistics (Multi-task - Single-task) by k and gamma:")
    print(summary_stats)
    
    # Create figure with subplots for different metrics by k
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Task vs Single-Task Feature Selection by Number of Features (k)\n(Positive = Multi-task Better)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Accuracy Difference by k
    ax1 = axes[0, 0]
    for gamma in results_df['gamma'].unique():
        data_gamma = results_df[results_df['gamma'] == gamma]
        means = data_gamma.groupby('k')['acc_avg_diff'].mean()
        stds = data_gamma.groupby('k')['acc_avg_diff'].std()
        ax1.errorbar(means.index, means.values, yerr=stds.values, 
                    marker='o', label=f'γ={gamma}', linewidth=2, markersize=8)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Number of Features (k)')
    ax1.set_ylabel('Accuracy Difference')
    ax1.set_title('Average Accuracy Difference vs Number of Features')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. F1-Score Difference by k
    ax2 = axes[0, 1]
    for gamma in results_df['gamma'].unique():
        data_gamma = results_df[results_df['gamma'] == gamma]
        means = data_gamma.groupby('k')['f1_avg_diff'].mean()
        stds = data_gamma.groupby('k')['f1_avg_diff'].std()
        ax2.errorbar(means.index, means.values, yerr=stds.values, 
                    marker='s', label=f'γ={gamma}', linewidth=2, markersize=8)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Number of Features (k)')
    ax2.set_ylabel('F1-Score Difference')
    ax2.set_title('Average F1-Score Difference vs Number of Features')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Resource Usage - Stages by k
    ax3 = axes[1, 0]
    sns.swarmplot(ax=ax3, data = results_df, x= 'k', y = 'stages_total_diff', hue = 'gamma')
    '''for gamma in results_df['gamma'].unique():
        data_gamma = results_df[results_df['gamma'] == gamma]
        means = data_gamma.groupby('k')['stages_total_diff'].mean()
        stds = data_gamma.groupby('k')['stages_total_diff'].std()
        ax3.errorbar(means.index, means.values, yerr=stds.values, 
                    marker='^', label=f'γ={gamma}', linewidth=2, markersize=8)'''
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Number of Features (k)')
    ax3.set_ylabel('Stages Difference')
    ax3.set_title('Average Stages Difference vs Number of Features')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Resource Usage - Blocks by k
    ax4 = axes[1, 1]
    sns.swarmplot(ax=ax4, data = results_df, x= 'k', y = 'blocks_total_diff', hue = 'gamma')
    '''for gamma in results_df['gamma'].unique():
        data_gamma = results_df[results_df['gamma'] == gamma]
        means = data_gamma.groupby('k')['blocks_total_diff'].mean()
        stds = data_gamma.groupby('k')['blocks_total_diff'].std()
        ax4.errorbar(means.index, means.values, yerr=stds.values, 
                    marker='d', label=f'γ={gamma}', linewidth=2, markersize=8)'''
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Number of Features (k)')
    ax4.set_ylabel('Blocks Difference')
    ax4.set_title('Average Blocks Difference vs Number of Features')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(name, dpi=300, bbox_inches='tight')
        print("\nPlot saved as 'multitask_vs_singletask_by_k.png'")
    
    #plt.show()
    
    # Create heatmap showing performance differences
    '''fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy heatmap
    acc_pivot = results_df.groupby(['k', 'gamma'])['acc_avg_diff'].mean().reset_index()
    acc_heatmap = acc_pivot.pivot(index='k', columns='gamma', values='acc_avg_diff')
    
    sns.heatmap(acc_heatmap, annot=True, fmt='.4f', cmap='RdBu_r', center=0, 
                ax=axes2[0], cbar_kws={'label': 'Accuracy Difference'})
    axes2[0].set_title('Average Accuracy Difference\n(Multi-task - Single-task)')
    axes2[0].set_xlabel('Gamma (γ)')
    axes2[0].set_ylabel('Number of Features (k)')
    
    # F1-score heatmap
    f1_pivot = results_df.groupby(['k', 'gamma'])['f1_avg_diff'].mean().reset_index()
    f1_heatmap = f1_pivot.pivot(index='k', columns='gamma', values='f1_avg_diff')
    
    sns.heatmap(f1_heatmap, annot=True, fmt='.4f', cmap='RdBu_r', center=0, 
                ax=axes2[1], cbar_kws={'label': 'F1-Score Difference'})
    axes2[1].set_title('Average F1-Score Difference\n(Multi-task - Single-task)')
    axes2[1].set_xlabel('Gamma (γ)')
    axes2[1].set_ylabel('Number of Features (k)')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('performance_heatmaps_by_k.png', dpi=300, bbox_inches='tight')
        print("Heatmaps saved as 'performance_heatmaps_by_k.png'")
    
    plt.show()'''
    
    return summary_stats


def create_multidim_visualizations(df, analysis_results, k_values):
    """
    Create comprehensive visualizations
    """
    # 1. Pareto fronts for each k

    plt.rcParams['font.family'] = 'Times New Roman'

    fig, axes = plt.subplots(3, 3, figsize=(16, 8))
    axes = axes.flatten()

    colors = ['#4682B4', '#FF8C00']  # Steel Blue, Dark Orange
    # Colors for individual tasks - using different shades
    task_colors = {
        'single': {'ddos': '#6495ED', 'app': '#87CEEB'},  # Cornflower blue, Sky blue
        'multi': {'ddos': '#FF8C00', 'app': '#FFB347'}     # Dark orange, Light orange
    }

    label_mapping = {
        'multi': 'Joint',
        'single': 'Independent'
    }

    for i, k in enumerate(k_values):
        ax = axes[i]
        k_results = analysis_results[f'k_{k}']

        # Create secondary y-axis for individual task accuracies
        ax2 = ax.twinx()

        # Extract data for this k
        k_df = df[df['k'] == k]

        # Plot Pareto fronts and individual task accuracies
        for j, (approach, front) in enumerate(k_results['pareto_fronts'].items()):
            if front:
                accs, mems = zip(*front)
                if k == 17:
                    filtered = [(acc, mem) for acc, mem in zip(accs, mems) if acc >= 0.8]
                    if filtered:
                        accs, mems = zip(*filtered)
                    else:
                        continue

                # Use the mapped label instead of the original approach name
                display_label = label_mapping.get(approach)
                ax.plot(mems, accs, 'o-', label=display_label, markersize=6, color=colors[j], zorder=3, linewidth=2)

                # Get individual task accuracies for Pareto front points
                approach_df = k_df[k_df['method'] == approach]
                acc_ddos_list = []
                acc_app_list = []
                sorted_mems = []

                for acc, mem in zip(accs, mems):
                    # Find matching points in the dataframe
                    matching = approach_df[
                        (np.abs(approach_df['blocks'] - mem) < 0.1) &
                        (np.abs((approach_df['acc_app'] + approach_df['acc_ddos']) / 2 - acc) < 0.001)
                    ]

                    if not matching.empty:
                        row = matching.iloc[0]
                        acc_ddos_list.append(row['acc_ddos'])
                        acc_app_list.append(row['acc_app'])
                        sorted_mems.append(mem)

                # Plot individual task accuracies on secondary axis
                if acc_ddos_list:
                    ax2.plot(sorted_mems, acc_ddos_list, 's--', markersize=4,
                            color=task_colors[approach]['ddos'], alpha=0.7, linewidth=1.5, zorder=2,
                            label=f'{display_label} DDoS' if i == 0 else '')
                    ax2.plot(sorted_mems, acc_app_list, '^--', markersize=4,
                            color=task_colors[approach]['app'], alpha=0.7, linewidth=1.5, zorder=2,
                            label=f'{display_label} App' if i == 0 else '')

        # Only show x-axis label on bottom row
        if i > 5:  # Last subplot (bottom)
            ax.set_xlabel('Memory (TCAM blocks)', fontsize = 20)

        # Only show y-axis label on left column
        if i == 0 or i == 3 or i == 6:  # First subplot (leftmost)
            ax.set_ylabel('Average accuracy', fontsize = 20)

        # Only show secondary y-axis label on right column
        if i == 2 or i == 5 or i == 8:  # Rightmost column
            ax2.set_ylabel('Task accuracy', fontsize = 18, rotation=270, labelpad=25)
        else:
            ax2.set_yticklabels([])

        ax.set_title(f'k = {k} features', fontsize = 20)
        if i == 0:
            # Create combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize = 14, loc='lower right', ncol=2)

        ax.set_xlim(0, 100)

        if i > 5:
            ax.set_ylim(0.7, 1.0)
            ax.set_yticks(np.arange(0.7, 1.0, 0.05))
        else:
            ax.set_ylim(0.7, 1.0)
            ax.set_yticks(np.arange(0.7, 1.0, 0.05))

        # Set limits for secondary axis to match primary axis
        if i > 5:
            ax2.set_ylim(0.7, 1.0)
        else:
            ax2.set_ylim(0.7, 1.0)

        ax.set_facecolor('#f0f0f0')
        ax.grid(True, alpha=1.0, color='white', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=19)
        ax2.tick_params(axis='y', which='major', labelsize=17)

    plt.tight_layout()
    plt.savefig('pareto_fronts_by_k.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    
    # 2. Hypervolume comparison
    fig = plt.figure(figsize=(16, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.2)  # Reduced wspace, adjusted ratios
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Left plot: Hypervolume gain
    k_list = []
    hv_gain_percent = []

    for k in k_values:
        k_results = analysis_results[f'k_{k}']
        k_list.append(k)
        
        hv_single = k_results['hypervolume']['single']
        hv_multi = k_results['hypervolume']['multi']
        
        gain_percent = ((hv_multi - hv_single) / hv_single) * 100
        hv_gain_percent.append(gain_percent)

    x = np.arange(len(k_list))

    ax1.bar(x, hv_gain_percent, color='#228B22', alpha=0.7)
    ax1.set_xlabel('Number of features (k)', fontsize=20)
    ax1.set_ylabel('Hypervolume gain, %', fontsize=20)
    ax1.set_title('Hypervolume Gain (Joint vs Independent)', fontsize=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(k_list)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Right plot: Coverage ratios heatmap
    coverage_matrix = np.zeros((len(k_values), 2))
    for i, k in enumerate(k_values):
        k_results = analysis_results[f'k_{k}']
        coverage = k_results['coverage_ratio']
        coverage_matrix[i, 0] = coverage['single_covers_multi']
        coverage_matrix[i, 1] = coverage['multi_covers_single']

    xlabels = ['Independent\ncovers Joint', 'Joint covers\nIndependent']

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax2)
    cbar_ax = divider.append_axes("right", size="6%", pad=0.6)

    heatmap = sns.heatmap(
        coverage_matrix,
        xticklabels=xlabels,
        yticklabels=[f'{k}' for k in k_values],
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        ax=ax2,
        annot_kws={"size": 17},
        cbar=True,
        cbar_ax=cbar_ax
    )

    # Colorbar tick font size
    cbar_ax.tick_params(labelsize=16)

    # Labels ABOVE the colorbar - increased font size
    cbar_ax.text(0.5, 1.03, 'Outperforms all\ncompeting solutions', 
                ha='center', va='bottom', fontsize=18, transform=cbar_ax.transAxes)
    cbar_ax.text(0.5, -0.03, 'Outperforms none\ncompeting solutions', 
                ha='center', va='top', fontsize=18, transform=cbar_ax.transAxes)

    ax2.set_title('Coverage Ratios', fontsize=20)
    ax2.set_ylabel('Number of features (k)', fontsize=20)
    ax2.tick_params(axis='x', which='major', labelsize=18, rotation=0)
    ax2.tick_params(axis='y', which='major', labelsize=19)

    # Subplot labels
    ax1.text(-0.05, 1.1, 'a)', transform=ax1.transAxes, fontsize=21, 
            verticalalignment='top', horizontalalignment='left')
    ax2.text(-0.05, 1.1, 'b)', transform=ax2.transAxes, fontsize=21,
            verticalalignment='top', horizontalalignment='left')

    plt.tight_layout()

    plt.tight_layout()
    plt.savefig('combined_metrics.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Summary statistics table
    print_multidim_summary_table(analysis_results, k_values)


def print_multidim_summary_table(analysis_results, k_values):
    """
    Print a summary table of key metrics
    """
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    # Overall results
    all_k = analysis_results['all_k']
    
    print("\nOVERALL METRICS (All k combined):")
    print(f"Hypervolume - Single: {all_k['hypervolume']['single']:.4f}")
    print(f"Hypervolume - Multi: {all_k['hypervolume']['multi']:.4f}")
    
    coverage = all_k['coverage_ratio']
    print(f"\nCoverage Ratios:")
    print(f"  Single covers Multi: {coverage['single_covers_multi']:.3f}")
    print(f"  Multi covers Single: {coverage['multi_covers_single']:.3f}")
    
    if 'overall' in all_k['statistical_tests']:
        stat_test = all_k['statistical_tests']['overall']
        print(f"\nStatistical Test (Wilcoxon):")
        print(f"  p-value: {stat_test['p_value']:.4f}")
        print(f"  Significant: {stat_test['significant']}")
        print(f"  Median difference: {stat_test['median_diff']:.4f}")
    
    # Results by k
    print("\n" + "-"*80)
    print("RESULTS BY NUMBER OF FEATURES (k):")
    print("-"*80)
    
    headers = ['k', 'HV Single', 'HV Multi', 'Coverage S→M', 'Coverage M→S']
    print(f"{headers[0]:>3} | {headers[1]:>10} | {headers[2]:>10} | {headers[3]:>12} | {headers[4]:>12}")
    print("-"*60)
    
    for k in k_values:
        k_results = analysis_results[f'k_{k}']
        hv_s = k_results['hypervolume']['single']
        hv_m = k_results['hypervolume']['multi']
        cov = k_results['coverage_ratio']
        
        print(f"{k:3d} | {hv_s:10.4f} | {hv_m:10.4f} | {cov['single_covers_multi']:12.3f} | {cov['multi_covers_single']:12.3f}")

    # 3. Print summary of accuracy differences across all k
    print('\n' + '='*70)
    print('SUMMARY: Accuracy Differences Across All k Values')
    print('='*70)

    for approach in ['single', 'multi']:
        approach_name = 'Independent' if approach == 'single' else 'Joint'

        # Collect all avg and max values across k
        all_avgs = []
        all_maxs = []
        total_points = 0

        for k in k_values:
            acc_diff = analysis_results[f'k_{k}']['accuracy_differences'][approach]
            if acc_diff['n_points'] > 0:
                all_avgs.append(acc_diff['avg'])
                all_maxs.append(acc_diff['max'])
                total_points += acc_diff['n_points']

        print(f"\n{approach_name} Encoding:")
        if all_avgs:
            print(all_avgs)
            print(all_maxs)
            print(f"  Average across k: {np.mean(all_avgs):.4f}")
            print(f"  Max across k: {np.max(all_maxs):.4f}")
            print(f"  Total Pareto front points across all k: {total_points}")
        else:
            print(f"  No data available")

        # Also report from the combined 'all_k' analysis
        all_k_diff = analysis_results['all_k']['accuracy_differences'][approach]
        print(f"\n  Combined analysis (all k together):")
        print(f"    Average: {all_k_diff['avg']:.4f}")
        print(f"    Max: {all_k_diff['max']:.4f}")
        print(f"    Pareto front points: {all_k_diff['n_points']}")

    print('\n' + '='*70)
