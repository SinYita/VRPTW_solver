import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Parse RC101 time limit results (50 customers)
def parse_rc101_results():
    results = [
        {'time_limit': 3, 'exec_time': 3.025, 'distance': 1175.82, 'vehicles': 11},
        {'time_limit': 5, 'exec_time': 5.028, 'distance': 1022.05, 'vehicles': 10},
        {'time_limit': 7, 'exec_time': 7.096, 'distance': 1014.41, 'vehicles': 10},
        {'time_limit': 10, 'exec_time': 10.027, 'distance': 965.61, 'vehicles': 9},
        {'time_limit': 15, 'exec_time': 15.032, 'distance': 827.99, 'vehicles': 8},
        {'time_limit': 30, 'exec_time': 30.040, 'distance': 709.86, 'vehicles': 6},
    ]
    return pd.DataFrame(results)


def plot_time_analysis(df):
    """Create comprehensive time limit analysis"""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Total Distance vs Time Limit
    ax1 = axes[0, 0]
    ax1.plot(df['time_limit'], df['distance'], marker='o', linewidth=3, 
             markersize=10, color='#E74C3C', label='Total Distance')
    ax1.fill_between(df['time_limit'], df['distance'], alpha=0.3, color='#E74C3C')
    ax1.set_xlabel('Time Limit (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Distance', fontsize=12, fontweight='bold')
    ax1.set_title('Solution Quality vs Time Limit\nRC101 (50 customers)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add improvement percentages
    for i in range(1, len(df)):
        improvement = ((df.iloc[i-1]['distance'] - df.iloc[i]['distance']) / 
                      df.iloc[i-1]['distance'] * 100)
        mid_x = (df.iloc[i-1]['time_limit'] + df.iloc[i]['time_limit']) / 2
        mid_y = (df.iloc[i-1]['distance'] + df.iloc[i]['distance']) / 2
        ax1.annotate(f'-{improvement:.1f}%', xy=(mid_x, mid_y), 
                    fontsize=9, color='green', fontweight='bold')
    
    # Plot 2: Number of Vehicles vs Time Limit
    ax2 = axes[0, 1]
    bars = ax2.bar(df['time_limit'], df['vehicles'], color='#3498DB', 
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_xlabel('Time Limit (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Vehicles', fontsize=12, fontweight='bold')
    ax2.set_title('Vehicle Count vs Time Limit\nRC101 (50 customers)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 3: Distance per Vehicle
    ax3 = axes[0, 2]
    df['distance_per_vehicle'] = df['distance'] / df['vehicles']
    ax3.plot(df['time_limit'], df['distance_per_vehicle'], marker='s', 
             linewidth=3, markersize=10, color='#9B59B6')
    ax3.fill_between(df['time_limit'], df['distance_per_vehicle'], alpha=0.3, color='#9B59B6')
    ax3.set_xlabel('Time Limit (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Distance per Vehicle', fontsize=12, fontweight='bold')
    ax3.set_title('Route Efficiency vs Time Limit\nRC101 (50 customers)', 
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Improvement Rate
    ax4 = axes[1, 0]
    df['improvement_rate'] = 0.0
    for i in range(1, len(df)):
        time_diff = df.iloc[i]['time_limit'] - df.iloc[i-1]['time_limit']
        dist_diff = df.iloc[i-1]['distance'] - df.iloc[i]['distance']
        df.loc[i, 'improvement_rate'] = dist_diff / time_diff
    
    colors = ['green' if x > 0 else 'red' for x in df['improvement_rate'].iloc[1:]]
    bars = ax4.bar(df['time_limit'].iloc[1:], df['improvement_rate'].iloc[1:], 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_xlabel('Time Limit (seconds)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Distance Improvement per Second', fontsize=12, fontweight='bold')
    ax4.set_title('Marginal Improvement Rate\nRC101 (50 customers)', 
                  fontsize=14, fontweight='bold')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=9, fontweight='bold')
    
    # Plot 5: Cumulative Improvement
    ax5 = axes[1, 1]
    df['total_improvement'] = df['distance'].iloc[0] - df['distance']
    df['improvement_pct'] = (df['total_improvement'] / df['distance'].iloc[0]) * 100
    
    ax5.plot(df['time_limit'], df['improvement_pct'], marker='D', 
             linewidth=3, markersize=10, color='#16A085')
    ax5.fill_between(df['time_limit'], df['improvement_pct'], alpha=0.3, color='#16A085')
    ax5.set_xlabel('Time Limit (seconds)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Total Improvement (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Cumulative Distance Reduction\nRC101 (50 customers)', 
                  fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, row in df.iterrows():
        ax5.annotate(f'{row["improvement_pct"]:.1f}%', 
                    xy=(row['time_limit'], row['improvement_pct']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    # Plot 6: Summary Table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            f"{int(row['time_limit'])}s",
            f"{row['distance']:.2f}",
            f"{int(row['vehicles'])}",
            f"{row['improvement_pct']:.1f}%"
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Time', 'Distance', 'Vehicles', 'Improve'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header
    for i in range(4):
        table[(0, i)].set_facecolor('#3498DB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style the cells
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ECF0F1')
    
    ax6.set_title('Summary Statistics\nRC101 (50 customers)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('plots/rc101_time_analysis.png', dpi=300, bbox_inches='tight')
    print("Time analysis plot saved to 'plots/rc101_time_analysis.png'")
    plt.close()


def plot_comparative_analysis(df):
    """Create comparative analysis plots"""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Multi-metric comparison
    ax1 = axes[0]
    
    # Normalize all metrics to 0-100 scale for comparison
    df_norm = df.copy()
    df_norm['distance_norm'] = 100 * (1 - (df['distance'] - df['distance'].min()) / 
                                       (df['distance'].max() - df['distance'].min()))
    df_norm['vehicles_norm'] = 100 * (1 - (df['vehicles'] - df['vehicles'].min()) / 
                                       (df['vehicles'].max() - df['vehicles'].min()))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df_norm['distance_norm'], width, 
                    label='Solution Quality', color='#E74C3C', alpha=0.8)
    bars2 = ax1.bar(x + width/2, df_norm['vehicles_norm'], width, 
                    label='Vehicle Efficiency', color='#3498DB', alpha=0.8)
    
    ax1.set_xlabel('Time Limit (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Score (0-100)', fontsize=12, fontweight='bold')
    ax1.set_title('Multi-Metric Performance Comparison\nRC101 (50 customers)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(t)}s" for t in df['time_limit']])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Cost-Benefit Analysis
    ax2 = axes[1]
    
    # Calculate benefit (distance reduction) vs cost (time)
    df['benefit'] = df['distance'].iloc[0] - df['distance']
    df['cost_benefit_ratio'] = df['benefit'] / df['time_limit']
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df)))
    scatter = ax2.scatter(df['time_limit'], df['benefit'], 
                         s=df['cost_benefit_ratio']*50, 
                         c=range(len(df)), cmap='RdYlGn',
                         alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, row in df.iterrows():
        ax2.annotate(f"{row['cost_benefit_ratio']:.1f}",
                    xy=(row['time_limit'], row['benefit']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Time Investment (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Distance Reduction', fontsize=12, fontweight='bold')
    ax2.set_title('Cost-Benefit Analysis\nRC101 (50 customers)\n(bubble size = benefit/cost ratio)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/rc101_comparative_analysis.png', dpi=300, bbox_inches='tight')
    print("Comparative analysis plot saved to 'plots/rc101_comparative_analysis.png'")
    plt.close()


def print_detailed_analysis(df):
    """Print detailed statistical analysis"""
    # Calculate derived metrics
    df['improvement_rate'] = 0.0
    for i in range(1, len(df)):
        time_diff = df.iloc[i]['time_limit'] - df.iloc[i-1]['time_limit']
        dist_diff = df.iloc[i-1]['distance'] - df.iloc[i]['distance']
        df.loc[i, 'improvement_rate'] = dist_diff / time_diff
    
    df['benefit'] = df['distance'].iloc[0] - df['distance']
    df['cost_benefit_ratio'] = df['benefit'] / df['time_limit']
    
    print("\n" + "="*80)
    print("RC101 TIME LIMIT ANALYSIS (50 Customers)")
    print("="*80)
    
    print("\nDETAILED RESULTS:")
    print("-"*80)
    print(f"{'Time':<8} {'Distance':<12} {'Vehicles':<10} {'Exec Time':<12} {'Status':<15}")
    print("-"*80)
    for _, row in df.iterrows():
        status = "Time Limit" if row['exec_time'] >= row['time_limit'] - 0.1 else "Early Stop"
        print(f"{int(row['time_limit'])}s{'':<5} {row['distance']:<12.2f} {int(row['vehicles']):<10} "
              f"{row['exec_time']:<12.3f} {status:<15}")
    
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS:")
    print("="*80)
    
    baseline_distance = df['distance'].iloc[0]
    baseline_vehicles = df['vehicles'].iloc[0]
    
    for i in range(1, len(df)):
        prev_row = df.iloc[i-1]
        curr_row = df.iloc[i]
        
        time_increase = curr_row['time_limit'] - prev_row['time_limit']
        dist_improvement = prev_row['distance'] - curr_row['distance']
        dist_improvement_pct = (dist_improvement / prev_row['distance']) * 100
        vehicle_reduction = prev_row['vehicles'] - curr_row['vehicles']
        
        print(f"\n{int(prev_row['time_limit'])}s → {int(curr_row['time_limit'])}s "
              f"(+{time_increase:.0f}s additional time):")
        print(f"  Distance: {prev_row['distance']:.2f} → {curr_row['distance']:.2f} "
              f"(-{dist_improvement:.2f}, -{dist_improvement_pct:.1f}%)")
        print(f"  Vehicles: {int(prev_row['vehicles'])} → {int(curr_row['vehicles'])} "
              f"({int(vehicle_reduction):+d})")
        print(f"  Marginal benefit: {dist_improvement/time_increase:.2f} distance units per second")
    
    print("\n" + "="*80)
    print("OVERALL SUMMARY:")
    print("="*80)
    
    total_improvement = baseline_distance - df['distance'].iloc[-1]
    total_improvement_pct = (total_improvement / baseline_distance) * 100
    vehicle_reduction = int(baseline_vehicles - df['vehicles'].iloc[-1])
    
    print(f"\nFrom 3s to 30s time limit:")
    print(f"  Distance improvement: {total_improvement:.2f} ({total_improvement_pct:.1f}%)")
    print(f"  Vehicle reduction: {vehicle_reduction} ({vehicle_reduction/baseline_vehicles*100:.1f}%)")
    print(f"  Average improvement rate: {total_improvement/27:.2f} distance units per second")
    
    # Find best cost-benefit ratio
    best_idx = df['cost_benefit_ratio'].iloc[1:].idxmax()
    best_row = df.loc[best_idx]
    print(f"\nBest cost-benefit ratio: {int(best_row['time_limit'])}s time limit")
    print(f"  (Ratio: {best_row['cost_benefit_ratio']:.2f} distance units per second)")
    
    # Diminishing returns analysis
    print("\n" + "="*80)
    print("DIMINISHING RETURNS ANALYSIS:")
    print("="*80)
    
    for i in range(1, len(df)):
        rate = df.loc[i, 'improvement_rate']
        if i == 1:
            print(f"\nInitial improvement rate (3s→{int(df.loc[i, 'time_limit'])}s): {rate:.2f} units/s")
        else:
            prev_rate = df.loc[i-1, 'improvement_rate']
            decline = ((prev_rate - rate) / prev_rate) * 100 if prev_rate != 0 else 0
            print(f"Rate at {int(df.loc[i, 'time_limit'])}s: {rate:.2f} units/s "
                  f"({decline:+.1f}% vs previous)")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    df = parse_rc101_results()
    
    # Print analysis
    print_detailed_analysis(df)
    
    # Create visualizations
    plot_time_analysis(df)
    plot_comparative_analysis(df)
    
    # Save data
    df.to_csv('rc101_time_analysis.csv', index=False)
    print("\nData saved to 'rc101_time_analysis.csv'")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - plots/rc101_time_analysis.png")
    print("  - plots/rc101_comparative_analysis.png")
    print("  - rc101_time_analysis.csv")
