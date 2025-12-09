import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Parse the result.txt file
def parse_results():
    results = []
    
    # C101 results
    c101_data = [
        {'dataset': 'C101', 'customers': 10, 'time': 0.037, 'distance': 58.33, 'vehicles': 1},
        {'dataset': 'C101', 'customers': 20, 'time': 0.123, 'distance': 162.04, 'vehicles': 2},
        {'dataset': 'C101', 'customers': 30, 'time': 0.265, 'distance': 207.40, 'vehicles': 3},
        {'dataset': 'C101', 'customers': 40, 'time': 0.766, 'distance': 346.15, 'vehicles': 5},
        {'dataset': 'C101', 'customers': 50, 'time': 0.919, 'distance': 364.76, 'vehicles': 5},
    ]
    
    # R101 results
    r101_data = [
        {'dataset': 'R101', 'customers': 10, 'time': 0.051, 'distance': 253.07, 'vehicles': 3},
        {'dataset': 'R101', 'customers': 20, 'time': 0.385, 'distance': 480.43, 'vehicles': 6},
        {'dataset': 'R101', 'customers': 30, 'time': 0.451, 'distance': 657.33, 'vehicles': 9},
        {'dataset': 'R101', 'customers': 40, 'time': 0.629, 'distance': 820.47, 'vehicles': 9},
        {'dataset': 'R101', 'customers': 50, 'time': 2.120, 'distance': 985.37, 'vehicles': 11},
    ]
    
    # RC101 results
    rc101_data = [
        {'dataset': 'RC101', 'customers': 10, 'time': 0.299, 'distance': 183.14, 'vehicles': 2},
        {'dataset': 'RC101', 'customers': 20, 'time': 10.011, 'distance': 328.33, 'vehicles': 3},
        {'dataset': 'RC101', 'customers': 30, 'time': 10.012, 'distance': 488.79, 'vehicles': 4},
        {'dataset': 'RC101', 'customers': 40, 'time': 10.018, 'distance': 606.43, 'vehicles': 5},
        {'dataset': 'RC101', 'customers': 50, 'time': 10.030, 'distance': 974.52, 'vehicles': 9},
    ]
    
    results.extend(c101_data)
    results.extend(r101_data)
    results.extend(rc101_data)
    
    return pd.DataFrame(results)


def plot_comprehensive_analysis(df):
    """Create comprehensive visualization of the results"""
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Total Distance vs Customer Number
    ax1 = fig.add_subplot(gs[0, 0])
    for dataset in df['dataset'].unique():
        data_subset = df[df['dataset'] == dataset]
        ax1.plot(data_subset['customers'], data_subset['distance'], 
                marker='o', label=dataset, linewidth=2.5, markersize=8)
    ax1.set_xlabel('Number of Customers', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Distance', fontsize=11, fontweight='bold')
    ax1.set_title('Total Distance vs Problem Scale', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of Vehicles vs Customer Number
    ax2 = fig.add_subplot(gs[0, 1])
    for dataset in df['dataset'].unique():
        data_subset = df[df['dataset'] == dataset]
        ax2.plot(data_subset['customers'], data_subset['vehicles'], 
                marker='s', label=dataset, linewidth=2.5, markersize=8)
    ax2.set_xlabel('Number of Customers', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Vehicles', fontsize=11, fontweight='bold')
    ax2.set_title('Vehicle Usage vs Problem Scale', fontsize=12, fontweight='bold')
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Execution Time vs Customer Number
    ax3 = fig.add_subplot(gs[0, 2])
    for dataset in df['dataset'].unique():
        data_subset = df[df['dataset'] == dataset]
        ax3.plot(data_subset['customers'], data_subset['time'], 
                marker='^', label=dataset, linewidth=2.5, markersize=8)
    ax3.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Time Limit', alpha=0.6)
    ax3.set_xlabel('Number of Customers', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax3.set_title('Computational Time vs Problem Scale', fontsize=12, fontweight='bold')
    ax3.legend(frameon=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Distance per Customer
    ax4 = fig.add_subplot(gs[1, 0])
    df['distance_per_customer'] = df['distance'] / df['customers']
    for dataset in df['dataset'].unique():
        data_subset = df[df['dataset'] == dataset]
        ax4.plot(data_subset['customers'], data_subset['distance_per_customer'], 
                marker='D', label=dataset, linewidth=2.5, markersize=8)
    ax4.set_xlabel('Number of Customers', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Distance per Customer', fontsize=11, fontweight='bold')
    ax4.set_title('Average Distance per Customer', fontsize=12, fontweight='bold')
    ax4.legend(frameon=True, shadow=True)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Customers per Vehicle
    ax5 = fig.add_subplot(gs[1, 1])
    df['customers_per_vehicle'] = df['customers'] / df['vehicles']
    for dataset in df['dataset'].unique():
        data_subset = df[df['dataset'] == dataset]
        ax5.plot(data_subset['customers'], data_subset['customers_per_vehicle'], 
                marker='p', label=dataset, linewidth=2.5, markersize=8)
    ax5.set_xlabel('Number of Customers', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Customers per Vehicle', fontsize=11, fontweight='bold')
    ax5.set_title('Vehicle Utilization Efficiency', fontsize=12, fontweight='bold')
    ax5.legend(frameon=True, shadow=True)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Bar chart comparison at customer=50
    ax6 = fig.add_subplot(gs[1, 2])
    df_50 = df[df['customers'] == 50]
    x_pos = np.arange(len(df_50))
    width = 0.25
    
    ax6.bar(x_pos - width, df_50['distance']/10, width, label='Distance (÷10)', alpha=0.8)
    ax6.bar(x_pos, df_50['vehicles']*10, width, label='Vehicles (×10)', alpha=0.8)
    ax6.bar(x_pos + width, df_50['time']*10, width, label='Time (×10)', alpha=0.8)
    
    ax6.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Scaled Values', fontsize=11, fontweight='bold')
    ax6.set_title('Comparison at 50 Customers (Scaled)', fontsize=12, fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(df_50['dataset'])
    ax6.legend(frameon=True, shadow=True)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Heatmap - Distance
    ax7 = fig.add_subplot(gs[2, 0])
    pivot_distance = df.pivot(index='dataset', columns='customers', values='distance')
    sns.heatmap(pivot_distance, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax7, 
                cbar_kws={'label': 'Distance'}, linewidths=0.5)
    ax7.set_title('Distance Heatmap', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Number of Customers', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Dataset', fontsize=11, fontweight='bold')
    
    # Plot 8: Heatmap - Vehicles
    ax8 = fig.add_subplot(gs[2, 1])
    pivot_vehicles = df.pivot(index='dataset', columns='customers', values='vehicles')
    sns.heatmap(pivot_vehicles, annot=True, fmt='d', cmap='Blues', ax=ax8, 
                cbar_kws={'label': 'Vehicles'}, linewidths=0.5)
    ax8.set_title('Vehicle Count Heatmap', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Number of Customers', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Dataset', fontsize=11, fontweight='bold')
    
    # Plot 9: Heatmap - Time
    ax9 = fig.add_subplot(gs[2, 2])
    pivot_time = df.pivot(index='dataset', columns='customers', values='time')
    sns.heatmap(pivot_time, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax9, 
                cbar_kws={'label': 'Time (s)'}, linewidths=0.5)
    ax9.set_title('Execution Time Heatmap', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Number of Customers', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Dataset', fontsize=11, fontweight='bold')
    
    plt.suptitle('VRPTW Solver Performance Analysis: C101 vs R101 vs RC101', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('plots/results_analysis.png', dpi=300, bbox_inches='tight')
    print("Comprehensive analysis plot saved to 'plots/results_analysis.png'")
    plt.close()


def plot_scaling_analysis(df):
    """Create scaling efficiency plots"""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Distance growth rate
    ax1 = axes[0, 0]
    for dataset in df['dataset'].unique():
        data_subset = df[df['dataset'] == dataset].sort_values('customers')
        growth_rate = data_subset['distance'].pct_change() * 100
        ax1.plot(data_subset['customers'].iloc[1:], growth_rate.iloc[1:], 
                marker='o', label=dataset, linewidth=2.5, markersize=8)
    ax1.set_xlabel('Number of Customers', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Distance Growth Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Distance Growth Rate Between Scales', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot 2: Complexity comparison (Distance × Vehicles × Time)
    ax2 = axes[0, 1]
    df['complexity'] = df['distance'] * df['vehicles'] * df['time']
    for dataset in df['dataset'].unique():
        data_subset = df[df['dataset'] == dataset]
        ax2.plot(data_subset['customers'], data_subset['complexity'], 
                marker='s', label=dataset, linewidth=2.5, markersize=8)
    ax2.set_xlabel('Number of Customers', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Complexity Index', fontsize=12, fontweight='bold')
    ax2.set_title('Problem Complexity (Distance × Vehicles × Time)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Efficiency scatter plot
    ax3 = axes[1, 0]
    colors = {'C101': 'red', 'R101': 'blue', 'RC101': 'green'}
    for dataset in df['dataset'].unique():
        data_subset = df[df['dataset'] == dataset]
        scatter = ax3.scatter(data_subset['time'], data_subset['distance'], 
                             s=data_subset['vehicles']*50, alpha=0.6, 
                             c=colors[dataset], label=dataset, edgecolors='black', linewidth=1.5)
    ax3.set_xlabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Total Distance', fontsize=12, fontweight='bold')
    ax3.set_title('Solution Quality vs Time (bubble size = vehicles)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Normalized comparison
    ax4 = axes[1, 1]
    df_normalized = df.copy()
    for col in ['distance', 'vehicles', 'time']:
        df_normalized[f'{col}_norm'] = df.groupby('dataset')[col].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
    
    x = np.arange(len(df['customers'].unique()))
    width = 0.25
    
    for i, dataset in enumerate(df['dataset'].unique()):
        data_subset = df_normalized[df_normalized['dataset'] == dataset]
        avg_norm = (data_subset['distance_norm'] + data_subset['vehicles_norm'] + 
                   data_subset['time_norm']) / 3
        ax4.plot(data_subset['customers'], avg_norm, 
                marker='o', label=dataset, linewidth=2.5, markersize=8)
    
    ax4.set_xlabel('Number of Customers', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Normalized Average Score', fontsize=12, fontweight='bold')
    ax4.set_title('Overall Normalized Performance', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/scaling_analysis.png', dpi=300, bbox_inches='tight')
    print("Scaling analysis plot saved to 'plots/scaling_analysis.png'")
    plt.close()


def print_statistics(df):
    """Print summary statistics"""
    # Calculate derived metrics
    df['distance_per_customer'] = df['distance'] / df['customers']
    df['customers_per_vehicle'] = df['customers'] / df['vehicles']
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for dataset in df['dataset'].unique():
        data_subset = df[df['dataset'] == dataset]
        print(f"\n{dataset}:")
        print(f"  Distance range: {data_subset['distance'].min():.2f} - {data_subset['distance'].max():.2f}")
        print(f"  Vehicle range: {data_subset['vehicles'].min()} - {data_subset['vehicles'].max()}")
        print(f"  Time range: {data_subset['time'].min():.3f}s - {data_subset['time'].max():.3f}s")
        print(f"  Avg distance per customer: {data_subset['distance_per_customer'].mean():.2f}")
        print(f"  Avg customers per vehicle: {data_subset['customers_per_vehicle'].mean():.2f}")
    
    print("\n" + "="*70)
    print("COMPARISON AT 50 CUSTOMERS")
    print("="*70)
    df_50 = df[df['customers'] == 50].sort_values('distance')
    print(f"\n{'Dataset':<10} {'Distance':<12} {'Vehicles':<10} {'Time (s)':<10}")
    print("-"*50)
    for _, row in df_50.iterrows():
        print(f"{row['dataset']:<10} {row['distance']:<12.2f} {row['vehicles']:<10} {row['time']:<10.3f}")
    
    print("\n" + "="*70)
    print("GROWTH ANALYSIS (from 10 to 50 customers)")
    print("="*70)
    for dataset in df['dataset'].unique():
        data_subset = df[df['dataset'] == dataset].sort_values('customers')
        dist_growth = ((data_subset['distance'].iloc[-1] / data_subset['distance'].iloc[0]) - 1) * 100
        veh_growth = ((data_subset['vehicles'].iloc[-1] / data_subset['vehicles'].iloc[0]) - 1) * 100
        time_growth = ((data_subset['time'].iloc[-1] / data_subset['time'].iloc[0]) - 1) * 100
        print(f"\n{dataset}:")
        print(f"  Distance growth: {dist_growth:.1f}%")
        print(f"  Vehicle growth: {veh_growth:.1f}%")
        print(f"  Time growth: {time_growth:.1f}%")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Load and parse data
    df = parse_results()
    
    # Print statistics
    print_statistics(df)
    
    # Create visualizations
    plot_comprehensive_analysis(df)
    plot_scaling_analysis(df)
    
    # Save data to CSV
    df.to_csv('parsed_results.csv', index=False)
    print("\nParsed data saved to 'parsed_results.csv'")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
