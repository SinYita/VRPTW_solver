import os
os.environ['GRB_LICENSE_FILE'] = '/Users/weiyuandu/gurobi.lic'
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from Solver import readData, Model, GRB, quicksum, tuplelist, product, getSolution


def solve_vrptw(data, M=10000, time_limit=None):
    """使用与Solver.py相同的求解逻辑"""
    model = Model('VRPTW')
    model.setParam('MIPGap', 0.05)
    model.setParam('OutputFlag', 0)
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)
    
    # Step1: 建立变量索引集合
    X_set = []
    S_set = []
    k_set = [k for k in range(data.vehicleNum)]
    i_set = [i for i in range(data.nodeNum - 1)]
    j_set = [j for j in range(data.nodeNum)]
    
    for k in k_set:
        for i in i_set:
            for j in j_set:
                if i != j:
                    X_set.append((i, j, k))
    
    for i in range(data.nodeNum):
        for k in range(data.vehicleNum):
            S_set.append((i, k))
    
    X_set_tplst = tuplelist(X_set)
    S_set_tplst = tuplelist(S_set)
    
    # Step2: 定义变量
    x = model.addVars(X_set_tplst, vtype=GRB.BINARY, name='x')
    s = model.addVars(S_set_tplst, vtype=GRB.CONTINUOUS, lb=0.0, name='s')
    model.update()
    
    # 定义目标函数
    model.setObjective(
        quicksum(x[i, j, k] * data.distanceMatrix[i][j] for i, j, k in X_set_tplst),
        sense=GRB.MINIMIZE
    )
    
    # 定义约束条件
    customer_ids = [i for i in range(1, data.nodeNum - 1)]
    model.addConstrs(
        (quicksum(x[i, j, k] for i, j, k in X_set_tplst.select(I, '*', '*')) == 1 for I in customer_ids),
        'customer_once'
    )
    
    model.addConstrs(
        (quicksum(x[i, j, k] for i, j, k in X_set_tplst.select(0, '*', K)) == 1 for K in k_set),
        'start_depot'
    )
    
    model.addConstrs(
        (quicksum(x[i, j, k] for i, j, k in X_set_tplst.select('*', data.nodeNum - 1, K)) == 1 for K in k_set),
        'end_depot'
    )
    
    model.addConstrs(
        (quicksum(x[i, h, k] for i, h, k in X_set_tplst.select('*', H, K)) - 
         quicksum(x[h, j, k] for h, j, k in X_set_tplst.select(H, '*', K)) == 0 
         for H, K in product(customer_ids, k_set)),
        'flow_balance'
    )
    
    model.addConstrs(
        (s[i, k] + data.distanceMatrix[i][j] - M * (1 - x[i, j, k]) <= s[j, k] 
         for i, j, k in X_set_tplst),
        'time_window_constraint'
    )
    
    model.addConstrs(
        (s[i, k] >= data.readyTime[i] for i, k in S_set_tplst),
        'ready_time'
    )
    model.addConstrs(
        (s[i, k] <= data.dueTime[i] for i, k in S_set_tplst),
        'due_time'
    )
    
    model.addConstrs(
        (quicksum(data.demand[i] * x[i, j, k] for i, j, k in X_set_tplst.select('*', '*', K)) <= data.capacity 
         for K in k_set),
        'capacity'
    )
    
    # 求解
    start_time = time.time()
    model.optimize()
    exec_time = time.time() - start_time
    
    # 提取解
    if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
        solution = getSolution(data, model)
        return {
            'total_distance': model.ObjVal,
            'num_vehicles': solution.routeNum,
            'exec_time': exec_time,
            'status': 'Optimal' if model.status == GRB.OPTIMAL else 'Time Limit'
        }
    else:
        return {
            'total_distance': None,
            'num_vehicles': None,
            'exec_time': exec_time,
            'status': 'No Solution'
        }


def run_benchmark_customers():
    """Test r101/c101/rc101 with different customer numbers (10, 30, 50, 70) with 15s time limit"""
    data_types = ['r101', 'c101', 'rc101']
    customer_nums = [10, 30, 50, 70]
    time_limit = 15  # max 15 seconds
    M = 10000
    
    results = []
    
    print("=" * 60)
    print("Benchmark 1: Testing different customer numbers (15s limit)")
    print("=" * 60)
    
    for data_type in data_types:
        for customer_num in customer_nums:
            print(f"\nTesting {data_type} with {customer_num} customers...")
            
            data_path = f'data/solomon/{data_type}.txt'
            data = readData(data_path, customer_num)
            
            # 使用统一的求解函数
            result = solve_vrptw(data, M, time_limit)
            
            results.append({
                'Dataset': data_type.upper(),
                'Customer Number': customer_num,
                'Total Distance': result['total_distance'],
                'Number of Vehicles': result['num_vehicles'],
                'Execution Time (s)': result['exec_time'],
                'Status': result['status']
            })
            
            print(f"  Distance: {result['total_distance']}, Vehicles: {result['num_vehicles']}, "
                  f"Time: {result['exec_time']:.2f}s, Status: {result['status']}")
    
    df_customers = pd.DataFrame(results)
    df_customers.to_csv('benchmark_results_customers.csv', index=False)
    print("\n" + "=" * 60)
    print("Results saved to 'benchmark_results_customers.csv'")
    print("=" * 60)
    
    return df_customers


def run_benchmark_time():
    """Test r101/c101/rc101 with same customer number (25) at different time limits"""
    data_types = ['r101', 'c101', 'rc101']
    customer_num = 25
    time_limits = [3, 5, 7, 10, 15]  # seconds
    M = 10000
    
    results = []
    
    print("\n" + "=" * 60)
    print("Benchmark 2: Testing different time limits (25 customers)")
    print("=" * 60)
    
    for data_type in data_types:
        for time_limit in time_limits:
            print(f"\nTesting {data_type} with {customer_num} customers, time limit: {time_limit}s...")
            
            data_path = f'data/solomon/{data_type}.txt'
            data = readData(data_path, customer_num)
            
            # 使用统一的求解函数
            result = solve_vrptw(data, M, time_limit)
            
            results.append({
                'Dataset': data_type.upper(),
                'Time Limit (s)': time_limit,
                'Total Distance': result['total_distance'],
                'Number of Vehicles': result['num_vehicles'],
                'Actual Time (s)': result['exec_time'],
                'Status': result['status']
            })
            
            print(f"  Distance: {result['total_distance']}, Vehicles: {result['num_vehicles']}, "
                  f"Time: {result['exec_time']:.2f}s, Status: {result['status']}")
    
    df_time = pd.DataFrame(results)
    df_time.to_csv('benchmark_results_time.csv', index=False)
    print("\n" + "=" * 60)
    print("Results saved to 'benchmark_results_time.csv'")
    print("=" * 60)
    
    return df_time


def plot_customer_benchmark(df):
    """Create plots for customer number benchmark"""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Total Distance vs Customer Number
    ax1 = axes[0, 0]
    for dataset in df['Dataset'].unique():
        data_subset = df[df['Dataset'] == dataset]
        ax1.plot(data_subset['Customer Number'], data_subset['Total Distance'], 
                marker='o', label=dataset, linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Customers', fontsize=12)
    ax1.set_ylabel('Total Distance', fontsize=12)
    ax1.set_title('Total Distance vs Number of Customers (15s limit)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of Vehicles vs Customer Number
    ax2 = axes[0, 1]
    for dataset in df['Dataset'].unique():
        data_subset = df[df['Dataset'] == dataset]
        ax2.plot(data_subset['Customer Number'], data_subset['Number of Vehicles'], 
                marker='s', label=dataset, linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Customers', fontsize=12)
    ax2.set_ylabel('Number of Vehicles', fontsize=12)
    ax2.set_title('Number of Vehicles vs Number of Customers', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Execution Time vs Customer Number
    ax3 = axes[1, 0]
    for dataset in df['Dataset'].unique():
        data_subset = df[df['Dataset'] == dataset]
        ax3.plot(data_subset['Customer Number'], data_subset['Execution Time (s)'], 
                marker='^', label=dataset, linewidth=2, markersize=8)
    ax3.axhline(y=15, color='r', linestyle='--', label='Time Limit', alpha=0.5)
    ax3.set_xlabel('Number of Customers', fontsize=12)
    ax3.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax3.set_title('Execution Time vs Number of Customers', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Heatmap of Total Distance
    ax4 = axes[1, 1]
    pivot_data = df.pivot(index='Dataset', columns='Customer Number', values='Total Distance')
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Total Distance'})
    ax4.set_title('Total Distance Heatmap', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Number of Customers', fontsize=12)
    ax4.set_ylabel('Dataset', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('plots/benchmark_customers.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to 'plots/benchmark_customers.png'")
    plt.close()


def plot_time_benchmark(df):
    """Create plots for time limit benchmark"""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Total Distance vs Time Limit
    ax1 = axes[0, 0]
    for dataset in df['Dataset'].unique():
        data_subset = df[df['Dataset'] == dataset]
        ax1.plot(data_subset['Time Limit (s)'], data_subset['Total Distance'], 
                marker='o', label=dataset, linewidth=2, markersize=8)
    ax1.set_xlabel('Time Limit (seconds)', fontsize=12)
    ax1.set_ylabel('Total Distance', fontsize=12)
    ax1.set_title('Solution Quality vs Time Limit (25 customers)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of Vehicles vs Time Limit
    ax2 = axes[0, 1]
    for dataset in df['Dataset'].unique():
        data_subset = df[df['Dataset'] == dataset]
        ax2.plot(data_subset['Time Limit (s)'], data_subset['Number of Vehicles'], 
                marker='s', label=dataset, linewidth=2, markersize=8)
    ax2.set_xlabel('Time Limit (seconds)', fontsize=12)
    ax2.set_ylabel('Number of Vehicles', fontsize=12)
    ax2.set_title('Number of Vehicles vs Time Limit', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Actual Execution Time
    ax3 = axes[1, 0]
    for dataset in df['Dataset'].unique():
        data_subset = df[df['Dataset'] == dataset]
        ax3.plot(data_subset['Time Limit (s)'], data_subset['Actual Time (s)'], 
                marker='^', label=dataset, linewidth=2, markersize=8)
    ax3.plot(df['Time Limit (s)'].unique(), df['Time Limit (s)'].unique(), 
            'k--', label='Time Limit', alpha=0.5)
    ax3.set_xlabel('Time Limit (seconds)', fontsize=12)
    ax3.set_ylabel('Actual Execution Time (seconds)', fontsize=12)
    ax3.set_title('Actual vs Allowed Time', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Bar plot comparing datasets
    ax4 = axes[1, 1]
    x_pos = range(len(df['Dataset'].unique()))
    width = 0.2
    time_limits_to_compare = [3, 7, 15]
    
    for i, time_limit in enumerate(time_limits_to_compare):
        distances = []
        for dataset in df['Dataset'].unique():
            dist = df[(df['Dataset'] == dataset) & (df['Time Limit (s)'] == time_limit)]['Total Distance'].values
            distances.append(dist[0] if len(dist) > 0 else 0)
        ax4.bar([x + width * i for x in x_pos], distances, width, 
               label=f'{time_limit}s', alpha=0.8)
    
    ax4.set_xlabel('Dataset', fontsize=12)
    ax4.set_ylabel('Total Distance', fontsize=12)
    ax4.set_title('Distance Comparison at Different Time Limits', fontsize=14, fontweight='bold')
    ax4.set_xticks([x + width for x in x_pos])
    ax4.set_xticklabels(df['Dataset'].unique())
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/benchmark_time.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'plots/benchmark_time.png'")
    plt.close()


if __name__ == '__main__':
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Run benchmarks
    print("Starting benchmarking process...\n")
    
    # Benchmark 1: Different customer numbers
    df_customers = run_benchmark_customers()
    print("\n")
    print(df_customers.to_string(index=False))
    plot_customer_benchmark(df_customers)
    
    # Benchmark 2: Different time limits
    df_time = run_benchmark_time()
    print("\n")
    print(df_time.to_string(index=False))
    plot_time_benchmark(df_time)
    
    print("\n" + "=" * 60)
    print("Benchmarking complete!")
    print("=" * 60)
