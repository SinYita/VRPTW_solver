"""
VRPTW Instance Visualizer

This script visualizes VRPTW instances from CSV or TXT files.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import glob


def resolve_file_path(input_str):
    """
    Resolve the file path based on the input string.
    
    Parameters:
    -----------
    input_str : str
        Input string (e.g., 'n11', 'r101', or full path)
    
    Returns:
    --------
    file_path : str
        Resolved file path
    """
    # If it's already a full path, return it
    if os.path.exists(input_str):
        return input_str
    
    # Check if it starts with 'n' (instance files)
    if input_str.startswith('i'):
        pattern = f"../data/instance/{input_str}.txt"
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
        else:
            raise FileNotFoundError(f"No instance file found matching '{input_str}'")
    
    # Check if it's a Solomon benchmark (like r101, c101, rc101)
    else:
        file_path = f"../data/solomon/{input_str}.txt"
        if os.path.exists(file_path):
            return file_path
        else:
            # Try lowercase
            file_path_lower = f"../data/solomon/{input_str.lower()}.txt"
            if os.path.exists(file_path_lower):
                return file_path_lower
            else:
                raise FileNotFoundError(f"Solomon benchmark file '{input_str}' not found")


def read_solomon_file(file_path):
    """
    Read a VRPTW instance file in Solomon format.
    
    Parameters:
    -----------
    file_path : str
        Path to the Solomon format file
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame containing the instance data
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the start of the customer data
    start_idx = None
    for i, line in enumerate(lines):
        if 'CUST NO.' in line:
            start_idx = i + 2  # Skip header and empty line
            break
    
    if start_idx is None:
        raise ValueError("Could not find customer data in file")
    
    # Parse data
    nodes = []
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        parts = re.split(r'\s+', line)
        if len(parts) >= 7:
            nodes.append({
                'CUST NO.': int(parts[0]),
                'XCOORD.': int(parts[1]),
                'YCOORD.': int(parts[2]),
                'DEMAND': int(parts[3]),
                'READY TIME': int(parts[4]),
                'DUE DATE': int(parts[5]),
                'SERVICE TIME': int(parts[6])
            })
    
    df = pd.DataFrame(nodes)
    return df


def visualize_vrptw_instance(df_all, output_file=None, x_range=None, y_range=None, routes=None):
    """
    Visualize a VRPTW instance.
    
    Parameters:
    -----------
    df_all : pd.DataFrame
        DataFrame containing the instance data
    output_file : str, optional
        Path to save the plot. If None, plot is displayed but not saved.
    x_range : tuple, optional
        Range for x axis. If None, will be auto-calculated from data.
    y_range : tuple, optional
        Range for y axis. If None, will be auto-calculated from data.
    routes : list of lists, optional
        List of routes, where each route is a list of customer IDs.
        If provided, will draw lines connecting customers in each route.
    """
    sns.set_style("whitegrid")
    
    # Auto-calculate ranges if not provided
    if x_range is None:
        x_min, x_max = df_all['XCOORD.'].min(), df_all['XCOORD.'].max()
        x_range = (int(x_min), int(x_max))
    if y_range is None:
        y_min, y_max = df_all['YCOORD.'].min(), df_all['YCOORD.'].max()
        y_range = (int(y_min), int(y_max))
    
    # Add type column for visualization
    df_all['type'] = df_all['DEMAND'].apply(lambda x: 'Depot' if x == 0 else 'Customer')
    
    # Define colors and markers
    palette = {'Depot': '#D62728', 'Customer': '#1F77B4'}
    markers = {'Depot': 's', 'Customer': 'o'}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Use seaborn to plot
    sns.scatterplot(
        data=df_all, 
        x='XCOORD.', 
        y='YCOORD.', 
        hue='type', 
        style='type', 
        s=2000, 
        palette=palette,
        markers=markers,
        zorder=2,
        legend='brief',
        ax=ax
    )
    
    # Post-process to make markers hollow and set different line widths
    for i, collection in enumerate(ax.collections):
        colors = collection.get_facecolors()
        collection.set_edgecolors(colors)
        collection.set_facecolors('white')
        if i == 1:  # Customer markers
            collection.set_linewidth(6)
        else:  # Depot marker
            collection.set_linewidth(4)
    
    # Draw routes if provided
    if routes is not None:
        colors_list = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
        for route_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            
            route_color = colors_list[route_idx % len(colors_list)]
            
            # Draw lines connecting customers in the route
            for i in range(len(route) - 1):
                node_from = route[i]
                node_to = route[i + 1]
                
                # Get coordinates
                from_row = df_all[df_all['CUST NO.'] == node_from]
                to_row = df_all[df_all['CUST NO.'] == node_to]
                
                if not from_row.empty and not to_row.empty:
                    x_coords = [from_row['XCOORD.'].values[0], to_row['XCOORD.'].values[0]]
                    y_coords = [from_row['YCOORD.'].values[0], to_row['YCOORD.'].values[0]]
                    
                    ax.plot(x_coords, y_coords, color=route_color, linewidth=2, 
                           alpha=0.7, zorder=1, label=f'Route {route_idx + 1}' if i == 0 else '')
    
    # Annotate nodes
    for i in range(df_all.shape[0]):
        row = df_all.iloc[i]
        color = palette[row['type']]
        
        # Plot ID inside the marker
        ax.text(
            row['XCOORD.'], 
            row['YCOORD.'], 
            str(int(row['CUST NO.'])), 
            horizontalalignment='center', 
            verticalalignment='center', 
            size='large', 
            color=color, 
            weight='bold',
            zorder=3,
            clip_on=False
        )
        
        # Plot details next to the marker
        if row['type'] == 'Depot':
            ax.text(
                row['XCOORD.'] + 0.35, 
                row['YCOORD.'], 
                "Depot", 
                horizontalalignment='left', 
                verticalalignment='center', 
                size='x-large', 
                color=color, 
                weight='bold',
                zorder=3,
                clip_on=False
            )
        else:
            info_text = f"[{int(row['READY TIME'])}, {int(row['DUE DATE'])}]"
            ax.text(
                row['XCOORD.'] + 0.35, 
                row['YCOORD.'], 
                info_text, 
                horizontalalignment='left', 
                verticalalignment='center', 
                size='medium', 
                color='#333333',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='#cccccc', boxstyle='round,pad=0.2'),
                zorder=3,
                clip_on=False
            )
    
    # Beautify the grid and layout
    ax.set_title('VRPTW Instance Visualization', fontsize=20, weight='bold', pad=20)
    ax.set_xlabel('X Coordinate', fontsize=14)
    ax.set_ylabel('Y Coordinate', fontsize=14)
    
    # Set view range
    ax.set_xlim(x_range[0] - 0.5, x_range[1] + 0.5)
    ax.set_ylim(y_range[0] - 0.5, y_range[1] + 0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Customize grid
    ax.set_xticks(range(x_range[0], x_range[1] + 1))
    ax.set_yticks(range(y_range[0], y_range[1] + 1))
    
    # Draw grid lines
    ax.grid(False)
    for x in range(x_range[0], x_range[1] + 1):
        ax.axvline(x=x, ymin=(0.5 - y_range[0]) / (y_range[1] - y_range[0] + 1), 
                   ymax=(y_range[1] + 0.5 - y_range[0]) / (y_range[1] - y_range[0] + 1), 
                   linestyle='-', alpha=0.5, color='black', linewidth=1, zorder=1)
    for y in range(y_range[0], y_range[1] + 1):
        ax.axhline(y=y, xmin=(0.5 - x_range[0]) / (x_range[1] - x_range[0] + 1), 
                   xmax=(x_range[1] + 0.5 - x_range[0]) / (x_range[1] - x_range[0] + 1), 
                   linestyle='-', alpha=0.5, color='black', linewidth=1, zorder=1)
    
    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Fix legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title='Location Type', 
              title_fontsize='12', fontsize='10', 
              bbox_to_anchor=(1.02, 1), loc='upper left', 
              borderaxespad=0., markerscale=0.3)
    
    fig.tight_layout()
    sns.despine(left=False, bottom=False)
    
    # Save or show
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        plt.close(fig)  # Close figure to free memory
    else:
        plt.show()


if __name__ == '__main__':
    # Example usage
    input_str = input("Please enter the instance name (e.g., 'n11', 'r101') or full path: ")
    
    # Resolve file path
    try:
        file_path = resolve_file_path(input_str)
        print(f"Loading file: {file_path}")
    except FileNotFoundError as e:
        print(e)
        exit(1)
    
    # Read the file
    if file_path.endswith('.txt'):
        df_all = read_solomon_file(file_path)
    else:
        print("Unsupported file format. Please use .txt files.")
        exit(1)
    
    print(f"Loaded {len(df_all)} nodes.")
    print(df_all)
    
    n = len(df_all)
    output_file = f'plot/vrptw_instance_{os.path.basename(file_path).replace(".txt", "")}.png'
    visualize_vrptw_instance(df_all, output_file=output_file, routes=routes)
