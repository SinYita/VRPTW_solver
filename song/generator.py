"""
VRPTW Instance Generator

This script generates random VRPTW (Vehicle Routing Problem with Time Windows) instances
and saves them in Solomon format.
"""

import numpy as np
import pandas as pd
import os


def generate_vrptw_instance(n, vehicle_capacity, x_range=(0, 8), y_range=(0, 8), 
                           d_range=(1, 15), time_range=(0, 10)):
    """
    Generate a random VRPTW instance.
    
    Parameters:
    -----------
    n : int
        Total number of nodes (including depot)
    vehicle_capacity : int
        Capacity of each vehicle
    x_range : tuple
        Range for x coordinates (min, max)
    y_range : tuple
        Range for y coordinates (min, max)
    d_range : tuple
        Range for customer demand (min, max)
    time_range : tuple
        Range for time windows (min, max)
    
    Returns:
    --------
    df_all : pd.DataFrame
        DataFrame containing all nodes
    """
    n_customers = n - 1
    nodes = []
    used_coordinates = set()
    
    # 1. Generate Depot
    depot_x = int(np.random.randint(x_range[0], x_range[1] + 1))
    depot_y = int(np.random.randint(y_range[0], y_range[1] + 1))
    used_coordinates.add((depot_x, depot_y))
    nodes.append({
        'CUST NO.': 0,
        'XCOORD.': depot_x,
        'YCOORD.': depot_y,
        'DEMAND': 0,
        'READY TIME': 0,
        'DUE DATE': time_range[1],
        'SERVICE TIME': 0
    })
    
    # 2. Generate Customers
    for i in range(1, n_customers + 1):
        # Generate unique coordinates
        while True:
            x = int(np.random.randint(x_range[0], x_range[1] + 1))
            y = int(np.random.randint(y_range[0], y_range[1] + 1))
            if (x, y) not in used_coordinates:
                used_coordinates.add((x, y))
                break
        
        d = int(np.random.randint(d_range[0], d_range[1] + 1))
        
        # Time window generation
        a = int(np.random.randint(time_range[0], time_range[1]))
        duration = int(np.random.randint(1, 11))
        b = int(min(a + duration, time_range[1]))
        
        nodes.append({
            'CUST NO.': i,
            'XCOORD.': x,
            'YCOORD.': y,
            'DEMAND': d,
            'READY TIME': a,
            'DUE DATE': b,
            'SERVICE TIME': 0
        })
    
    df_all = pd.DataFrame(nodes)
    return df_all


def save_to_solomon_format(df_all, n, vehicle_capacity, output_dir='../data/instance'):
    """
    Save the generated instance to a file in Solomon format.
    
    Parameters:
    -----------
    df_all : pd.DataFrame
        DataFrame containing all nodes
    n : int
        Total number of nodes
    vehicle_capacity : int
        Capacity of each vehicle
    output_dir : str
        Directory to save the output file
    """
    n_customers = n - 1
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    txt_filename = f'{output_dir}/i{n}.txt'
    
    with open(txt_filename, 'w') as f:
        # Write instance name
        f.write(f"i{n}\n")
        f.write("\n")
        
        # Write vehicle information
        f.write("VEHICLE\n")
        f.write("NUMBER     CAPACITY\n")
        f.write(f"  {n_customers}         {vehicle_capacity}\n")
        f.write("\n")
        
        # Write customer header
        f.write("CUSTOMER\n")
        f.write("CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME\n")
        f.write(" \n")
        
        # Write data rows
        for _, row in df_all.iterrows():
            f.write(f"{int(row['CUST NO.']):5d}      "
                    f"{int(row['XCOORD.']):2d}         "
                    f"{int(row['YCOORD.']):2d}         "
                    f"{int(row['DEMAND']):2d}        "
                    f"{int(row['READY TIME']):4d}       "
                    f"{int(row['DUE DATE']):4d}        "
                    f"{int(row['SERVICE TIME']):3d}   \n")
    
    print(f"Data saved to {txt_filename}")
    return txt_filename


if __name__ == '__main__':
    # Parameters
    n = int(input("Please enter the total number of nodes (n): "))
    vehicle_capacity = int(input("Please enter the vehicle capacity: "))
    
    # Generate instance
    df_all = generate_vrptw_instance(n, vehicle_capacity)
    print(f"Generated {len(df_all)} nodes (1 Depot + {n - 1} Customers).")
    
    # Save to file
    txt_filename = save_to_solomon_format(df_all, n, vehicle_capacity)
    
    # Display DataFrame
    print("\nGenerated nodes:")
    print(df_all)
