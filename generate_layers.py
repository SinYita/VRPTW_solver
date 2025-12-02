#!/usr/bin/env python3
"""
Quick script to generate separate plots for each layer
"""

from colorful_visualization_en import ColorfulVisualization

def main():
    print("Generating separate plots for each layer...")
    
    # Create visualization with default settings
    viz = ColorfulVisualization(width=20, height=12, max_time_windows=5)
    
    # Generate random data
    viz.generate_random_points()
    print()
    
    # Generate separate plot for each layer
    print(f"Creating {viz.max_time_windows} separate layer plots...")
    for i in range(viz.max_time_windows):
        filename = f"layer_{i + 1}.png"
        print(f"  Generating {filename}...")
        viz.create_single_time_window_plot(i, filename)
    
    print(f"\nCompleted! Generated {viz.max_time_windows} separate layer plots:")
    for i in range(viz.max_time_windows):
        print(f"  - layer_{i + 1}.png (Time Window {i + 1})")

if __name__ == "__main__":
    main()