#!/usr/bin/env python3
"""
2D Layered Model Visualization Tool - Python Version
Generate images using matplotlib, similar to the C++ version
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from typing import List, Tuple, Dict
import os

class Point:
    """Represents a data point"""
    def __init__(self, x: int, y: int, time_window: int, color: str):
        self.x = x
        self.y = y
        self.time_window = time_window
        self.color = color

class ColorfulVisualization:
    """2D Layered Model Visualization Class"""
    
    def __init__(self, width: int = 20, height: int = 12, max_time_windows: int = 5):
        self.width = width
        self.height = height
        self.max_time_windows = max_time_windows
        self.all_points = []
        
        # Define color mapping (corresponding to C++ version colors)
        self.colors = [
            '#FF6B6B',  # Bright Red
            '#4ECDC4',  # Bright Cyan
            '#45B7D1',  # Bright Blue
            '#96CEB4',  # Bright Green
            '#FECA57',  # Bright Yellow
            '#FF9FF3',  # Bright Purple
            '#54A0FF',  # Sky Blue
            '#5F27CD',  # Deep Purple
        ]
        
    def generate_random_points(self, min_points: int = 3, max_points: int = 8) -> None:
        """Generate random data points"""
        self.all_points.clear()
        
        print("Generating point data...")
        for time_window in range(self.max_time_windows):
            num_points = random.randint(min_points, max_points)
            color = self.colors[time_window % len(self.colors)]
            
            points_in_window = []
            for _ in range(num_points):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                point = Point(x, y, time_window, color)
                self.all_points.append(point)
                points_in_window.append(f"({x},{y})")
            
            print(f"Time Window {time_window + 1}: {' '.join(points_in_window)}")
    
    def create_single_time_window_plot(self, time_window: int, save_path: str = None) -> None:
        """Create visualization image for a single time window"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up grid
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        
        # Invert Y axis so (0,0) is at top-left (consistent with C++ version)
        ax.invert_yaxis()
        
        # Draw grid lines
        for i in range(self.width + 1):
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.7)
        for i in range(self.height + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.7)
        
        # Create grid background
        for y in range(self.height):
            for x in range(self.width):
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                       linewidth=1, edgecolor='black', 
                                       facecolor='white', alpha=0.8)
                ax.add_patch(rect)
        
        # Draw points for current time window
        current_points = [p for p in self.all_points if p.time_window == time_window]
        
        for point in current_points:
            rect = patches.Rectangle((point.x - 0.5, point.y - 0.5), 1, 1,
                                   linewidth=1, edgecolor='black',
                                   facecolor=point.color, alpha=0.8)
            ax.add_patch(rect)
        
        # Set axis labels
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.set_xlabel('X Axis', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Axis', fontsize=12, fontweight='bold')
        
        # Set title
        title = f'2D Layered Model Visualization - Time Window {time_window + 1}/{self.max_time_windows}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add statistics
        stats_text = f'Current Window Points: {len(current_points)}\\nGrid Size: {self.width}×{self.height}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add legend
        legend_elements = [patches.Patch(color=self.colors[time_window % len(self.colors)], 
                                       label=f'Time Window {time_window + 1}')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Image saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_all_time_windows_plot(self, save_path: str = None) -> None:
        """Create comprehensive image showing all time windows"""
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Set up grid
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.invert_yaxis()
        
        # Draw grid lines
        for i in range(self.width + 1):
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.7)
        for i in range(self.height + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.7)
        
        # Create grid background
        for y in range(self.height):
            for x in range(self.width):
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                       linewidth=1, edgecolor='black',
                                       facecolor='white', alpha=0.8)
                ax.add_patch(rect)
        
        # Draw all points (with transparency to show overlaps)
        for point in self.all_points:
            rect = patches.Rectangle((point.x - 0.5, point.y - 0.5), 1, 1,
                                   linewidth=1, edgecolor='black',
                                   facecolor=point.color, alpha=0.6)
            ax.add_patch(rect)
        
        # Set axis labels
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.set_xlabel('X Axis', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Axis', fontsize=12, fontweight='bold')
        
        # Set title
        ax.set_title('2D Layered Model Visualization - All Time Windows', fontsize=16, fontweight='bold', pad=20)
        
        # Add statistics
        total_points = len(self.all_points)
        stats_text = f'Total Points: {total_points}\\nTime Windows: {self.max_time_windows}\\nGrid Size: {self.width}×{self.height}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Add legend
        legend_elements = []
        for i in range(self.max_time_windows):
            color = self.colors[i % len(self.colors)]
            legend_elements.append(patches.Patch(color=color, label=f'Time Window {i + 1}'))
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive image saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_separate_layer_plots(self, output_dir: str = "layers") -> None:
        """Create separate plot for each layer/time window"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Generating separate layer plots to directory: {output_dir}")
        
        for time_window in range(self.max_time_windows):
            layer_path = os.path.join(output_dir, f"layer_{time_window + 1}.png")
            print(f"  Creating layer {time_window + 1}...")
            self.create_single_time_window_plot(time_window, layer_path)
        
        print(f"All {self.max_time_windows} layer plots generated!")
    
    def create_animation_frames(self, output_dir: str = "frames") -> None:
        """Create animation frame image sequence"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Generating animation frames to directory: {output_dir}")
        
        for time_window in range(self.max_time_windows):
            frame_path = os.path.join(output_dir, f"frame_{time_window + 1:02d}.png")
            self.create_single_time_window_plot(time_window, frame_path)
        
        # Also generate a comprehensive image
        summary_path = os.path.join(output_dir, "summary_all_windows.png")
        self.create_all_time_windows_plot(summary_path)
        
        print(f"All frames generated! Total {self.max_time_windows + 1} files")

def main():
    """Main function"""
    print("=" * 60)
    print("  Welcome to 2D Layered Model Visualization Tool (Python)  ")
    print("=" * 60)
    
    # Create visualization object
    viz = ColorfulVisualization(width=20, height=12, max_time_windows=5)
    
    print(f"\\nConfiguration:")
    print(f"- Grid Size: {viz.width}x{viz.height}")
    print(f"- Time Windows: {viz.max_time_windows}")
    print()
    
    # Generate random data
    viz.generate_random_points()
    print()
    
    while True:
        print("Select operation mode:")
        print("1. Generate single time window image")
        print("2. Generate separate plots for all layers")
        print("3. Generate comprehensive image (all time windows)")
        print("4. Generate animation frame sequence")
        print("5. Regenerate data")
        print("6. Exit")
        
        choice = input("\\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            try:
                time_window = int(input(f"Enter time window (1-{viz.max_time_windows}): ")) - 1
                if 0 <= time_window < viz.max_time_windows:
                    filename = f"time_window_{time_window + 1}.png"
                    viz.create_single_time_window_plot(time_window, filename)
                else:
                    print("Time window number out of range!")
            except ValueError:
                print("Please enter a valid number!")
        
        elif choice == '2':
            print(f"Generating separate plots for all {viz.max_time_windows} layers...")
            for i in range(viz.max_time_windows):
                filename = f"layer_{i + 1}.png"
                print(f"  Creating {filename}...")
                viz.create_single_time_window_plot(i, filename)
            print(f"All {viz.max_time_windows} layer plots generated successfully!")
        
        elif choice == '3':
            filename = "all_time_windows.png"
            viz.create_all_time_windows_plot(filename)
        
        elif choice == '4':
            viz.create_animation_frames()
        
        elif choice == '5':
            viz.generate_random_points()
            print()
        
        elif choice == '6':
            print("Thank you for using!")
            break
        
        else:
            print("Invalid choice, please try again!")
        
        print()

if __name__ == "__main__":
    main()