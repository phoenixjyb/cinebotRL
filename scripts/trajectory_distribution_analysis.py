#!/usr/bin/env python3
"""
Trajectory Distribution Analysis
Creates detailed statistics and visualizations for trajectory analysis.

Usage:
    python scripts/trajectory_distribution_analysis.py
"""

import csv
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_analysis_results(csv_path: str = "trajectory_analysis_results.csv") -> list:
    """Load trajectory analysis results from CSV"""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Convert numeric fields
    for row in data:
        row['Index'] = int(row['Index'])
        row['Num Waypoints'] = int(row['Num Waypoints'])
        row['X Change'] = float(row['X Change'])
        row['Y Change'] = float(row['Y Change'])
        row['Z Change'] = float(row['Z Change'])
        row['X Range'] = float(row['X Range'])
        row['Y Range'] = float(row['Y Range'])
        row['Z Range'] = float(row['Z Range'])
        row['Path Length'] = float(row['Path Length'])
        row['Chassis Movement Score'] = float(row['Chassis Movement Score'])
        row['Requires Chassis'] = row['Requires Chassis'] == 'True'
    
    return data


def analyze_distribution(data: list):
    """Analyze the distribution of trajectories"""
    
    print("\n" + "="*80)
    print("DETAILED TRAJECTORY DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Chassis movement requirement analysis
    chassis_required = [row for row in data if row['Requires Chassis']]
    chassis_not_required = [row for row in data if not row['Requires Chassis']]
    
    print(f"\n--- Chassis Movement Requirement ---")
    print(f"Total trajectories: {len(data)}")
    print(f"Require chassis movement: {len(chassis_required)} ({100*len(chassis_required)/len(data):.1f}%)")
    print(f"Don't require chassis: {len(chassis_not_required)} ({100*len(chassis_not_required)/len(data):.1f}%)")
    
    # X-direction bins
    print(f"\n--- X-Direction Change Distribution ---")
    x_changes = [row['X Change'] for row in data]
    x_bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
    x_counts, _ = np.histogram(x_changes, bins=x_bins)
    
    for i in range(len(x_bins)-1):
        count = x_counts[i]
        pct = 100 * count / len(data)
        print(f"{x_bins[i]:.1f}m - {x_bins[i+1]:.1f}m: {count:4d} trajectories ({pct:5.1f}%)")
    
    # Trajectory type analysis (from names)
    print(f"\n--- Trajectory Type Analysis (from names) ---")
    
    type_patterns = [
        'arc_left', 'arc_right', 'orbit_left', 'orbit_right', 
        'push', 'pull', 'round', 'retreat', 'approach'
    ]
    
    for pattern in type_patterns:
        pattern_data = [row for row in data if pattern in row['Trajectory Name'].lower()]
        if len(pattern_data) > 0:
            chassis_needed = sum(1 for row in pattern_data if row['Requires Chassis'])
            pct = 100 * chassis_needed / len(pattern_data)
            avg_x = np.mean([row['X Change'] for row in pattern_data])
            print(f"{pattern:15s}: {len(pattern_data):4d} total, {chassis_needed:4d} need chassis ({pct:5.1f}%), avg X={avg_x:.3f}m")
    
    # Scene analysis
    print(f"\n--- Scene-wise Analysis ---")
    scenes = {}
    for row in data:
        scene = row['Scene']
        if scene not in scenes:
            scenes[scene] = []
        scenes[scene].append(row)
    
    for scene in sorted(scenes.keys()):
        scene_data = scenes[scene]
        chassis_needed = sum(1 for row in scene_data if row['Requires Chassis'])
        pct = 100 * chassis_needed / len(scene_data) if len(scene_data) > 0 else 0
        
        x_changes_scene = [row['X Change'] for row in scene_data]
        y_changes_scene = [row['Y Change'] for row in scene_data]
        path_lengths_scene = [row['Path Length'] for row in scene_data]
        
        print(f"\n{scene}:")
        print(f"  Total: {len(scene_data)}")
        print(f"  Chassis required: {chassis_needed} ({pct:.1f}%)")
        print(f"  X change: mean={np.mean(x_changes_scene):.3f}m, median={np.median(x_changes_scene):.3f}m, max={np.max(x_changes_scene):.3f}m")
        print(f"  Y change: mean={np.mean(y_changes_scene):.3f}m, median={np.median(y_changes_scene):.3f}m, max={np.max(y_changes_scene):.3f}m")
        print(f"  Path length: mean={np.mean(path_lengths_scene):.3f}m, median={np.median(path_lengths_scene):.3f}m")
    
    # Extreme cases
    print(f"\n--- Extreme Cases ---")
    print(f"\nTop 10 Longest X-direction changes:")
    sorted_by_x = sorted(data, key=lambda r: r['X Change'], reverse=True)[:10]
    for row in sorted_by_x:
        print(f"  [{row['Index']:4d}] {row['Scene']:12s} {row['Trajectory Name']:45s} X={row['X Change']:6.3f}m Score={row['Chassis Movement Score']:6.3f}m")
    
    print(f"\nTop 10 Longest path lengths:")
    sorted_by_path = sorted(data, key=lambda r: r['Path Length'], reverse=True)[:10]
    for row in sorted_by_path:
        print(f"  [{row['Index']:4d}] {row['Scene']:12s} {row['Trajectory Name']:45s} Path={row['Path Length']:7.3f}m X={row['X Change']:6.3f}m")
    
    print(f"\nTop 10 Largest Y-direction changes (lateral):")
    sorted_by_y = sorted(data, key=lambda r: r['Y Change'], reverse=True)[:10]
    for row in sorted_by_y:
        print(f"  [{row['Index']:4d}] {row['Scene']:12s} {row['Trajectory Name']:45s} Y={row['Y Change']:6.3f}m X={row['X Change']:6.3f}m")
    
    # Correlation analysis
    print(f"\n--- Correlation Analysis ---")
    x_changes_all = np.array([row['X Change'] for row in data])
    y_changes_all = np.array([row['Y Change'] for row in data])
    path_lengths_all = np.array([row['Path Length'] for row in data])
    chassis_scores_all = np.array([row['Chassis Movement Score'] for row in data])
    
    corr_x_path = np.corrcoef(x_changes_all, path_lengths_all)[0, 1]
    corr_x_y = np.corrcoef(x_changes_all, y_changes_all)[0, 1]
    corr_path_chassis = np.corrcoef(path_lengths_all, chassis_scores_all)[0, 1]
    
    print(f"Correlation between X change and path length: {corr_x_path:.3f}")
    print(f"Correlation between X change and Y change: {corr_x_y:.3f}")
    print(f"Correlation between path length and chassis score: {corr_path_chassis:.3f}")
    
    # Percentiles
    print(f"\n--- Percentile Analysis (X Change) ---")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(x_changes_all, p)
        print(f"{p:3d}th percentile: {value:.3f}m")


def export_chassis_indices(data: list, output_file: str = "chassis_required_indices.txt"):
    """Export just the indices of trajectories requiring chassis movement"""
    chassis_data = [row for row in data if row['Requires Chassis']]
    chassis_data.sort(key=lambda r: r['Chassis Movement Score'], reverse=True)
    
    with open(output_file, 'w') as f:
        f.write(f"# Trajectory Indices Requiring Chassis Movement\n")
        f.write(f"# Total: {len(chassis_data)} out of {len(data)}\n")
        f.write(f"# Sorted by chassis movement score (descending)\n\n")
        
        # Write as Python list
        indices = [row['Index'] for row in chassis_data]
        f.write(f"CHASSIS_REQUIRED_INDICES = [\n")
        for i in range(0, len(indices), 10):
            chunk = indices[i:i+10]
            f.write("    " + ", ".join(f"{idx:4d}" for idx in chunk) + ",\n")
        f.write("]\n\n")
        
        # Write summary by trajectory type
        f.write(f"# Breakdown by trajectory type:\n")
        
        type_patterns = [
            'arc_left', 'arc_right', 'orbit_left', 'orbit_right', 
            'push', 'pull', 'round', 'retreat', 'approach'
        ]
        
        for pattern in type_patterns:
            type_data = [row for row in chassis_data if pattern in row['Trajectory Name'].lower()]
            if len(type_data) > 0:
                f.write(f"# {pattern}: {len(type_data)} trajectories\n")
    
    print(f"\nChassis-required indices saved to: {output_file}")


def main():
    csv_path = "trajectory_analysis_results.csv"
    
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found. Please run analyze_trajectories.py first.")
        return
    
    # Load data
    print(f"Loading data from {csv_path}...")
    data = load_analysis_results(csv_path)
    
    # Analyze
    analyze_distribution(data)
    
    # Export indices
    export_chassis_indices(data)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
