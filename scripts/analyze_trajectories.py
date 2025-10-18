#!/usr/bin/env python3
"""
Trajectory Analysis Script
Analyzes recorded trajectories to identify those requiring significant chassis base movement.

Usage:
    python scripts/analyze_trajectories.py
    python scripts/analyze_trajectories.py --threshold 2.0 --output results.csv
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
import argparse
import csv


@dataclass
class TrajectoryStats:
    """Statistics for a single trajectory"""
    file_path: str
    scene_id: str
    trajectory_name: str
    num_waypoints: int
    
    # Position changes
    x_start: float
    x_end: float
    x_range: float  # max - min
    x_total_change: float  # end - start
    
    y_start: float
    y_end: float
    y_range: float
    y_total_change: float
    
    z_start: float
    z_end: float
    z_range: float
    z_total_change: float
    
    # 3D distances
    euclidean_distance: float  # Straight-line distance start to end
    path_length: float  # Sum of all segment lengths
    
    # Chassis movement requirement
    requires_chassis_movement: bool
    chassis_movement_score: float  # How much chassis movement needed


def load_trajectory(json_path: Path) -> List[Dict]:
    """Load trajectory from JSON file"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get('poses', [])
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return []


def calculate_trajectory_stats(json_path: Path, chassis_threshold: float = 2.0) -> TrajectoryStats:
    """
    Calculate comprehensive statistics for a trajectory
    
    Args:
        json_path: Path to trajectory JSON file
        chassis_threshold: X-direction change threshold (meters) for requiring chassis movement
    
    Returns:
        TrajectoryStats object with all statistics
    """
    poses = load_trajectory(json_path)
    
    if not poses:
        # Return empty stats for invalid trajectories
        return None
    
    # Extract positions
    positions = np.array([pose['position'] for pose in poses])
    num_waypoints = len(positions)
    
    # Start and end positions
    start_pos = positions[0]
    end_pos = positions[-1]
    
    # Per-axis statistics
    x_vals = positions[:, 0]
    y_vals = positions[:, 1]
    z_vals = positions[:, 2]
    
    x_start, x_end = x_vals[0], x_vals[-1]
    y_start, y_end = y_vals[0], y_vals[-1]
    z_start, z_end = z_vals[0], z_vals[-1]
    
    x_range = x_vals.max() - x_vals.min()
    y_range = y_vals.max() - y_vals.min()
    z_range = z_vals.max() - z_vals.min()
    
    x_total_change = abs(x_end - x_start)
    y_total_change = abs(y_end - y_start)
    z_total_change = abs(z_end - z_start)
    
    # Euclidean distance (straight line from start to end)
    euclidean_distance = np.linalg.norm(end_pos - start_pos)
    
    # Path length (sum of all segment lengths)
    path_length = 0.0
    for i in range(1, num_waypoints):
        segment_length = np.linalg.norm(positions[i] - positions[i-1])
        path_length += segment_length
    
    # Chassis movement requirement
    # Primary criterion: X-axis (longitudinal) movement
    # Secondary: Total 2D horizontal movement (XY plane)
    horizontal_distance = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    
    requires_chassis = x_total_change >= chassis_threshold
    
    # Chassis movement score: combination of X change and horizontal distance
    chassis_movement_score = max(x_total_change, horizontal_distance)
    
    # Parse file path for scene and trajectory info
    path_parts = json_path.parts
    scene_id = "unknown"
    trajectory_name = json_path.stem
    
    for i, part in enumerate(path_parts):
        if part.startswith('scene_'):
            scene_id = part
            break
    
    return TrajectoryStats(
        file_path=str(json_path),
        scene_id=scene_id,
        trajectory_name=trajectory_name,
        num_waypoints=num_waypoints,
        x_start=x_start,
        x_end=x_end,
        x_range=x_range,
        x_total_change=x_total_change,
        y_start=y_start,
        y_end=y_end,
        y_range=y_range,
        y_total_change=y_total_change,
        z_start=z_start,
        z_end=z_end,
        z_range=z_range,
        z_total_change=z_total_change,
        euclidean_distance=euclidean_distance,
        path_length=path_length,
        requires_chassis_movement=requires_chassis,
        chassis_movement_score=chassis_movement_score
    )


def analyze_all_trajectories(
    trajectory_dir: Path,
    chassis_threshold: float = 2.0
) -> Tuple[List[TrajectoryStats], List[TrajectoryStats]]:
    """
    Analyze all trajectories in directory
    
    Returns:
        Tuple of (all_stats, chassis_required_stats)
    """
    # Find all JSON files
    json_files = list(trajectory_dir.rglob("*.json"))
    
    # Filter out __MACOSX files
    json_files = [f for f in json_files if '__MACOSX' not in str(f)]
    
    print(f"Found {len(json_files)} trajectory files")
    print(f"Analyzing with chassis threshold: {chassis_threshold}m X-direction change\n")
    
    all_stats = []
    
    for i, json_path in enumerate(json_files):
        if (i + 1) % 100 == 0:
            print(f"Processing: {i+1}/{len(json_files)}")
        
        stats = calculate_trajectory_stats(json_path, chassis_threshold)
        if stats is not None:
            all_stats.append(stats)
    
    # Separate trajectories requiring chassis movement
    chassis_required = [s for s in all_stats if s.requires_chassis_movement]
    
    # Sort by chassis movement score (descending)
    chassis_required.sort(key=lambda s: s.chassis_movement_score, reverse=True)
    all_stats.sort(key=lambda s: s.chassis_movement_score, reverse=True)
    
    return all_stats, chassis_required


def print_summary(all_stats: List[TrajectoryStats], chassis_required: List[TrajectoryStats], threshold: float):
    """Print analysis summary"""
    print("\n" + "="*80)
    print("TRAJECTORY ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nTotal trajectories analyzed: {len(all_stats)}")
    print(f"Trajectories requiring chassis movement (X change >= {threshold}m): {len(chassis_required)}")
    print(f"Percentage requiring chassis: {100*len(chassis_required)/len(all_stats):.1f}%")
    
    # Overall statistics
    x_changes = [s.x_total_change for s in all_stats]
    y_changes = [s.y_total_change for s in all_stats]
    z_changes = [s.z_total_change for s in all_stats]
    path_lengths = [s.path_length for s in all_stats]
    
    print(f"\n--- Overall Statistics ---")
    print(f"X-direction change (longitudinal):")
    print(f"  Mean: {np.mean(x_changes):.3f}m | Median: {np.median(x_changes):.3f}m")
    print(f"  Min: {np.min(x_changes):.3f}m | Max: {np.max(x_changes):.3f}m")
    print(f"  Std: {np.std(x_changes):.3f}m")
    
    print(f"\nY-direction change (lateral):")
    print(f"  Mean: {np.mean(y_changes):.3f}m | Median: {np.median(y_changes):.3f}m")
    print(f"  Min: {np.min(y_changes):.3f}m | Max: {np.max(y_changes):.3f}m")
    
    print(f"\nZ-direction change (vertical):")
    print(f"  Mean: {np.mean(z_changes):.3f}m | Median: {np.median(z_changes):.3f}m")
    print(f"  Min: {np.min(z_changes):.3f}m | Max: {np.max(z_changes):.3f}m")
    
    print(f"\nPath length:")
    print(f"  Mean: {np.mean(path_lengths):.3f}m | Median: {np.median(path_lengths):.3f}m")
    print(f"  Min: {np.min(path_lengths):.3f}m | Max: {np.max(path_lengths):.3f}m")
    
    # Scene breakdown
    scene_counts = {}
    scene_chassis_counts = {}
    
    for stats in all_stats:
        scene_counts[stats.scene_id] = scene_counts.get(stats.scene_id, 0) + 1
    
    for stats in chassis_required:
        scene_chassis_counts[stats.scene_id] = scene_chassis_counts.get(stats.scene_id, 0) + 1
    
    print(f"\n--- Breakdown by Scene ---")
    for scene in sorted(scene_counts.keys()):
        total = scene_counts[scene]
        chassis = scene_chassis_counts.get(scene, 0)
        pct = 100 * chassis / total if total > 0 else 0
        print(f"{scene}: {chassis}/{total} require chassis ({pct:.1f}%)")


def print_top_trajectories(chassis_required: List[TrajectoryStats], top_n: int = 20):
    """Print top N trajectories requiring most chassis movement"""
    print(f"\n" + "="*80)
    print(f"TOP {min(top_n, len(chassis_required))} TRAJECTORIES REQUIRING CHASSIS MOVEMENT")
    print("="*80)
    print(f"\n{'#':<4} {'Scene':<12} {'Trajectory Name':<40} {'X Change':<10} {'Score':<10}")
    print("-"*80)
    
    for i, stats in enumerate(chassis_required[:top_n]):
        print(f"{i+1:<4} {stats.scene_id:<12} {stats.trajectory_name:<40} {stats.x_total_change:>8.3f}m {stats.chassis_movement_score:>8.3f}m")


def save_to_csv(all_stats: List[TrajectoryStats], output_path: Path):
    """Save analysis results to CSV"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Index', 'Scene', 'Trajectory Name', 'File Path',
            'Num Waypoints',
            'X Start', 'X End', 'X Change', 'X Range',
            'Y Start', 'Y End', 'Y Change', 'Y Range',
            'Z Start', 'Z End', 'Z Change', 'Z Range',
            'Euclidean Distance', 'Path Length',
            'Requires Chassis', 'Chassis Movement Score'
        ])
        
        # Data rows
        for i, stats in enumerate(all_stats):
            writer.writerow([
                i,
                stats.scene_id,
                stats.trajectory_name,
                stats.file_path,
                stats.num_waypoints,
                f"{stats.x_start:.4f}", f"{stats.x_end:.4f}", f"{stats.x_total_change:.4f}", f"{stats.x_range:.4f}",
                f"{stats.y_start:.4f}", f"{stats.y_end:.4f}", f"{stats.y_total_change:.4f}", f"{stats.y_range:.4f}",
                f"{stats.z_start:.4f}", f"{stats.z_end:.4f}", f"{stats.z_total_change:.4f}", f"{stats.z_range:.4f}",
                f"{stats.euclidean_distance:.4f}",
                f"{stats.path_length:.4f}",
                stats.requires_chassis_movement,
                f"{stats.chassis_movement_score:.4f}"
            ])
    
    print(f"\nResults saved to: {output_path}")


def save_chassis_required_ids(chassis_required: List[TrajectoryStats], output_path: Path):
    """Save list of trajectory IDs/names that require chassis movement"""
    with open(output_path, 'w') as f:
        f.write(f"# Trajectories Requiring Chassis Movement\n")
        f.write(f"# Total: {len(chassis_required)}\n\n")
        
        # Group by scene
        by_scene = {}
        for stats in chassis_required:
            if stats.scene_id not in by_scene:
                by_scene[stats.scene_id] = []
            by_scene[stats.scene_id].append(stats)
        
        for scene in sorted(by_scene.keys()):
            f.write(f"\n## {scene} ({len(by_scene[scene])} trajectories)\n")
            for stats in by_scene[scene]:
                f.write(f"{stats.trajectory_name} (X: {stats.x_total_change:.3f}m, Score: {stats.chassis_movement_score:.3f}m)\n")
    
    print(f"Chassis-required trajectory IDs saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trajectories to identify those requiring chassis movement"
    )
    parser.add_argument(
        '--trajectory_dir',
        type=str,
        default='trajectoryToLearn/world_json',
        help='Directory containing trajectory JSON files (default: trajectoryToLearn/world_json)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=2.0,
        help='X-direction change threshold in meters for chassis movement (default: 2.0)'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=20,
        help='Number of top trajectories to display (default: 20)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='trajectory_analysis_results.csv',
        help='Output CSV file path (default: trajectory_analysis_results.csv)'
    )
    parser.add_argument(
        '--output_ids',
        type=str,
        default='chassis_required_trajectories.txt',
        help='Output text file for chassis-required IDs (default: chassis_required_trajectories.txt)'
    )
    
    args = parser.parse_args()
    
    # Convert to paths
    trajectory_dir = Path(args.trajectory_dir)
    output_csv = Path(args.output_csv)
    output_ids = Path(args.output_ids)
    
    if not trajectory_dir.exists():
        print(f"Error: Directory not found: {trajectory_dir}")
        return
    
    # Analyze trajectories
    all_stats, chassis_required = analyze_all_trajectories(trajectory_dir, args.threshold)
    
    # Print results
    print_summary(all_stats, chassis_required, args.threshold)
    print_top_trajectories(chassis_required, args.top_n)
    
    # Save results
    save_to_csv(all_stats, output_csv)
    save_chassis_required_ids(chassis_required, output_ids)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
