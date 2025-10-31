"""Visualization and analysis script for evaluation results.

This script creates comprehensive plots and analysis from the quantitative evaluation data:
- Tracking error distributions and trends
- Joint usage analysis
- Velocity profiles
- Reward component breakdown
- Success/failure analysis

Usage:
    python scripts/reinforcement_learning/sb3/visualize_eval_results.py \
        --input evaluation_results/eval_summary_20251029_143022.json
    
    # Or analyze all results in a directory:
    python scripts/reinforcement_learning/sb3/visualize_eval_results.py \
        --input_dir evaluation_results
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--input", type=str, help="Path to eval_summary_*.json file")
    parser.add_argument("--input_dir", type=str, help="Directory containing evaluation results")
    parser.add_argument("--output_dir", type=str, default="evaluation_plots", help="Output directory for plots")
    return parser.parse_args()


def load_evaluation_data(summary_file: Path) -> Dict:
    """Load evaluation summary and associated data files."""
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Extract timestamp from filename
    timestamp = summary_file.stem.split('_', 2)[-1]
    
    # Load CSV files
    base_dir = summary_file.parent
    episodes_file = base_dir / f"episodes_{timestamp}.csv"
    steps_file = base_dir / f"steps_{timestamp}.csv"
    arrays_file = base_dir / f"arrays_{timestamp}.npz"
    
    data = {'summary': summary}
    
    if episodes_file.exists():
        data['episodes'] = pd.read_csv(episodes_file)
    
    if steps_file.exists():
        data['steps'] = pd.read_csv(steps_file)
    
    if arrays_file.exists():
        arrays = np.load(arrays_file)
        data['arrays'] = {key: arrays[key] for key in arrays.keys()}
    
    return data


def plot_tracking_errors(data: Dict, output_dir: Path):
    """Plot tracking error distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Tracking Error Analysis', fontsize=16, fontweight='bold')
    
    # Position errors
    if 'arrays' in data and 'tracking_errors_pos' in data['arrays']:
        pos_errors = data['arrays']['tracking_errors_pos'] * 100  # Convert to cm
        
        # Histogram
        ax = axes[0, 0]
        ax.hist(pos_errors, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(pos_errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(pos_errors):.2f} cm')
        ax.axvline(np.median(pos_errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(pos_errors):.2f} cm')
        ax.set_xlabel('Position Error (cm)')
        ax.set_ylabel('Count')
        ax.set_title('Position Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # CDF
        ax = axes[0, 1]
        sorted_errors = np.sort(pos_errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, cdf * 100, linewidth=2)
        ax.axhline(95, color='r', linestyle='--', alpha=0.5, label='95th percentile')
        ax.axvline(np.percentile(pos_errors, 95), color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Position Error (cm)')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title('Position Error CDF')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.text(np.percentile(pos_errors, 95), 50, f'P95: {np.percentile(pos_errors, 95):.2f} cm', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Orientation errors
    if 'arrays' in data and 'tracking_errors_ori' in data['arrays']:
        ori_errors = np.rad2deg(data['arrays']['tracking_errors_ori'])
        
        # Histogram
        ax = axes[1, 0]
        ax.hist(ori_errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
        ax.axvline(np.mean(ori_errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ori_errors):.2f}°')
        ax.axvline(np.median(ori_errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(ori_errors):.2f}°')
        ax.set_xlabel('Orientation Error (degrees)')
        ax.set_ylabel('Count')
        ax.set_title('Orientation Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # CDF
        ax = axes[1, 1]
        sorted_errors = np.sort(ori_errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, cdf * 100, linewidth=2, color='orange')
        ax.axhline(95, color='r', linestyle='--', alpha=0.5, label='95th percentile')
        ax.axvline(np.percentile(ori_errors, 95), color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Orientation Error (degrees)')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title('Orientation Error CDF')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.text(np.percentile(ori_errors, 95), 50, f'P95: {np.percentile(ori_errors, 95):.2f}°', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tracking_errors.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'tracking_errors.png'}")
    plt.close()


def plot_joint_analysis(data: Dict, output_dir: Path):
    """Plot joint angle and velocity analysis."""
    if 'arrays' not in data or 'joint_angles' not in data['arrays']:
        return
    
    joint_angles = data['arrays']['joint_angles']
    joint_names = [f'Joint {i}' for i in range(joint_angles.shape[1])]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Joint Usage Analysis', fontsize=16, fontweight='bold')
    
    # Joint angle distributions
    for i in range(min(6, joint_angles.shape[1])):
        ax = axes[i // 3, i % 3]
        angles_deg = np.rad2deg(joint_angles[:, i])
        ax.hist(angles_deg, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(angles_deg), color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(angles_deg):.1f}°')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Count')
        ax.set_title(f'{joint_names[i]} Angle Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add range info
        range_deg = np.ptp(angles_deg)
        ax.text(0.05, 0.95, f'Range: {range_deg:.1f}°', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'joint_angles.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'joint_angles.png'}")
    plt.close()
    
    # Joint velocities
    if 'joint_velocities' in data['arrays']:
        joint_vels = data['arrays']['joint_velocities']
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Joint Velocity Analysis', fontsize=16, fontweight='bold')
        
        for i in range(min(6, joint_vels.shape[1])):
            ax = axes[i // 3, i % 3]
            vels_deg = np.rad2deg(joint_vels[:, i])
            ax.hist(vels_deg, bins=50, alpha=0.7, edgecolor='black', color='orange')
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax.set_xlabel('Velocity (deg/s)')
            ax.set_ylabel('Count')
            ax.set_title(f'{joint_names[i]} Velocity Distribution')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            max_vel = np.max(np.abs(vels_deg))
            ax.text(0.05, 0.95, f'Max: {max_vel:.1f} deg/s', transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'joint_velocities.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_dir / 'joint_velocities.png'}")
        plt.close()


def plot_base_velocities(data: Dict, output_dir: Path):
    """Plot base velocity analysis."""
    if 'arrays' not in data or 'base_velocities' not in data['arrays']:
        return
    
    base_vels = data['arrays']['base_velocities']
    
    # Handle both 1D and 2D arrays
    if base_vels.ndim == 1:
        # If 1D, assume it's aggregated statistics - skip plotting
        print("⚠️ Base velocities data is 1D, skipping plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Base Velocity Analysis', fontsize=16, fontweight='bold')
    
    # Linear X
    ax = axes[0, 0]
    ax.hist(base_vels[:, 0], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Linear X Velocity (m/s)')
    ax.set_ylabel('Count')
    ax.set_title('Linear X Velocity Distribution')
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'Max: {np.max(np.abs(base_vels[:, 0])):.2f} m/s', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Linear Y
    ax = axes[0, 1]
    ax.hist(base_vels[:, 1], bins=50, alpha=0.7, edgecolor='black', color='green')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Linear Y Velocity (m/s)')
    ax.set_ylabel('Count')
    ax.set_title('Linear Y Velocity Distribution')
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'Max: {np.max(np.abs(base_vels[:, 1])):.2f} m/s', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Angular Z
    ax = axes[1, 0]
    angular_deg = np.rad2deg(base_vels[:, 2])
    ax.hist(angular_deg, bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Angular Z Velocity (deg/s)')
    ax.set_ylabel('Count')
    ax.set_title('Angular Z Velocity Distribution')
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'Max: {np.max(np.abs(angular_deg)):.1f} deg/s', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2D velocity scatter
    ax = axes[1, 1]
    scatter = ax.scatter(base_vels[:, 0], base_vels[:, 1], alpha=0.1, s=1)
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('Linear X Velocity (m/s)')
    ax.set_ylabel('Linear Y Velocity (m/s)')
    ax.set_title('Base Linear Velocity (X vs Y)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'base_velocities.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'base_velocities.png'}")
    plt.close()


def plot_reward_components(data: Dict, output_dir: Path):
    """Plot reward component breakdown."""
    if 'summary' not in data or 'reward_components' not in data['summary']['statistics']:
        return
    
    reward_comps = data['summary']['statistics']['reward_components']
    
    # Extract data
    components = []
    means = []
    stds = []
    
    for key, vals in reward_comps.items():
        components.append(key.replace('_', ' ').title())
        means.append(vals['mean'])
        stds.append(vals['std'])
    
    # Sort by mean value
    sorted_indices = np.argsort(means)[::-1]
    components = [components[i] for i in sorted_indices]
    means = [means[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, max(6, len(components) * 0.4)))
    
    y_pos = np.arange(len(components))
    colors = ['green' if m > 0 else 'red' for m in means]
    
    ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.7, capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(components)
    ax.set_xlabel('Mean Reward Value')
    ax.set_title('Reward Component Breakdown', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_components.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'reward_components.png'}")
    plt.close()


def plot_episode_statistics(data: Dict, output_dir: Path):
    """Plot episode-level statistics."""
    if 'episodes' not in data:
        return
    
    episodes_df = data['episodes']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Episode Statistics', fontsize=16, fontweight='bold')
    
    # Reward distribution
    ax = axes[0, 0]
    ax.hist(episodes_df['total_reward'], bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(episodes_df['total_reward'].mean(), color='r', linestyle='--', linewidth=2, 
               label=f'Mean: {episodes_df["total_reward"].mean():.2f}')
    ax.set_xlabel('Total Reward')
    ax.set_ylabel('Count')
    ax.set_title('Episode Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Episode length distribution
    ax = axes[0, 1]
    ax.hist(episodes_df['length'], bins=30, alpha=0.7, edgecolor='black', color='green')
    ax.axvline(episodes_df['length'].mean(), color='r', linestyle='--', linewidth=2, 
               label=f'Mean: {episodes_df["length"].mean():.1f}')
    ax.set_xlabel('Episode Length (steps)')
    ax.set_ylabel('Count')
    ax.set_title('Episode Length Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Reward over time
    ax = axes[1, 0]
    ax.plot(episodes_df['episode'], episodes_df['total_reward'], alpha=0.5, linewidth=1)
    # Add moving average
    window = 20
    if len(episodes_df) >= window:
        moving_avg = episodes_df['total_reward'].rolling(window=window).mean()
        ax.plot(episodes_df['episode'], moving_avg, color='red', linewidth=2, label=f'{window}-episode MA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward Over Episodes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Success rate (if available)
    ax = axes[1, 1]
    if 'success' in episodes_df.columns:
        success_rate = episodes_df['success'].mean() * 100
        ax.bar(['Success', 'Failure'], 
               [episodes_df['success'].sum(), (~episodes_df['success']).sum()],
               color=['green', 'red'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Count')
        ax.set_title(f'Success Rate: {success_rate:.1f}%')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'Success data not available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'episode_statistics.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'episode_statistics.png'}")
    plt.close()


def generate_summary_report(data: Dict, output_dir: Path):
    """Generate a text summary report."""
    report_file = output_dir / 'evaluation_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        summary = data['summary']
        stats = summary['statistics']
        
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(f"Checkpoint: {summary['checkpoint']}\n")
        f.write(f"Episodes: {summary['num_episodes']}\n")
        f.write(f"Environments: {summary['num_envs']}\n")
        f.write(f"Trajectory Type: {summary['trajectory_type']}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("TRACKING ACCURACY\n")
        f.write("=" * 80 + "\n\n")
        
        if 'position_error' in stats:
            f.write("Position Error (cm):\n")
            f.write(f"  Mean:   {stats['position_error']['mean_cm']:.2f}\n")
            f.write(f"  Median: {stats['position_error']['median_cm']:.2f}\n")
            f.write(f"  Std:    {stats['position_error']['std_m'] * 100:.2f}\n")
            f.write(f"  P95:    {stats['position_error']['p95_cm']:.2f}\n")
            f.write(f"  P99:    {stats['position_error']['p99_m'] * 100:.2f}\n")
            f.write(f"  Max:    {stats['position_error']['max_m'] * 100:.2f}\n\n")
        
        if 'orientation_error' in stats:
            f.write("Orientation Error (degrees):\n")
            f.write(f"  Mean:   {stats['orientation_error']['mean_deg']:.2f}\n")
            f.write(f"  Median: {stats['orientation_error']['median_deg']:.2f}\n")
            f.write(f"  Std:    {np.rad2deg(stats['orientation_error']['std_rad']):.2f}\n")
            f.write(f"  P95:    {stats['orientation_error']['p95_deg']:.2f}\n")
            f.write(f"  Max:    {np.rad2deg(stats['orientation_error']['max_rad']):.2f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("EPISODE STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        if 'episodes' in stats:
            f.write(f"Total Episodes: {stats['episodes']['count']}\n")
            f.write(f"Mean Reward: {stats['episodes']['mean_reward']:.2f} ± {stats['episodes']['std_reward']:.2f}\n")
            f.write(f"Median Reward: {stats['episodes']['median_reward']:.2f}\n")
            f.write(f"Reward Range: [{stats['episodes']['min_reward']:.2f}, {stats['episodes']['max_reward']:.2f}]\n")
            f.write(f"Mean Length: {stats['episodes']['mean_length']:.1f} ± {stats['episodes']['std_length']:.1f} steps\n")
            
            if 'success_rate' in stats['episodes']:
                f.write(f"\nSuccess Rate: {stats['episodes']['success_rate'] * 100:.1f}%\n")
                f.write(f"Success Count: {stats['episodes']['success_count']}/{stats['episodes']['count']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("ASSESSMENT\n")
        f.write("=" * 80 + "\n\n")
        
        # Add assessment based on metrics
        if 'position_error' in stats:
            mean_error = stats['position_error']['mean_cm']
            if mean_error < 10:
                f.write("✅ Position tracking: EXCELLENT (< 10 cm)\n")
            elif mean_error < 20:
                f.write("✅ Position tracking: GOOD (< 20 cm)\n")
            elif mean_error < 50:
                f.write("⚠️  Position tracking: ACCEPTABLE (< 50 cm)\n")
            else:
                f.write("❌ Position tracking: NEEDS IMPROVEMENT (> 50 cm)\n")
        
        if 'orientation_error' in stats:
            mean_error = stats['orientation_error']['mean_deg']
            if mean_error < 5:
                f.write("✅ Orientation tracking: EXCELLENT (< 5°)\n")
            elif mean_error < 10:
                f.write("✅ Orientation tracking: GOOD (< 10°)\n")
            elif mean_error < 20:
                f.write("⚠️  Orientation tracking: ACCEPTABLE (< 20°)\n")
            else:
                f.write("❌ Orientation tracking: NEEDS IMPROVEMENT (> 20°)\n")
        
        if 'episodes' in stats and 'success_rate' in stats['episodes']:
            success_rate = stats['episodes']['success_rate'] * 100
            if success_rate > 90:
                f.write("✅ Success rate: EXCELLENT (> 90%)\n")
            elif success_rate > 75:
                f.write("✅ Success rate: GOOD (> 75%)\n")
            elif success_rate > 50:
                f.write("⚠️  Success rate: ACCEPTABLE (> 50%)\n")
            else:
                f.write("❌ Success rate: NEEDS IMPROVEMENT (< 50%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✓ Saved: {report_file}")


def main():
    args = parse_args()
    
    if not args.input and not args.input_dir:
        print("❌ Error: Must provide either --input or --input_dir")
        return
    
    # Find summary files
    summary_files = []
    if args.input:
        summary_files.append(Path(args.input))
    elif args.input_dir:
        input_path = Path(args.input_dir)
        summary_files = list(input_path.glob("eval_summary_*.json"))
    
    if not summary_files:
        print(f"❌ Error: No evaluation summary files found")
        return
    
    print(f"Found {len(summary_files)} evaluation result(s)")
    
    # Process each summary file
    for summary_file in summary_files:
        print(f"\n{'='*80}")
        print(f"Processing: {summary_file.name}")
        print(f"{'='*80}")
        
        # Load data
        data = load_evaluation_data(summary_file)
        
        # Create output directory with same structure as input
        # If input is evaluation_results/20251028_200923/eval_summary_xxx.json
        # Output will be evaluation_plots/20251028_200923/
        model_folder = summary_file.parent.name
        input_path = Path(args.input) if args.input is not None else None
        if input_path is not None and model_folder != input_path.parent.name:  # Has model subfolder
            output_dir = Path(args.output_dir) / model_folder
        else:
            output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        print("\n📊 Generating plots...")
        print(f"Output directory: {output_dir}")
        plot_tracking_errors(data, output_dir)
        plot_joint_analysis(data, output_dir)
        plot_base_velocities(data, output_dir)
        plot_reward_components(data, output_dir)
        plot_episode_statistics(data, output_dir)
        
        # Generate summary report
        print("\n📄 Generating summary report...")
        generate_summary_report(data, output_dir)
        
        print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
