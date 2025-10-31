#!/usr/bin/env python3
"""
Compare multiple training sessions side by side.
Generates comparison plots showing metrics from different sessions.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


def load_session_data(json_path: Path) -> Dict:
    """Load evaluation summary from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
        # Return statistics section if it exists, otherwise return full data
        return data.get('statistics', data)


def plot_tracking_error_comparison(sessions: Dict[str, Dict], output_dir: Path):
    """Compare tracking errors across sessions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    session_names = list(sessions.keys())
    colors = sns.color_palette("husl", len(sessions))
    
    # Position error comparison
    ax = axes[0, 0]
    pos_data = []
    for name in session_names:
        stats = sessions[name]['position_error']
        pos_data.append([stats['mean_cm'], stats['median_cm'], stats['p95_cm']])
    
    x = np.arange(3)
    width = 0.25
    for i, name in enumerate(session_names):
        offset = (i - len(session_names)/2 + 0.5) * width
        ax.bar(x + offset, pos_data[i], width, label=name, color=colors[i], alpha=0.8)
    
    ax.set_ylabel('Position Error (cm)', fontsize=12, fontweight='bold')
    ax.set_title('Position Tracking Error Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Mean', 'Median', 'P95'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Orientation error comparison
    ax = axes[0, 1]
    ori_data = []
    for name in session_names:
        stats = sessions[name]['orientation_error']
        ori_data.append([stats['mean_deg'], stats['median_deg'], stats['p95_deg']])
    
    for i, name in enumerate(session_names):
        offset = (i - len(session_names)/2 + 0.5) * width
        ax.bar(x + offset, ori_data[i], width, label=name, color=colors[i], alpha=0.8)
    
    ax.set_ylabel('Orientation Error (degrees)', fontsize=12, fontweight='bold')
    ax.set_title('Orientation Tracking Error Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Mean', 'Median', 'P95'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Joint velocity comparison
    ax = axes[1, 0]
    for i, name in enumerate(session_names):
        joint_vels = []
        for j in range(6):
            vel_stats = sessions[name]['joint_velocities'][f'joint_{j}']
            joint_vels.append(vel_stats['mean_rad_s'])
        ax.plot(range(6), joint_vels, marker='o', label=name, color=colors[i], linewidth=2)
    
    ax.set_xlabel('Joint Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Velocity (rad/s)', fontsize=12, fontweight='bold')
    ax.set_title('Joint Velocity Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(6))
    
    # Success rate comparison (if available)
    ax = axes[1, 1]
    metrics = ['Position\nError', 'Orientation\nError', 'Mean\nReward', 'Episode\nLength']
    
    # Normalize metrics for comparison (0-1 scale, lower is better for errors)
    normalized_data = []
    for name in session_names:
        data = sessions[name]
        pos_norm = 1.0 - min(data['position_error']['mean_cm'] / 500.0, 1.0)  # Lower is better
        ori_norm = 1.0 - min(data['orientation_error']['mean_deg'] / 90.0, 1.0)  # Lower is better
        reward_norm = min((data['episodes']['mean_reward'] + 1000) / 2000.0, 1.0)  # Higher is better
        length_norm = min(data['episodes']['mean_length'] / 1000.0, 1.0)  # Normalize to 0-1
        normalized_data.append([pos_norm, ori_norm, reward_norm, length_norm])
    
    x = np.arange(len(metrics))
    for i, name in enumerate(session_names):
        offset = (i - len(session_names)/2 + 0.5) * width
        ax.bar(x + offset, normalized_data[i], width, label=name, color=colors[i], alpha=0.8)
    
    ax.set_ylabel('Normalized Score (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    output_path = output_dir / 'session_comparison_tracking.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_reward_comparison(sessions: Dict[str, Dict], output_dir: Path):
    """Compare reward components across sessions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    session_names = list(sessions.keys())
    colors = sns.color_palette("husl", len(sessions))
    
    # Mean episode reward
    ax = axes[0, 0]
    rewards = [sessions[name]['episodes']['mean_reward'] for name in session_names]
    bars = ax.bar(session_names, rewards, color=colors, alpha=0.8)
    ax.set_ylabel('Mean Episode Reward', fontsize=12, fontweight='bold')
    ax.set_title('Mean Episode Reward Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Reward components breakdown
    ax = axes[0, 1]
    reward_components = ['position_tracking', 'orientation_tracking', 'joint_velocity', 
                        'base_velocity', 'reachability_penalty']
    
    x = np.arange(len(reward_components))
    width = 0.25
    
    for i, name in enumerate(session_names):
        if 'reward_components' in sessions[name]:
            comp_data = []
            for comp in reward_components:
                if comp in sessions[name]['reward_components']:
                    comp_data.append(sessions[name]['reward_components'][comp]['mean'])
                else:
                    comp_data.append(0.0)
            
            offset = (i - len(session_names)/2 + 0.5) * width
            ax.bar(x + offset, comp_data, width, label=name, color=colors[i], alpha=0.8)
    
    ax.set_ylabel('Mean Reward Component', fontsize=12, fontweight='bold')
    ax.set_title('Reward Components Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(reward_components, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Episode length comparison
    ax = axes[1, 0]
    lengths = [sessions[name]['episodes']['mean_length'] for name in session_names]
    bars = ax.bar(session_names, lengths, color=colors, alpha=0.8)
    ax.set_ylabel('Mean Episode Length (steps)', fontsize=12, fontweight='bold')
    ax.set_title('Episode Length Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, lengths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = []
    headers = ['Metric'] + session_names
    
    metrics_to_show = [
        ('Pos Error (cm)', lambda s: f"{s['position_error']['mean_cm']:.1f}"),
        ('Ori Error (deg)', lambda s: f"{s['orientation_error']['mean_deg']:.1f}"),
        ('Mean Reward', lambda s: f"{s['episodes']['mean_reward']:.1f}"),
        ('Episode Length', lambda s: f"{s['episodes']['mean_length']:.0f}"),
    ]
    
    for metric_name, metric_func in metrics_to_show:
        row = [metric_name]
        for name in session_names:
            row.append(metric_func(sessions[name]))
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.3] + [0.35/len(session_names)] * len(session_names))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / 'session_comparison_rewards.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_joint_comparison(sessions: Dict[str, Dict], output_dir: Path):
    """Compare joint usage across sessions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    session_names = list(sessions.keys())
    colors = sns.color_palette("husl", len(sessions))
    
    for joint_idx in range(6):
        ax = axes[joint_idx]
        
        x = np.arange(3)  # mean, std, range
        width = 0.25
        
        for i, name in enumerate(session_names):
            joint_data = sessions[name]['joint_angles'][f'joint_{joint_idx}']
            values = [joint_data['mean_rad'], joint_data['std_rad'], joint_data['range_rad']]
            
            offset = (i - len(session_names)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=name, color=colors[i], alpha=0.8)
        
        ax.set_title(f'Joint {joint_idx} Angle Usage', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Mean', 'Std', 'Range'])
        ax.set_ylabel('Angle (rad)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'session_comparison_joints.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_comparison_report(sessions: Dict[str, Dict], output_dir: Path):
    """Generate text report comparing sessions."""
    output_path = output_dir / 'comparison_report.txt'
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SESSION COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        session_names = list(sessions.keys())
        
        # Tracking Performance
        f.write("TRACKING PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Session':<20} {'Pos Error (cm)':<20} {'Ori Error (deg)':<20}\n")
        f.write(f"{'':<20} {'Mean/Med/P95':<20} {'Mean/Med/P95':<20}\n")
        f.write("-" * 80 + "\n")
        
        for name in session_names:
            pos = sessions[name]['position_error']
            ori = sessions[name]['orientation_error']
            f.write(f"{name:<20} {pos['mean_cm']:>6.1f}/{pos['median_cm']:>6.1f}/{pos['p95_cm']:>6.1f}  ")
            f.write(f"{ori['mean_deg']:>6.1f}/{ori['median_deg']:>6.1f}/{ori['p95_deg']:>6.1f}\n")
        
        f.write("\n")
        
        # Overall Performance
        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Session':<20} {'Mean Reward':<20} {'Episode Length':<20}\n")
        f.write("-" * 80 + "\n")
        
        for name in session_names:
            f.write(f"{name:<20} {sessions[name]['episodes']['mean_reward']:>12.2f}     ")
            f.write(f"{sessions[name]['episodes']['mean_length']:>12.1f}\n")
        
        f.write("\n")
        
        # Best performing session
        f.write("ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        # Find best for each metric
        best_pos = min(session_names, key=lambda n: sessions[n]['position_error']['mean_cm'])
        best_ori = min(session_names, key=lambda n: sessions[n]['orientation_error']['mean_deg'])
        best_reward = max(session_names, key=lambda n: sessions[n]['episodes']['mean_reward'])
        
        f.write(f"Best Position Tracking:    {best_pos} ")
        f.write(f"({sessions[best_pos]['position_error']['mean_cm']:.1f} cm)\n")
        
        f.write(f"Best Orientation Tracking: {best_ori} ")
        f.write(f"({sessions[best_ori]['orientation_error']['mean_deg']:.1f} deg)\n")
        
        f.write(f"Best Mean Reward:          {best_reward} ")
        f.write(f"({sessions[best_reward]['episodes']['mean_reward']:.2f})\n")
        
        f.write("\n")
        
        # Relative improvements
        if len(session_names) >= 2:
            f.write("RELATIVE CHANGES (vs first session)\n")
            f.write("-" * 80 + "\n")
            baseline = session_names[0]
            
            for name in session_names[1:]:
                pos_change = ((sessions[name]['position_error']['mean_cm'] - 
                              sessions[baseline]['position_error']['mean_cm']) / 
                             sessions[baseline]['position_error']['mean_cm'] * 100)
                
                ori_change = ((sessions[name]['orientation_error']['mean_deg'] - 
                              sessions[baseline]['orientation_error']['mean_deg']) / 
                             sessions[baseline]['orientation_error']['mean_deg'] * 100)
                
                reward_change = ((sessions[name]['episodes']['mean_reward'] - 
                                 sessions[baseline]['episodes']['mean_reward']) / 
                                abs(sessions[baseline]['episodes']['mean_reward']) * 100)
                
                f.write(f"\n{name} vs {baseline}:\n")
                f.write(f"  Position Error:    {pos_change:+.1f}% ")
                f.write(f"({'BETTER' if pos_change < 0 else 'WORSE'})\n")
                
                f.write(f"  Orientation Error: {ori_change:+.1f}% ")
                f.write(f"({'BETTER' if ori_change < 0 else 'WORSE'})\n")
                
                f.write(f"  Mean Reward:       {reward_change:+.1f}% ")
                f.write(f"({'BETTER' if reward_change > 0 else 'WORSE'})\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple training sessions")
    parser.add_argument('--sessions', nargs='+', required=True,
                       help='Paths to eval_summary JSON files (format: name:path)')
    parser.add_argument('--output_dir', type=str, default='evaluation_plots/comparison',
                       help='Output directory for comparison plots')
    
    args = parser.parse_args()
    
    # Parse session arguments (format: name:path)
    sessions = {}
    for session_arg in args.sessions:
        if ':' in session_arg:
            name, path = session_arg.split(':', 1)
        else:
            # Use filename as name if not specified
            path = session_arg
            name = Path(path).parent.name
        
        print(f"Loading: {name} from {path}")
        sessions[name] = load_session_data(Path(path))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nComparing {len(sessions)} sessions...")
    print(f"Output directory: {output_dir}\n")
    
    # Generate comparison plots
    plot_tracking_error_comparison(sessions, output_dir)
    plot_reward_comparison(sessions, output_dir)
    plot_joint_comparison(sessions, output_dir)
    
    # Generate report
    generate_comparison_report(sessions, output_dir)
    
    print(f"\nComparison complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
