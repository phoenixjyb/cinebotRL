"""
Validate Session 8 reward weights by computing projected reward/penalty balance.

This script uses the actual evaluation data from Session 7d to predict what
the rewards would be under Session 8 weights, helping verify the design is sound.

Usage:
    python scripts/reinforcement_learning/sb3/validate_session8_weights.py \
        --eval_summary evaluation_results/20251028_200923/eval_summary_20251029_131728.json

Author: Generated for Session 8 validation
Date: October 29, 2025
"""

import argparse
import json
from pathlib import Path
from typing import Dict


def load_session7d_data(eval_file: Path) -> Dict[str, float]:
    """Load Session 7d evaluation data."""
    with open(eval_file, 'r') as f:
        data = json.load(f)
    
    # Extract mean reward components (these are per-step values)
    reward_comps = data['statistics']['reward_components']
    components = {
        'position_tracking': reward_comps['position_tracking']['mean'],
        'orientation_tracking': reward_comps['orientation_tracking']['mean'],
        'progress_bonus': reward_comps['progress_bonus']['mean'],
        'base_mobilization': reward_comps['base_mobilization']['mean'],
        'base_target_alignment': reward_comps['base_target_alignment']['mean'],
        'target_distance_penalty': -reward_comps['target_distance_penalty']['mean'],  # Negate (it's a penalty)
        'excessive_base_movement_penalty': -reward_comps['excessive_base_movement_penalty']['mean'],
        'action_magnitude_penalty': -reward_comps['action_magnitude_penalty']['mean'],
        'action_rate_penalty': -reward_comps['action_rate_penalty']['mean'],
        'action_smoothness_penalty': -reward_comps['action_smoothness_penalty']['mean'],
        'velocity_limit_penalty': -reward_comps['velocity_limit_penalty']['mean'],
        'acceleration_limit_penalty': -reward_comps['acceleration_limit_penalty']['mean'],
        'jerk_penalty': -reward_comps['jerk_penalty']['mean'],
        'joint_limit_penalty': -reward_comps['joint_limit_penalty']['mean'],
        'lateral_motion_penalty': -reward_comps['lateral_motion_penalty']['mean'],
        'self_collision_penalty': -reward_comps['self_collision_penalty']['mean'],
        'stability_penalty': -reward_comps['stability_penalty']['mean'],
    }
    
    return components


def compute_session8_projections(session7d_components: Dict[str, float]) -> Dict[str, float]:
    """
    Project what Session 8 rewards would be, given Session 7d behavior.
    
    Key assumptions:
    1. Tracking error magnitudes stay similar initially (policy hasn't learned yet)
    2. Violation rates stay similar (robot still moves at similar speeds)
    3. Weight changes directly scale the reward components
    
    This gives us a LOWER BOUND on Session 8 performance (actual should be better
    as policy adapts to new weights).
    """
    
    # Session 7d → Session 8 weight ratios
    weight_ratios = {
        'position_tracking': 150.0 / 100.0,        # 1.5×
        'orientation_tracking': 75.0 / 2.0,        # 37.5×! 🔥
        'progress_bonus': 5.0 / 1.0,               # 5×
        'base_mobilization': 400.0 / 250.0,        # 1.6× (scales base_progress_reward)
        'base_target_alignment': 30.0 / 10.0,      # 3×
        'target_distance_penalty': 1.0 / 3.0,      # 0.33× (reduction)
        'excessive_base_movement_penalty': 5.0 / 10.0,  # 0.5× (reduction)
        'action_magnitude_penalty': 0.002 / 0.005,  # 0.4× (reduction)
        'action_rate_penalty': 0.005 / 0.01,       # 0.5× (reduction)
        'action_smoothness_penalty': 0.05 / 0.15,  # 0.33× (reduction)
        'velocity_limit_penalty': 1.5 / 5.0,       # 0.3× (70% reduction!) 🔥
        'acceleration_limit_penalty': 1.5 / 5.0,   # 0.3× (70% reduction)
        'jerk_penalty': 0.01 / 0.05,               # 0.2× (80% reduction!) 🔥
        'joint_limit_penalty': 5.0 / 10.0,         # 0.5× (reduction)
        'lateral_motion_penalty': 1.0 / 2.0,       # 0.5× (reduction)
        'self_collision_penalty': 1.0 / 0.5,       # 2× (increase)
        'stability_penalty': 0.2 / 0.1,            # 2× (increase)
    }
    
    # Compute projected Session 8 values
    session8_components = {}
    for component, session7d_value in session7d_components.items():
        if component in weight_ratios:
            session8_components[component] = session7d_value * weight_ratios[component]
        else:
            # Component not affected by weight change (shouldn't happen)
            session8_components[component] = session7d_value
    
    return session8_components


def print_comparison_table(session7d: Dict[str, float], session8: Dict[str, float]) -> None:
    """Print side-by-side comparison table."""
    
    print("\n" + "="*100)
    print("SESSION 8 REWARD PROJECTION (Based on Session 7d Evaluation Data)")
    print("="*100)
    print("\nASSUMPTIONS:")
    print("  - Robot behavior initially similar (policy hasn't adapted yet)")
    print("  - These are LOWER BOUND estimates (actual should improve as policy learns)")
    print("  - Per-step values (multiply by ~400 steps for episode total)")
    print("\n" + "-"*100)
    print(f"{'Component':<40} {'Session 7d':>15} {'Session 8':>15} {'Change':>15} {'Impact'}")
    print("-"*100)
    
    # Separate into rewards and penalties
    rewards = []
    penalties = []
    
    for component in session7d.keys():
        s7d_val = session7d[component]
        s8_val = session8[component]
        change = s8_val - s7d_val
        
        # Categorize
        if s7d_val >= 0:
            rewards.append((component, s7d_val, s8_val, change))
        else:
            penalties.append((component, s7d_val, s8_val, change))
    
    # Print rewards
    print("\n🎁 REWARDS (Positive contributions):")
    print("-"*100)
    total_reward_s7d = 0
    total_reward_s8 = 0
    for component, s7d_val, s8_val, change in sorted(rewards, key=lambda x: -x[2]):
        pct_change = ((s8_val / s7d_val) - 1) * 100 if s7d_val != 0 else 999
        impact = "🔥🔥🔥" if pct_change > 1000 else "🔥🔥" if pct_change > 100 else "🔥" if pct_change > 50 else "✅" if change > 0 else ""
        print(f"{component:<40} {s7d_val:>15.2f} {s8_val:>15.2f} {change:>+15.2f} {impact}")
        total_reward_s7d += s7d_val
        total_reward_s8 += s8_val
    
    print("-"*100)
    print(f"{'TOTAL REWARDS':<40} {total_reward_s7d:>15.2f} {total_reward_s8:>15.2f} {total_reward_s8 - total_reward_s7d:>+15.2f}")
    
    # Print penalties
    print("\n❌ PENALTIES (Negative contributions):")
    print("-"*100)
    total_penalty_s7d = 0
    total_penalty_s8 = 0
    for component, s7d_val, s8_val, change in sorted(penalties, key=lambda x: x[1]):
        pct_change = ((s8_val / s7d_val) - 1) * 100 if s7d_val != 0 else 999
        impact = "🔥🔥" if abs(change) > 10 else "🔥" if abs(change) > 5 else "✅" if change > 0 else ""
        print(f"{component:<40} {s7d_val:>15.2f} {s8_val:>15.2f} {change:>+15.2f} {impact}")
        total_penalty_s7d += s7d_val
        total_penalty_s8 += s8_val
    
    print("-"*100)
    print(f"{'TOTAL PENALTIES':<40} {total_penalty_s7d:>15.2f} {total_penalty_s8:>15.2f} {total_penalty_s8 - total_penalty_s7d:>+15.2f}")
    
    # Print totals
    print("\n" + "="*100)
    print("SUMMARY (per step):")
    print("="*100)
    net_s7d = total_reward_s7d + total_penalty_s7d
    net_s8 = total_reward_s8 + total_penalty_s8
    ratio_s7d = abs(total_reward_s7d / total_penalty_s7d) if total_penalty_s7d != 0 else float('inf')
    ratio_s8 = abs(total_reward_s8 / total_penalty_s8) if total_penalty_s8 != 0 else float('inf')
    
    print(f"Total Rewards:        {total_reward_s7d:>10.2f}  →  {total_reward_s8:>10.2f}  ({(total_reward_s8/total_reward_s7d - 1)*100:+.1f}%)")
    print(f"Total Penalties:      {total_penalty_s7d:>10.2f}  →  {total_penalty_s8:>10.2f}  ({(total_penalty_s8/total_penalty_s7d - 1)*100:+.1f}%)")
    print(f"Net Reward:           {net_s7d:>10.2f}  →  {net_s8:>10.2f}  ({'✅ POSITIVE!' if net_s8 > 0 else '❌ NEGATIVE'})")
    print(f"Reward/Penalty Ratio: {ratio_s7d:>10.2f}  →  {ratio_s8:>10.2f}  ({'✅ REWARDS DOMINATE' if ratio_s8 > 2 else '⚠️ STILL IMBALANCED'})")
    
    # Episode projections (assuming 400 steps per episode)
    steps_per_episode = 400
    print(f"\nEPISODE PROJECTION (~{steps_per_episode} steps):")
    print(f"Session 7d:  {net_s7d * steps_per_episode:>10.0f}  (actual: -5,120)")
    print(f"Session 8:   {net_s8 * steps_per_episode:>10.0f}  {'✅ MUCH BETTER!' if net_s8 * steps_per_episode > -1000 else '⚠️ STILL NEEDS WORK'}")
    
    print("\n" + "="*100)
    print("INTERPRETATION:")
    print("="*100)
    if net_s8 > 0 and ratio_s8 > 2:
        print("✅ **EXCELLENT**: Session 8 weights look good!")
        print("   - Net reward is POSITIVE")
        print("   - Rewards dominate penalties (ratio > 2:1)")
        print("   - Policy should learn to maximize tracking, not minimize penalties")
        print("\n👉 **RECOMMENDATION**: Proceed with 10M validation run, then full training")
    elif net_s8 > -2:
        print("⚠️ **ACCEPTABLE**: Session 8 weights are much better but may need fine-tuning")
        print("   - Net reward is close to zero or slightly negative")
        print("   - Still much better than Session 7d")
        print("\n👉 **RECOMMENDATION**: Run 10M validation, monitor closely, adjust if needed")
    else:
        print("❌ **NEEDS WORK**: Session 8 weights still have issues")
        print("   - Net reward still significantly negative")
        print("   - May need further weight adjustments")
        print("\n👉 **RECOMMENDATION**: Review weight ratios before training")
    
    print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate Session 8 reward weights")
    parser.add_argument(
        '--eval_summary',
        type=str,
        default='evaluation_results/20251028_200923/eval_summary_20251029_131728.json',
        help='Path to Session 7d evaluation summary JSON'
    )
    args = parser.parse_args()
    
    eval_file = Path(args.eval_summary)
    if not eval_file.exists():
        print(f"❌ Error: Evaluation file not found: {eval_file}")
        return 1
    
    print(f"\n📊 Loading Session 7d evaluation data from: {eval_file}")
    session7d_components = load_session7d_data(eval_file)
    
    print(f"🔮 Computing Session 8 projections...")
    session8_components = compute_session8_projections(session7d_components)
    
    print_comparison_table(session7d_components, session8_components)
    
    return 0


if __name__ == '__main__':
    exit(main())
