"""Analyze trajectory to determine if base movement is required."""
import json
import numpy as np
from pathlib import Path

# Load trajectory
traj_file = Path(__file__).parent.parent / "trajectoryToLearn" / "1_pull_world_scaled.json"
with open(traj_file) as f:
    traj = json.load(f)

# Extract waypoints
if 'trajectory' in traj:
    waypoints = traj['trajectory']
elif 'poses' in traj:
    waypoints = traj['poses']
elif isinstance(traj, list):
    waypoints = traj
else:
    raise ValueError(f"Unknown trajectory format: {list(traj.keys())}")

# Assume starting base at trajectory start (typical setup)
first_wp = waypoints[0]
if 'ee_pos' in first_wp:
    base_pos = np.array(first_wp['ee_pos'])
elif 'position' in first_wp:
    base_pos = np.array(first_wp['position'])
else:
    base_pos = np.array([0, 0, 0.86])

print(f"\nTrajectory Analysis: {traj_file.name}")
print(f"{'='*80}")
print(f"Base starting position: {base_pos}")
print(f"Total waypoints: {len(waypoints)}")

# Extract all positions
positions = []
for wp in waypoints:
    if 'ee_pos' in wp:
        positions.append(wp['ee_pos'])
    elif 'position' in wp:
        positions.append(wp['position'])
    else:
        print(f"Warning: waypoint missing position: {wp}")
        
positions = np.array(positions)

# Compute 2D distances from base (XY plane, ignore Z)
distances_2d = np.linalg.norm(positions[:, :2] - base_pos[:2], axis=1)
distances_3d = np.linalg.norm(positions - base_pos, axis=1)

# Arm reach thresholds
ARM_REACH_CONSERVATIVE = 0.5  # meters (safe estimate)
ARM_REACH_NOMINAL = 0.6  # meters (typical estimate)
ARM_REACH_MAX = 0.8  # meters (maximum stretch)

print(f"\n📏 Distance Statistics (2D - XY plane):")
print(f"{'='*80}")
print(f"  Min:    {distances_2d.min():.3f} m")
print(f"  Mean:   {distances_2d.mean():.3f} m")
print(f"  Median: {np.median(distances_2d):.3f} m")
print(f"  Max:    {distances_2d.max():.3f} m")
print(f"  Std:    {distances_2d.std():.3f} m")

print(f"\n📏 Distance Statistics (3D - full space):")
print(f"{'='*80}")
print(f"  Min:    {distances_3d.min():.3f} m")
print(f"  Mean:   {distances_3d.mean():.3f} m")
print(f"  Median: {np.median(distances_3d):.3f} m")
print(f"  Max:    {distances_3d.max():.3f} m")
print(f"  Std:    {distances_3d.std():.3f} m")

print(f"\n🎯 Reachability Analysis:")
print(f"{'='*80}")
for reach, label in [(ARM_REACH_CONSERVATIVE, "Conservative (0.5m)"),
                     (ARM_REACH_NOMINAL, "Nominal (0.6m)"),
                     (ARM_REACH_MAX, "Maximum (0.8m)")]:
    within = (distances_2d <= reach).sum()
    beyond = (distances_2d > reach).sum()
    pct_within = within / len(distances_2d) * 100
    pct_beyond = beyond / len(distances_2d) * 100
    
    print(f"\n  {label}:")
    print(f"    Within reach: {within:4d} waypoints ({pct_within:5.1f}%)")
    print(f"    Out of reach: {beyond:4d} waypoints ({pct_beyond:5.1f}%)")

# Find longest consecutive out-of-reach sequence
out_of_reach = distances_2d > ARM_REACH_NOMINAL
consecutive_counts = []
current_count = 0
for is_out in out_of_reach:
    if is_out:
        current_count += 1
    else:
        if current_count > 0:
            consecutive_counts.append(current_count)
        current_count = 0
if current_count > 0:
    consecutive_counts.append(current_count)

if consecutive_counts:
    max_consecutive = max(consecutive_counts)
    avg_consecutive = np.mean(consecutive_counts)
    print(f"\n⏱️  Out-of-Reach Dwell Time (@ {ARM_REACH_NOMINAL}m reach):")
    print(f"{'='*80}")
    print(f"  Longest consecutive OOR sequence: {max_consecutive} waypoints")
    print(f"  Average OOR sequence length: {avg_consecutive:.1f} waypoints")
    print(f"  Number of OOR sequences: {len(consecutive_counts)}")
    print(f"  @ 0.1s/waypoint: {max_consecutive * 0.1:.2f}s longest, {avg_consecutive * 0.1:.2f}s average")
else:
    print(f"\n✅ All waypoints within reach! No base movement needed.")

print(f"\n💡 Conclusions:")
print(f"{'='*80}")
if distances_2d.max() <= ARM_REACH_NOMINAL:
    print(f"  ✅ ENTIRE TRAJECTORY within nominal arm reach ({ARM_REACH_NOMINAL}m)")
    print(f"  ➡️  Base movement NOT required - policy is CORRECT to stay still!")
elif (distances_2d > ARM_REACH_NOMINAL).mean() < 0.2:
    print(f"  ⚠️  Only {(distances_2d > ARM_REACH_NOMINAL).mean()*100:.1f}% of trajectory out of reach")
    print(f"  ➡️  Base movement OPTIONAL - arm-only strategy viable")
elif (distances_2d > ARM_REACH_NOMINAL).mean() < 0.5:
    print(f"  ⚠️  {(distances_2d > ARM_REACH_NOMINAL).mean()*100:.1f}% of trajectory out of reach")
    print(f"  ➡️  Base movement BENEFICIAL but not critical")
else:
    print(f"  🚨 {(distances_2d > ARM_REACH_NOMINAL).mean()*100:.1f}% of trajectory out of reach")
    print(f"  ➡️  Base movement REQUIRED - investigate why policy not learning it!")

print(f"\n")
