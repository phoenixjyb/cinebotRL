"""Quick debug script to check what first waypoints are actually loaded."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from rl_platform.tasks.mobile_mm.trajectories import TrajectoryManager

# Test configuration
traj_dir = Path(__file__).parent.parent / "trajectoryToLearn" / "world_json"
device = "cuda" if torch.cuda.is_available() else "cpu"
num_envs = 16

print("\n" + "="*80)
print("TESTING TRAJECTORY LOADING - FIRST WAYPOINTS")
print("="*80)

# Create trajectory manager
manager = TrajectoryManager(
    traj_type="multi_recorded",
    num_envs=num_envs,
    device=device,
    dt=0.05,
    trajectory_dir=str(traj_dir),
    trajectory_pattern="**/*.json",
)

print(f"\n✓ TrajectoryManager initialized with {len(manager.multi_loader.trajectories)} trajectories")

# Get first waypoints for all environments
print(f"\nFirst waypoints for all {num_envs} environments:")
print("-"*80)
for i in range(num_envs):
    first_wp = manager.recorded_positions[i, 0].cpu().numpy()
    print(f"Env {i:2d}: [{first_wp[0]:7.3f}, {first_wp[1]:7.3f}, {first_wp[2]:7.3f}]")

# Now test reset
print("\n" + "="*80)
print("TESTING RESET - Resampling environments 0, 5, 10")
print("="*80)

env_ids = torch.tensor([0, 5, 10], device=device)
manager.reset(env_ids)

print(f"\nFirst waypoints after reset:")
print("-"*80)
for i in [0, 5, 10]:
    first_wp = manager.recorded_positions[i, 0].cpu().numpy()
    print(f"Env {i:2d}: [{first_wp[0]:7.3f}, {first_wp[1]:7.3f}, {first_wp[2]:7.3f}]")

# Test get_target_pose
print("\n" + "="*80)
print("TESTING get_target_pose() - Should return first waypoints")
print("="*80)

target_pos, _ = manager.get_target_pose()
print(f"\nget_target_pose() results:")
print("-"*80)
for i in [0, 5, 10]:
    wp = target_pos[i].cpu().numpy()
    print(f"Env {i:2d}: [{wp[0]:7.3f}, {wp[1]:7.3f}, {wp[2]:7.3f}]")

print("\n" + "="*80)
print("DONE")
print("="*80)
