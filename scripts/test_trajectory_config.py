#!/usr/bin/env python3
"""
Test if trajectory configuration is being applied correctly during environment creation.
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

print("="*70)
print("TESTING TRAJECTORY CONFIG")
print("="*70)
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")
print()

# Test 1: Create TrajectoryConfig
print("\n[1] Creating TrajectoryConfig...")
from rl_platform.tasks.mobile_mm.config import TrajectoryConfig

traj_cfg = TrajectoryConfig(
    type="multi_recorded",
    trajectory_dir="trajectoryToLearn/world_json",
    trajectory_pattern="**/*.json",
    trajectory_filter_indices=None,
    max_trajectories=None,
)

print(f"  ✓ TrajectoryConfig created:")
print(f"    type: {traj_cfg.type}")
print(f"    trajectory_dir: {traj_cfg.trajectory_dir}")
print(f"    trajectory_pattern: {traj_cfg.trajectory_pattern}")

# Test 2: Create MobileMMTrackEEEnvCfg
print("\n[2] Creating MobileMMTrackEEEnvCfg...")
from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnvCfg

env_cfg = MobileMMTrackEEEnvCfg()
env_cfg.scene.num_envs = 4
env_cfg.task_config.trajectory = traj_cfg

print(f"  ✓ EnvCfg created:")
print(f"    num_envs: {env_cfg.scene.num_envs}")
print(f"    trajectory type: {env_cfg.task_config.trajectory.type}")
print(f"    trajectory dir: {env_cfg.task_config.trajectory.trajectory_dir}")

# Test 3: Check if trajectory directory exists
print("\n[3] Checking trajectory directory...")
traj_dir = Path(traj_cfg.trajectory_dir)
if traj_dir.exists():
    json_files = list(traj_dir.glob(traj_cfg.trajectory_pattern))
    # Filter __MACOSX
    json_files = [f for f in json_files if '__MACOSX' not in str(f)]
    print(f"  ✓ Directory exists: {traj_dir.absolute()}")
    print(f"  ✓ Found {len(json_files)} JSON files")
else:
    print(f"  ✗ Directory NOT found: {traj_dir.absolute()}")

print("\n" + "="*70)
print("CONFIG TEST COMPLETE - No Isaac Sim needed!")
print("="*70)
