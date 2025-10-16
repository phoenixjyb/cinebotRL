"""Check the exact joint ordering in Isaac Lab for our robot.

This verifies:
1. What order joints appear in robot.data.joint_pos
2. Which indices correspond to base (PPR) vs arm joints
3. Joint limits for each joint
"""

import argparse
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Check joint ordering")
parser.add_argument("--headless", action="store_true", help="Run headless")
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Register task
from task_spec import register_isaac_lab_tasks
register_isaac_lab_tasks()

import gymnasium as gym

print("\n" + "="*70)
print("JOINT ORDER VERIFICATION")
print("="*70)

# Create environment
env = gym.make("MobileMMTrackEE-v0", num_envs=1, headless=args_cli.headless)
print(f"\n✓ Environment created\n")

# Access robot
robot = env.unwrapped.robot

print("Joint Information:")
print("-" * 70)
print(f"Total joints: {robot.num_joints}")
print(f"\nJoint names (in order):")

for i, name in enumerate(robot.joint_names):
    print(f"  [{i}] {name}")

print(f"\n" + "-" * 70)

# Get joint positions and limits
joint_pos = robot.data.joint_pos[0].cpu().numpy()
joint_vel = robot.data.joint_vel[0].cpu().numpy()

if hasattr(robot, 'data') and hasattr(robot.data, 'joint_limits'):
    joint_limits = robot.data.joint_limits
    print(f"\nJoint Positions, Velocities, and Limits:")
    print("-" * 70)
    for i, name in enumerate(robot.joint_names):
        pos = joint_pos[i]
        vel = joint_vel[i]
        if joint_limits is not None:
            lower = joint_limits[i, 0].item() if i < len(joint_limits) else "N/A"
            upper = joint_limits[i, 1].item() if i < len(joint_limits) else "N/A"
            print(f"  [{i}] {name:20s} | pos={pos:8.4f} | vel={vel:8.4f} | limits=[{lower:8.2f}, {upper:8.2f}]")
        else:
            print(f"  [{i}] {name:20s} | pos={pos:8.4f} | vel={vel:8.4f}")

print("\n" + "="*70)
print("JOINT CATEGORIZATION")
print("="*70)

# Identify base vs arm joints
base_joint_names = ["joint_x", "joint_y", "joint_theta"]
arm_joint_names = [f"left_arm_joint{i}" for i in range(1, 7)]

base_indices = []
arm_indices = []

for name in base_joint_names:
    if name in robot.joint_names:
        idx = robot.joint_names.index(name)
        base_indices.append(idx)
        print(f"Base joint: [{idx}] {name}")
    else:
        print(f"⚠️  Base joint NOT FOUND: {name}")

print()

for name in arm_joint_names:
    if name in robot.joint_names:
        idx = robot.joint_names.index(name)
        arm_indices.append(idx)
        print(f"Arm joint:  [{idx}] {name}")
    else:
        print(f"⚠️  Arm joint NOT FOUND: {name}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Base joint indices: {base_indices}")
print(f"Arm joint indices:  {arm_indices}")
print()

# Verify expected order
expected_base = [0, 1, 2]  # joint_x, joint_y, joint_theta
expected_arm = [3, 4, 5, 6, 7, 8]  # left_arm_joint1-6

if base_indices == expected_base:
    print("✅ Base joints are at indices 0-2 (CORRECT)")
else:
    print(f"⚠️  Base joints NOT at expected indices!")
    print(f"   Expected: {expected_base}")
    print(f"   Actual:   {base_indices}")

if arm_indices == expected_arm:
    print("✅ Arm joints are at indices 3-8 (CORRECT)")
else:
    print(f"⚠️  Arm joints NOT at expected indices!")
    print(f"   Expected: {expected_arm}")
    print(f"   Actual:   {arm_indices}")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

if base_indices == expected_base and arm_indices == expected_arm:
    print("✅ Joint ordering is correct!")
    print()
    print("Code should use:")
    print("  - base_joints = joint_pos[:, 0:3]  # [joint_x, joint_y, joint_theta]")
    print("  - arm_joints  = joint_pos[:, 3:9]  # [left_arm_joint1-6]")
else:
    print("⚠️  Joint ordering is NOT as expected!")
    print()
    print("Code should use dynamic lookup:")
    print("  - Find indices by name using robot.joint_names.index(name)")
    print("  - Store indices in _base_joint_ids and _arm_joint_ids")

env.close()
simulation_app.close()
