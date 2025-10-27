#!/usr/bin/env python3
"""Quick validation test for base movement control fix.

Tests that the base actually moves when commanded after fixing the dual control bug.

Expected behavior after fix:
- Base displacement: ~5.0m for 10 seconds at 0.5 m/s
- Root velocity: ~0.5 m/s during motion
- PPR joints: Stay near zero (< 0.01m drift)

Run:
    I:\\isaaclab\\isaaclab.bat -p scripts/test_base_movement_fix.py --headless
"""

import sys
from pathlib import Path

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent  # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent  # project root
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import argparse
from isaaclab.app import AppLauncher

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Quick base movement validation test")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
parser.add_argument("--headless", action="store_true", help="Run headless")
args_cli = parser.parse_args()

# Initialize app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import Isaac Lab modules and custom task
from task_spec import register_isaac_lab_tasks
import gymnasium as gym

# Register tasks
register_isaac_lab_tasks()


def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from quaternion (same as env.py)."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))


def main():
    """Run base movement validation test."""
    
    # Create environment with disabled termination for testing
    env = gym.make(
        "MobileMMTrackEE-v0",
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        headless=args_cli.headless,
    )
    
    # Disable collision termination for test (base-ground contact is normal!)
    env.unwrapped.cfg.terminations = {}  # Disable all terminations
    
    print("\n" + "="*80)
    print("Base Movement Validation Test")
    print("="*80)
    print(f"Device: {args_cli.device}")
    print(f"Num envs: {args_cli.num_envs}")
    print(f"Test duration: 10 seconds")
    print(f"Command: 0.5 m/s forward (normalized: {0.5/1.5:.3f})")
    print(f"⚠️  Terminations disabled for test")
    print("="*80 + "\n")
    
    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]  # Extract policy observation tensor
    
    # Record initial state (USE ROOT STATE, NOT JOINT STATE!)
    robot = env.unwrapped.scene["robot"]
    initial_root_pos = robot.data.root_pos_w.clone()
    initial_root_quat = robot.data.root_quat_w.clone()
    initial_joint_pos = robot.data.joint_pos[:, 0:3].clone()  # PPR joints for monitoring
    
    print("Initial state:")
    print(f"  Root position (world): {initial_root_pos[0, :3]}")
    print(f"  Root orientation (yaw): {quat_to_yaw(initial_root_quat[0:1]).item():.3f} rad")
    print(f"  PPR joint positions: {initial_joint_pos[0]}")
    print()
    
    # Run for 10 seconds with constant forward command
    dt = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation  # ~0.1s per step
    num_steps = int(10.0 / dt)
    
    # Action space: [arm_joints (0-5), base_vx (6), base_wz (7)]
    # Base actions in indices 6 and 7, normalized to [-1, 1]
    # Scaled by max_linear_velocity (1.5 m/s) and max_angular_velocity (1.0 rad/s)
    # To get 0.5 m/s forward: 0.5 / 1.5 = 0.333 normalized
    actions = torch.zeros(args_cli.num_envs, 8, device=args_cli.device)
    actions[:, 6] = 0.5 / 1.5  # Index 6: Normalized forward velocity → 0.5 m/s
    actions[:, 7] = 0.0         # Index 7: No rotation
    # Arm joints (0-5) stay at current positions (zeros are fine)
    
    print(f"Running {num_steps} steps at {dt:.3f}s per step...")
    print(f"  Action indices: arm=[0-5], base_vx=[6], base_wz=[7]")
    print(f"  Commanding: base_vx={actions[0, 6].item():.3f} (normalized) → 0.5 m/s\n")
    
    for step in range(num_steps):
        obs_dict, reward, terminated, truncated, info = env.step(actions)
        obs = obs_dict["policy"]  # Extract policy observation
        
        # Print progress every 2 seconds
        if (step + 1) % int(2.0 / dt) == 0:
            current_root_pos = robot.data.root_pos_w
            displacement = current_root_pos - initial_root_pos
            current_vel = robot.data.root_lin_vel_w
            print(f"  Step {step+1}/{num_steps}: displacement = {displacement[0, :2]}, velocity = {current_vel[0, :2]}")
    
    # Record final state
    final_root_pos = robot.data.root_pos_w.clone()
    final_root_quat = robot.data.root_quat_w.clone()
    final_joint_pos = robot.data.joint_pos[:, 0:3].clone()
    final_vel = robot.data.root_lin_vel_w.clone()
    
    # Calculate metrics (USE ROOT STATE!)
    displacement = final_root_pos - initial_root_pos
    mean_displacement_x = displacement[:, 0].mean().item()
    mean_displacement_y = displacement[:, 1].mean().item()
    mean_displacement_z = displacement[:, 2].mean().item()
    mean_velocity_x = final_vel[:, 0].mean().item()
    mean_velocity_y = final_vel[:, 1].mean().item()
    
    # PPR joint drift
    joint_drift = (final_joint_pos - initial_joint_pos).abs()
    max_joint_drift = joint_drift.max().item()
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Base displacement (from root_pos_w):")
    print(f"  X: {mean_displacement_x:.3f} m")
    print(f"  Y: {mean_displacement_y:.3f} m")
    print(f"  Z: {mean_displacement_z:.3f} m")
    print(f"Final velocity (from root_lin_vel_w):")
    print(f"  X: {mean_velocity_x:.3f} m/s")
    print(f"  Y: {mean_velocity_y:.3f} m/s")
    print(f"Max PPR joint drift: {max_joint_drift:.5f} m")
    print("="*80 + "\n")
    
    # Validation checks
    print("VALIDATION:")
    checks = []
    
    # Check 1: Forward displacement should be ~5.0m (0.5 m/s * 10s)
    expected_displacement = 5.0
    tolerance = 0.5
    check1 = abs(mean_displacement_x - expected_displacement) < tolerance
    checks.append(check1)
    print(f"  1. Forward displacement: {mean_displacement_x:.3f}m (expected {expected_displacement}±{tolerance}m) - {'✅ PASS' if check1 else '❌ FAIL'}")
    
    # Check 2: Velocity should be ~0.5 m/s
    expected_vel = 0.5
    vel_tolerance = 0.1
    check2 = abs(mean_velocity_x - expected_vel) < vel_tolerance
    checks.append(check2)
    print(f"  2. Forward velocity: {mean_velocity_x:.3f}m/s (expected {expected_vel}±{vel_tolerance}m/s) - {'✅ PASS' if check2 else '❌ FAIL'}")
    
    # Check 3: PPR joints should stay near zero (< 1cm drift)
    max_drift_threshold = 0.01
    check3 = max_joint_drift < max_drift_threshold
    checks.append(check3)
    print(f"  3. PPR joint drift: {max_joint_drift:.5f}m (threshold {max_drift_threshold}m) - {'✅ PASS' if check3 else '❌ FAIL'}")
    
    # Check 4: Lateral drift should be small
    lateral_threshold = 0.5
    check4 = abs(mean_displacement_y) < lateral_threshold
    checks.append(check4)
    print(f"  4. Lateral drift: {abs(mean_displacement_y):.3f}m (threshold {lateral_threshold}m) - {'✅ PASS' if check4 else '❌ FAIL'}")
    
    # Check 5: Vertical drift should be minimal
    vertical_threshold = 0.1
    check5 = abs(mean_displacement_z) < vertical_threshold
    checks.append(check5)
    print(f"  5. Vertical drift: {abs(mean_displacement_z):.3f}m (threshold {vertical_threshold}m) - {'✅ PASS' if check5 else '❌ FAIL'}")
    
    # Overall result
    all_passed = all(checks)
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Base movement fix is working correctly!")
    else:
        print("❌ SOME CHECKS FAILED - Base movement fix may have issues!")
    print("="*80 + "\n")
    
    env.close()
    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    
    exit(exit_code)
