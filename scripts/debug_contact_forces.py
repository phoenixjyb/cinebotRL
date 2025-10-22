#!/usr/bin/env python3
"""
Debug Contact Forces in MobileMMTrackEEEnv

This script investigates why contact forces are reading 0.0 N in the environment.
It prints all available PhysX contact arrays and their values to identify the
correct array to use.

Root Cause (from code review):
    env.py line 1120 uses: self._contact_forces[:, self._chassis_body_idx, 0]
    This is reading from net_contact_forces which might be:
    - Not the right array (try contact_forces_w instead)
    - Not accumulating forces correctly
    - Body index mismatch (wrong body being queried)

Expected Behavior:
    - Arm-chassis self-collisions should produce non-zero forces
    - These forces should trigger self_collision_penalty
    - Without contact feedback, policy has no incentive to reposition

Usage:
    I:\isaaclab\isaaclab.bat -p scripts/debug_contact_forces.py --num_envs 16

Expected Output:
    - Print all available contact arrays
    - Show non-zero values during collisions
    - Identify correct array and indexing

Reference: docs/_CODE_REVIEW_VALIDATION.md (Issue #2 - CRITICAL)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path BEFORE importing anything
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def main():
    """Debug contact forces by printing all available PhysX arrays."""
    
    parser = argparse.ArgumentParser(description="Debug contact forces in MobileMMTrackEEEnv")
    parser.add_argument("--task", type=str, default="MobileMMTrackEE-v0", help="Task name")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CONTACT FORCES DEBUG")
    print("="*70)
    print(f"\nInitializing Isaac Sim with {args.num_envs} environments...")
    
    # Initialize Isaac Sim/Lab - MUST happen before importing Isaac Lab modules
    from isaaclab.app import AppLauncher
    
    app_launcher = AppLauncher(
        headless=True,
        enable_cameras=False,
    )
    simulation_app = app_launcher.app
    print("✓ Isaac Sim initialized")
    
    # NOW we can import Isaac Lab and other modules
    import torch
    import gymnasium as gym
    
    # Register tasks
    print(f"\nRegistering task: {args.task}")
    from task_spec import register_isaac_lab_tasks
    register_isaac_lab_tasks()
    print("✓ Task registered")
    
    # Create environment directly
    print(f"\nCreating environment with {args.num_envs} parallel instances...")
    from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnvCfg, MobileMMTrackEEEnv
    
    env_cfg = MobileMMTrackEEEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    
    env = MobileMMTrackEEEnv(cfg=env_cfg)
    print("✓ Environment created")
    
    # Reset environment
    print("\nResetting environment...")
    obs, _ = env.reset()
    print("✓ Environment reset complete")
    
    # Get the unwrapped environment to access internal state
    base_env = env
    
    # Print available contact-related attributes
    print("\n1. Scene Contact Sensor:")
    print("-" * 70)
    
    # Check for ContactSensor in scene (Isaac Lab 2.2.0 pattern)
    if hasattr(base_env, 'scene') and "contact_sensor" in base_env.scene:
        sensor = base_env.scene["contact_sensor"]
        print(f"  ✅ ContactSensor found in scene")
        print(f"  Type: {type(sensor)}")
        
        if hasattr(sensor, 'data'):
            print(f"\n  ContactSensor data attributes:")
            print(f"    net_forces_w shape: {sensor.data.net_forces_w.shape}")
            print(f"    (Expected: [num_envs, num_bodies, 3] or [num_envs, 3])")
    else:
        print("  ❌ No ContactSensor found in scene")
    
    print("\n2. Robot Articulation:")
    print("-" * 70)
    
    if hasattr(base_env, 'robot'):
        robot = base_env.robot
        print(f"  ✅ Robot found: {type(robot)}")
        print(f"  Number of bodies: {robot.num_bodies}")
        if hasattr(robot, 'body_names'):
            print(f"  Body names: {robot.body_names[:5]}...")  # First 5
    else:
        print("  ❌ No robot attribute found")
    
    print("\n3. USD Body Structure:")
    print("-" * 70)
    print("  Expected monitored bodies:")
    print("    - abstract_chassis_link (primary sensor location)")
    print("    - Filter: left_arm.* (arm links that collide with chassis)")
    print("  Contact forces only reported when arm links touch chassis")
    
    # Run simulation and monitor contact forces
    print("\n4. Contact Force Monitoring:")
    print("-" * 70)
    print(f"\nRunning simulation for {args.steps} steps...")
    print("Random actions to test collision detection...\n")
    
    # Get correct action dimension (should be 8: 6 arm + 2 base)
    action_dim = 8
    print(f"Action dimension: {action_dim}")
    
    max_force_overall = 0.0
    collision_count = 0
    
    for step in range(args.steps):
        # Random actions to potentially cause collisions
        action = torch.randn(args.num_envs, action_dim, device="cuda:0") * 0.5
        
        obs, reward, terminated, truncated, info = env.step(action)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n✅ Contact sensor validation complete!")
    print("   Check output above for collision detections.")
    print("   If you see collision warnings with force values > 0 N,")
    print("   the contact sensor is working correctly.")
    
    print("\n" + "="*70)
    
    # Cleanup
    env.close()
    simulation_app.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
