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
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    import gymnasium as gym
    
    # Register tasks
    print(f"\nRegistering task: {args.task}")
    from task_spec import register_isaac_lab_tasks
    register_isaac_lab_tasks()
    print("✓ Task registered")
    
    # Create environment
    print(f"\nCreating environment with {args.num_envs} parallel instances...")
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    print("✓ Environment created")
    
    # Reset environment
    print("\nResetting environment...")
    obs, _ = env.reset()
    print("✓ Environment reset complete")
    
    # Get the unwrapped environment to access internal state
    base_env = env.unwrapped
    
    # Print available contact-related attributes
    print("\n1. Available Contact Force Arrays:")
    print("-" * 70)
    
    contact_attrs = []
    for attr_name in dir(base_env):
        if 'contact' in attr_name.lower() or 'force' in attr_name.lower():
            if not attr_name.startswith('_') and not callable(getattr(base_env, attr_name)):
                contact_attrs.append(attr_name)
    
    if contact_attrs:
        for attr in contact_attrs:
            attr_val = getattr(base_env, attr)
            if isinstance(attr_val, torch.Tensor):
                print(f"  {attr:30s}: shape={attr_val.shape}, dtype={attr_val.dtype}")
            else:
                print(f"  {attr:30s}: {type(attr_val)}")
    else:
        print("  No contact/force attributes found at top level")
    
    # Try to access PhysX contact sensor directly
    print("\n2. PhysX Contact Sensor:")
    print("-" * 70)
    
    if hasattr(base_env, '_contact_sensor'):
        sensor = base_env._contact_sensor
        print(f"  Contact sensor found: {type(sensor)}")
        
        # Print sensor data attributes
        if hasattr(sensor, 'data'):
            print("\n  Sensor data attributes:")
            for attr_name in dir(sensor.data):
                if not attr_name.startswith('_'):
                    try:
                        attr_val = getattr(sensor.data, attr_name)
                        if isinstance(attr_val, torch.Tensor):
                            print(f"    {attr_name:30s}: shape={attr_val.shape}")
                    except:
                        pass
    else:
        print("  No _contact_sensor attribute found")
    
    # Try to access robot articulation
    print("\n3. Robot Articulation:")
    print("-" * 70)
    
    if hasattr(base_env, '_robot'):
        robot = base_env._robot
        print(f"  Robot found: {type(robot)}")
        
        # Check for contact-related methods
        if hasattr(robot, 'root_physx_view'):
            print("  Has root_physx_view")
            physx_view = robot.root_physx_view
            
            # Print available contact arrays
            print("\n  PhysX View Contact Arrays:")
            for attr_name in dir(physx_view):
                if 'contact' in attr_name.lower() or 'force' in attr_name.lower():
                    if not attr_name.startswith('_') and not callable(getattr(physx_view, attr_name)):
                        print(f"    {attr_name}")
    
    # Run simulation and monitor contact forces
    print("\n4. Contact Force Values During Simulation:")
    print("-" * 70)
    print("\nRunning simulation for", args.steps, "steps...")
    print("Looking for non-zero contact forces...\n")
    
    max_forces_seen = {}
    
    for step in range(args.steps):
        # Random actions to potentially cause collisions
        action = torch.randn(args.num_envs, env.action_space.shape[0], device="cuda:0") * 0.5
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check contact forces from the environment's internal state
        if hasattr(base_env, '_contact_forces'):
            forces = base_env._contact_forces
            max_force = forces.abs().max().item()
            
            if step not in max_forces_seen:
                max_forces_seen[step] = max_force
            
            if max_force > 0.01:  # Non-trivial force
                print(f"  Step {step:4d}: max contact force = {max_force:8.3f} N (shape={forces.shape})")
        
        # Also check if contact sensor exists
        if hasattr(base_env, '_contact_sensor'):
            sensor = base_env._contact_sensor
            if hasattr(sensor, 'data'):
                # Try different potential arrays
                for attr_name in ['net_forces_w', 'force_matrix_w', 'forces_w']:
                    if hasattr(sensor.data, attr_name):
                        forces = getattr(sensor.data, attr_name)
                        max_force = forces.abs().max().item()
                        
                        if max_force > 0.01:
                            print(f"  Step {step:4d}: sensor.data.{attr_name} max = {max_force:8.3f} N")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if max_forces_seen:
        overall_max = max(max_forces_seen.values())
        print(f"\nMax contact force seen: {overall_max:.3f} N")
        
        if overall_max < 0.01:
            print("\n❌ CRITICAL BUG CONFIRMED!")
            print("   Contact forces are reading ~0.0 N throughout simulation")
            print("   This means self_collision_penalty is always 0.0")
            print("\n   POSSIBLE CAUSES:")
            print("   1. Using wrong PhysX array (net_forces vs forces_w)")
            print("   2. Wrong body index (chassis_body_idx mismatch)")
            print("   3. Contact sensor not configured correctly")
            print("   4. PhysX collision detection disabled")
            print("\n   FIX: Check sensor.data for available arrays above")
        else:
            print(f"\n✅ Contact forces ARE being recorded!")
            print(f"   Max force: {overall_max:.3f} N")
            print("   Check if env is using the correct array")
    else:
        print("\n⚠️  Could not access contact forces during simulation")
        print("   Check PhysX sensor configuration")
    
    # Print body indices for debugging
    print("\n5. Body Indices (for reference):")
    print("-" * 70)
    
    if hasattr(base_env, '_chassis_body_idx'):
        print(f"  chassis_body_idx = {base_env._chassis_body_idx}")
    
    if hasattr(base_env, '_robot'):
        robot = base_env._robot
        if hasattr(robot, 'body_names'):
            print(f"\n  Robot body names ({len(robot.body_names)}):")
            for i, name in enumerate(robot.body_names):
                print(f"    [{i:2d}] {name}")
    
    print("\n" + "="*70)
    
    # Cleanup
    env.close()
    simulation_app.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
