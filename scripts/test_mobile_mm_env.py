#!/usr/bin/env python3
"""Test MobileMMTrackEE-v0 environment with Isaac Lab 2.2.0.

This test follows Isaac Lab's app launcher pattern for proper initialization.
"""

import argparse
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Accept EULA
os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"

def main():
    """Run environment test."""
    print("=" * 70)
    print("MobileMMTrackEE-v0 Environment Test (Isaac Lab 2.2.0)")
    print("=" * 70)
    print()
    
    # Parse arguments for Isaac Lab app launcher
    parser = argparse.ArgumentParser(description="Test custom MobileMMTrackEE environment")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--steps", type=int, default=10, help="Number of test steps")
    args_cli = parser.parse_args()
    
    # Import Isaac Lab app launcher
    print("[1/8] Initializing Isaac Lab...")
    try:
        from omni.isaac.lab.app import AppLauncher
        
        # Create launcher args
        app_launcher = AppLauncher(headless=args_cli.headless)
        simulation_app = app_launcher.app
        
        print("    ✓ Isaac Lab initialized")
    except Exception as e:
        print(f"    ✗ Failed to initialize Isaac Lab: {e}")
        return 1
    
    # Now import the rest (must be after app launcher)
    print("[2/8] Importing dependencies...")
    try:
        import gymnasium as gym
        import torch
        print("    ✓ Dependencies imported")
    except Exception as e:
        print(f"    ✗ Failed to import dependencies: {e}")
        simulation_app.close()
        return 1
    
    # Register task
    print("[3/8] Registering MobileMMTrackEE-v0 task...")
    try:
        from src.task_spec import register_isaac_lab_tasks
        register_isaac_lab_tasks()
        
        if 'MobileMMTrackEE-v0' in gym.envs.registry:
            print("    ✓ Task registered successfully")
        else:
            print("    ✗ Task not found in registry")
            simulation_app.close()
            return 1
    except Exception as e:
        print(f"    ✗ Failed to register task: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return 1
    
    # Create environment
    print(f"[4/8] Creating environment ({args_cli.num_envs} env(s))...")
    try:
        env = gym.make(
            'MobileMMTrackEE-v0',
            num_envs=args_cli.num_envs,
            headless=args_cli.headless,
        )
        print("    ✓ Environment created")
        print(f"    - Observation space: {env.observation_space.shape}")
        print(f"    - Action space: {env.action_space.shape}")
    except Exception as e:
        print(f"    ✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return 1
    
    # Reset
    print("[5/8] Resetting environment...")
    try:
        obs, info = env.reset()
        print("    ✓ Reset successful")
        print(f"    - Observation shape: {obs['policy'].shape}")
    except Exception as e:
        print(f"    ✗ Failed to reset: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        simulation_app.close()
        return 1
    
    # Check robot structure
    print("[6/8] Checking robot structure...")
    try:
        unwrapped_env = env.unwrapped
        if hasattr(unwrapped_env, '_ee_body_idx'):
            ee_idx = unwrapped_env._ee_body_idx
            print(f"    ✓ End-effector body index: {ee_idx}")
        
        if hasattr(unwrapped_env, 'robot'):
            robot = unwrapped_env.robot
            print(f"    ✓ Robot loaded")
            if hasattr(robot, 'body_names') and 'left_gripper_link' in robot.body_names:
                idx = robot.body_names.index('left_gripper_link')
                print(f"    ✓ Found 'left_gripper_link' at index {idx}")
    except Exception as e:
        print(f"    ⚠ Could not fully check robot: {e}")
    
    # Execute steps
    print(f"[7/8] Testing environment stepping ({args_cli.steps} steps)...")
    try:
        for i in range(args_cli.steps):
            actions = torch.zeros((args_cli.num_envs, env.action_space.shape[0]), device=env.unwrapped.device)
            obs, reward, terminated, truncated, info = env.step(actions)
            
            if i == 0:
                print(f"    Step 0: reward={reward[0].item():.4f}")
        
        print(f"    ✓ Completed {args_cli.steps} steps")
        print(f"    Final reward: {reward[0].item():.4f}")
    except Exception as e:
        print(f"    ✗ Failed during stepping: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        simulation_app.close()
        return 1
    
    # Cleanup
    print("[8/8] Cleanup...")
    env.close()
    simulation_app.close()
    print("    ✓ Cleanup complete")
    
    print()
    print("=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Test with more environments:")
    print(f"     python {__file__} --num_envs 4")
    print()
    print("  2. Run short training:")
    print("     python scripts/reinforcement_learning/sb3/train.py \\")
    print("         --task MobileMMTrackEE-v0 --num_envs 16 --total_timesteps 10000")
    print()
    print("  3. Full training:")
    print("     python scripts/reinforcement_learning/sb3/train.py \\")
    print("         --task MobileMMTrackEE-v0 --num_envs 1024 --headless")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
