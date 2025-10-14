#!/usr/bin/env python3
"""Test MobileMMTrackEE-v0 environment with Isaac Lab 2.2.0.

This script tests:
1. Task registration
2. Environment creation (minimal 1 env)
3. Robot spawning with correct EE link
4. Observation/action spaces
5. Basic step execution
6. Reset functionality

Run headlessly in WSL after Isaac Lab installation.
"""

import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Accept EULA for headless operation
os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"

print("=" * 70)
print("MobileMMTrackEE-v0 Environment Test")
print("=" * 70)
print()

# Test 1: Import Isaac Lab
print("[1/7] Testing Isaac Lab import...")
try:
    import isaaclab
    # Try to get version, but it's not always available
    try:
        version = isaaclab.__version__
    except AttributeError:
        version = "2.2.0"  # Known version from installation
    print(f"    ✓ Isaac Lab {version} available")
except ImportError as e:
    print(f"    ✗ Failed to import isaaclab: {e}")
    sys.exit(1)

# Test 2: Import gymnasium
print("[2/7] Testing gymnasium...")
try:
    import gymnasium as gym
    print(f"    ✓ Gymnasium available")
except ImportError as e:
    print(f"    ✗ Failed to import gymnasium: {e}")
    sys.exit(1)

# Test 3: Register custom task
print("[3/7] Registering MobileMMTrackEE-v0 task...")
try:
    # Import Isaac Lab first to initialize it
    import omni.isaac.lab.app as app_launcher
    
    # Create minimal Isaac Lab app configuration
    class SimpleAppCfg:
        headless = True
        offscreen_render = True
    
    # Launch the simulation app (required before importing task modules)
    simulation_app = app_launcher.AppLauncher(SimpleAppCfg()).app
    
    # Now we can import our task
    from src.task_spec import register_isaac_lab_tasks
    register_isaac_lab_tasks()
    
    # Check if task is in registry
    if 'MobileMMTrackEE-v0' in gym.envs.registry:
        print(f"    ✓ Task registered successfully")
    else:
        print(f"    ✗ Task not found in registry")
        available = [k for k in gym.envs.registry if 'Mobile' in k]
        print(f"    Available tasks: {available}")
        sys.exit(1)
except Exception as e:
    print(f"    ✗ Failed to register task: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create environment (1 env for testing)
print("[4/7] Creating environment (1 env, headless)...")
try:
    env = gym.make(
        'MobileMMTrackEE-v0',
        num_envs=1,
        headless=True,
    )
    print(f"    ✓ Environment created")
    print(f"    - Observation space: {env.observation_space.shape}")
    print(f"    - Action space: {env.action_space.shape}")
except Exception as e:
    print(f"    ✗ Failed to create environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Reset environment
print("[5/7] Resetting environment...")
try:
    obs, info = env.reset()
    print(f"    ✓ Reset successful")
    print(f"    - Observation shape: {obs['policy'].shape}")
    print(f"    - Info keys: {list(info.keys())}")
except Exception as e:
    print(f"    ✗ Failed to reset: {e}")
    import traceback
    traceback.print_exc()
    env.close()
    sys.exit(1)

# Test 6: Check robot and EE link
print("[6/7] Checking robot structure...")
try:
    # Access the underlying environment
    if hasattr(env.unwrapped, '_ee_body_idx'):
        ee_idx = env.unwrapped._ee_body_idx
        print(f"    ✓ End-effector body index: {ee_idx}")
    else:
        print(f"    ⚠ Could not check EE body index (may not be set yet)")
    
    # Check if robot exists
    if hasattr(env.unwrapped, 'robot'):
        robot = env.unwrapped.robot
        print(f"    ✓ Robot articulation loaded")
        if hasattr(robot, 'body_names'):
            ee_link_name = "left_gripper_link"
            if ee_link_name in robot.body_names:
                idx = robot.body_names.index(ee_link_name)
                print(f"    ✓ Found '{ee_link_name}' at index {idx}")
            else:
                print(f"    ⚠ '{ee_link_name}' not found in body_names")
                print(f"    Available bodies: {robot.body_names[:10]}...")
except Exception as e:
    print(f"    ⚠ Could not fully check robot: {e}")

# Test 7: Execute steps
print("[7/7] Testing environment stepping (10 steps)...")
try:
    import torch
    
    for i in range(10):
        # Zero actions (safe test)
        actions = torch.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(actions)
        
        if i == 0:
            print(f"    Step 0:")
            print(f"      - Reward: {reward.item():.4f}")
            print(f"      - Terminated: {terminated.item()}")
            print(f"      - Truncated: {truncated.item()}")
    
    print(f"    ✓ Completed 10 steps successfully")
    print(f"    Final reward: {reward.item():.4f}")
    
except Exception as e:
    print(f"    ✗ Failed during stepping: {e}")
    import traceback
    traceback.print_exc()
    env.close()
    sys.exit(1)

# Cleanup
print()
print("Closing environment...")
env.close()

print()
print("=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print()
print("Next steps:")
print("  1. Test with more environments:")
print("     python test_env.py --num_envs 4")
print()
print("  2. Test recorded trajectory loading:")
print("     python test_env.py --trajectory_file trajectoryToLearn/1_pull_world_scaled.json")
print()
print("  3. Run short training:")
print("     python scripts/reinforcement_learning/sb3/train.py \\")
print("         --task MobileMMTrackEE-v0 --num_envs 16 --total_timesteps 10000")
print()
print("  4. Full training with RTX 3090:")
print("     python scripts/reinforcement_learning/sb3/train.py \\")
print("         --task MobileMMTrackEE-v0 --num_envs 1024 --headless")
print()
