#!/usr/bin/env python3
"""Test MobileMMTrackEE-v0 environment with Isaac Lab 2.2.0.

This test follows Isaac Lab's app launcher pattern for proper initialization.
"""

import argparse
import sys
import os

#!/usr/bin/env python3
"""Test MobileMMTrackEE-v0 environment with Isaac Lab 2.2.0.

This test runs natively on Windows with full GPU support.
No WSL-specific workarounds needed!
"""

import argparse
import sys
import os

# Add project root to path (must be done before importing project modules)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Accept EULA
os.environ["ACCEPT_EULA"] = "YES"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"

# CRITICAL: Prevent ale_py crash by disabling Gymnasium auto-registration
# This must be set BEFORE importing gymnasium/Isaac Sim
os.environ["GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS"] = "1"
print("[DEBUG] Disabled Gymnasium plugin entrypoints to prevent ale_py crash")

# ============================================================================
# GPU Auto-detection for Windows
# ============================================================================
# System has 2 GPUs:
#   GPU 0: RTX 3090 (CUDA capability 8.6 - Training GPU)
#   GPU 1: Quadro P2000 (CUDA capability 6.1 - Display GPU, not supported by PyTorch 2.7+)
#
# Note: Windows GPU enumeration may differ from WSL2!
# Auto-detection strategy:
#   1. Let PyTorch detect available GPUs
#   2. Select GPU with highest compute capability >= 7.0
#   3. On Windows, Vulkan and Warp work natively - no special setup needed!
# ============================================================================

def get_best_gpu_device():
    """Automatically detect the best GPU device for training.
    
    Returns:
        str: Device string like "cuda:0" or "cuda:1"
    """
    try:
        import torch
        if not torch.cuda.is_available():
            print("    ⚠️  No CUDA devices available, using CPU")
            return "cpu"
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            return "cuda:0"
        
        # Multiple GPUs: find the one with highest compute capability
        best_device = 0
        best_compute_cap = 0.0
        
        for i in range(num_gpus):
            compute_cap = torch.cuda.get_device_capability(i)
            compute_cap_value = compute_cap[0] + compute_cap[1] * 0.1
            device_name = torch.cuda.get_device_name(i)
            
            print(f"    GPU {i}: {device_name} (compute {compute_cap[0]}.{compute_cap[1]})")
            
            # Only consider GPUs with compute capability >= 7.0 (Volta+)
            if compute_cap_value >= 7.0 and compute_cap_value > best_compute_cap:
                best_compute_cap = compute_cap_value
                best_device = i
        
        device_str = f"cuda:{best_device}"
        print(f"    ✓ Selected {torch.cuda.get_device_name(best_device)} as {device_str}")
        return device_str
        
    except ImportError:
        # Torch not yet imported (before Isaac Lab initialization)
        # Fallback: assume device 0 is the best (Windows native enumeration)
        return "cuda:0"

# ============================================================================
# GPU Auto-detection for Windows/WSL
# ============================================================================
# System has 2 GPUs:
#   GPU 0: Quadro P2000 (CUDA capability 6.1 - NOT supported by PyTorch 2.7+)
#   GPU 1: RTX 3090 (CUDA capability 8.6 - SUPPORTED)
#
# Auto-detection strategy:
#   1. Detect available GPUs with compute capability >= 7.0
#   2. Select RTX 3090 automatically (highest compute capability)
#   3. Works on both Windows and WSL without hardcoding device IDs
# ============================================================================

def get_best_gpu_device():
    """Automatically detect the best GPU device for training.
    
    Returns:
        str: Device string like "cuda:1" or "cuda" if only one suitable GPU
    """
    try:
        import torch
        if not torch.cuda.is_available():
            print("    ⚠️  No CUDA devices available, using CPU")
            return "cpu"
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            return "cuda:0"
        
        # Multiple GPUs: find the one with highest compute capability
        best_device = 0
        best_compute_cap = 0.0
        
        for i in range(num_gpus):
            compute_cap = torch.cuda.get_device_capability(i)
            compute_cap_value = compute_cap[0] + compute_cap[1] * 0.1
            device_name = torch.cuda.get_device_name(i)
            
            print(f"    GPU {i}: {device_name} (compute {compute_cap[0]}.{compute_cap[1]})")
            
            # Only consider GPUs with compute capability >= 7.0 (Volta+)
            if compute_cap_value >= 7.0 and compute_cap_value > best_compute_cap:
                best_compute_cap = compute_cap_value
                best_device = i
        
        device_str = f"cuda:{best_device}"
        print(f"    ✓ Selected {torch.cuda.get_device_name(best_device)} as {device_str}")
        return device_str
        
    except ImportError:
        # Torch not yet imported (before Isaac Lab initialization)
        return "cuda:1"  # Fallback to device 1 (RTX 3090)

# Accept EULA
os.environ["ACCEPT_EULA"] = "YES"
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
        from isaaclab.app import AppLauncher
        
        # Auto-detect best GPU device
        print("    Detecting GPU configuration...")
        gpu_device = get_best_gpu_device()
        
        # Create launcher args
        # Note: ale_py may crash during initialization but Isaac Sim will continue
        print("    Note: If you see an ale_py crash, it's non-fatal and can be ignored")
        app_launcher = AppLauncher(
            headless=args_cli.headless,
            enable_cameras=False,  # No camera rendering needed
            device=gpu_device,  # Auto-selected GPU (RTX 3090)
        )
        simulation_app = app_launcher.app
        
        print("    ✓ Isaac Lab initialized")
    except SystemExit as e:
        # ale_py crash causes SystemExit, but Isaac Sim may have loaded successfully
        # Check if we can continue
        if e.code == 1:
            print("    ⚠️  ale_py crash detected (non-fatal), attempting to continue...")
            # The app may have been created before the crash
            try:
                simulation_app = app_launcher.app
                print("    ✓ Isaac Lab initialized despite ale_py crash")
            except:
                print("    ✗ Failed to initialize Isaac Lab")
                return 1
        else:
            raise
    except Exception as e:
        print(f"    ✗ Failed to initialize Isaac Lab: {e}")
        import traceback
        traceback.print_exc()
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
        # Ensure project paths are in sys.path
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        
        src_path = os.path.join(PROJECT_ROOT, "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            print(f"    Added to sys.path: {src_path}")
        
        # Now import and register
        from task_spec import register_isaac_lab_tasks
        register_isaac_lab_tasks()
        
        if 'MobileMMTrackEE-v0' in gym.envs.registry:
            print("    ✓ Task registered successfully")
        else:
            print("    ✗ Task not found in registry")
            print(f"    Available tasks: {[k for k in gym.envs.registry.keys() if 'Mobile' in k]}")
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
        
        # Enable debug visualization to show trajectory markers
        if not args_cli.headless and hasattr(env.unwrapped, 'set_debug_vis'):
            env.unwrapped.set_debug_vis(True)
            print("    ✓ Debug visualization enabled (trajectory markers will be visible)")
        
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
        # Get action dimension - handle both wrapped and unwrapped cases
        if hasattr(env.action_space, 'shape'):
            action_dim = env.action_space.shape[-1] if len(env.action_space.shape) > 1 else env.action_space.shape[0]
        else:
            action_dim = 8  # Fallback: 6 arm + 2 base
        
        for i in range(args_cli.steps):
            actions = torch.zeros((args_cli.num_envs, action_dim), device=env.unwrapped.device)
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
