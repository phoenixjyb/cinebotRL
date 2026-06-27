#!/usr/bin/env python3
"""Smoke-test the Proto2-backed MobileMMTrackEE Isaac Lab environment.

This follows Isaac Lab's app launcher pattern and works with either the legacy
MobileMMTrackEE-v0 task ID or the RecomoProto2TrackEE-v0 alias.
"""

import argparse
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
# GPU auto-detection
# ============================================================================
# Select the highest-compute CUDA device >= 7.0. This avoids hardcoding the
# Windows/WSL GPU enumeration order on the .98 machine.

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
        return "cuda:0"

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
    parser.add_argument("--task", type=str, default="RecomoProto2TrackEE-v0", help="Gym task ID")
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
    print(f"[3/8] Registering {args_cli.task} task...")
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

        if args_cli.task in gym.envs.registry:
            print("    ✓ Task registered successfully")
        else:
            print("    ✗ Task not found in registry")
            print(f"    Available tasks: {[k for k in gym.envs.registry.keys() if 'TrackEE' in k]}")
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
            args_cli.task,
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
        if env.action_space.shape[-1] != 9:
            raise AssertionError(f"Expected 9-action Proto2 v3 policy, got {env.action_space.shape}")
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
            required_bodies = ["base_link", "arm_link_1", "arm_link_2", "arm_link_3",
                               "arm_link_4", "arm_link_5", "arm_link_6", "cam_link"]
            missing_bodies = [name for name in required_bodies if name not in robot.body_names]
            if missing_bodies:
                raise AssertionError(f"Missing Proto2 bodies: {missing_bodies}")
            print("    ✓ Required Proto2 bodies present")
    except Exception as e:
        print(f"    ✗ Robot structure check failed: {e}")
        env.close()
        simulation_app.close()
        return 1

    # Execute steps
    print(f"[7/8] Testing environment stepping ({args_cli.steps} steps)...")
    try:
        # Get action dimension - handle both wrapped and unwrapped cases
        if hasattr(env.action_space, 'shape'):
            action_dim = env.action_space.shape[-1] if len(env.action_space.shape) > 1 else env.action_space.shape[0]
        else:
            action_dim = 9  # Fallback: 6 arm + 3 base

        for i in range(args_cli.steps):
            actions = torch.zeros((args_cli.num_envs, action_dim), device=env.unwrapped.device)
            obs, reward, terminated, truncated, info = env.step(actions)

            if i == 0:
                print(f"    Step 0: reward={reward[0].item():.4f}")

        print(f"    ✓ Completed {args_cli.steps} steps")
        print(f"    Final reward: {reward[0].item():.4f}")

        from rl_platform.tasks.mobile_mm.joint_names import PASSIVE_JOINT_NAMES

        robot = env.unwrapped.robot
        passive_ids = [robot.joint_names.index(name) for name in PASSIVE_JOINT_NAMES]
        passive_pos = robot.data.joint_pos[:, passive_ids]
        max_passive_abs = passive_pos.abs().max().item()
        print(f"    Passive joint max abs position: {max_passive_abs:.6f}")
        if max_passive_abs > 1e-3:
            raise AssertionError(
                f"Passive Proto2 joints drifted from zero: {PASSIVE_JOINT_NAMES} max={max_passive_abs}"
            )
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
    print("         --task RecomoProto2TrackEE-v0 --num_envs 16 --total_timesteps 10000")
    print()
    print("  3. Full training:")
    print("     python scripts/reinforcement_learning/sb3/train.py \\")
    print("         --task RecomoProto2TrackEE-v0 --num_envs 1024 --headless")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
