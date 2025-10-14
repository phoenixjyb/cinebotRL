#!/usr/bin/env python3
"""Quick Isaac Lab installation verification for WSL.

This script checks:
1. Python version compatibility
2. PyTorch + CUDA availability
3. Isaac Lab import
4. Our custom task registration
5. Basic environment creation

Run this after installing Isaac Lab to verify the setup.
"""

import sys
from typing import List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is 3.10 or 3.11."""
    version = sys.version_info
    if version.major == 3 and version.minor in [10, 11]:
        return True, f"✓ Python {version.major}.{version.minor}.{version.micro}"
    return False, f"✗ Python {version.major}.{version.minor}.{version.micro} (need 3.10 or 3.11)"


def check_pytorch() -> Tuple[bool, str]:
    """Check PyTorch and CUDA availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            return True, f"✓ PyTorch {torch.__version__} with CUDA ({gpu_name})"
        return False, f"✗ PyTorch {torch.__version__} but CUDA not available"
    except ImportError:
        return False, "✗ PyTorch not installed"


def check_isaac_lab() -> Tuple[bool, str]:
    """Check Isaac Lab installation."""
    try:
        import omni.isaac.lab
        return True, f"✓ Isaac Lab {omni.isaac.lab.__version__}"
    except ImportError:
        try:
            # Try alternative import
            import isaaclab
            return True, f"✓ Isaac Lab (version detection failed)"
        except ImportError:
            return False, "✗ Isaac Lab not installed"


def check_task_registration() -> Tuple[bool, str]:
    """Check if our custom task can be registered."""
    try:
        import gymnasium as gym
        from src.task_spec import register_isaac_lab_tasks
        
        register_isaac_lab_tasks()
        
        # Check if task is in registry
        available_tasks = [k for k in gym.envs.registry if 'MobileMM' in k]
        if 'MobileMMTrackEE-v0' in available_tasks:
            return True, f"✓ Task registered: MobileMMTrackEE-v0"
        return False, f"✗ Task not found in registry (available: {available_tasks})"
    except Exception as e:
        return False, f"✗ Task registration failed: {str(e)}"


def check_env_creation() -> Tuple[bool, str]:
    """Try creating a single environment instance."""
    try:
        import gymnasium as gym
        import torch
        from src.task_spec import register_isaac_lab_tasks
        
        register_isaac_lab_tasks()
        
        # Try to create environment with minimal config
        env = gym.make('MobileMMTrackEE-v0', num_envs=1, headless=True)
        env.reset()
        
        # Try a single step
        actions = torch.zeros(env.action_space.shape)
        obs, reward, done, truncated, info = env.step(actions)
        
        env.close()
        return True, "✓ Environment creation successful"
    except Exception as e:
        return False, f"✗ Environment creation failed: {str(e)}"


def check_usd_assets() -> Tuple[bool, str]:
    """Check if USD robot assets are accessible."""
    from pathlib import Path
    
    usd_path = Path("/mnt/c/Users/yanbo/wSpace/cinebotRL/assets_own/usd/mobile_manipulator_PPR_base_corrected.usd")
    if usd_path.exists():
        size_mb = usd_path.stat().st_size / (1024 * 1024)
        return True, f"✓ USD asset found ({size_mb:.1f} MB)"
    return False, f"✗ USD asset not found at {usd_path}"


def check_dependencies() -> Tuple[bool, str]:
    """Check training dependencies."""
    missing = []
    
    try:
        import stable_baselines3
    except ImportError:
        missing.append("stable-baselines3")
    
    try:
        import gymnasium
    except ImportError:
        missing.append("gymnasium")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    if missing:
        return False, f"✗ Missing packages: {', '.join(missing)}"
    return True, "✓ Training dependencies installed"


def main():
    """Run all checks and print summary."""
    print("=" * 70)
    print("Isaac Lab WSL Installation Verification")
    print("=" * 70)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch + CUDA", check_pytorch),
        ("Isaac Lab", check_isaac_lab),
        ("USD Assets", check_usd_assets),
        ("Dependencies", check_dependencies),
        ("Task Registration", check_task_registration),
        ("Environment Creation", check_env_creation),
    ]
    
    results: List[Tuple[str, bool, str]] = []
    
    for name, check_func in checks:
        print(f"Checking {name}...", end=" ", flush=True)
        try:
            success, message = check_func()
            results.append((name, success, message))
            print(message)
        except Exception as e:
            results.append((name, False, f"✗ Error: {str(e)}"))
            print(f"✗ Error: {str(e)}")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, message in results:
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {name}: {message}")
    
    print()
    print(f"Result: {passed}/{total} checks passed")
    
    if passed == total:
        print()
        print("🎉 All checks passed! Isaac Lab is ready for training.")
        print()
        print("Next steps:")
        print("  1. Run a short training test:")
        print("     python scripts/reinforcement_learning/sb3/train.py \\")
        print("         --task MobileMMTrackEE-v0 --num_envs 4 --total_timesteps 10000")
        print()
        print("  2. Monitor with TensorBoard:")
        print("     tensorboard --logdir logs/sb3/MobileMMTrackEE-v0")
        print()
        print("  3. Scale up to full training:")
        print("     python scripts/reinforcement_learning/sb3/train.py \\")
        print("         --task MobileMMTrackEE-v0 --num_envs 1024 --headless")
        return 0
    else:
        print()
        print("⚠️  Some checks failed. Please review the errors above.")
        print()
        print("Common fixes:")
        print("  - Install Isaac Lab: pip install isaaclab")
        print("  - Install dependencies: pip install stable-baselines3[extra]")
        print("  - Check CUDA: nvidia-smi should show GPU")
        print("  - Set Python 3.11: pyenv local 3.11.10")
        return 1


if __name__ == "__main__":
    sys.exit(main())
