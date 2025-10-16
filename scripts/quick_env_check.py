"""Quick environment diversity check.

Tests if parallel environments are truly independent by checking:
1. Observation diversity
2. Reward diversity  
3. Action response independence
"""

import argparse
import torch
import gymnasium as gym
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Isaac Lab 5.0 import
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Quick environment diversity check")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run headless")
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Register task
from task_spec import register_isaac_lab_tasks
register_isaac_lab_tasks()

def quick_diversity_check():
    """Quick check if environments are independent."""
    print("\n" + "="*70)
    print("QUICK ENVIRONMENT DIVERSITY CHECK")
    print("="*70)
    
    num_envs = args_cli.num_envs
    print(f"\nTesting {num_envs} parallel environments\n")
    
    # Create environment
    env = gym.make("MobileMMTrackEE-v0", num_envs=num_envs, headless=args_cli.headless)
    print(f"✓ Created {num_envs} environments\n")
    
    # Reset and get initial observations
    print("[1] Initial Observations After Reset")
    print("-" * 70)
    obs, _ = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation device: {obs.device}")
    
    # Check observation diversity
    obs_mean = obs.mean(dim=0)
    obs_std = obs.std(dim=0)
    mean_std = obs_std.mean().item()
    
    print(f"\nStatistics across {num_envs} environments:")
    print(f"  Mean observation std: {mean_std:.6f}")
    print(f"  Max observation std:  {obs_std.max():.6f}")
    print(f"  Min observation std:  {obs_std.min():.6f}")
    
    # Count unique observations
    unique_obs = torch.unique(obs, dim=0)
    print(f"  Unique observations:  {unique_obs.shape[0]}/{num_envs}")
    
    # Print first 3 observations for manual inspection
    print(f"\nFirst 3 observations (first 10 dims):")
    for i in range(min(3, num_envs)):
        print(f"  Env {i}: {obs[i, :10].tolist()}")
    
    # Test action responses
    print(f"\n[2] Action Response Diversity")
    print("-" * 70)
    
    # Apply different random actions to each environment
    actions = torch.randn(num_envs, env.action_space.shape[0], device=obs.device)
    print(f"Action shape: {actions.shape}")
    
    # Step environment
    obs_new, rewards, dones, truncs, info = env.step(actions)
    
    # Check reward diversity
    reward_mean = rewards.mean().item()
    reward_std = rewards.std().item()
    
    print(f"\nReward statistics:")
    print(f"  Mean: {reward_mean:.4f}")
    print(f"  Std:  {reward_std:.4f}")
    print(f"  Min:  {rewards.min().item():.4f}")
    print(f"  Max:  {rewards.max().item():.4f}")
    
    # Print individual rewards
    print(f"\nIndividual rewards:")
    for i in range(num_envs):
        print(f"  Env {i}: {rewards[i].item():.6f}")
    
    # Check observation changes
    obs_change = (obs_new - obs).abs()
    obs_change_mean = obs_change.mean(dim=1)
    
    print(f"\nObservation changes (L1 norm):")
    for i in range(num_envs):
        print(f"  Env {i}: {obs_change_mean[i].item():.6f}")
    
    # Final verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Observation diversity
    tests_total += 1
    if mean_std > 1e-6 or unique_obs.shape[0] > 1:
        print("✓ PASS: Observations show diversity")
        tests_passed += 1
    else:
        print("✗ FAIL: All observations identical")
    
    # Test 2: Reward diversity
    tests_total += 1
    if reward_std > 1e-6:
        print("✓ PASS: Rewards show diversity")
        tests_passed += 1
    else:
        print("✗ FAIL: All rewards identical")
    
    # Test 3: Independent changes
    tests_total += 1
    change_std = obs_change_mean.std().item()
    if change_std > 1e-6:
        print("✓ PASS: Environments respond independently")
        tests_passed += 1
    else:
        print("✗ FAIL: All environments change identically")
    
    print(f"\nResult: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed == tests_total:
        print("\n✅ Environments are properly independent and diverse")
    elif tests_passed > 0:
        print("\n⚠️  Partial diversity - some environments may share state")
    else:
        print("\n❌ No diversity detected - all environments identical!")
    
    print("\nNote: For deterministic trajectories (circle, line),")
    print("      environments start with same target but different random seeds.")
    
    env.close()
    print()

if __name__ == "__main__":
    try:
        quick_diversity_check()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
