"""
Simple test to verify contact force API is working by adding diagnostics
to the environment during a short training run.

Run with: isaaclab.bat -p scripts/test_contact_forces_simple.py
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"

import torch

print("\n" + "=" * 80)
print("CONTACT FORCE API - SIMPLE VERIFICATION")
print("=" * 80)
print()

# Import environment
try:
    from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnvCfg, MobileMMTrackEEEnv
    from rl_platform.tasks.mobile_mm.config import TrajectoryConfig
    print("✓ Imports successful\n")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Create config
print("Creating environment config...")
env_cfg = MobileMMTrackEEEnvCfg()
env_cfg.num_envs = 16  # Small number for testing
env_cfg.task_config.trajectory = TrajectoryConfig(
    type="circle",  # Simple trajectory
)
print("✓ Config created\n")

# Create environment
print("Creating environment...")
try:
    env = MobileMMTrackEEEnv(cfg=env_cfg)
    print("✓ Environment created\n")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test contact force API
print("-" * 80)
print("TESTING CONTACT FORCE API")
print("-" * 80)

robot = env.robot
api_working = False
working_method = None

# Try both APIs
try:
    forces = robot.root_physx_view.get_net_contact_forces()
    print(f"✓ Method 1 (root_physx_view): WORKS")
    print(f"  Shape: {forces.shape}")
    api_working = True
    working_method = "root_physx_view"
except AttributeError as e:
    print(f"✗ Method 1: {e}")

try:
    forces = robot.data.body_net_contact_force_w
    print(f"✓ Method 2 (body_net_contact_force_w): WORKS")
    print(f"  Shape: {forces.shape}")
    if not api_working:
        api_working = True
        working_method = "body_net_contact_force_w"
except AttributeError as e:
    print(f"✗ Method 2: {e}")

if not api_working:
    print("\n❌ NO CONTACT FORCE API AVAILABLE!")
    env.close()
    sys.exit(1)

print(f"\n✓ Using: {working_method}\n")

# Run simulation and check forces
print("-" * 80)
print("RUNNING SIMULATION")
print("-" * 80)

obs_dict, extras = env.reset()
print("Environment reset")

max_forces_seen = []

print("\nRunning 100 steps with random actions...")
for step in range(100):
    # Random actions
    actions = torch.rand((env.num_envs, env.num_actions), device=env.device) * 2 - 1
    
    obs_dict, rewards, dones, extras = env.step(actions)
    
    # Get contact forces
    if working_method == "root_physx_view":
        forces = robot.root_physx_view.get_net_contact_forces()
    else:
        forces = robot.data.body_net_contact_force_w
    
    # Calculate magnitudes
    force_mag = torch.norm(forces, dim=-1)  # [num_envs, num_bodies]
    max_force_per_env = torch.max(force_mag, dim=-1)[0]  # [num_envs]
    overall_max = max_force_per_env.max().item()
    overall_mean = max_force_per_env.mean().item()
    
    max_forces_seen.append(overall_max)
    
    if step % 20 == 0:
        print(f"  Step {step:3d}: Max={overall_max:6.2f}N, Mean={overall_mean:6.2f}N")

print(f"\nResults over 100 steps:")
print(f"  Peak force seen: {max(max_forces_seen):.2f} N")
print(f"  Mean max force:  {sum(max_forces_seen)/len(max_forces_seen):.2f} N")
print(f"  Steps with force > 1.0N: {sum(1 for f in max_forces_seen if f > 1.0)}")
print(f"  Steps with force > 5.0N: {sum(1 for f in max_forces_seen if f > 5.0)}")

# Check reward components
print("\n" + "-" * 80)
print("REWARD COMPONENTS CHECK")
print("-" * 80)

if "reward_components" in extras:
    comps = extras["reward_components"]
    print("Collision-related rewards:")
    for key, val in comps.items():
        if "collision" in key.lower():
            print(f"  {key}: {val:.4f}")
else:
    print("⚠  No reward_components in extras")

# Final verdict
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

peak = max(max_forces_seen)
steps_with_contact = sum(1 for f in max_forces_seen if f > 1.0)

if peak > 5.0:
    print("✅ PASS: Contact forces ARE WORKING!")
    print(f"   Peak force: {peak:.2f} N")
    print("   Self-collision detection is functional ✓")
elif peak > 1.0:
    print("⚠️  PARTIAL: Some contact forces detected")
    print(f"   Peak force: {peak:.2f} N")
    print(f"   {steps_with_contact} steps had force > 1.0N")
    print("   API works but may need sensitivity tuning")
elif peak > 0.1:
    print("⚠️  WEAK: Very low forces detected")
    print(f"   Peak force: {peak:.2f} N")
    print("   API may be working but very insensitive")
else:
    print("❌ FAIL: NO contact forces detected!")
    print(f"   Peak force: {peak:.2f} N (essentially zero)")
    print("   API is NOT working - returns all zeros!")

print("=" * 80 + "\n")

env.close()
