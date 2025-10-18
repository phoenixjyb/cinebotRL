#!/usr/bin/env python3
"""
Analyze what actions the trained policy is actually producing.
Focus on base actions (vx, wz) to see if they're being used.
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Parse arguments FIRST
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_steps", type=int, default=200)
args = parser.parse_args()

# Validate checkpoint exists
if not os.path.exists(args.checkpoint):
    print(f"❌ Checkpoint not found: {args.checkpoint}")
    sys.exit(1)

# Initialize Isaac Sim using AppLauncher
print("=" * 70)
print("ACTION ANALYSIS: What is the policy doing?")
print("=" * 70)
print(f"Checkpoint: {args.checkpoint}")
print(f"Analyzing {args.num_steps} steps\n")

print("[1/4] Initializing Isaac Sim...")
try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError:
    print("    ✗ Could not import isaaclab.app. Make sure you're running with isaaclab.bat")
    sys.exit(1)

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app
print("    ✓ Isaac Sim initialized\n")

# Register custom tasks
print("[2/4] Registering custom tasks...")
try:
    from task_spec import register_isaac_lab_tasks
    register_isaac_lab_tasks()
    print("    ✓ Tasks registered\n")
except Exception as e:
    print(f"    ✗ Failed to register tasks: {e}")
    simulation_app.close()
    sys.exit(1)

# Create environment
print("[3/4] Creating environment...")
try:
    from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnvCfg, MobileMMTrackEEEnv
    from rl_platform.tasks.mobile_mm.config import TrajectoryConfig
    from stable_baselines3 import PPO
    
    # Create custom environment configuration
    env_cfg = MobileMMTrackEEEnvCfg()
    env_cfg.scene.num_envs = 1
    
    # Configure trajectory
    env_cfg.task_config.trajectory = TrajectoryConfig(
        type="multi_recorded",
        trajectory_dir="trajectoryToLearn/world_json",
        trajectory_pattern="**/*.json",
    )
    
    # Create environment directly with config
    env = MobileMMTrackEEEnv(cfg=env_cfg)
    print("    ✓ Environment created\n")
except Exception as e:
    print(f"    ✗ Failed to create environment: {e}")
    simulation_app.close()
    sys.exit(1)

# Load model
print("[4/4] Loading model and analyzing actions...")
try:
    from stable_baselines3 import PPO
    model = PPO.load(args.checkpoint)
    print("    ✓ Model loaded\n")
except Exception as e:
    print(f"    ✗ Failed to load model: {e}")
    simulation_app.close()
    sys.exit(1)

# Collect actions
print(f"Collecting actions from policy over {args.num_steps} steps...")
arm_actions_list = []
base_actions_list = []
arm_activity = []  # Track if arm is moving significantly

try:
    obs = env.reset()
    
    # Handle new Gymnasium API: reset() returns (obs, info)
    if isinstance(obs, tuple):
        obs, info = obs
    
    for step in range(args.num_steps):
        # obs is a dict of torch tensors from Isaac Lab
        # Get the policy observation
        if isinstance(obs, dict):
            obs_tensor = obs.get("policy", list(obs.values())[0])
        else:
            obs_tensor = obs
        
        # Convert to numpy if needed for SB3
        if hasattr(obs_tensor, 'cpu'):
            obs_np = obs_tensor.cpu().numpy()
        else:
            obs_np = np.array(obs_tensor)
        
        # Predict action (SB3 expects numpy array)
        action, _ = model.predict(obs_np, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(torch.from_numpy(action).float().to(env.device))
        
        # Split actions: first 6 for arm, last 2 for base (vx, wz)
        if isinstance(action, np.ndarray):
            if len(action.shape) > 1:
                # [num_envs, action_dim] -> take first env
                action_single = action[0]
            else:
                action_single = action
            
            arm_actions_list.append(action_single[:6])
            base_actions_list.append(action_single[6:])
            
            # Check if arm is moving significantly
            arm_magnitude = np.linalg.norm(action_single[:6])
            arm_activity.append(arm_magnitude > 0.01)  # Threshold for "moving"
        
        # Reset if episode done
        if terminated.any() if hasattr(terminated, 'any') else terminated:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs, info = obs

    # Convert to numpy arrays
    arm_actions = np.array(arm_actions_list)
    base_actions = np.array(base_actions_list)
    arm_activity = np.array(arm_activity)
    
    print("\n" + "=" * 70)
    print("POLICY ACTION ANALYSIS")
    print("=" * 70)
    print(f"\n📊 COLLECTED DATA:")
    print(f"   Steps analyzed: {len(arm_actions)}")
    print(f"   Action dim: 6 (arm) + 2 (base) = 8 total")
    
    print(f"\n🤖 ARM ACTIONS (DOF 0-5):")
    print(f"   Shape: {arm_actions.shape}")
    print(f"   Mean: {arm_actions.mean(axis=0)}")
    print(f"   Std:  {arm_actions.std(axis=0)}")
    print(f"   Min:  {arm_actions.min(axis=0)}")
    print(f"   Max:  {arm_actions.max(axis=0)}")
    arm_moving_pct = (arm_activity.sum() / len(arm_activity)) * 100
    print(f"   % Steps with arm activity (|action| > 0.01): {arm_moving_pct:.1f}%")
    
    print(f"\n🚀 BASE ACTIONS (vx, wz):")
    print(f"   Shape: {base_actions.shape}")
    print(f"   vx (linear velocity):")
    print(f"     Mean: {base_actions[:, 0].mean():.6f}")
    print(f"     Std:  {base_actions[:, 0].std():.6f}")
    print(f"     Min:  {base_actions[:, 0].min():.6f}")
    print(f"     Max:  {base_actions[:, 0].max():.6f}")
    print(f"   wz (angular velocity):")
    print(f"     Mean: {base_actions[:, 1].mean():.6f}")
    print(f"     Std:  {base_actions[:, 1].std():.6f}")
    print(f"     Min:  {base_actions[:, 1].min():.6f}")
    print(f"     Max:  {base_actions[:, 1].max():.6f}")
    
    # Count steps with meaningful base motion
    base_magnitude = np.linalg.norm(base_actions, axis=1)
    meaningful_base = (base_magnitude > 0.01).sum()
    pct_meaningful = (meaningful_base / len(base_actions)) * 100
    
    print(f"\n   Base motion magnitude:")
    print(f"     Mean: {base_magnitude.mean():.6f}")
    print(f"     Std:  {base_magnitude.std():.6f}")
    print(f"     Min:  {base_magnitude.min():.6f}")
    print(f"     Max:  {base_magnitude.max():.6f}")
    print(f"   % Steps with base activity (magnitude > 0.01): {pct_meaningful:.1f}%")
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    if pct_meaningful < 1.0:
        print("🔴 CRITICAL: Policy is NOT using base actions at all!")
        print("   → The policy learned to ignore chassis movement")
        print("   → Possible causes:")
        print("      1. Reward function doesn't incentivize base motion")
        print("      2. Network initialization didn't explore base actions")
        print("      3. Trajectory targets don't require base movement (unlikely)")
    elif pct_meaningful < 10.0:
        print("🟡 WARNING: Policy barely uses base actions (<10% of time)")
        print("   → Policy is learning to mostly use arm only")
        print("   → Check reward scaling for base motion components")
    else:
        print("🟢 OK: Policy is using base actions regularly")
        print(f"   → {pct_meaningful:.1f}% of steps have meaningful base movement")
        if arm_moving_pct > 50:
            print(f"   → Arm also active in {arm_moving_pct:.1f}% of steps")
            print("   → Policy appears to be coordinating arm and base")
        else:
            print(f"   → Arm less active ({arm_moving_pct:.1f}% of steps)")
            print("   → Policy prioritizing base motion")
    
    print("\n" + "=" * 70)

except Exception as e:
    import traceback
    print(f"\n❌ ERROR during analysis: {e}")
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)

finally:
    simulation_app.close()
    print("\n✓ Analysis complete!")
