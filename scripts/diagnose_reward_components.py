#!/usr/bin/env python3
"""
Diagnose reward components to understand why policy doesn't move forward.
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

print("=" * 70)
print("REWARD COMPONENT ANALYSIS")
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
print("[4/4] Loading model and analyzing reward components...")
try:
    from stable_baselines3 import PPO
    model = PPO.load(args.checkpoint)
    print("    ✓ Model loaded\n")
except Exception as e:
    print(f"    ✗ Failed to load model: {e}")
    simulation_app.close()
    sys.exit(1)

# Collect reward components
print(f"Collecting reward components over {args.num_steps} steps...\n")

reward_history = []
component_history = {
    "position_tracking": [],
    "orientation_tracking": [],
    "progress_bonus": [],
    "action_magnitude_penalty": [],
    "action_rate_penalty": [],
    "action_smoothness_penalty": [],
    "velocity_limit_penalty": [],
    "acceleration_limit_penalty": [],
    "jerk_penalty": [],
    "joint_limit_penalty": [],
    "lateral_motion_penalty": [],
    "self_collision_penalty": [],
    "stability_penalty": [],
}

action_history = {"base_vx": [], "base_wz": [], "arm_actions": []}
velocity_history = {"base_vx": [], "base_wz": []}

try:
    obs = env.reset()
    
    # Handle new Gymnasium API: reset() returns (obs, info)
    if isinstance(obs, tuple):
        obs, info = obs
    
    for step in range(args.num_steps):
        # obs is a dict of torch tensors from Isaac Lab
        if isinstance(obs, dict):
            obs_tensor = obs.get("policy", list(obs.values())[0])
        else:
            obs_tensor = obs
        
        # Convert to numpy if needed for SB3
        if hasattr(obs_tensor, 'cpu'):
            obs_np = obs_tensor.cpu().numpy()
        else:
            obs_np = np.array(obs_tensor)
        
        # Predict action
        action, _ = model.predict(obs_np, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(torch.from_numpy(action).float().to(env.device))
        
        # Collect reward components
        if hasattr(env, 'reward_components'):
            reward_history.append(reward.cpu().item() if hasattr(reward, 'cpu') else reward)
            
            for key, value in env.reward_components.items():
                if key in component_history:
                    val = value.cpu().item() if hasattr(value, 'cpu') else value[0].item()
                    component_history[key].append(val)
        
        # Collect actions
        if isinstance(action, np.ndarray):
            if len(action.shape) > 1:
                action_single = action[0]
            else:
                action_single = action
            action_history["base_vx"].append(action_single[6])
            action_history["base_wz"].append(action_single[7])
            action_history["arm_actions"].append(np.linalg.norm(action_single[:6]))
        
        # Get base velocities from diagnostics
        if hasattr(env, 'extras') and 'base_diagnostics' in env.extras:
            diag = env.extras['base_diagnostics']
            velocity_history["base_vx"].append(diag.get("base_vel_x_mean", 0.0))
            velocity_history["base_wz"].append(diag.get("base_vel_z_mean", 0.0))
        
        # Reset if episode done
        if terminated.any() if hasattr(terminated, 'any') else terminated:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs, info = obs
    
    # Convert to numpy arrays
    reward_history = np.array(reward_history)
    for key in component_history:
        component_history[key] = np.array(component_history[key])
    for key in action_history:
        action_history[key] = np.array(action_history[key])
    for key in velocity_history:
        velocity_history[key] = np.array(velocity_history[key])
    
    print("\n" + "=" * 70)
    print("REWARD COMPONENT BREAKDOWN")
    print("=" * 70)
    
    print(f"\n📊 OVERALL REWARD:")
    print(f"   Mean: {reward_history.mean():.2f}")
    print(f"   Min:  {reward_history.min():.2f}")
    print(f"   Max:  {reward_history.max():.2f}")
    
    print(f"\n✅ POSITIVE COMPONENTS (should incentivize good behavior):")
    positive_components = {
        "position_tracking": component_history["position_tracking"],
        "orientation_tracking": component_history["orientation_tracking"],
        "progress_bonus": component_history["progress_bonus"],
    }
    for name, values in positive_components.items():
        print(f"   {name:30s}: mean={values.mean():8.4f}, max={values.max():8.4f}")
    
    print(f"\n❌ NEGATIVE COMPONENTS (penalties that reduce reward):")
    negative_components = {
        "action_magnitude_penalty": component_history["action_magnitude_penalty"],
        "action_rate_penalty": component_history["action_rate_penalty"],
        "action_smoothness_penalty": component_history["action_smoothness_penalty"],
        "velocity_limit_penalty": component_history["velocity_limit_penalty"],
        "acceleration_limit_penalty": component_history["acceleration_limit_penalty"],
        "jerk_penalty": component_history["jerk_penalty"],
        "joint_limit_penalty": component_history["joint_limit_penalty"],
        "lateral_motion_penalty": component_history["lateral_motion_penalty"],
        "self_collision_penalty": component_history["self_collision_penalty"],
        "stability_penalty": component_history["stability_penalty"],
    }
    
    for name, values in negative_components.items():
        print(f"   {name:30s}: mean={values.mean():8.4f}, max={values.max():8.4f}")
    
    print(f"\n🔍 ACTION ANALYSIS:")
    print(f"   Base vx (forward):  mean={action_history['base_vx'].mean():8.6f}, "
          f"max={np.abs(action_history['base_vx']).max():8.6f}")
    print(f"   Base wz (rotation): mean={action_history['base_wz'].mean():8.6f}, "
          f"max={np.abs(action_history['base_wz']).max():8.6f}")
    print(f"   Arm magnitude:      mean={action_history['arm_actions'].mean():8.6f}, "
          f"max={action_history['arm_actions'].max():8.6f}")
    
    print(f"\n🚀 VELOCITY ANALYSIS (what actually happened):")
    print(f"   Base vx velocity:   mean={velocity_history['base_vx'].mean():8.6f}")
    print(f"   Base wz velocity:   mean={velocity_history['base_wz'].mean():8.6f}")
    
    print(f"\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    # Analyze which penalty is dominant
    penalties = {name: values.mean() for name, values in negative_components.items()}
    max_penalty_name = max(penalties, key=penalties.get)
    max_penalty_value = penalties[max_penalty_name]
    
    print(f"\n🎯 DOMINANT PENALTY: {max_penalty_name} (mean={max_penalty_value:.4f})")
    
    if component_history["action_magnitude_penalty"].mean() > 0.5:
        print(f"⚠️  Action magnitude penalty is HIGH ({component_history['action_magnitude_penalty'].mean():.4f})")
        print(f"   This penalizes ANY movement, including necessary base motion")
        print(f"   Suggestion: Reduce action_magnitude weight or separate arm/base penalties")
    
    if component_history["velocity_limit_penalty"].mean() > 0.5:
        print(f"⚠️  Velocity limit penalty is HIGH ({component_history['velocity_limit_penalty'].mean():.4f})")
        print(f"   Policy is hitting velocity limits")
        print(f"   Suggestion: Check if base velocity normalization is correct")
    
    if component_history["position_tracking"].mean() < 0.1:
        print(f"⚠️  Position tracking reward is LOW ({component_history['position_tracking'].mean():.4f})")
        print(f"   Policy not making good progress toward target")
        print(f"   Suggestion: Increase position_tracking weight or fix trajectory scaling")
    
    if action_history["base_vx"].mean() < 0.001 and component_history["progress_bonus"].mean() < 0.01:
        print(f"🔴 CRITICAL: Policy not attempting forward movement AND not getting progress bonus")
        print(f"   Policy learned to ignore forward motion component")
        print(f"   Root cause: Likely reward structure doesn't incentivize base movement")
    
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
