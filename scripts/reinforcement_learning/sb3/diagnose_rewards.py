"""Diagnostic script to understand reward breakdown during evaluation.

This script will help identify which reward components are causing the massive negative rewards.
"""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=100, help="Steps to run for diagnostics")
    args = parser.parse_args()
    
    print("="*80)
    print("REWARD DIAGNOSTIC TOOL")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num envs: {args.num_envs}")
    print(f"Steps to analyze: {args.num_steps}")
    print()
    
    # Initialize Isaac Sim
    print("[1/4] Initializing Isaac Sim...")
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import PPO
    
    # Register tasks
    print("[2/4] Registering tasks...")
    from task_spec import register_isaac_lab_tasks
    register_isaac_lab_tasks()
    
    # Create environment
    print("[3/4] Creating environment...")
    env = gym.make(
        "MobileMMTrackEE-v0",
        num_envs=args.num_envs,
        headless=True,
        trajectory_type="multi_recorded",
        use_all_trajectories=True,
    )
    
    # Load model
    print("[4/4] Loading model...")
    model = PPO.load(args.checkpoint, device="cuda")
    
    print("\n" + "="*80)
    print("RUNNING DIAGNOSTIC...")
    print("="*80)
    
    # Reset environment
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs
    if isinstance(obs, dict):
        obs = obs.get("policy", list(obs.values())[0])
    if hasattr(obs, 'cpu'):
        obs = obs.cpu().numpy()
    
    # Track reward components
    reward_components = {
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
        "obstacle_reward": [],
        "total": [],
    }
    
    print(f"\nRunning {args.num_steps} steps...")
    for step in range(args.num_steps):
        # Get action
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        actions_torch = torch.from_numpy(action).float().to("cuda:0")
        result = env.unwrapped.step(actions_torch)
        
        if len(result) == 5:
            obs_dict, rewards, terminated, truncated, infos = result
        else:
            obs_dict, rewards, dones, infos = result
        
        # Extract reward components from environment
        if hasattr(env.unwrapped, 'reward_components'):
            components = env.unwrapped.reward_components
            for key in reward_components.keys():
                if key == "total":
                    reward_components["total"].append(rewards.cpu().numpy() if hasattr(rewards, 'cpu') else rewards)
                elif key in components:
                    val = components[key]
                    if hasattr(val, 'cpu'):
                        val = val.cpu().numpy()
                    reward_components[key].append(val)
        
        # Convert observation for next step
        if isinstance(obs_dict, dict):
            obs_tensor = obs_dict.get("policy", list(obs_dict.values())[0])
            if hasattr(obs_tensor, 'cpu'):
                obs = obs_tensor.cpu().numpy()
            else:
                obs = np.array(obs_tensor)
        else:
            obs = obs_dict
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step + 1}/{args.num_steps}")
    
    # Analyze results
    print("\n" + "="*80)
    print("REWARD COMPONENT ANALYSIS")
    print("="*80)
    print(f"{'Component':<30} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print("-"*80)
    
    for key, values in reward_components.items():
        if len(values) > 0:
            arr = np.concatenate([v.flatten() for v in values])
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            min_val = np.min(arr)
            max_val = np.max(arr)
            
            # Highlight problematic components
            indicator = ""
            if key != "total":
                if key.endswith("penalty") and mean_val < -100:
                    indicator = " ⚠️  VERY HIGH!"
                elif key.endswith("penalty") and mean_val < -10:
                    indicator = " ⚠️"
                elif not key.endswith("penalty") and mean_val < 0.1:
                    indicator = " (low reward)"
            
            print(f"{key:<30} {mean_val:<15.2f} {std_val:<15.2f} {min_val:<15.2f} {max_val:<15.2f}{indicator}")
    
    print("="*80)
    print("\nINTERPRETATION:")
    print("  - Penalties should be negative (subtracted from reward)")
    print("  - Large negative means that penalty term is dominating")
    print("  - Check components marked with ⚠️")
    print("  - If self_collision_penalty is zero, contact forces not working!")
    print("="*80)
    
    # Cleanup
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
