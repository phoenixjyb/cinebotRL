#!/usr/bin/env python3
"""Diagnose reward components during evaluation to find the cause of massive negative rewards."""

import argparse
import torch
import numpy as np
from pathlib import Path
from omni.isaac.lab.app import AppLauncher

# Setup paths
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.task_spec import MobileMMTrackConfig
from src.rl_platform.tasks.mobile_mm.env import MobileMMTrackEEEnv


def main():
    """Run diagnostic evaluation."""
    parser = argparse.ArgumentParser(description="Diagnose reward components")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--num_episodes", type=int, default=1,
                       help="Number of episodes to run")
    parser.add_argument("--num_steps", type=int, default=100,
                       help="Max steps per episode")
    args = parser.parse_args()

    # Setup Isaac Sim
    app_launcher = AppLauncher(headless=False)
    sim = app_launcher.app

    # Create environment
    print("\n[1/3] Creating environment...")
    cfg = MobileMMTrackConfig()
    cfg.env.num_envs = 1
    cfg.task.trajectory.type = "multi_recorded"
    cfg.task.trajectory.use_all_trajectories = True
    
    env = MobileMMTrackEEEnv(cfg=cfg.task)
    print("✓ Environment created")

    # Load model
    print("\n[2/3] Loading model...")
    from stable_baselines3 import PPO
    model = PPO.load(args.checkpoint, device="cuda:0")
    print("✓ Model loaded")

    # Run diagnostic steps
    print("\n[3/3] Running diagnostic evaluation...")
    print("=" * 100)
    
    for episode in range(args.num_episodes):
        print(f"\n📊 EPISODE {episode + 1}")
        print("=" * 100)
        
        obs, _ = env.reset()
        episode_total_reward = 0
        reward_components_sum = {k: 0 for k in [
            "position_tracking", "orientation_tracking", "progress_bonus",
            "action_magnitude_penalty", "action_rate_penalty", "action_smoothness_penalty",
            "velocity_limit_penalty", "acceleration_limit_penalty", "jerk_penalty",
            "joint_limit_penalty", "lateral_motion_penalty", "self_collision_penalty",
            "stability_penalty", "obstacle_reward"
        ]}
        
        for step in range(args.num_steps):
            # Get action from model
            action, _ = model.predict(obs, deterministic=False)
            
            # Convert to torch
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float().to("cuda:0")
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            episode_total_reward += reward[0].item() if hasattr(reward, 'item') else reward
            
            # Extract reward components
            if hasattr(env, 'reward_components') and env.reward_components is not None:
                for key, val in env.reward_components.items():
                    if key in reward_components_sum:
                        reward_components_sum[key] += val[0].item()
            
            # Print first step detail
            if step == 0:
                print(f"\n🔹 STEP 0 - Detailed Breakdown:")
                print(f"   Total Reward: {reward[0].item():+.2f}")
                if hasattr(env, 'reward_components') and env.reward_components is not None:
                    for key, val in sorted(env.reward_components.items()):
                        v = val[0].item() if hasattr(val, 'item') else val[0]
                        sign = "+" if v >= 0 else ""
                        print(f"   {key:35s}: {sign}{v:+10.4f}")
            
            # Print summary every 10 steps
            if (step + 1) % 10 == 0:
                print(f"\n✓ Steps {step - 9:3d}-{step + 1:3d}: Avg reward = {episode_total_reward / (step + 1):+10.2f}")
        
        # Episode summary
        print("\n" + "=" * 100)
        print(f"📊 EPISODE SUMMARY:")
        print(f"   Total Steps: {step + 1}")
        print(f"   Episode Total Reward: {episode_total_reward:+.2f}")
        print(f"   Average Reward/Step: {episode_total_reward / (step + 1):+.4f}")
        
        print(f"\n📈 Component Averages (over {step + 1} steps):")
        for key, total in sorted(reward_components_sum.items()):
            avg = total / (step + 1)
            sign = "+" if avg >= 0 else ""
            print(f"   {key:35s}: {sign}{avg:+10.4f}")
    
    print("\n" + "=" * 100)
    print("Diagnosis complete!")
    
    # Cleanup
    env.close()
    app_launcher.close()


if __name__ == "__main__":
    main()
