#!/usr/bin/env python3
"""
Debug script to check if trained model uses chassis and what rewards look like.
"""
import argparse

# Parse arguments FIRST (before Isaac imports)
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--num_steps", type=int, default=500, help="Number of steps to run")
args_cli = parser.parse_args()

# Launch Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Now import everything else
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO

# Register custom tasks
import src.rl_platform.tasks.mobile_mm  # noqa: F401

print("=" * 70)
print("DEBUG: Trained Model Chassis & Reward Analysis")
print("=" * 70)
print(f"Checkpoint: {args_cli.checkpoint}")
print(f"Steps to analyze: {args_cli.num_steps}")
print()

# Create environment
print("[1/3] Creating environment...")
env = gym.make(
    "MobileMMTrackEE-v0",
    num_envs=1,  # Single env for clarity
    trajectory_type="multi_recorded",
    use_all_trajectories=True,
)
print(f"    ✓ Environment created")

# Load model
print("\n[2/3] Loading model...")
model = PPO.load(args_cli.checkpoint, env=env)
print(f"    ✓ Model loaded")

# Run and analyze
print(f"\n[3/3] Running {args_cli.num_steps} steps and collecting stats...")
print()

obs = env.reset()
all_actions = []
all_rewards = []
step_count = 0

print("Step | Arm Actions (6) | Base Actions (vx, wz) | Reward | Cumulative")
print("-" * 80)

cumulative_reward = 0.0

while step_count < args_cli.num_steps:
    # Get action
    action, _ = model.predict(obs, deterministic=True)
    
    # Step
    obs, reward, done, info = env.step(action)
    
    # Record
    all_actions.append(action[0])  # First env
    all_rewards.append(reward[0])
    cumulative_reward += reward[0]
    
    # Print every 50 steps
    if step_count % 50 == 0:
        arm_actions = action[0, :6]
        base_actions = action[0, 6:]
        print(f"{step_count:4d} | {arm_actions} | {base_actions} | {reward[0]:8.2f} | {cumulative_reward:10.2f}")
    
    step_count += 1
    
    if done[0]:
        print(f"     | [EPISODE DONE - resetting]")
        obs = env.reset()
        cumulative_reward = 0.0

print()
print("=" * 70)
print("ANALYSIS")
print("=" * 70)

all_actions = np.array(all_actions)
all_rewards = np.array(all_rewards)

# Action statistics
arm_actions = all_actions[:, :6]
base_actions = all_actions[:, 6:]

print(f"\nArm Actions (joints 0-5):")
print(f"  Mean magnitude: {np.abs(arm_actions).mean():.4f}")
print(f"  Max magnitude:  {np.abs(arm_actions).max():.4f}")
print(f"  Std deviation:  {arm_actions.std():.4f}")

print(f"\nBase Actions (vx, wz):")
print(f"  Mean magnitude: {np.abs(base_actions).mean():.4f}")
print(f"  Max magnitude:  {np.abs(base_actions).max():.4f}")
print(f"  Std deviation:  {base_actions.std():.4f}")
print(f"  vx mean: {base_actions[:, 0].mean():.4f}, std: {base_actions[:, 0].std():.4f}")
print(f"  wz mean: {base_actions[:, 1].mean():.4f}, std: {base_actions[:, 1].std():.4f}")

print(f"\n🔍 Base action magnitude > 0.1: {(np.abs(base_actions) > 0.1).any(axis=1).sum()} / {len(base_actions)} steps")
print(f"   -> Robot using chassis: {'✅ YES' if (np.abs(base_actions) > 0.1).any() else '❌ NO - FROZEN!'}")

print(f"\nRewards:")
print(f"  Mean: {all_rewards.mean():.2f}")
print(f"  Std:  {all_rewards.std():.2f}")
print(f"  Min:  {all_rewards.min():.2f}")
print(f"  Max:  {all_rewards.max():.2f}")
print(f"  Total: {all_rewards.sum():.2f}")

print("\n" + "=" * 70)
print("✓ Analysis complete!")
print("=" * 70)

env.close()
simulation_app.close()
