#!/usr/bin/env python3
"""
Test script to visualize the trained mobile manipulator model in Isaac Sim
"""

import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import time
import sys
import os

# Add src to path to import our task
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Isaac Lab imports
from omni.isaac.lab.app import AppLauncher

def main():
    """Main function to test the trained model with visualization."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test trained mobile manipulator model")
    parser.add_argument("--model_path", type=str, 
                       default="c:/Users/yanbo/wSpace/cinebotRL/logs/sb3/mobilemmtrackee_v0/20251017_211012/final_model.zip",
                       help="Path to the trained model")
    parser.add_argument("--norm_path", type=str,
                       default="c:/Users/yanbo/wSpace/cinebotRL/logs/sb3/mobilemmtrackee_v0/20251017_211012/vec_normalize.pkl", 
                       help="Path to the normalization parameters")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (use 1 for visualization)")
    parser.add_argument("--episode_length", type=int, default=1000, help="Length of each episode")
    args = parser.parse_args()

    # Configure AppLauncher for visualization
    app_launcher = AppLauncher(headless=False)  # Enable visualization
    simulation_app = app_launcher.app

    print("=" * 80)
    print("TESTING TRAINED MOBILE MANIPULATOR MODEL")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Normalization path: {args.norm_path}")
    print(f"Num envs: {args.num_envs}")
    print()

    # Import Isaac Lab after AppLauncher
    import omni.isaac.lab.envs  # noqa: F401
    from omni.isaac.lab.envs import DirectRLEnvCfg
    from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
    import gymnasium as gym
    
    # Import our task
    import task_spec

    # Create environment for testing (with visualization)
    print("[1/5] Creating environment with visualization...")
    
    # Create environment directly
    env = gym.make("MobileMMTrackEE-v0", num_envs=args.num_envs, headless=False)
    env = Sb3VecEnvWrapper(env)
    
    print(f"✓ Environment created with {env.num_envs} environments")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Load normalization parameters
    print("\n[2/5] Loading normalization parameters...")
    try:
        env = VecNormalize.load(args.norm_path, env)
        env.training = False  # Don't update normalization during testing
        env.norm_reward = False  # Don't normalize rewards during testing
        print("✓ Normalization parameters loaded")
    except Exception as e:
        print(f"⚠ Could not load normalization parameters: {e}")
        print("  Continuing without normalization...")
    
    # Load the trained model
    print("\n[3/5] Loading trained model...")
    try:
        model = PPO.load(args.model_path, env=env)
        print("✓ Model loaded successfully")
        print(f"  Policy architecture: {model.policy}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Test the model
    print("\n[4/5] Starting model testing...")
    print("=" * 60)
    
    obs = env.reset()
    episode_rewards = []
    episode_steps = []
    current_episode_reward = 0
    current_episode_steps = 0
    episode_count = 0
    
    print("Press Ctrl+C to stop testing...")
    
    try:
        for step in range(10000):  # Run for many steps
            # Get action from the trained model
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, rewards, dones, infos = env.step(action)
            
            current_episode_reward += rewards[0]
            current_episode_steps += 1
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"Step {step:4d} | Reward: {current_episode_reward:8.3f} | Episode steps: {current_episode_steps:4d}")
            
            # Check for episode completion
            if dones[0]:
                episode_count += 1
                episode_rewards.append(current_episode_reward)
                episode_steps.append(current_episode_steps)
                
                print(f"\n🏁 Episode {episode_count} completed!")
                print(f"   Reward: {current_episode_reward:.3f}")
                print(f"   Steps: {current_episode_steps}")
                
                if len(episode_rewards) > 1:
                    avg_reward = sum(episode_rewards) / len(episode_rewards)
                    avg_steps = sum(episode_steps) / len(episode_steps)
                    print(f"   Average reward: {avg_reward:.3f}")
                    print(f"   Average steps: {avg_steps:.1f}")
                print()
                
                current_episode_reward = 0
                current_episode_steps = 0
            
            # Small delay for visualization
            time.sleep(0.05)  # 20 FPS (matches 20Hz control frequency)
            
    except KeyboardInterrupt:
        print("\n🛑 Testing stopped by user")
    
    # Final statistics
    print("\n[5/5] Testing completed!")
    print("=" * 60)
    if episode_rewards:
        print(f"Episodes completed: {len(episode_rewards)}")
        print(f"Average reward: {sum(episode_rewards) / len(episode_rewards):.3f}")
        print(f"Average episode length: {sum(episode_steps) / len(episode_steps):.1f}")
        print(f"Best episode reward: {max(episode_rewards):.3f}")
    else:
        print("No episodes completed")
    
    print("\n✓ Model testing finished!")


if __name__ == "__main__":
    main()