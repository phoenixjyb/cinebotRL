"""Evaluation script for trained MobileMMTrackEE models.

This script loads a trained model and runs it in the environment with visualization
to see how well it performs on the end-effector tracking task.

Usage:
    # Visualize trained model with GUI:
    I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/evaluate.py \\
        --checkpoint H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251016_184941\final_model.zip \\
        --num_envs 4
    
    # Run headless evaluation (get metrics only):
    I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/evaluate.py \\
        --checkpoint H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251016_184941\final_model.zip \\
        --num_envs 16 \\
        --headless \\
        --num_episodes 100
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}")
print(f"[DEBUG] sys.path includes: {PROJECT_ROOT}, {PROJECT_ROOT / 'src'}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained mobile manipulator policy"
    )
    
    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.zip file)",
    )
    
    # Environment settings
    parser.add_argument(
        "--task",
        type=str,
        default="MobileMMTrackEE-v0",
        help="Task ID to evaluate on",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4,
        help="Number of parallel environments (use fewer for better visualization)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)",
    )
    
    # Evaluation settings
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (no exploration noise)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run on",
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    print("=" * 70)
    print("MobileMMTrackEE Policy Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Task: {args.task}")
    print(f"Num envs: {args.num_envs}")
    print(f"Headless: {args.headless}")
    print(f"Deterministic: {args.deterministic}")
    print()
    
    # Initialize Isaac Sim
    print("[1/5] Initializing Isaac Sim...")
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        print("    ✗ Could not import isaaclab.app. Make sure you're running with isaaclab.bat")
        sys.exit(1)
    
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    print("    ✓ Isaac Sim initialized")
    
    # Now we can import Isaac Lab modules
    import torch
    import gymnasium as gym
    from stable_baselines3 import PPO
    
    # Register custom tasks
    print("\n[2/5] Registering custom tasks...")
    try:
        from task_spec import register_isaac_lab_tasks
        register_isaac_lab_tasks()
        print(f"    ✓ Registered task: {args.task}")
    except Exception as e:
        print(f"    ✗ Failed to register tasks: {e}")
        simulation_app.close()
        sys.exit(1)
    
    # Set device
    print("\n[3/5] Setting up device...")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"    ✓ Using device: {device}")
    
    # Create environment
    print(f"\n[4/5] Creating environment ({args.task})...")
    try:
        env = gym.make(args.task, num_envs=args.num_envs, headless=args.headless)
        print(f"    ✓ Environment created with {args.num_envs} parallel instances")
    except Exception as e:
        print(f"    ✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
    
    # Load trained model
    print(f"\n[5/5] Loading trained model...")
    try:
        model = PPO.load(args.checkpoint, env=env, device=device)
        print(f"    ✓ Model loaded successfully")
        print(f"    Policy network: {model.policy}")
    except Exception as e:
        print(f"    ✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        simulation_app.close()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("Starting Evaluation...")
    print("=" * 70)
    if not args.headless:
        print("🎨 VISUALIZATION ENABLED:")
        print("   🔴 Red spheres = Trajectory targets")
        print("   🟢 Green spheres = Robot end-effector")
        print("   Watch if green follows red!")
        print()
    
    # Evaluation loop
    episode_rewards = []
    episode_lengths = []
    tracking_errors = []
    
    obs = env.reset()
    episode_count = 0
    current_episode_reward = np.zeros(args.num_envs)
    current_episode_length = np.zeros(args.num_envs)
    
    print(f"Running {args.num_episodes} episodes...")
    
    while episode_count < args.num_episodes:
        # Get action from policy
        action, _states = model.predict(obs, deterministic=args.deterministic)
        
        # Step environment
        obs, rewards, dones, infos = env.step(action)
        
        # Track metrics
        current_episode_reward += rewards
        current_episode_length += 1
        
        # Handle episode termination
        for i, done in enumerate(dones):
            if done:
                episode_rewards.append(current_episode_reward[i])
                episode_lengths.append(current_episode_length[i])
                
                episode_count += 1
                print(f"  Episode {episode_count}/{args.num_episodes}: "
                      f"Reward={current_episode_reward[i]:.2f}, "
                      f"Length={int(current_episode_length[i])}")
                
                current_episode_reward[i] = 0
                current_episode_length[i] = 0
                
                if episode_count >= args.num_episodes:
                    break
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Episodes completed: {len(episode_rewards)}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("=" * 70)
    
    # Cleanup
    print("\nCleaning up...")
    env.close()
    simulation_app.close()
    
    print("✓ Evaluation complete!")


if __name__ == "__main__":
    main()
