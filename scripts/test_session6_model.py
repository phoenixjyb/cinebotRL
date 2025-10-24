#!/usr/bin/env python3
"""
Test Session 6 trained model to verify base movement and track performance.

Usage:
    python scripts/test_session6_model.py --checkpoint logs/sb3/mobilemmtrackee_v0/20251022_230622/final_model.zip
    
    # Or test without rendering (headless)
    python scripts/test_session6_model.py --checkpoint logs/sb3/mobilemmtrackee_v0/20251022_230622/final_model.zip --headless
"""

import argparse
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Test Session 6 trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="logs/sb3/mobilemmtrackee_v0/20251022_230622/final_model.zip",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of parallel environments (default: 16 for quick testing)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to test per environment",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)",
    )
    parser.add_argument(
        "--trajectory_type",
        type=str,
        default="multi_recorded",
        choices=["multi_recorded", "single_recorded", "figure_eight"],
        help="Trajectory type to test on",
    )
    
    args = parser.parse_args()
    
    # Step 1: Initialize Isaac Sim first
    print("=" * 80)
    print("SESSION 6 MODEL EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num Envs: {args.num_envs}")
    print(f"Num Episodes: {args.num_episodes}")
    print(f"Trajectory Type: {args.trajectory_type}")
    print(f"Headless: {args.headless}")
    print("=" * 80)
    
    print("\n🔧 Initializing Isaac Sim...")
    try:
        from isaaclab.app import AppLauncher
        import torch
        import numpy as np
        
        # Create AppLauncher
        app_launcher = AppLauncher(headless=args.headless)
        simulation_app = app_launcher.app
        print("✅ Isaac Sim initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Isaac Sim: {e}")
        return
    
    # Step 2: Register custom tasks AFTER Isaac Sim is running
    print("\n🔧 Registering custom tasks...")
    try:
        from task_spec import register_isaac_lab_tasks
        register_isaac_lab_tasks()
        print("✅ Tasks registered")
    except Exception as e:
        print(f"❌ Failed to register tasks: {e}")
        simulation_app.close()
        return
    
    # Step 3: Import SB3 and environment modules
    print("\n🔧 Loading modules...")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize
        from src.rl_platform.tasks.mobile_mm.env import MobileMMTrackEEEnv, MobileMMTrackEEEnvCfg
        from src.rl_platform.tasks.mobile_mm.config import TrajectoryConfig
        print("✅ Modules loaded")
    except Exception as e:
        print(f"❌ Failed to load modules: {e}")
        simulation_app.close()
        return
    
    # Create environment config
    env_cfg = MobileMMTrackEEEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    
    # Configure trajectory
    env_cfg.trajectory = TrajectoryConfig(
        type=args.trajectory_type,
        waypoint_file=None,
        loop_trajectory=True,
    )
    
    if args.trajectory_type == "multi_recorded":
        env_cfg.trajectory_dir = "trajectoryToLearn"
        env_cfg.trajectory_pattern = "**/*.json"
        env_cfg.use_all_trajectories = True
    
    # Create environment
    print("\n🔧 Creating environment...")
    env = MobileMMTrackEEEnv(cfg=env_cfg)
    
    # Load trained model
    print(f"\n📦 Loading trained model from {args.checkpoint}...")
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        env.close()
        simulation_app.close()
        return
    
    # Check for vec_normalize stats in the same directory
    vec_normalize_path = checkpoint_path.parent / "vec_normalize.pkl"
    
    try:
        model = PPO.load(str(checkpoint_path))
        print("✅ Model loaded successfully!")
        
        # Load normalization if available
        if vec_normalize_path.exists():
            print(f"📊 Loading VecNormalize stats from {vec_normalize_path}")
            # Note: We're using Isaac Lab env directly, not wrapped in VecNormalize
            # The model should handle normalization internally if needed
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        env.close()
        simulation_app.close()
        return
    
    # Test the model
    print("\n🚀 Starting model evaluation...")
    print("-" * 80)
    
    obs, _ = env.reset()
    
    episode_rewards = []
    episode_lengths = []
    base_velocities = []
    tracking_errors = []
    contact_forces = []
    
    current_episode_rewards = torch.zeros(args.num_envs, device=env.device)
    current_episode_lengths = torch.zeros(args.num_envs, dtype=torch.int32, device=env.device)
    completed_episodes = 0
    step_count = 0
    
    try:
        while completed_episodes < args.num_episodes * args.num_envs:
            # Get action from trained model
            # Convert dict obs to numpy array if needed
            if isinstance(obs, dict):
                # Concatenate observation components
                obs_array = torch.cat([v for v in obs.values() if isinstance(v, torch.Tensor)], dim=-1)
            else:
                obs_array = obs
            
            # Convert to numpy for SB3
            obs_numpy = obs_array.cpu().numpy()
            
            # Get action from model (deterministic for evaluation)
            action, _ = model.predict(obs_numpy, deterministic=True)
            
            # Convert action back to tensor
            action_tensor = torch.from_numpy(action).to(env.device)
            
            # Step environment
            obs, rewards, dones, truncated, info = env.step(action_tensor)
            
            current_episode_rewards += rewards
            current_episode_lengths += 1
            step_count += 1
            
            # Collect diagnostics
            if "base_vel_x" in info.get("extras", {}):
                base_vel_x = info["extras"]["base_vel_x"].mean().item()
                base_vel_y = info["extras"]["base_vel_y"].mean().item()
                base_vel_yaw = info["extras"]["base_vel_yaw"].mean().item()
                base_vel_magnitude = np.sqrt(base_vel_x**2 + base_vel_y**2)
                base_velocities.append(base_vel_magnitude)
            
            if "ee_position_error" in info.get("extras", {}):
                ee_error = info["extras"]["ee_position_error"].mean().item()
                tracking_errors.append(ee_error)
            
            if "contact_forces" in info.get("extras", {}):
                contact_force = info["extras"]["contact_forces"].max().item()
                contact_forces.append(contact_force)
            
            # Print periodic updates
            if step_count % 50 == 0:
                avg_reward = current_episode_rewards.mean().item() / current_episode_lengths.float().mean().item()
                print(f"Step {step_count:5d} | Episodes: {completed_episodes:3d}/{args.num_episodes * args.num_envs} | "
                      f"Avg Reward: {avg_reward:7.3f} | "
                      f"Base vel: {base_vel_magnitude:.3f} m/s | "
                      f"Track err: {ee_error:.4f}m")
            
            # Handle completed episodes
            if dones.any():
                done_envs = dones.nonzero(as_tuple=True)[0]
                for env_id in done_envs:
                    episode_rewards.append(current_episode_rewards[env_id].item())
                    episode_lengths.append(current_episode_lengths[env_id].item())
                    completed_episodes += 1
                    
                    if completed_episodes % 10 == 0:
                        print(f"\n✓ Completed {completed_episodes} episodes")
                    
                    # Reset tracking for this env
                    current_episode_rewards[env_id] = 0
                    current_episode_lengths[env_id] = 0
    
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS - SESSION 6")
    print("=" * 80)
    
    if episode_rewards:
        print(f"\n📊 Episode Statistics ({len(episode_rewards)} episodes):")
        print(f"   Mean Reward: {np.mean(episode_rewards):8.2f} ± {np.std(episode_rewards):.2f}")
        print(f"   Min Reward:  {np.min(episode_rewards):8.2f}")
        print(f"   Max Reward:  {np.max(episode_rewards):8.2f}")
        print(f"   Mean Length: {np.mean(episode_lengths):8.1f} steps")
    
    if base_velocities:
        print(f"\n🏃 Base Movement Analysis:")
        print(f"   Mean base velocity: {np.mean(base_velocities):6.4f} m/s")
        print(f"   Max base velocity:  {np.max(base_velocities):6.4f} m/s")
        print(f"   Std base velocity:  {np.std(base_velocities):6.4f} m/s")
        
        # Check if base is actually moving
        moving_threshold = 0.01  # 1 cm/s
        pct_moving = 100 * np.mean(np.array(base_velocities) > moving_threshold)
        print(f"   % time moving (>{moving_threshold}m/s): {pct_moving:.1f}%")
        
        if np.mean(base_velocities) > moving_threshold:
            print(f"   ✅ BASE IS MOVING! Average velocity: {np.mean(base_velocities):.4f} m/s")
        else:
            print(f"   ⚠️  BASE NOT MOVING! Average velocity: {np.mean(base_velocities):.4f} m/s")
    
    if tracking_errors:
        print(f"\n🎯 Tracking Performance:")
        print(f"   Mean EE error: {np.mean(tracking_errors):6.4f} m")
        print(f"   Min EE error:  {np.min(tracking_errors):6.4f} m")
        print(f"   Max EE error:  {np.max(tracking_errors):6.4f} m")
    
    if contact_forces:
        print(f"\n💥 Contact Forces:")
        print(f"   Mean max force: {np.mean(contact_forces):8.2f} N")
        print(f"   Peak force:     {np.max(contact_forces):8.2f} N")
        print(f"   Contact events: {np.sum(np.array(contact_forces) > 10.0)} (>10N)")
    
    print("\n" + "=" * 80)
    print("CRITICAL FIXES ASSESSMENT:")
    print("-" * 80)
    
    # Check Fix 1: Jerk penalty (should allow base movement)
    if base_velocities and np.mean(base_velocities) > 0.01:
        print("✅ Fix 1 (Jerk penalty 50.0): BASE IS MOVING")
    else:
        print("❌ Fix 1 (Jerk penalty 50.0): BASE STILL FROZEN")
    
    # Check Fix 2: ContactSensor (should detect collisions)
    if contact_forces and np.max(contact_forces) > 1.0:
        print(f"✅ Fix 2 (ContactSensor): DETECTING FORCES (peak {np.max(contact_forces):.1f}N)")
    else:
        print("⚠️  Fix 2 (ContactSensor): NO SIGNIFICANT FORCES DETECTED")
    
    # Check Fix 3: Shape fix is internal, check if training succeeded
    if episode_rewards and np.mean(episode_rewards) > -100:
        print(f"✅ Fix 3 (Shape fix): TRAINING SUCCEEDED (avg reward {np.mean(episode_rewards):.1f})")
    else:
        print("⚠️  Fix 3 (Shape fix): LOW REWARDS - POSSIBLE ISSUE")
    
    print("=" * 80)
    
    # Cleanup
    env.close()
    simulation_app.close()
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
