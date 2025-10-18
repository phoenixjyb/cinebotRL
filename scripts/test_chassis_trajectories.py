#!/usr/bin/env python3
"""
Quick visual test of recorded trajectories requiring chassis movement.

This script loads chassis-required trajectories and visualizes the robot
attempting to track them. Perfect for validating base movement fixes!

Usage:
    # Test top 10 most challenging trajectories with 4 parallel environments
    python scripts/test_chassis_trajectories.py

    # Test specific number
    python scripts/test_chassis_trajectories.py --num 20 --envs 8
    
    # Use trained model
    python scripts/test_chassis_trajectories.py --checkpoint logs/final_model.zip
"""

import argparse


# Chassis-required trajectory indices (from analysis)
# These are the top trajectories requiring significant base movement (X change >= 2.0m)
CHASSIS_REQUIRED_INDICES = [
       0,    1,    2,    3,    4,    5,    6,    7,    8,    9,
      10,   11,   12,   13,   14,   15,   16,   17,   18,   19,
      20,   21,   22,   23,   24,   25,   26,   27,   28,   29,
      30,   31,   32,   33,   34,   35,   36,   37,   38,   39,
      40,   41,   42,   43,   44,   45,   46,   47,   48,   49,
      50,   51,   52,   53,   54,   55,   56,   57,   58,   59,
      60,   61,   62,   63,   64,   65,   66,   67,   68,   69,
      70,   71,   72,   73,   74,   75,   76,   77,   78,   79,
      80,   81,   82,   83,   84,   85,   86,   87,   88,   89,
      90,   91,   92,   93,   94,   95,   96,   97,   98,   99,
]  # ... and 419 more indices (truncated for brevity)


def main():
    parser = argparse.ArgumentParser(
        description="Visual test of chassis-requiring trajectories"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Number of trajectories to test (default: 10)"
    )
    parser.add_argument(
        "--envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (optional)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI"
    )
    
    args = parser.parse_args()
    
    # Select trajectory indices
    trajectory_indices = CHASSIS_REQUIRED_INDICES[:args.num]
    
    print("\n" + "="*80)
    print("🎬 CHASSIS-REQUIRING TRAJECTORY VISUAL TEST")
    print("="*80)
    print(f"Testing {len(trajectory_indices)} trajectories across {args.envs} parallel environments")
    print(f"Trajectory indices: {trajectory_indices}")
    print(f"Model: {'Random policy' if args.checkpoint is None else args.checkpoint}")
    print("="*80 + "\n")
    
    # Import Isaac Lab (must be after arg parsing due to AppLauncher)
    import sys
    from pathlib import Path
    
    # Add project to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Configure and launch Isaac Sim
    from omni.isaac.lab.app import AppLauncher
    
    app_launcher = AppLauncher({"headless": args.headless})
    simulation_app = app_launcher.app
    
    # Now import Isaac Lab modules
    import torch
    from src.rl_platform.tasks.mobile_mm.env import MobileMMTrackEEEnv, MobileMMTrackEEEnvCfg
    from src.rl_platform.tasks.mobile_mm.config import TrajectoryConfig
    
    # Create environment config
    env_cfg = MobileMMTrackEEEnvCfg()
    env_cfg.scene.num_envs = args.envs
    env_cfg.episode_length_s = 30.0  # Longer episodes for complex trajectories
    
    # Configure for multi-recorded trajectories with chassis filtering
    env_cfg.task_config.trajectory = TrajectoryConfig(
        type="multi_recorded",
        trajectory_dir="trajectoryToLearn/world_json",
        trajectory_pattern="**/*.json",
        trajectory_filter_indices=trajectory_indices,
        max_trajectories=None,  # Use all filtered
    )
    
    # Create environment
    print("🔧 Creating environment...")
    env = MobileMMTrackEEEnv(cfg=env_cfg)
    print("✓ Environment created\n")
    
    # Load model if provided
    model = None
    if args.checkpoint is not None:
        print(f"📦 Loading model from {args.checkpoint}...")
        try:
            from stable_baselines3 import PPO
            model = PPO.load(args.checkpoint, device="cuda")
            print("✓ Model loaded\n")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print("  Using random policy instead\n")
            model = None
    
    # Run visual test
    print("🎮 Starting visual test (Press Ctrl+C to stop)...\n")
    
    obs, _ = env.reset()
    step_count = 0
    episode_count = 0
    
    try:
        while episode_count < 100:  # Run many episodes
            # Get actions
            if model is not None:
                with torch.no_grad():
                    actions_np, _ = model.predict(obs.cpu().numpy(), deterministic=True)
                    actions = torch.from_numpy(actions_np).to(env.device)
            else:
                # Random actions (small for safety)
                actions = torch.randn_like(env.action_manager.action) * 0.1
            
            # Step environment
            obs, rewards, dones, truncated, info = env.step(actions)
            step_count += 1
            
            # Print diagnostics every 50 steps
            if step_count % 50 == 0:
                avg_reward = rewards.mean().item()
                
                # Extract base diagnostics if available
                base_info = ""
                if "base_vel_x" in info["extras"]:
                    vx = info["extras"]["base_vel_x"].mean().item()
                    vy = info["extras"]["base_vel_y"].mean().item()
                    vyaw = info["extras"]["base_vel_yaw"].mean().item()
                    base_info = f"| Base: vx={vx:+.2f} vy={vy:+.2f} ω={vyaw:+.2f}"
                
                print(f"Step {step_count:6d} | Reward: {avg_reward:8.2f} {base_info}")
            
            # Handle episode completion
            if dones.any():
                completed = dones.nonzero(as_tuple=True)[0]
                for env_id in completed:
                    episode_count += 1
                    final_reward = rewards[env_id].item()
                    
                    distance_info = ""
                    if "base_distance_traveled" in info["extras"]:
                        dist = info["extras"]["base_distance_traveled"][env_id].item()
                        distance_info = f", base traveled: {dist:.2f}m"
                    
                    print(f"\n  ✓ Episode {episode_count} complete (env {env_id}) | "
                          f"reward: {final_reward:.2f}{distance_info}\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    
    finally:
        print(f"\n{'='*80}")
        print(f"Test complete! Ran {episode_count} episodes, {step_count} total steps")
        print(f"{'='*80}\n")
        
        # Cleanup
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
