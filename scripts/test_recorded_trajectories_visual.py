#!/usr/bin/env python3
"""
Visual test of recorded trajectories with mobile manipulator.
Tests robot's ability to follow challenging chassis-requiring trajectories.

Usage:
    # Test top 10 chassis-requiring trajectories
    python scripts/test_recorded_trajectories_visual.py --num_trajectories 10 --num_envs 4
    
    # Test specific indices
    python scripts/test_recorded_trajectories_visual.py --indices 0 1 2 3 --num_envs 4
    
    # Load chassis-required indices from file
    python scripts/test_recorded_trajectories_visual.py --use_chassis_indices --max_trajectories 20
"""

import argparse
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_chassis_indices(file_path: str = "data/trajectory_filters/chassis_required_indices.txt", max_indices: int | None = None):
    """Load chassis-required indices from generated file."""
    import re
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract the Python list
    match = re.search(r'CHASSIS_REQUIRED_INDICES = \[(.*?)\]', content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find CHASSIS_REQUIRED_INDICES in {file_path}")
    
    # Parse indices
    indices_str = match.group(1)
    indices = [int(x.strip()) for x in indices_str.replace('\n', ' ').split(',') if x.strip()]
    
    if max_indices is not None:
        indices = indices[:max_indices]
    
    print(f"Loaded {len(indices)} chassis-required trajectory indices")
    return indices


def create_test_env(args):
    """Create environment with recorded trajectory configuration."""
    import torch
    from omni.isaac.lab.app import AppLauncher
    
    # Configure app
    app_launcher = AppLauncher(args_cli={"headless": args.headless})
    simulation_app = app_launcher.app
    
    # Import after app launch
    import omni.isaac.lab.envs.mdp as mdp
    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
    from src.rl_platform.tasks.mobile_mm.env import MobileMMTrackEEEnv, MobileMMTrackEEEnvCfg
    from src.rl_platform.tasks.mobile_mm.config import TrajectoryConfig
    
    # Create environment config
    env_cfg = MobileMMTrackEEEnvCfg()
    
    # Configure for recorded trajectories
    env_cfg.trajectory = TrajectoryConfig(
        type="multi_recorded",
        waypoint_file=None,  # Not used for multi_recorded
        loop_trajectory=True,
    )
    
    # Set trajectory directory and filtering
    env_cfg.trajectory_dir = args.trajectory_dir
    env_cfg.trajectory_pattern = "**/*.json"
    
    # Apply filtering
    if args.use_chassis_indices:
        indices = load_chassis_indices(args.chassis_indices_file, args.max_trajectories)
        env_cfg.trajectory_filter_indices = indices
    elif args.indices is not None:
        env_cfg.trajectory_filter_indices = args.indices
    elif args.num_trajectories is not None:
        # Load top N chassis-requiring trajectories
        indices = load_chassis_indices(args.chassis_indices_file, args.num_trajectories)
        env_cfg.trajectory_filter_indices = indices
    else:
        env_cfg.trajectory_filter_indices = None
    
    # Set number of environments
    env_cfg.scene.num_envs = args.num_envs
    
    # Increase episode length for complex trajectories
    env_cfg.episode_length_s = args.episode_length
    
    # Enable visualization
    env_cfg.viewer.eye = (5.0, 5.0, 3.0)  # Better view for base movement
    env_cfg.viewer.lookat = (0.0, 0.0, 1.0)
    
    print("\n" + "="*80)
    print("RECORDED TRAJECTORY VISUAL TEST")
    print("="*80)
    print(f"Trajectory directory: {args.trajectory_dir}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Episode length: {args.episode_length}s")
    
    if env_cfg.trajectory_filter_indices is not None:
        print(f"Filtered to {len(env_cfg.trajectory_filter_indices)} trajectories")
        print(f"Trajectory indices: {env_cfg.trajectory_filter_indices[:10]}{'...' if len(env_cfg.trajectory_filter_indices) > 10 else ''}")
    else:
        print(f"Using all available trajectories")
    
    print("="*80 + "\n")
    
    # Create environment
    env = MobileMMTrackEEEnv(cfg=env_cfg)
    
    return env, simulation_app


def run_visual_test(env, args):
    """Run visual test with recorded trajectories."""
    import torch
    
    print("\n🎬 Starting visual test...")
    print("Press Ctrl+C to stop\n")
    
    # Reset environment
    obs, _ = env.reset()
    
    episode_count = 0
    step_count = 0
    
    try:
        while episode_count < args.num_episodes:
            # Random actions initially (or you can load a trained model)
            if args.model_checkpoint is not None:
                # TODO: Load trained model and get actions
                print(f"Loading model from {args.model_checkpoint}")
                # actions = model.predict(obs)
                actions = torch.randn_like(env.action_space.sample())  # Placeholder
            else:
                # Random baseline
                actions = torch.randn_like(env.action_space.sample()) * 0.1
            
            # Step environment
            obs, rewards, dones, truncated, info = env.step(actions)
            step_count += 1
            
            # Print diagnostics
            if step_count % 100 == 0:
                # Extract base diagnostics
                if "base_vel_x" in info["extras"]:
                    base_vel_x = info["extras"]["base_vel_x"].mean().item()
                    base_vel_y = info["extras"]["base_vel_y"].mean().item()
                    base_vel_yaw = info["extras"]["base_vel_yaw"].mean().item()
                    
                    print(f"Step {step_count:5d} | Avg Reward: {rewards.mean():7.2f} | "
                          f"Base: vx={base_vel_x:+.3f} vy={base_vel_y:+.3f} vyaw={base_vel_yaw:+.3f}")
            
            # Check for episode completion
            if dones.any():
                completed_envs = dones.nonzero(as_tuple=True)[0]
                for env_id in completed_envs:
                    episode_count += 1
                    if episode_count <= args.num_episodes:
                        print(f"\n✓ Episode {episode_count} completed (env {env_id})")
                        print(f"  Final reward: {rewards[env_id]:.2f}")
                        if "base_distance_traveled" in info["extras"]:
                            distance = info["extras"]["base_distance_traveled"][env_id].item()
                            print(f"  Base distance traveled: {distance:.3f}m")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
    
    print(f"\n✓ Visual test complete! Ran {episode_count} episodes, {step_count} steps")


def main():
    parser = argparse.ArgumentParser(
        description="Visual test of mobile manipulator with recorded trajectories"
    )
    
    # Trajectory selection
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default="trajectoryToLearn/world_json",
        help="Directory containing trajectory JSON files"
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=None,
        help="Number of top chassis-requiring trajectories to test"
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific trajectory indices to test (e.g., --indices 0 1 2 3)"
    )
    parser.add_argument(
        "--use_chassis_indices",
        action="store_true",
        help="Use all chassis-required indices from file"
    )
    parser.add_argument(
        "--chassis_indices_file",
        type=str,
        default="data/trajectory_filters/chassis_required_indices.txt",
        help="File containing chassis-required trajectory indices"
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Maximum number of trajectories to load (when using --use_chassis_indices)"
    )
    
    # Environment settings
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4,
        help="Number of parallel environments (for visual comparison)"
    )
    parser.add_argument(
        "--episode_length",
        type=float,
        default=30.0,
        help="Episode length in seconds"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to run per environment"
    )
    
    # Model settings
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (optional, uses random policy if not provided)"
    )
    
    # Rendering
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_trajectories is None and args.indices is None and not args.use_chassis_indices:
        print("⚠ No trajectory selection specified. Using default: top 10 chassis-requiring trajectories")
        args.num_trajectories = 10
    
    # Create environment
    env, simulation_app = create_test_env(args)
    
    # Run test
    try:
        run_visual_test(env, args)
    finally:
        # Cleanup
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
